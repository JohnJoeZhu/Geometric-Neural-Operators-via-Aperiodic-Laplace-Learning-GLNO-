import sys
import os
import random

from Benchmark.CortexDiffusion.geometry import norm
from model.GLNO.pointnet import FPSPointNetModule
import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import numpy as np
import torch
import torch.nn as nn

from ..geometry import to_basis, from_basis, rotate
from einops.layers.torch import Rearrange

DEBUG=os.getenv('DEBUG_MODE', '0') == '1'

class Laplace_Transform_Layer(nn.Module):
    """
    Laplace Transform with learnable non-linear basis and learnable system poles and residues on spectral domain.

    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal

      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values
    """

    def __init__(self, config):
        super(Laplace_Transform_Layer, self).__init__()
        self.C_inout = config["C_width"]
        self.in_channels = config["C_width"]
        self.out_channels = config["C_width"]
        self.num_sigma = config["glno_sigma"]
        self.num_poles = config["glno_poles"]
        
        self.exp_x2_appr=config['exp_x2_appr']
        self.gaussian=config['gaussian']
        self.gaussian_factor=config.get('gaussian_factor',1)

        self.LT_kmin = config.get('k_min',0)
        self.LT_kmax = config.get('k_max',config['k_eig'])

        self.period_norm=config['glno_period_norm']
        self.aperiod_norm= config['glno_aperiod_norm']
        self.glno_norm=config['glno_norm']
        self.norm_period=NormLayer(self.period_norm,self.C_inout)
        self.norm_aperiod=NormLayer(self.aperiod_norm,self.C_inout)
        self.norm_glno=NormLayer(self.glno_norm,self.C_inout)

        self.scale=config.get("glno_scale",None)
        self.system_poles = nn.Parameter(torch.rand(self.C_inout, self.C_inout, self.num_poles, dtype=torch.cfloat)) #*5-2.5for car
        self.system_residues = nn.Parameter(torch.randn(self.C_inout, self.C_inout, self.num_poles, dtype=torch.cfloat))
        self.sigma = nn.Parameter(-(torch.rand(self.C_inout, self.num_sigma, dtype=torch.float))) #default 0.02
        if self.scale:
            self.system_poles.data *= self.scale
            self.system_residues.data *= self.scale
            self.sigma.data *= self.scale
            
        self.normalize_pole=config["normalize_pole"]
        self.normalize_k=self.LT_kmax
        self.normalize_basis=config.get("normalize_basis",False)
        self.safe_mode=config.get("safe_mode",True)
        self.basis_norm=config.get("basis_norm",1)

    def gaussian_spectral_filter(self, omega, evals, residue):
        """
        高斯核谱滤波方法

        参数:
            omega: 目标频率[c_in,c_out,num_pole]
            sigma: 高斯核带宽[batch,c_in,num_eig]
            evals: 特征值[batch,num_eig]
            evecs: 特征向量[batch,num_vectices,num_eig]
            residue: 留数[batch,c_in,c_out,num_eig]
            中间值：特征向量分量[batch,c_in,c_out,num_pole,num_eig]
            输出：特征向量[batch,c_in,c_out,num_pole,num_eig,num_vectices]

        返回: 近似特征函数，形状为 [n_vertices]
        """
        # 计算目标特征值 λ = ω²
        omega_ = omega ** 2
        batch, _ = evals.shape

        # gaussian kernel weights
        scale = omega_.unsqueeze(0).unsqueeze(-1) - evals.reshape(batch, 1, 1, 1, -1)
        if self.gaussian=='expx2':
            weights = torch.exp(-(scale*self.gaussian_factor)**2)
        elif self.gaussian=='x2':
            weights = scale**2
        elif self.gaussian=='expx':
            weights=torch.exp(-scale)
        else:
            raise ValueError("gaussian_x2 should be expx2/x2/expx")
        
        # normalize weights
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        if self.safe_mode:
            weights= torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
        weights = torch.where(weights < 1e-5, torch.zeros_like(weights), weights)

        efuc = torch.einsum("biopk,biop->biopk", weights, torch.real(residue))
        
        return efuc

    def pole_to_operator(self, pole, evals, evecs, residue, geo_feat, **kwargs):
        """
        参数:
            pole: 复数极点，实部x表示衰减，虚部y表示频率[c_in,c_out,num_pole]
            residue: 留数[batch,c_out,num_eig]
        """
        x, y = pole.real, pole.imag

        # 根据虚部y计算近似特征函数
        approx_func = self.gaussian_spectral_filter(y, evals, residue)#/self.C_inout

        # use Taylor expansion to approximate exponential operator
        term1=torch.einsum("biopk,iop->bok",approx_func,x)
        
        operator0 = torch.einsum("biopk,bnk->bon",approx_func,evecs)
        operator1 = torch.einsum("bok,bnk,bn->bon",term1,evecs,geo_feat)
        operator=operator0-operator1

        if self.exp_x2_appr:
            term2=torch.einsum("biopk,iop->bok",approx_func,x**2)
            operator2 = torch.einsum("bok,bnk,bn->bon",term2,evecs,geo_feat**2)
            operator+=operator2*0.5

        return operator#/self.num_poles

    def PoleRes(self, evals, x_spec, system_poles, system_residues):
        """
        Pole residue calculation

        参数:
            evals: LBO特征值，形状为 (batch,num_eigenvectors)
            sigma: 高斯核，形状为 (in_channels,sigma)
            x_spec: 输入谱系数，形状为 (batch_size, in_channels, num_eigenvectors*sigma)
            system_poles: 系统极点，形状为 (in_channels, out_channels, num_poles)
            system_residues: 系统留数，形状为 (in_channels, out_channels, num_poles)

        返回:
            output_residue1: periodic_signal
            output_residue2: aperiodic_signal
        """
        # batch_size, in_channels, num_eigenvectors = x_spec.shape
        # out_channels = system_poles.shape[1]

        # evals_extended: (1, 1, 1, num_eigenvectors)
        batch_size, in_channels, _ = x_spec.shape

        evals_extended = torch.sqrt(torch.abs(evals)).unsqueeze(1).unsqueeze(-1)

        if DEBUG:
           assert torch.isnan(evals_extended).any()==False, "evals_extended contains NaN"

        sigma = self.sigma.unsqueeze(1).unsqueeze(0)
        pole_in = evals_extended*1j + sigma
        pole_in = pole_in.reshape(batch_size, in_channels, 1, 1, -1)
        # [batch,inc,1,1,k_eig*sigma]

        # system_poles_extended: (in_channels, out_channels, num_poles, 1)
        system_poles_extended = system_poles.unsqueeze(-1).unsqueeze(0)

        # Hw = residue / (eval - pole)
        term1 = torch.div(1, torch.sub(pole_in, system_poles_extended))
        if self.safe_mode:
            term1_real = torch.nan_to_num(term1.real, nan=0.0)
            term1_imag = torch.nan_to_num(term1.imag, nan=0.0)
            term1 = torch.complex(term1_real, term1_imag)
        if DEBUG:
            assert torch.isnan(term1).any()==False, "term1 contains NaN"
        # [batch,inc,outc,numpole,k_eig*sigma]
        Hw = system_residues.unsqueeze(-1).unsqueeze(0) * term1
        # [batch,inc,outc,numpole,k_eig*sigma]
        # print(Hw.shape,x_spec.shape)
        # output_residue1 = α * H(λ) (periodic)
        output_residue1 = torch.einsum("bix,biokx->biox", x_spec, Hw)
        # output_residue2 = α * H(μ) (aperiodic)
        output_residue2 = torch.einsum("bix,biokx->biok", x_spec, -Hw)
        return output_residue1, output_residue2

    def forward(self, x, mass, evals, evecs, geo_feat):
        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        # Transform to spectral
        batch_size = x.shape[0]

        if evals.shape[-1]<self.LT_kmax:
            raise ValueError("evals.shape[-1]<self.LT_kmax")

        if self.normalize_pole:
            evals = (evals / evals[...,self.normalize_k-1].unsqueeze(-1)) * self.normalize_pole

        evals=evals[...,self.LT_kmin:self.LT_kmax]
        evecs=evecs[...,self.LT_kmin:self.LT_kmax]       

        # print(self.sigma.shape,geo_feat.shape)
        basis = torch.exp(-torch.einsum("cs,bn->bcsn", self.sigma, geo_feat))
        if self.normalize_basis:
            basis = basis / torch.norm(basis, dim=-1, keepdim=True) #normalize basis
        x_scale = torch.einsum("bnc,bcsn->bsnc", x, basis)
        x_spec = to_basis(x_scale, evecs, mass)
        # x_spec=[batch,sigma,num_eig,inc]

        if DEBUG:
            assert torch.isnan(x_scale).any()==False, "x_scale contains NaN"
            assert torch.isnan(basis).any()==False, "basis contains NaN"
            assert torch.isnan(x_spec).any()==False, "x_spec contains NaN"

        num_eig = x_spec.shape[-2]
        x_spec = x_spec.permute(0, 2, 1, 3).reshape(batch_size, -1, self.C_inout)
        # x_spec=[batch,num_eig*sigma,inc]
        x_spec_lno = x_spec.transpose(1, 2).to(torch.cfloat)
        # x_spec_lno=[batch,inc,num_eig*sigma]

        # Compute output residues
        output_residue1, output_residue2 = self.PoleRes(
            evals, x_spec_lno, self.system_poles, self.system_residues
        )
        
        if DEBUG:
            assert torch.isnan(output_residue1).any()==False, "output_residue1 contains NaN"
            assert torch.isnan(output_residue2).any()==False, "output_residue2 contains NaN"

        # Transform back to per-vertex
        # output_residue1=[batch,outchannel,num_eig*sigma]
        x1_spec = torch.real(output_residue1).reshape(batch_size, self.C_inout, self.C_inout, num_eig, self.num_sigma).permute(0,4,2,3,1)
        x1 = from_basis(x1_spec, evecs)
        # x1=[batch,sigma,out,num_vec,in]
        x_period = torch.einsum("bsoni,bisn->bno", x1, basis)
        # print(torch.max(x1_spec),torch.max(basis),torch.max(x_period))
        if self.period_norm:
            x_period=self.norm_period(x_period)
        else:
            x_period=x_period/self.num_sigma/self.C_inout
        
        x_aperiod = self.pole_to_operator(self.system_poles, evals, evecs, output_residue2, geo_feat)

        if self.aperiod_norm:
            x_aperiod=self.norm_aperiod(x_aperiod.transpose(1, 2))
        else:
            x_aperiod=x_aperiod.transpose(1, 2)/self.num_sigma/self.C_inout #原来没有这个

        x_glno = x_period + x_aperiod
        # x_glno = torch.nan_to_num(x_glno, nan=0.0)
        if DEBUG:
            assert torch.isnan(x_glno).any()==False

        if self.glno_norm:
            x_glno=self.norm_glno(x_glno)
            
        return x_glno  ##分开来并不能变好
    
class MLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''
    def __init__(self, layer_sizes, dropout=False, norm='layer', act='relu', name="MLP"):
        super(MLP, self).__init__()
        activation = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'tanh': nn.Tanh(),
                      'sigmoid': nn.Sigmoid(), 'leaky_relu': nn.LeakyReLU(),
                      'elu': nn.ELU(), 'softplus': nn.Softplus()}.get(act, None)
        
        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=dropout)
                )

            self.add_module(
                name + "_layer_{:03d}".format(i),
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
            )

            if not is_last and norm: ##增加了not is_last
                self.add_module(
                    name + f"_{norm}norm_{i:03d}",
                    NormLayer(norm, layer_sizes[i + 1])
                )   #原来是layer

            # Nonlinearity
            # (but not on the last layer)
            if not is_last and activation is not None:
                self.add_module(
                    name + "_act_{:03d}".format(i),
                    activation
                )

from torch_geometric.nn.conv.edge_conv import EdgeConv
class EdgeConvNet(nn.Module):
    """A simple GNN with configurable hidden layer sizes. Using MLP as MLP"""

    def __init__(self, in_channels, out_channels, hidden_dims, dropout=True,norm='layer', activation='relu'):
        super(EdgeConvNet, self).__init__()
        # print(in_channels,out_channels,hidden_dims,dropout,norm,activation)
        self.mlp = MLP([2*in_channels]+hidden_dims+[out_channels], dropout=dropout, norm=norm, act=activation, name="edge_conv_mlp")
        self.edge_conv = EdgeConv(nn=self.mlp, aggr='sum')

    def forward(self, x, edge_index=None):
        if edge_index is None:
            return x

        # KNN
        if x.shape[0] == 1:
            # edge_index_ = torch.tensor(edge_index[0],dtype=torch.long)
            x_out = self.edge_conv(x[0], edge_index[0]).unsqueeze(0)
        else:
            x_out = []
            for i in range(x.shape[0]):
                edge_index_ = torch.tensor(edge_index[i], dtype=torch.long).to(x.device)
                x_out.append(self.edge_conv(x[i], edge_index_))
            x_out = torch.stack(x_out)

        return x_out

class NormLayer(nn.Module):
    def __init__(self, method, num_features, affine=True):
        super(NormLayer, self).__init__()
        self.num_features=num_features
        if method == 'layer':
            self.norm = nn.LayerNorm(num_features, elementwise_affine=affine)
        elif method == 'batch':
            self.norm = nn.Sequential(
                        Rearrange('b n c -> b c n'),
                        nn.BatchNorm1d(num_features, track_running_stats=True, affine=affine),
                        Rearrange('b c n -> b n c')
                    )
        elif method == 'instance':
            self.norm=nn.Sequential(
                        Rearrange('b n c -> b c n'),
                        nn.InstanceNorm1d(num_features, affine=affine),
                        Rearrange('b c n -> b n c')
                    )
        elif method == 'group':
            self.norm=nn.Sequential(
                        Rearrange('b n c -> b c n'),
                        nn.GroupNorm(num_groups=16, num_channels=num_features, affine=affine),
                        Rearrange('b c n -> b n c')
                    )
        elif method is None:
            self.norm=None
        else:
            raise ValueError("invalid norm type")

    def forward(self, x):
        if x.shape[-1] != self.num_features and x.shape[-2] == self.num_features:
            x = x.transpose(-1, -2)
        elif x.shape[-1] != self.num_features:
            raise ValueError("input feature dimension does not match num_features")
        flag=False
        if len(x.shape)<3:
            x = x.unsqueeze(0)
            flag=True
        
        if self.norm is not None:
            x = self.norm(x)
        if flag:
            x = x.squeeze(0)
        return x

class GLNOBlock(nn.Module):
    def __init__(self, idx, config):
        super(GLNOBlock, self).__init__()

        self.C_width = config["C_width"]
        self.idx=idx

        # Operator
        self.LT = Laplace_Transform_Layer(config=config)

        self.learned_geo_feat=config['learned_geo_feat']
        if self.learned_geo_feat:
            self.geo_feat_dim=config['geo_feat_dim']
            self.geo_hidden_layers=config.get('geo_feat_hidden_layers',[self.C_width*2,self.C_width*2])
            self.geo_dropout=config.get('geo_feat_dropout',0.0)
            self.geo_norm=config.get('geo_feat_norm',None)
            self.geo_activation=config.get('geo_feat_activation',None)
            self.mlp_geo=MLP([self.geo_feat_dim]+self.geo_hidden_layers+[1], dropout=self.geo_dropout, norm=self.geo_norm,act=self.geo_activation)
            self.last_geo_norm=NormLayer(config.get('geo_feat_last_norm',None),1)

        self.high_fre=config['high_fre']
        if self.high_fre:
            self.k_eig_high_fre=config['k_eig_high_fre']
            self.high_fre_hidden_dims=config.get("high_fre_hidden_layers",[self.C_width])
            self.high_fre_dropout=config.get("high_fre_dropout",0.0)
            self.high_fre_norm=config.get("high_fre_norm",None)
            self.high_fre_activation=config.get("high_fre_activation",None)
            self.mlp_high=MLP([self.C_width]+ self.high_fre_hidden_dims + [self.C_width], dropout=self.high_fre_dropout, norm=self.high_fre_norm,act=self.high_fre_activation)

        self.connect_method = config["connect_method"]

        if self.connect_method:
            self.mlp_hidden_dims = config.get("mlp_hidden_layers",[self.C_width,self.C_width])
            self.mlp_dropout = config.get("mlp_dropout",0.0)
            self.mlp_norm = config.get("mlp_norm",None)
            if config.get('mlp_norm_last_disable',None)==True and idx < config['blocks']:
                self.mlp_norm=None
            self.mlp_activation = config.get("mlp_activation",None)
            self.MLP_C = 2* self.C_width+1
            # MLPs
            if self.connect_method=='mlp':
                self.mlp = MLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.mlp_dropout, norm=self.mlp_norm,act=self.mlp_activation)
            elif self.connect_method=='gnn':
                self.edge_conv = EdgeConvNet(self.MLP_C, self.C_width, hidden_dims=self.mlp_hidden_dims, dropout=self.mlp_dropout, norm=self.mlp_norm,activation=self.mlp_activation)
            else:
                raise ValueError("invalid connect_method")

        self.norm_feature=NormLayer(config["norm_feature"], self.MLP_C)        
        
        self.skip=config["skip"]

        
    def forward(self, x_in, mass, evals, evecs, geo_feat, pos=None, edges=None):
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {e}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width))
        if self.learned_geo_feat:
            if len(geo_feat.shape)==2:
                geo_feat=geo_feat.unsqueeze(-1)
            geo_feat=self.mlp_geo(geo_feat)
            if self.last_geo_norm:
                geo_feat=self.last_geo_norm(geo_feat)
            geo_feat=geo_feat.squeeze(-1)

        x_glno = self.LT(x_in, mass, evals, evecs, geo_feat)

        if self.high_fre:
            if evecs.shape[-1]<self.k_eig_high_fre:
                raise ValueError("k_eig_high_fre should be less than or equal to the number of eigenvectors")
            evecs=evecs[...,:self.k_eig_high_fre]
            x_low=from_basis(to_basis(x_in.unsqueeze(1), evecs, mass), evecs).squeeze(1)
            x_high=x_in-x_low
            x_high=self.mlp_high(x_high)
            x_glno+=x_high
        
        if DEBUG:
            assert torch.isnan(x_glno).any()==False, "LT output contains NaN"
        
        if len(geo_feat.shape)==2:
            geo_feat=geo_feat.unsqueeze(-1)
        feature_combined = torch.cat((x_in, x_glno, geo_feat), dim=-1) #high还是in好像也no difference， xin比较good #这里加x_high是useless
        feature_combined = self.norm_feature(feature_combined)         
        
        # Apply the connect method
        if self.connect_method=='mlp':
            x0_out = self.mlp(feature_combined)
        elif self.connect_method=='gnn':
            if edges is None:
                raise ValueError("edges must be provided for gnn connect_method")
            x0_out = self.edge_conv(feature_combined, edges)
        else:
            raise ValueError("invalid connect_method")

        # Skip connection
        if self.skip:
            x0_out = x0_out + x_in

        return x0_out


class GLNONet(nn.Module):
    def __init__(self, config):
        """
        Construct a GLNONet.

        Parameters loaed from config:
            C_in (int):                     input dimension
            C_out (int):                    output dimension
            last_activation (str)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (str):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            """

        super(GLNONet, self).__init__()

        # Load parameters
        self.device=config["device"]
        
        self.C_in = config["C_in"]
        self.C_width = config["C_width"]
        self.N_block = config["blocks"]
        self.C_out = config["C_out"]
        self.vertices_dim = config["vertices_dim"]
        self.rotate = config["rotate"]
        self.geo_pre = config.get("geo_preprocess",None)
        self.k_eig=config['k_eig']

        if config["last_activation"] is None:
            self.last_activation = None
        elif config["last_activation"] == "log_softmax":
            self.last_activation = nn.LogSoftmax(dim=-1)
        else:
            raise ValueError("invalid setting for last_activation")
        
        self.outputs_at = config["outputs_at"]
        if self.outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']:
            raise ValueError("invalid setting for outputs_at")
        
        ## Set up the network
        self.encoder_method=config.get('encoder',None)
        if self.encoder_method is None:
            self.encoder = nn.Linear(self.C_in, self.C_width)
        else:
            self.encoder_dropout=config["encoder_dropout"]
            self.encoder_norm=config["encoder_norm"]
            self.encoder_hidden_dims=config["encoder_hidden_layers"]
            self.encoder_activation=config["encoder_activation"]
            if config["encoder"]=='mlp':
                self.encoder = MLP([self.C_in] + self.encoder_hidden_dims + [self.C_width], dropout=self.encoder_dropout, norm=self.encoder_norm,act=self.encoder_activation)
            elif config["encoder"]=='gnn':
                self.encoder = EdgeConvNet(self.C_width, self.C_width, hidden_dims=self.encoder_hidden_dims, dropout=self.encoder_dropout,norm=self.encoder_norm,act=self.encoder_activation)
            else:
                raise ValueError("invalid encoder method")
        
        if config["decoder"]=='mlp':
            self.decoder_dropout=config["decoder_dropout"]
            self.decoder_norm=config["decoder_norm"]
            self.decoder_hidden_dims=config["decoder_hidden_layers"]
            self.decoder_activation=config["decoder_activation"]
            self.decoder = MLP([self.C_width*self.N_block] + self.decoder_hidden_dims + [self.C_out], dropout=self.decoder_dropout, norm=self.decoder_norm,act=self.decoder_activation)
        elif config['decoder'] is None:
           self.decoder=nn.Linear(self.C_width, self.C_out)
        else: 
            raise ValueError("invalid decoder method")
        
        self.blocks = []
        for i_block in range(self.N_block):
            block = GLNOBlock(i_block,config=config)
            self.blocks.append(block)
            self.add_module("block_" + str(i_block), self.blocks[-1])

    def forward(self, data):
        """
        In the notation in this document, dimension are:
            - C: channel dimension (C_in/C_out on construction)
            - N: the number of vertices/points, which CAN be different for each forward pass
            - B: is an OPTIONAL batch dimension
            - K/K_EIG: is the number of eigenvalues used for spectral acceleration
            - V is vertices_dim at most 3

        Parameters:
            data should be a dictionary with the following keys:
                vertice (required):   tensor of vertex positions, dimension [N,V] or [B,N,V]
                input:     tensor of input features, dimension [N,C] or [B,N,C]
                faces:     tensor of face indices, dimension [F,3] or [B,F,3]
                edges:     tensor of edge indices, dimension [E,2] or [B,E,2]
                mass (required):      Mass vector, dimension [N] or [B,N]
                evals (required):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
                evecs (required):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
                geo_feat (required):   tensor of geometric features, dimension [N,C] or [B,N,C] (C>0 if learned geometric features block is used)
            
        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """
        ## Real data
        x_in=None
        if "input" in data:
            x_in=data["input"].to(self.device)
        if "mass" in data:
            mass = data["mass"].to(self.device)
        else:
            raise ValueError("input data must contain'mass' key")
        if "evals" in data:
            evals = data["evals"].to(self.device)
            if evals.shape[-1]<self.k_eig:
                raise ValueError("input evals must have at least k_eig elements")
        else:
            raise ValueError("input data must contain 'evals' key")
        if "evecs" in data:
            evecs = data["evecs"].to(self.device)
        else:
            raise ValueError("input data must contain 'evecs' key")
        if "geo_feat" in data:
            geo_feat = data["geo_feat"].to(self.device)
            if self.geo_pre=='exp':
                geo_feat=torch.exp(-geo_feat)
            elif self.geo_pre=='max':
                geo_feat=geo_feat/torch.max(geo_feat,dim=-1,keepdim=True)
            elif self.geo_pre is not None:
                raise ValueError("geometry feature preprocessing method not recognized")
        else:
            raise ValueError("input data must contain 'geo_feat' key")
        
        edges,faces,vertices=None,None,None
        if "edges" in data:
            edges = data["edges"].to(self.device)
        if "faces" in data:
            faces = data["faces"].to(self.device)
        if "vertices" in data:
            vertices = data["vertices"].to(self.device)
            vertices = vertices[...,:self.vertices_dim]

            if self.rotate:
                vertices=rotate(vertices)
            if x_in is not None:
                x_in=torch.cat([x_in,vertices],dim=-1)
            else:
                x_in=vertices

        if x_in is None:
            raise ValueError("input data must contain 'input' or 'vertices' key")

        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in:
            raise ValueError(
                "DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in, x_in.shape[-1]))

        if len(x_in.shape) == 2: # add a batch dim
            appended_batch_dim = True
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            evals = evals.unsqueeze(0)
            evecs = evecs.unsqueeze(0)
            geo_feat = geo_feat.unsqueeze(0)
            if vertices is not None: vertices = vertices.unsqueeze(0)
            if edges is not None: edges = edges.unsqueeze(0)
            if faces is not None: faces = faces.unsqueeze(0)
        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        else:
            raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")

        ## Forward pass through the network
        if DEBUG:
            assert torch.isnan(x_in).any() == False, "input contains NaNs"
            assert torch.isnan(mass).any() == False, "mass contains NaNs"
            assert torch.isnan(evals).any() == False, "evals contains NaNs"
            assert torch.isnan(evecs).any() == False, "evecs contains NaNs"
            assert torch.isnan(geo_feat).any() == False, "geo_feat contains NaNs"
        
        x = self.encoder(x_in)
        if DEBUG:
            assert torch.isnan(x).any() == False, "encoder output contains NaNs"
        for b in self.blocks:
            x = b(x, mass, evals, evecs, geo_feat, vertices, edges)
        if DEBUG:
            assert torch.isnan(x).any() == False, "block output contains NaNs"
        x = self.decoder(x)
        if DEBUG:
            assert torch.isnan(x).any() == False, "decoder output contains NaNs"

        ## Remap output to requested output type
        if self.outputs_at == 'vertices':
            x_out = x

        elif self.outputs_at == 'edges':
            if edges is None:
                raise ValueError("edges must be provided for outputs_at='edges'")
            # Remap to edges
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)

        elif self.outputs_at == 'faces':
            if faces is None:
                raise ValueError("faces must be provided for outputs_at='faces'")
            # Remap to faces
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)

        elif self.outputs_at == 'global_mean':
            # Produce a single global mean ouput using a weighted mean according to the point mass/area which is discretization-invariant.
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)

        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out

