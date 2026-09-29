import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ====================================
#  Laplace layer: pole-residue operation is used to calculate the poles and residues of the output
# ====================================

class LT(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, sigma, sigma_scale, pole_scale=True):
        """
        Laplace Transform Layer based on LT_method.pdf

                Args:
                    in_channels:
                    out_channels: 
                    modes1: 
                    a_max:
                    sigma_max:
                    device: 
        """
        super(LT, self).__init__()
        self.modes1 = modes1
        self.in_channels = in_channels
        if pole_scale==True:
            self.scale = (1 / (in_channels * out_channels))
        else:
            self.scale = 1.0
        self.sigma_scale=sigma_scale
        self.num_sigma = sigma

        self.norm=nn.LayerNorm(out_channels)

        self.sigma_i = nn.Parameter(
            (torch.randn(in_channels, self.num_sigma, dtype=torch.float)) * self.sigma_scale,
            requires_grad=True)

        # System parameters (poles and residues)
        self.weights_pole = nn.Parameter(
            self.scale*torch.rand(in_channels, out_channels, 1, self.modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(
            self.scale*torch.rand(in_channels, out_channels, 1, self.modes1, dtype=torch.cfloat))

    def output_PR(self, input_pole, alpha, weights_pole, weights_residue):
        # input_pole (in_channels,1, num_sigma*num_omega)
        # alpha (batch,in_channels,freq)
        # weights_pole (in_channels,out_channels,1,modes)
        # weights_residue (in_channels,out_channels,modes)
        
        term1 = torch.div(1, torch.sub(input_pole, weights_pole))
        # (in_channels,out_channels,num_sigma*num_omega,modes)
        Hw = weights_residue*term1  
        
        output_residue1 = torch.einsum("bik,iokm->bok", alpha, Hw)
        # res1 (batch,outchannel,freq) ifft
        output_residue2 = torch.einsum("bik,iokm->bom", alpha, -Hw)
        # res2 (batch,out_channel,mode) aperiodic part
        return output_residue1, output_residue2
    
    def forward(self, x, grid):
        # Compute orthogonal basis functions [1,1,num_real_bxasis, seq_len]
        # x:[b,c,x] grid [b,x,1]
        batch=x.shape[0]
        channel=x.shape[1]
        basis = torch.exp(torch.einsum("cs,xk->csx", self.sigma_i, grid[0]))
        omega=torch.fft.fftfreq(x.shape[-1], d=grid[0,1].item()-grid[0,0].item()).to(x.device)
        # omega=omega/torch.max(omega)

        input_pole=self.sigma_i.unsqueeze(-1)+omega.reshape(1,1,-1)*2*np.pi*1j
        input_pole=input_pole.reshape(channel,1,-1,1)

        alpha = torch.fft.fft(torch.einsum('bcx,csx->bcsx',x,basis)).reshape(batch, channel, -1)
        # alpha=[batch,channls,s_i*omega_i]

        res1, res2 = self.output_PR(
            input_pole,
            alpha,
            self.weights_pole,
            self.weights_residue
        )

        # periodic
        x1 = torch.fft.ifft(res1.reshape(batch, channel, self.num_sigma, -1), n=x.shape[-1])
        x1 = torch.einsum("bcsx,csx->bcx", torch.real(x1), basis)

        # aperiodic
        term = torch.exp(torch.einsum("iovm,xw->iomx", self.weights_pole, grid[0].type(torch.complex64))) 
        x2 = torch.einsum("bim,iomx->box", res2, term)
        
        return x1/self.num_sigma + torch.real(x2)/x.shape[-1]/self.num_sigma 

class MLP1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, activate="sin", mlp_depth=2):
        super(MLP1D, self).__init__()

        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        if mlp_depth==3:
            self.mlp2 = nn.Conv1d(mid_channels, mid_channels*2, 1)
            self.mlp3 = nn.Conv1d(mid_channels*2, out_channels, 1)
        else:
            self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

        if activate=="gelu":
            self.activate=F.gelu
        elif activate=="relu":
            self.activate=F.relu
        elif activate=="sin":
            self.activate=torch.sin
        else:
            raise ValueError("Unsupported activation function")
        
        self.mlp_depth=mlp_depth


    def forward(self, x):

        x=self.mlp1(x)
        x = self.activate(x)
        x=self.mlp2(x)
        if self.mlp_depth==3:
            x = self.activate(x)
            x = self.mlp3(x)

        return x

class block_GLNO(nn.Module):
    def __init__(self, width, mid_c, modes, dropout, sigma, sigma_scale, pole_scale,activate, mlp_on=True, mlp_depth=2):
        super(block_GLNO, self).__init__()
        self.width = width
        self.modes1 = modes
        self.conv0 = LT(self.width, self.width, self.modes1, sigma, sigma_scale, pole_scale)
        self.dropout_rate=dropout
        self.dropout = nn.Dropout(dropout)
        if mlp_on:
            self.mlp0 = MLP1D(self.width, self.width, mid_c, activate=activate,  mlp_depth=mlp_depth)
        self.w0 = nn.Conv1d(self.width, self.width, 1)

        self.norm=nn.InstanceNorm1d(self.width)
        self.mlp_on = mlp_on

    def forward(self, x, grid, is_last):
        # x0=x
        #[b,c,t]
        # x = self.norm(x.permute(0,2,1)).permute(0,2,1)
        if self.dropout_rate>0:
            x = self.dropout(x) 
        # x=self.norm(x)
        x1=self.conv0(x, grid) 

        if self.mlp_on:
            x1 = self.norm(x1)
            x1 = self.mlp0(x1)

        x2 = self.w0(x)
        x = x1 + x2

        return x

class GLNO1D(nn.Module):
    def __init__(self, config):
        super(GLNO1D, self).__init__()
        self.block=config['blocks']
        self.width = config['width']
        self.modes = config['modes']
        self.device=config['device']
        self.C_in=config['C_in']
        self.C_out=config['C_out']
        self.dropout = config['dropout']
        self.C_mid = config['C_mid']
        self.C_last= config['C_last'] if config.get('C_last') else config['C_mid']
        mlp_depth=config.get('mlp_depth',2)
        self.activate=config['activate']
        for i in range(self.block):
            self.add_module(f'block{i}', block_GLNO(self.width,self.C_mid, self.modes, self.dropout, config['sigma'],config['sigma_scale'],config['pole_scale'],config['activate'], config.get("mlp_on",True), mlp_depth))

        self.p = nn.Linear(self.C_in,self.width) #MLP1D(self.C_in, self.width,128)
        self.last_mode=config.get('last_mode','q')
        if self.last_mode=='linear':
            self.fc1 = nn.Linear(self.width, self.C_last)
            self.fc2 = nn.Linear(self.C_last, self.C_out)
            if self.activate=='sin':
                self.activate=torch.sin
            elif self.activate=='relu':
                self.activate=F.relu
            elif self.activate=='gelu':
                self.activate=F.gelu
            else:   
                raise ValueError("Unsupported activation function")
        elif self.last_mode=='q':
            self.q = MLP1D(self.width, self.C_out, self.C_last, activate=config['activate'], mlp_depth=mlp_depth)    

    def forward(self, data):
        grid=data['grid_x'].unsqueeze(-1).to(self.device)
        x=data['inputs'].unsqueeze(-1).to(self.device)

        x = torch.cat((x, grid), dim=-1)
        
        x = self.p(x)
        x=x.permute(0,2,1)
        for i in range(self.block):
            x = self.__getattr__(f'block{i}')(x,grid,i==self.block-1)
        
        if self.last_mode=='linear':
            x=x.permute(0,2,1)
            x=self.fc1(x)
            x=self.activate(x)
            x=self.fc2(x)
        elif self.last_mode=='q':
            x=self.q(x).permute(0,2,1)
        else:
            raise ValueError("Unsupported last_mode")

        return x


# ====================================
#  Laplace layer: pole-residue operation is used to calculate the poles and residues of the output
# ====================================  

class LT2d(nn.Module):
    def __init__(self, config,in_channels, out_channels):
        super(LT2d, self).__init__()

        self.modes1 = config['modes1']
        self.modes2 = config['modes2']
        self.in_channels=in_channels
        self.num_sigma1=config['num_sigma_i']
        self.num_sigma2=config['num_sigma_j']
        self.num_sigma=self.num_sigma1*self.num_sigma2
        self.scale = (1 / (in_channels*out_channels))
        # self.scale = 2.0
        self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 1,  dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, 1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  self.modes2, dtype=torch.cfloat))
        self.sigma_i = nn.Parameter((torch.rand(in_channels,self.num_sigma1, dtype=torch.float)-0.5)*config['sigma_scale'])#.requires_grad_(False)
        self.sigma_j = nn.Parameter((torch.rand(in_channels, self.num_sigma2, dtype=torch.float)-0.5)*config['sigma_scale'])#.requires_grad_(False)
    
    def output_PR(self, lambda1, lambda2, alpha, weights_pole1, weights_pole2, weights_residue):
        #lambda1=[inc,1,1,s_i*t_1],lambda2=[inc,1,1,s_j*t_2]#s_i*t_1=x,s_j*t_2=y
        #alpha=[btch,c,s_i*t_1,s_j*t_2]
        #weights_pole1=[inc,ouc,mo_i,1],weights_pole2=[inc,ouc,mo_j,1]#mo_i=m mo_j=n
        #res=[inc,ouc,mo_i,mo_j]
        # Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],lambda1.shape[0], lambda2.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.einsum("iomx,iony->iomnxy",torch.sub(lambda1,weights_pole1),torch.sub(lambda2,weights_pole2)))
        Hw=torch.einsum("iomn,iomnxy->iomnxy",weights_residue,term1)
        Pk=Hw  # for ode, Pk=-Hw; for 2d pde, Pk=Hw; for 3d pde, Pk=-Hw; 
        output_residue1=torch.einsum("bixy,iomnxy->boxy", alpha, Hw)
        output_residue2=torch.einsum("bixy,iomnxy->bomn", alpha, Pk)
        return output_residue1,output_residue2

    def forward(self, x, grid_x, grid_y):#[btch,c,t_1,t_2]
        tx=grid_x[0,:,0,0]
        ty=grid_y[0,0,:,0]
        #Compute input poles and resudes by FFT
        dty=(ty[1]-ty[0]).item()  # locatiDon interval
        dtx=(tx[1]-tx[0]).item()  # time interval
        
        term1 = torch.einsum("cs,t->cst",self.sigma_i,tx)#sigma_i=[inc,num_sigma_i],tx=[x_num,1]
        basis1 = torch.exp(term1)
        term2 = torch.einsum("cs,t->cst", self.sigma_j, ty)
        basis2 = torch.exp(term2)
        basis = torch.einsum("cxi,cyj->cxyij",basis1,basis2)#[1,c,s_i,s_j,t_1,t_2]

        scale_x=torch.einsum("bcij,cxyij->bcxyij",x,basis)#[btch,c,s_i,s_j,t_1,t_2]
        alpha = torch.fft.fft2(scale_x, dim=[-2,-1]).reshape(x.shape[0],x.shape[1],-1,scale_x.shape[3],scale_x.shape[5]).reshape(x.shape[0],x.shape[1],scale_x.shape[2]*scale_x.shape[4],-1)
        #alpha=[batch,inc,s_i*t_1,s_j*t_2]
        omega1=torch.fft.fftfreq(tx.shape[0], dtx).to(x.device)*2*np.pi*1j  # location frequency  [t_1]
        omega2=torch.fft.fftfreq(ty.shape[0], dty).to(x.device)*2*np.pi*1j   # time frequency  [t_2]
        omega1=self.sigma_i.reshape(self.in_channels,-1,1)+omega1.reshape(1,1,-1)
        omega2=self.sigma_j.reshape(self.in_channels,-1,1)+omega2.reshape(1,1,-1)
        lambda1=omega1.reshape(self.in_channels,1,1,-1)
        lambda2=omega2.reshape(self.in_channels,1,1,-1)
 
        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2 = self.output_PR(lambda1, lambda2, alpha, self.weights_pole1, self.weights_pole2, self.weights_residue)
        #res1: sigma pole, res2: weight pole
        # Obtain time histories of transient response and steady-state response
        res1=output_residue1.reshape(x.shape[0],x.shape[1],output_residue1.shape[2],scale_x.shape[3],-1).reshape(x.shape[0],x.shape[1],scale_x.shape[2],scale_x.shape[3],-1,scale_x.shape[5])
        x1_ifft = torch.fft.ifft2(res1, s=(x.size(-2), x.size(-1)))
        x1_ifft = torch.real(x1_ifft)
        x1 = torch.einsum("bcxyij,cxyij->bcij",x1_ifft,basis)/self.num_sigma
        term1=torch.einsum("bipf,z->bipz", self.weights_pole1, tx.type(torch.complex64))
        term2=torch.einsum("biqf,x->biqx", self.weights_pole2, ty.type(torch.complex64))
        term3=torch.einsum("bipz,biqx->bipqzx", torch.exp(term1),torch.exp(term2))
        x2=torch.einsum("bimn,iomnzx->bozx", output_residue2,term3)
        x2=torch.real(x2)
        x2=x2/x.size(-1)/x.size(-2)#/self.num_sigma1/self.num_sigma2
        return x1+x2

class MLP2D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, act='sin'):
        super(MLP2D, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.act = act
        # self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        # x=self.dropout(x)
        x = self.mlp1(x)
        if self.act=='sin':
            x = torch.sin(x) 
        else:
            x = F.gelu(x)
        x = self.mlp2(x)
        return x

class block_GLNO_2D(nn.Module):
    def __init__(self, config):
        super(block_GLNO_2D, self).__init__()
        self.width = config['width']
        self.conv0 = LT2d(config, self.width, self.width)
        self.dropout = nn.Dropout(0.1)
        self.mlp0 = MLP2D(self.width, self.width, self.width*2, act=config.get('gelu','sin'))
        # self.mlp1=MLP2D(self.width, self.width, self.width*2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        # self.activate=nn.ReLU()
        # self.norm=nn.LayerNorm(self.width)
        self.norm=nn.InstanceNorm2d(self.width)

    def forward(self, x, grid_x, grid_y, is_last):
        #[b,c,t]
        # x = self.norm(x.permute(0,2,1)).permute(0,2,1)
        # x = self.dropout(x)
        x1 = self.conv0(x, grid_x, grid_y)
        x1=self.norm(x1) 
        x1=self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2 
        return x

class GLNO2D(nn.Module):
    def __init__(self, config):
        super(GLNO2D, self).__init__()

        self.width = config['width']
        self.device = config['device']
        self.C_in = config['C_in']
        self.C_out = config['C_out']
        self.dropout = config['dropout']
        self.block = config['blocks']
        self.act=config.get('gelu','sin')

        self.fc0 = nn.Linear(self.C_in, self.width) 

        for i in range(self.block):
            self.add_module(f'block{i}', block_GLNO_2D(config))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.C_out)

    def forward(self, data):
        x=data['inputs'].to(self.device)
        if len(x.shape)==3:
            x=x.unsqueeze(-1)
        batch_size=x.shape[0]
        size_x=x.shape[1]
        size_y=x.shape[2]
        grid_x=data['grid_x'].view(batch_size,size_x,1,1).repeat(1,1,size_y,1).to(self.device)
        grid_y=data['grid_y'].view(batch_size,1,size_y,1).repeat(1,size_x,1,1).to(self.device)

        x = torch.cat((x, grid_x, grid_y), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.block):
            x = self.__getattr__(f'block{i}')(x,grid_x,grid_y,i==self.block-1)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        if self.act=='sin':
            x = torch.sin(x)
        else:
            x=F.gelu(x)
        x = self.fc2(x)
        return x