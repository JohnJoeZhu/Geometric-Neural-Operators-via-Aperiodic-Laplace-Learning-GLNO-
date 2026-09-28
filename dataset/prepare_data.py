import shutil
import os
import sys
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import potpourri3d as pp3d

import diffusion_net
from diffusion_net.utils import toNP
from tqdm import tqdm

class CylinderFlowBase(Dataset):
    def __init__(self, root_dir, data=None, mode='train', k_eig=128, use_cache=True, op_cache_dir=None, **kwargs):
        self.mode = mode
        self.all_data = data
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-vertex 
        self.edges_list = []
        self.curv_list = []
        self.x_list = []

        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            print(os.path.exists(train_cache),train_cache)
            val_cache=os.path.join(self.cache_dir, "val.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            if mode=='train':
                load_cache = train_cache
            elif mode=='val':
                load_cache = val_cache
            elif mode=='test':
                load_cache = test_cache
            else:
                raise ValueError("Invalid mode: {}".format(mode))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.x_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.labels_list, self.disnorm_list = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        for data in self.all_data:
            x=data.x
            label=data.y
            pos=data.pos
            zeros=torch.zeros(pos.shape[0])
            pos=torch.cat([pos,zeros.unsqueeze(-1)],dim=-1)
            faces=data.cell
            assert faces.shape[-1]==3

            self.verts_list.append(pos)
            self.faces_list.append(faces)
            self.labels_list.append(label)
            self.x_list.append(x)

        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.disnorm_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            print("caching to", self.cache_dir)
            train_cache = os.path.join(self.cache_dir, "train.pt")
            val_cache=os.path.join(self.cache_dir, "val.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            if mode=='train':
                load_cache = train_cache
            elif mode=='val':
                load_cache = val_cache
            elif mode=='test':
                load_cache = test_cache
            else:
                raise ValueError("Invalid mode: {}".format(mode))
            
            torch.save((self.verts_list, self.x_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.labels_list, self.disnorm_list), load_cache)   

    def __len__(self):
        return len(self.verts_list)
    
    def __getitem__(self, idx):
        return Data(pos=self.verts_list[idx], x=self.x_list[idx], faces=self.faces_list[idx], 
                    mass=self.massvec_list[idx], L=self.L_list[idx], evals=self.evals_list[idx],
                    evecs=self.evecs_list[idx], y=self.labels_list[idx], edge_index=self.edges_list[idx],
                    curv=self.disnorm_list[idx])


class PoissonBase(Dataset):
    def __init__(self, root_dir, data=None, mode='train', k_eig=128, use_cache=True, op_cache_dir=None, **kwargs):
        self.mode = mode
        self.all_data = data
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-vertex 
        self.edges_list = []
        self.disnorm_list = []
        self.x_list = []

        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            val_cache=os.path.join(self.cache_dir, "val.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            if mode=='train':
                load_cache = train_cache
            elif mode=='val':
                load_cache = val_cache
            elif mode=='test':
                load_cache = test_cache
            else:
                raise ValueError("Invalid mode: {}".format(mode))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.x_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.labels_list,self.edges_list, self.disnorm_list = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        for data in self.all_data:
            x=data.x
            label=data.y
            pos=data.pos
            zeros=torch.zeros(pos.shape[0])
            pos=torch.cat([pos,zeros.unsqueeze(-1)],dim=-1)
            faces=data.cell
            assert faces.shape[-1]==3

            self.verts_list.append(pos)
            self.faces_list.append(faces)
            self.labels_list.append(label)
            self.x_list.append(x)

        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list,self.edges_list, self.disnorm_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            print("caching to", self.cache_dir)
            train_cache = os.path.join(self.cache_dir, "train.pt")
            val_cache=os.path.join(self.cache_dir, "val.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            if mode=='train':
                load_cache = train_cache
            elif mode=='val':
                load_cache = val_cache
            elif mode=='test':
                load_cache = test_cache
            else:
                raise ValueError("Invalid mode: {}".format(mode))
            
            torch.save((self.verts_list, self.x_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.labels_list, self.edges_list, self.disnorm_list), load_cache)   

    def __len__(self):
        return len(self.verts_list)
    
    def __getitem__(self, idx):
        return self.verts_list[idx], self.x_list[idx], self.faces_list[idx], \
               self.massvec_list[idx], self.evals_list[idx], \
               self.evecs_list[idx], self.labels_list[idx], self.edges_list[idx], \
               self.disnorm_list[idx]
    


class DeformingPlateBase(Dataset):
    def __init__(self, root_dir, data=None, mode='train', k_eig=128, use_cache=True, op_cache_dir=None, **kwargs):
        self.mode = mode
        self.all_data = data
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-vertex 
        self.edges_list = []
        self.disnorm_list = []
        self.x_list = []

        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            val_cache=os.path.join(self.cache_dir, "val.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            if mode=='train':
                load_cache = train_cache
            elif mode=='val':
                load_cache = val_cache
            elif mode=='test':
                load_cache = test_cache
            else:
                raise ValueError("Invalid mode: {}".format(mode))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.x_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.labels_list,self.edges_list, self.disnorm_list = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        for data in self.all_data:
            if mode=='train':
                x=data[0].x
                label=data[0].y
                pos=data[0].world_pos
                faces=data[0].cell
            else:
                x=data.x
                label=data.y
                pos=data.world_pos
                faces=data.cell
            assert faces.shape[-1]==3

            self.verts_list.append(pos)
            self.faces_list.append(faces)
            self.labels_list.append(label)
            self.x_list.append(x)

        # Precompute operators
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list,self.edges_list, self.disnorm_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            print("caching to", self.cache_dir)
            train_cache = os.path.join(self.cache_dir, "train.pt")
            val_cache=os.path.join(self.cache_dir, "val.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            if mode=='train':
                load_cache = train_cache
            elif mode=='val':
                load_cache = val_cache
            elif mode=='test':
                load_cache = test_cache
            else:
                raise ValueError("Invalid mode: {}".format(mode))
            
            torch.save((self.verts_list, self.x_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.labels_list, self.edges_list, self.disnorm_list), load_cache)   

    def __len__(self):
        return len(self.verts_list)
    
    def __getitem__(self, idx):
        return self.verts_list[idx], self.x_list[idx], self.faces_list[idx], \
               self.massvec_list[idx], self.evals_list[idx], \
               self.evecs_list[idx], self.labels_list[idx], self.edges_list[idx], \
               self.disnorm_list[idx]


class CarCFDBase(Dataset):
    def __init__(self, root_dir, data=None, mode='train', k_eig=128, use_cache=True, op_cache_dir=None, **kwargs):
        self.mode = mode
        self.all_data = data
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-vertex 
        self.edges_list = []
        self.disnorm_list = []
        self.x_list = []

        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            val_cache=os.path.join(self.cache_dir, "val.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            if mode=='train':
                load_cache = train_cache
            elif mode=='val':
                load_cache = val_cache
            elif mode=='test':
                load_cache = test_cache
            else:
                raise ValueError("Invalid mode: {}".format(mode))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.x_list, self.massvec_list, self.evals_list, self.evecs_list, self.labels_list, self.disnorm_list = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        for data in tqdm(self.all_data):
            x=data.x
            label=data.y
            pos=data.pos
            surf=data.surf

            pos=pos[surf]
            x=x[surf]
            label=label[surf]

            self.verts_list.append(pos)
            self.labels_list.append(label)
            self.x_list.append(x)

        # Precompute operators
        a, self.massvec_list, b, self.evals_list, self.evecs_list, self.disnorm_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            print("caching to", self.cache_dir)
            train_cache = os.path.join(self.cache_dir, "train.pt")
            val_cache=os.path.join(self.cache_dir, "val.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            if mode=='train':
                load_cache = train_cache
            elif mode=='val':
                load_cache = val_cache
            elif mode=='test':
                load_cache = test_cache
            else:
                raise ValueError("Invalid mode: {}".format(mode))
            
            torch.save((self.verts_list, self.x_list, self.massvec_list, self.evals_list, self.evecs_list, self.labels_list, self.disnorm_list), load_cache)   

    def __len__(self):
        return len(self.verts_list)
    
    def __getitem__(self, idx):
        return self.verts_list[idx], self.x_list[idx], \
               self.massvec_list[idx], self.evals_list[idx], \
               self.evecs_list[idx], self.labels_list[idx],  \
               self.disnorm_list[idx]


class AIRFOILBase(Dataset):
    def __init__(self, root_dir, data=None, mode='train', k_eig=128, use_cache=True, op_cache_dir=None, **kwargs):
        self.mode = mode
        self.all_data = data
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-vertex
        self.edges_list = []
        self.disnorm_list = []
        self.x_list = []

        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            val_cache = os.path.join(self.cache_dir, "val.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            if mode == 'train':
                load_cache = train_cache
            elif mode == 'val':
                load_cache = val_cache
            elif mode == 'test':
                load_cache = test_cache
            else:
                raise ValueError("Invalid mode: {}".format(mode))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                self.verts_list, self.x_list, self.labels_list = torch.load(
                    load_cache)
                self.frames, self.mass, self.L, self.evals, self.evecs, self.dis_norm = torch.load(
                    os.path.join(self.cache_dir, "global.pt"))
                return
            print("  --> dataset not in cache, repopulating")

        faces=self.all_data[0].cell
        for data in tqdm(self.all_data):
            x = data.x
            label = data.y
            pos=data.pos

            self.verts_list.append(pos)
            self.labels_list.append(label)
            self.x_list.append(x)

        # Precompute operators
        # a, self.massvec_list, b, self.evals_list, self.evecs_list, self.disnorm_list = diffusion_net.geometry.get_all_operators(
        #     self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
        input_pos=torch.cat([self.verts_list[0],torch.zeros([self.verts_list[0].shape[0],1])],dim=-1)
        print(input_pos.shape)
        self.frames, self.mass, self.L, self.evals, self.evecs, self.dis_norm=diffusion_net.geometry.get_operators(input_pos, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            print("caching to", self.cache_dir)
            train_cache = os.path.join(self.cache_dir, "train.pt")
            val_cache = os.path.join(self.cache_dir, "val.pt")
            test_cache = os.path.join(self.cache_dir, "test.pt")
            gloabl_cache = os.path.join(self.cache_dir, "global.pt")
            if mode == 'train':
                load_cache = train_cache
            elif mode == 'val':
                load_cache = val_cache
            elif mode == 'test':
                load_cache = test_cache
            else:
                raise ValueError("Invalid mode: {}".format(mode))

            torch.save((self.verts_list, self.x_list, self.labels_list), load_cache)
            torch.save((self.frames,self.mass,self.L,self.evals,self.evecs,self.dis_norm),  gloabl_cache)

    def __len__(self):
        if self.mode=='train':
            return 100
        else:
            return 80
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.x_list[idx], \
            self.mass, self.evals, \
            self.evecs, self.labels_list[idx], \
            self.dis_norm


