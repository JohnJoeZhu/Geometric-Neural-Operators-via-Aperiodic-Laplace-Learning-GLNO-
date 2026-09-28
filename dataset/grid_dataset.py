import shutil
import os
import sys
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import h5py

class Grid1DDataset(Dataset):
    def __init__(self, mode='train', dataset_path=None, **kwargs):
        self.mode = mode
        self.dataset_path = dataset_path

        train_cache = os.path.join(self.dataset_path, "train.pt")
        val_cache=os.path.join(self.dataset_path, "vali.pt")
        test_cache = os.path.join(self.dataset_path, "test.pt")
        if mode=='train':
            load_cache = train_cache
        elif mode=='val':
            load_cache = val_cache
        elif mode=='test':
            load_cache = test_cache
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        if os.path.exists(load_cache):
            print(f"  --> loading dataset from cache {load_cache}")
            data=torch.load(load_cache)
            self.input_data=data['x']
            self.output_data=data['y']
            self.grid=data['grid_x']
            return

    def __len__(self):
        return self.input_data.shape[0]
    
    def __getitem__(self, idx):
        return {'inputs':self.input_data[idx], 'labels':self.output_data[idx], 'grid_x':self.grid}
    
class Grid2DDataset(Dataset):
    def __init__(self, mode='train', dataset_path=None, **kwargs):
        self.mode = mode
        self.dataset_path = dataset_path

        train_cache = os.path.join(self.dataset_path, "train.pt")
        val_cache=os.path.join(self.dataset_path, "vali.pt")
        test_cache = os.path.join(self.dataset_path, "test.pt")
        if mode=='train':
            load_cache = train_cache
        elif mode=='val':
            load_cache = val_cache
        elif mode=='test':
            load_cache = test_cache
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        if os.path.exists(load_cache):
            print(f"  --> loading dataset from cache {load_cache}")
            data=torch.load(load_cache)
            self.input_data=data['x']
            self.output_data=data['y']
            self.grid_x=data['grid_x']
            self.grid_y=data['grid_y']
            return

    def __len__(self):
        return self.input_data.shape[0]
    
    def __getitem__(self, idx):
        return {'inputs':self.input_data[idx], 'labels':self.output_data[idx], 'grid_x':self.grid_x, 'grid_y':self.grid_y}

class WellDataset(Dataset):
    """
    用于加载 The Well 数据集的 PyTorch Dataset 类
    
    Args:
        mode: 'train', 'val', 或 'test'
        dataset_path: 数据集根目录，如 '/data/well/turbulent_radiative_layer_2D'
        cache_dir: 缓存文件保存目录，默认为 dataset_path
        fields: 要加载的物理场列表，可选 ['density', 'pressure', 'velocity']
        time_steps: 要使用的时间步，None表示全部
    """
    def __init__(self, mode='train', dataset_path=None, 
                 fields=['density', 'pressure', 'velocity'], time_steps=None, **kwargs):
        self.mode = mode
        self.dataset_path = dataset_path
        self.fields = fields
        self.time_steps = time_steps
        
        cache_dir=dataset_path
        
        # 缓存文件名
        cache_file = cache_dir + f"/{mode}.pt"
        
        # 尝试加载缓存
        if os.path.exists(cache_file):
            print(f"  --> loading dataset from cache {cache_file}")
            data = torch.load(cache_file)
            self.input_data = data['x']
            self.output_data = data['y']
            self.grid_x = data['grid_x']
            self.grid_y = data['grid_y']
            self.num_trajectories = data.get('num_trajectories', len(self.input_data))
            return
        
        raise ValueError(f"Cache file {cache_file} not found.")

    def __len__(self):
        return self.input_data.shape[0]
    
    def __getitem__(self, idx):
        return {
            'inputs': self.input_data[idx].permute(1,2,0),     # (H, W, C)
            'labels': self.output_data[idx].permute(1,2,0),    # (H, W, C)
            'grid_x': self.grid_x,              # (W,)
            'grid_y': self.grid_y               # (H,)
        }