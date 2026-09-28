import shutil
import os
import sys
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import potpourri3d as pp3d
import scipy.io


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            print(self.file_path)
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            # self.data = h5py.File(self.file_path)
            print("UNABLE")
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

class TurnMatInPT(Dataset):
    def __init__(self, dataset_path=None, **kwargs):
        self.dataset_path = dataset_path

        # reader = MatReader(self.dataset_path+'/data.mat')#1d
        # x_train = reader.read_field('f_train')
        # y_train = reader.read_field('u_train')
        # grid_x_train = reader.read_field('x_train').squeeze(1)

        # x_vali = reader.read_field('f_vali')
        # y_vali = reader.read_field('u_vali')
        # grid_x_vali = reader.read_field('x_vali').squeeze(1)

        # x_test = reader.read_field('f_test')
        # y_test = reader.read_field('u_test')
        # grid_x_test = reader.read_field('x_test').squeeze(1)

        train=torch.load(self.dataset_path+'/train.pt')
        vali=torch.load(self.dataset_path+'/vali.pt')
        test=torch.load(self.dataset_path+'/test.pt')
        x_train=torch.stack(train['inputs'], dim=0)
        y_train=torch.stack(train['outputs'], dim=0)
        x_vali=torch.stack(vali['inputs'], dim=0)
        y_vali=torch.stack(vali['outputs'], dim=0)
        x_test=torch.stack(test['inputs'], dim=0)
        y_test=torch.stack(test['outputs'], dim=0)

        # reader = MatReader(self.dataset_path+'/data.mat')
        # x_train = reader.read_field('f_train')
        # y_train = reader.read_field('u_train')
        # T = reader.read_field('t').squeeze(0)
        # X = reader.read_field('x').squeeze(0)

        # x_vali = reader.read_field('f_vali')
        # y_vali = reader.read_field('u_vali')

        # x_test = reader.read_field('f_test')
        # y_test = reader.read_field('u_test')

        T=20.48
        n_points=2048
        grid_x_train=torch.tensor(np.linspace(0, T, n_points),dtype=torch.float)
        grid_x_vali=torch.tensor(np.linspace(0, T, n_points),dtype=torch.float)
        grid_x_test=torch.tensor(np.linspace(0, T, n_points),dtype=torch.float)

        print(x_train.shape,y_train.shape,grid_x_train.shape)
        print(x_vali.shape,y_vali.shape,grid_x_vali.shape)
        print(x_test.shape,y_test.shape,grid_x_test.shape)

        # x_train = x_train.reshape(x_train.shape[0],s,1)
        # x_vali = x_vali.reshape(x_vali.shape[0],s,1)
        # x_test = x_test.reshape(x_test.shape[0],s,1)

        train_path=self.dataset_path+'/train.pt'
        vali_path=self.dataset_path+'/vali.pt'
        test_path=self.dataset_path+'/test.pt'
        # torch.save({'x': x_train, 'y': y_train, 'grid_x': grid_x_train}, train_path)
        # torch.save({'x': x_vali, 'y': y_vali, 'grid_x': grid_x_vali}, vali_path)
        # torch.save({'x': x_test, 'y': y_test, 'grid_x': grid_x_test}, test_path)

        # torch.save({'x': x_train, 'y': y_train, 'grid_x': X, 'grid_y': T}, train_path)2d
        # torch.save({'x': x_vali, 'y': y_vali, 'grid_x': X, 'grid_y': T}, vali_path)
        # torch.save({'x': x_test, 'y': y_test, 'grid_x': X, 'grid_y': T}, test_path)
        

        # torch.save((self.verts_list, self.x_list, self.massvec_list, self.evals_list, self.evecs_list, self.labels_list, self.disnorm_list), load_cache)   

if __name__ == '__main__':
    dataset_path="../data/pendulum/c10"
    TurnMatInPT(dataset_path)