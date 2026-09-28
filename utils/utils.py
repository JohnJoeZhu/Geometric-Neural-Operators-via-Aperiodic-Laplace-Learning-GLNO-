import torch
import torch.distributed as dist
import hashlib
import numpy as np
import scipy
import logging
import os
from datetime import datetime
import yaml

def torch2np(tensor):
    assert isinstance(tensor, torch.Tensor)
    return tensor.detach().cpu().numpy()

def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

def sparse_torch_to_np(A):
    assert len(A.shape) == 2

    indices = torch2np(A.indices())
    values = torch2np(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()
    return mat

# Hash a list of numpy arrays
def hash_arrays(arrs):
    running_hash = hashlib.sha1()
    for arr in arrs:
        binarr = arr.view(np.uint8)
        running_hash.update(binarr)
    return running_hash.hexdigest()

def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device('cpu')).numpy()

def read_sparse_L(h5_group):
    # 检查必要数据集是否存在
    required_keys = ['L_data', 'L_indices', 'L_indptr', 'L_shape']
    for key in required_keys:
        if key not in h5_group:
            raise KeyError(f"Group missing required dataset: {key}")
    
    # 读取数据
    data = h5_group['L_data'][:]
    indices = h5_group['L_indices'][:]
    indptr = h5_group['L_indptr'][:]
    shape = tuple(h5_group['L_shape'][:])
    return scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)