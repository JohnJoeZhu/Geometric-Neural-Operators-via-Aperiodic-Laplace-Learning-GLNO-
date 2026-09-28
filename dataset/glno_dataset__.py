import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import potpourri3d as pp3d

from tqdm import tqdm
import h5py


class PoissonBase(Dataset):
    def __init__(self, mode='train', dataset_path=None, preload=False, features=None, device='cpu', **kwargs):
        self.mode = mode
        self.dataset_path = dataset_path
        self.device = device
        self.preload = preload
        self.features = features

        train_cache = os.path.join(self.dataset_path, "train.h5")
        val_cache = os.path.join(self.dataset_path, "val.h5")
        test_cache = os.path.join(self.dataset_path, "test.h5")

        if self.mode == 'train':
            self.load_cache = train_cache
        elif self.mode == 'val':
            self.load_cache = val_cache
        elif self.mode == 'test':
            self.load_cache = test_cache
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

        with h5py.File(self.load_cache, 'r') as f:
            # 获取所有样本组名（假设元数据组为 'metadata'，其余都是样本组）
            # 或者从 metadata/sample_names 读取，这里采用简单规则：排除 'metadata'
            self.meta={}
            meta=f['metadata']
            for feat,arr in meta.items():
                # print(feat,arr.shape)
                if 'L' in feat:
                    continue
                arr=arr[:]
                if arr.dtype.kind in {'U', 'S', 'O'}:
                    self.meta[feat] = arr
                else:
                    # if feat=='curv':
                    #     arr=enhance_curvature_values(arr)
                    self.meta[feat] = torch.from_numpy(arr).to(self.device)
                    if self.meta[feat].dtype==torch.double:
                        # print(":!",feat)
                        self.meta[feat]=self.meta[feat].to(torch.float)
                    elif self.meta[feat].dtype==torch.int32:
                        self.meta[feat]=self.meta[feat].to(torch.int64)
            # print(torch.max(self.meta['eval']))
            self.sample_groups = [name for name in f.keys() if name.startswith('sample_')]
            self.num_samples = len(self.sample_groups)

            if self.num_samples == 0:
                raise RuntimeError(f"No sample groups found in {self.load_cache}")

            # 获取第一个样本组，用于确定可用的特征
            first_grp = f[self.sample_groups[0]]
            available = set(first_grp.keys())

            # 处理稀疏矩阵 L：如果存在 L_data, L_indices, L_indptr, L_shape，则视为一个特征 'L'
            # sparse_L_components = {'L_data', 'L_indices', 'L_indptr', 'L_shape'}
            # if sparse_L_components.issubset(available):
            #     self.has_sparse_L = True
            #     available = available - sparse_L_components
            #     available.add('L')
            # else:
            #     self.has_sparse_L = False

            # 确定最终要返回的特征列表
            if self.features is None:
                self.feature_list = sorted(available)  # 按字母排序保持稳定
            else:
                # 检查用户请求的特征是否都存在
                for feat in self.features:
                    if feat == 'L':
                        if not self.has_sparse_L:
                            raise ValueError(f"Feature 'L' requested but sparse L components not found in file.")
                    elif feat not in available:
                        raise ValueError(f"Feature '{feat}' not found in file (available: {available})")
                self.feature_list = self.features
        print(self.feature_list)
        
    def compute_graph_laplacian_eigenvectors(self, edge_index, num_nodes):
        """简化的图拉普拉斯特征向量计算"""
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1
        adj = (adj + adj.T) / 2
        
        degree = adj.sum(dim=1)
        D_inv_sqrt = torch.diag(degree.pow(-0.5))
        L = torch.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt
        
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            U = eigenvectors[:, :32]
        except:
            print("Warning: failed to compute eigenvectors, using random vectors instead")
            U = torch.randn(num_nodes, 32)
        
        return U.unsqueeze(0)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        with h5py.File(self.load_cache, 'r') as f:
            grp = f[self.sample_groups[idx]]
            sample = {}
            for feat in self.feature_list:
                if feat == 'L' and self.has_sparse_L:
                    continue
                    # L = sparse_np_to_torch(read_sparse_L(grp))
                    # sample['L'] = L
                else:
                    arr = grp[feat][:]
                    if arr.dtype.kind in {'U', 'S', 'O'}:
                        sample[feat] = arr
                    else:
                        # if feat=='curv':
                        #     arr=enhance_curvature_values(arr)
                        sample[feat] = torch.from_numpy(arr).to(self.device)
                        if sample[feat].dtype==torch.double:
                            # print(":!",feat)
                            sample[feat]=sample[feat].to(torch.float)
                        elif sample[feat].dtype==torch.int32:
                            sample[feat]=sample[feat].to(torch.int64)
        # print(torch.max(sample['eval']))
        # print(torch.max(sample['curv']))
        # curv=sample['curv']/torch.max(sample['curv'])
        # dist=sample['distance']/torch.max(sample['distance'])
        # print(curv.shape,sample['vertice'].shape)
        return {"input":sample['input'],"vertices":self.meta['vertice'], \
                "mass":self.meta['mass'], "evals":self.meta['eval'].view(-1), \
                "evecs":self.meta['evec'], "labels":sample['output'],\
                "geo_feat":self.meta['distance']}

class CarCFD(Dataset):
    def __init__(self, mode='train', dataset_path=None, preload=False, features=None, device='cpu', **kwargs):
        self.mode = mode
        self.dataset_path = dataset_path
        self.device=device
        self.preload=preload
        self.features=features

        train_cache = os.path.join(self.dataset_path, "train.h5")
        val_cache=os.path.join(self.dataset_path, "val.h5")
        test_cache = os.path.join(self.dataset_path, "test.h5")
        self.len_list={'train':690,'val':99,'test':100}
        if mode=='train':
            self.load_cache = train_cache
        elif mode=='val':
            self.load_cache = val_cache
        elif mode=='test':
            self.load_cache = test_cache
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        
        with h5py.File(self.load_cache, 'r') as f:
            # 获取所有样本组名（假设元数据组为 'metadata'，其余都是样本组）
            # 或者从 metadata/sample_names 读取，这里采用简单规则：排除 'metadata'
            self.sample_groups = [name for name in f.keys() if name.startswith('sample_')]
            self.num_samples = len(self.sample_groups)

            if self.num_samples == 0:
                raise RuntimeError(f"No sample groups found in {self.load_cache}")

            # 获取第一个样本组，用于确定可用的特征
            first_grp = f[self.sample_groups[0]]
            available = set(first_grp.keys())

            # 处理稀疏矩阵 L：如果存在 L_data, L_indices, L_indptr, L_shape，则视为一个特征 'L'
            sparse_L_components = {'L_data', 'L_indices', 'L_indptr', 'L_shape'}
            if sparse_L_components.issubset(available):
                self.has_sparse_L = True
                available = available - sparse_L_components
                available.add('L')
            else:
                self.has_sparse_L = False

            # 确定最终要返回的特征列表
            if self.features is None:
                self.feature_list = sorted(available)  # 按字母排序保持稳定
            else:
                # 检查用户请求的特征是否都存在
                for feat in self.features:
                    if feat == 'L':
                        if not self.has_sparse_L:
                            raise ValueError(f"Feature 'L' requested but sparse L components not found in file.")
                    elif feat not in available:
                        raise ValueError(f"Feature '{feat}' not found in file (available: {available})")
                self.feature_list = self.features
        print(self.feature_list)

    def __len__(self):
        return self.len_list[self.mode]
    
    def __getitem__(self, idx):
        with h5py.File(self.load_cache, 'r') as f:
            grp = f[self.sample_groups[idx]]
            sample = {}
            for feat in self.feature_list:
                if feat == 'L' and self.has_sparse_L:
                    continue
                    # L = sparse_np_to_torch(read_sparse_L(grp))
                    # sample['L'] = L
                else:
                    arr = grp[feat][:]
                    if arr.dtype.kind in {'U', 'S', 'O'}:
                        sample[feat] = arr
                    else:
                        # if feat=='curv':
                        #     arr=enhance_curvature_values(arr)
                        sample[feat] = torch.from_numpy(arr).to(self.device)
                        if sample[feat].dtype==torch.double:
                            # print(":!",feat)
                            sample[feat]=sample[feat].to(torch.float)
                        elif sample[feat].dtype==torch.int32:
                            sample[feat]=sample[feat].to(torch.int64)
        # print(torch.max(sample['eval']))
        # print(torch.max(sample['curv']))
        # curv=sample['curv']/torch.max(sample['curv'])
        dist=sample['distance']/torch.max(sample['distance'])
        # print(sample['input'].shape)
        print(torch.max(sample['input'],dim=0),torch.min(sample['input'],dim=0))
        return {"input":sample['input'],"vertices":sample['vertice'], \
                "mass":sample['mass'], "evals":sample['eval'].view(-1), \
                "evecs":sample['evec'], "labels":sample['output'],\
                "geo_feat":dist}

class CylinderFlow(Dataset):
    def __init__(self, mode='train', dataset_path=None, preload=False, features=None, device='cpu', **kwargs):
        self.mode = mode
        self.dataset_path = dataset_path
        self.device=device
        self.preload=preload
        self.features=features

        train_cache = os.path.join(self.dataset_path, "train.h5")
        val_cache=os.path.join(self.dataset_path, "val.h5")
        test_cache = os.path.join(self.dataset_path, "test.h5")
        self.len_list={'train':7600,'val':1000,'test':1000}
        if mode=='train':
            self.load_cache = train_cache
        elif mode=='val':
            self.load_cache = val_cache
        elif mode=='test':
            self.load_cache = test_cache
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        
        with h5py.File(self.load_cache, 'r') as f:
            # 获取所有样本组名（假设元数据组为 'metadata'，其余都是样本组）
            # 或者从 metadata/sample_names 读取，这里采用简单规则：排除 'metadata'
            self.sample_groups = [name for name in f.keys() if name.startswith('sample_')]
            self.num_samples = len(self.sample_groups)

            if self.num_samples == 0:
                raise RuntimeError(f"No sample groups found in {self.load_cache}")

            # 获取第一个样本组，用于确定可用的特征
            first_grp = f[self.sample_groups[0]]
            available = set(first_grp.keys())

            # 处理稀疏矩阵 L：如果存在 L_data, L_indices, L_indptr, L_shape，则视为一个特征 'L'
            sparse_L_components = {'L_data', 'L_indices', 'L_indptr', 'L_shape'}
            if sparse_L_components.issubset(available):
                self.has_sparse_L = True
                available = available - sparse_L_components
                available.add('L')
            else:
                self.has_sparse_L = False

            # 确定最终要返回的特征列表
            if self.features is None:
                self.feature_list = sorted(available)  # 按字母排序保持稳定
            else:
                # 检查用户请求的特征是否都存在
                for feat in self.features:
                    if feat == 'L':
                        if not self.has_sparse_L:
                            raise ValueError(f"Feature 'L' requested but sparse L components not found in file.")
                    elif feat not in available:
                        raise ValueError(f"Feature '{feat}' not found in file (available: {available})")
                self.feature_list = self.features
        print(self.feature_list)

    def __len__(self):
        return self.len_list[self.mode]
    
    def __getitem__(self, idx):
        with h5py.File(self.load_cache, 'r') as f:
            grp = f[self.sample_groups[idx]]
            sample = {}
            for feat in self.feature_list:
                if feat == 'L' and self.has_sparse_L:
                    continue
                    # L = sparse_np_to_torch(read_sparse_L(grp))
                    # sample['L'] = L
                elif feat =='downsample1' or feat=='downsample2':
                    continue
                elif feat =='edge' or feat=='face':
                    continue
                else:
                    arr = grp[feat][:]
                    if arr.dtype.kind in {'U', 'S', 'O'}:
                        sample[feat] = arr
                    else:
                        # if feat=='curv':
                        #     arr=enhance_curvature_values(arr)
                        sample[feat] = torch.from_numpy(arr).to(self.device)
                        if sample[feat].dtype==torch.double:
                            # print(":!",feat)
                            sample[feat]=sample[feat].to(torch.float)
                        elif sample[feat].dtype==torch.int32:
                            sample[feat]=sample[feat].to(torch.int64)
        # print(torch.max(sample['eval']))
        return {"input":sample['input'],"vertices":sample['vertice'], \
                "mass":sample['mass'], "evals":sample['eval'], \
                "evecs":sample['evec'], "labels":sample['output'],\
                "geo_feat":torch.norm(sample['vertice'],dim=-1)}

class AIRFOILBase(Dataset):
    def __init__(self, mode='train', dataset_path=None, **kwargs):
        self.mode = mode
        self.dataset_path = dataset_path

        train_cache = os.path.join(self.dataset_path, "train.pt")
        val_cache=os.path.join(self.dataset_path, "val.pt")
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
            print("  --> loading dataset from cache")
            self.verts_list, self.x_list, self.labels_list = torch.load(
                load_cache)
            self.frames, self.mass, self.L, self.evals, self.evecs, self.dis_norm = torch.load(
                os.path.join(self.dataset_path, "global.pt"))
            return
            
        raise ValueError("No data! Run prepare_data first! ")

    def __len__(self):
        # if self.mode=='train':
        #     return 200
        return len(self.verts_list)

    def __getitem__(self, idx):
        return {"input":self.x_list[idx], \
            "mass":self.mass, "evals":self.evals, \
            "evecs":self.evecs, "labels":self.labels_list[idx], \
            "geo_feat":self.dis_norm}


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

# import scipy.sparse as sp
# import scipy.sparse.linalg as spla

# class RNAMeshDataset(Dataset):
#     """
#     RNA segmentation dataset from Poulenard et al., 3DV 2019.

#     See https://github.com/nmwsharp/RNA-Surface-Segmentation-Dataset
#     """

#     def __init__(self, mode='train', dataset_path=None, **kwargs):
#         self.mode = mode
#         self.dataset_path = dataset_path

#         # store in memory
#         self.verts_list = []
#         self.faces_list = []
#         self.labels_list = []  # per-vertex 
#         self.edges_list = []
#         self.curv_list = []
#         self.divide=False
#         self.device=torch.device('cuda:2')
#         self.num_freq=32
#         train_cache = os.path.join(self.dataset_path, "train_sp2gno.pt")
#         val_cache=os.path.join(self.dataset_path, "val_sp2gno.pt")
#         test_cache = os.path.join(self.dataset_path, "test_sp2gno.pt")
#         if mode=='train':
#             load_cache = train_cache
#         elif mode=='val':
#             load_cache = train_cache
#         elif mode=='test':
#             load_cache = test_cache
#         else:
#             raise ValueError("Invalid mode: {}".format(mode))
#         if os.path.exists(load_cache):
#             data=torch.load(load_cache)
#             self.verts_list=data['verts_list']
#             self.faces_list=data['faces_list']
#             self.labels_list=data['labels_list']
#             self.edges_list=data['edges_list']
#             self.curv_list=data['disnorm_list']
#             self.massvec_list=data['massvec_list']
#             self.graph_L=data['graph_L']
#             # for i in range(len(self.graph_L)):
#             #     print(self.graph_L[i].shape)
#             return
        
#         raise ValueError("No data! Run prepare_data first! ")
#         if os.path.exists(load_cache):
#             print(f"  --> loading dataset from cache {load_cache}")
#             self.verts_list, self.faces_list, self.frames_list, self.massvec_list,\
#             self.L_list, self.evals_list, self.evecs_list, self.labels_list,\
#             self.edges_list, self.curv_list = torch.load(load_cache)
#             self.graph_L=[]
#             for i in tqdm(range(len(self.verts_list))):
#                 self.graph_L.append(self.compute_graph_laplacian_eigenvectors(self.edges_list[i], self.verts_list[i].shape[0]))
#             saving_dict={'verts_list':self.verts_list, 'faces_list':self.faces_list, 'massvec_list':self.massvec_list, 'labels_list':self.labels_list, 'edges_list':self.edges_list, 'disnorm_list':self.curv_list, 'graph_L':self.graph_L}
#             torch.save(saving_dict, load_cache.split('.pt')[0]+'_sp2gno.pt')
#             return
        
#         raise ValueError(f"No data under {self.dataset_path}! Run prepare_data first! ")
    
#     def compute_graph_laplacian_eigenvectors(self, edge_index, num_nodes):
#         """使用SciPy稀疏特征值计算"""
#         try:
#             # 将边索引转换为numpy数组
#             edge_index_np = edge_index.cpu().numpy()
            
#             # 构建稀疏邻接矩阵
#             data = np.ones(edge_index_np.shape[1])
#             adj = sp.csr_matrix((data, (edge_index_np[0], edge_index_np[1])), 
#                             shape=(num_nodes, num_nodes))
            
#             # 对称化
#             adj = adj + adj.T
#             adj.data = np.ones_like(adj.data)  # 二值化
            
#             # 计算度矩阵
#             degree = np.array(adj.sum(axis=1)).flatten()
#             degree[degree == 0] = 1  # 避免除零
            
#             # 构建归一化拉普拉斯矩阵
#             D_inv_sqrt = sp.diags(degree ** -0.5)
#             L = sp.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt
            
#             # 使用ARPACK计算最小的几个特征值和特征向量
#             # 注意：我们计算最小的特征值对应的特征向量
#             eigenvalues, eigenvectors = spla.eigsh(L, k=self.num_freq, which='SM')
#             print(eigenvalues.shape, eigenvectors.shape)
#             # 转换为PyTorch张量并返回
#             U = torch.from_numpy(eigenvectors).float().to(self.device)
            
#             return U
            
#         except Exception as e:
#             print(f"Warning: SciPy computation failed: {e}, using random vectors")
#             return torch.randn(num_nodes, self.num_freq, device=self.device)
        
#     def __len__(self):
#         if self.divide:
#             return len(self.divide_list)
#         return len(self.verts_list)
    
#     def split_val(self, index_list):
#         self.divide=True
#         self.divide_list=index_list

#     def __getitem__(self, idx):   
#         if self.divide:
#             idx=self.divide_list[idx]    
#         return {"vertices":self.verts_list[idx],\
#                "mass":self.massvec_list[idx], \
#                "labels":self.labels_list[idx],\
#                "geo_feat":self.curv_list[idx],'edges':self.edges_list[idx],'graph_L':self.graph_L[idx]}

class RNAMeshDataset(Dataset):
    """
    RNA segmentation dataset from Poulenard et al., 3DV 2019.

    See https://github.com/nmwsharp/RNA-Surface-Segmentation-Dataset
    """

    def __init__(self, mode='train', dataset_path=None, **kwargs):
        self.mode = mode
        self.dataset_path = dataset_path

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-vertex 
        self.edges_list = []
        self.curv_list = []
        self.divide=False
        self.device=torch.device('cuda:2')
        self.num_freq=32
        train_cache = os.path.join(self.dataset_path, "train.pt")
        test_cache = os.path.join(self.dataset_path, "test.pt")
        if mode=='train':
            load_cache = train_cache
        elif mode=='val':
            load_cache = train_cache
        elif mode=='test':
            load_cache = test_cache
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        if os.path.exists(load_cache):
            print(f"  --> loading dataset from cache {load_cache}")
            self.verts_list, self.faces_list, self.frames_list, self.massvec_list,\
            self.L_list, self.evals_list, self.evecs_list, self.labels_list,\
            self.edges_list, self.curv_list = torch.load(load_cache)
            self.disnorm_list=compute_disnorm_list(self.verts_list,self.faces_list)
            mesh_max=0
            mesh_min=1e9
            for vert in self.verts_list:
                mesh_max=max(mesh_max,vert.shape[0])
                mesh_min=min(mesh_min,vert.shape[0])
            print(f"mesh_max={mesh_max}, mesh_min={mesh_min}")
            return
        
        raise ValueError("No data! Run prepare_data first! ")
        
    def __len__(self):
        if self.divide:
            return len(self.divide_list)
        return len(self.verts_list)
    
    def split_val(self, index_list):
        self.divide=True
        self.divide_list=index_list

    def __getitem__(self, idx):   
        if self.divide:
            idx=self.divide_list[idx]    
        return {"vertices":self.verts_list[idx],'faces':self.faces_list[idx],\
                "evals":self.evals_list[idx], "evecs":self.evecs_list[idx], \
               "labels":self.labels_list[idx],"geo_feat":self.disnorm_list[idx],"mass":self.massvec_list[idx],\
               'edges':self.edges_list[idx]}



# class HumanMeshDataset(Dataset): ### sp2gno version
#     """
#     RNA segmentation dataset from Poulenard et al., 3DV 2019.

#     See https://github.com/nmwsharp/RNA-Surface-Segmentation-Dataset
#     """

#     def __init__(self, mode='train', dataset_path=None, **kwargs):
#         self.mode = mode
#         self.dataset_path = dataset_path

#         # store in memory
#         self.verts_list = []
#         self.faces_list = []
#         self.labels_list = []  # per-vertex 
#         self.edges_list = []
#         self.curv_list = []
#         self.divide=False
#         self.num_freq=32
#         self.device=torch.device('cuda:2')

#         train_cache = os.path.join(self.dataset_path, "train_sp2gno.pt")
#         val_cache=os.path.join(self.dataset_path, "train_sp2gno.pt")
#         test_cache = os.path.join(self.dataset_path, "test_sp2gno.pt")
#         if mode=='train':
#             load_cache = train_cache
#         elif mode=='val':
#             load_cache = train_cache
#         elif mode=='test':
#             load_cache = test_cache
#         else:
#             raise ValueError("Invalid mode: {}".format(mode))
#         if os.path.exists(load_cache):
#             data=torch.load(load_cache)
#             self.verts_list=data['verts_list']
#             self.faces_list=data['faces_list']
#             self.labels_list=data['labels_list']
#             self.edges_list=data['edges_list']
#             self.curv_list=data['disnorm_list']
#             self.massvec_list=data['massvec_list']
#             self.graph_L=data['graph_L']
#             # for i in range(len(self.graph_L)):
#             #     print(self.graph_L[i].shape)
#             return
        
#         if os.path.exists(load_cache):
#             print(f"  --> loading dataset from cache {load_cache}")
#             self.verts_list, self.faces_list, self.frames_list, self.massvec_list,\
#             self.L_list, self.evals_list, self.evecs_list, self.edges_list, \
#             self.disnorm_list, self.labels_list = torch.load(load_cache)
#             self.disnorm_list=compute_disnorm_list(self.verts_list,self.faces_list)
#             self.edges_list=compute_edges_list_batch(self.faces_list)
#             self.graph_L=[]
#             for i in tqdm(range(len(self.verts_list))):
#                 self.graph_L.append(self.compute_graph_laplacian_eigenvectors(self.edges_list[i], self.verts_list[i].shape[0]))
#             saving_dict={'verts_list':self.verts_list, 'faces_list':self.faces_list, 'massvec_list':self.massvec_list, 'labels_list':self.labels_list, 'edges_list':self.edges_list, 'disnorm_list':self.curv_list, 'graph_L':self.graph_L}
#             torch.save(saving_dict, load_cache.split('.pt')[0]+'_sp2gno.pt')
#             return
        
#         raise ValueError(f"No data under {self.dataset_path}! Run prepare_data first! ")
        
#     def __len__(self):
#         if self.divide:
#             return len(self.divide_list)
#         return len(self.verts_list)
    
#     def split_val(self, index_list):
#         self.divide=True
#         self.divide_list=index_list

#     def compute_graph_laplacian_eigenvectors(self, edge_index, num_nodes):
#         """使用SciPy稀疏特征值计算"""
#         try:
#             # 将边索引转换为numpy数组
#             edge_index_np = edge_index.cpu().numpy()
            
#             # 构建稀疏邻接矩阵
#             data = np.ones(edge_index_np.shape[1])
#             adj = sp.csr_matrix((data, (edge_index_np[0], edge_index_np[1])), 
#                             shape=(num_nodes, num_nodes))
            
#             # 对称化
#             adj = adj + adj.T
#             adj.data = np.ones_like(adj.data)  # 二值化
            
#             # 计算度矩阵
#             degree = np.array(adj.sum(axis=1)).flatten()
#             degree[degree == 0] = 1  # 避免除零
            
#             # 构建归一化拉普拉斯矩阵
#             D_inv_sqrt = sp.diags(degree ** -0.5)
#             L = sp.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt
            
#             # 使用ARPACK计算最小的几个特征值和特征向量
#             # 注意：我们计算最小的特征值对应的特征向量
#             eigenvalues, eigenvectors = spla.eigsh(L, k=self.num_freq, which='SM')
            
#             # 转换为PyTorch张量并返回
#             U = torch.from_numpy(eigenvectors).float().to(self.device)
            
#             return U
            
#         except Exception as e:
#             print(f"Warning: SciPy computation failed: {e}, using random vectors")
#             return torch.randn(num_nodes, self.num_freq, device=self.device)

#     def __getitem__(self, idx):   
#         if self.divide:
#             idx=self.divide_list[idx]    
#         return {"vertices":self.verts_list[idx],'faces':self.faces_list[idx],\
#                "mass":self.massvec_list[idx], \
#                "labels":self.labels_list[idx],\
#                'edges':self.edges_list[idx],'graph_L':self.graph_L[idx]}
    


class HumanMeshDataset(Dataset): ### normal version
    """
    RNA segmentation dataset from Poulenard et al., 3DV 2019.

    See https://github.com/nmwsharp/RNA-Surface-Segmentation-Dataset
    """

    def __init__(self, mode='train', dataset_path=None, **kwargs):
        self.mode = mode
        self.dataset_path = dataset_path

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-vertex 
        self.edges_list = []
        self.curv_list = []
        self.divide=False
        self.num_freq=32
        self.device=torch.device('cuda:1')

        train_cache = os.path.join(self.dataset_path, "train.pt")
        test_cache = os.path.join(self.dataset_path, "test.pt")
        if mode=='train':
            load_cache = train_cache
        elif mode=='val':
            load_cache = train_cache
        elif mode=='test':
            load_cache = test_cache
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        if os.path.exists(load_cache):
            self.verts_list, self.faces_list, self.frames_list, self.massvec_list,\
            self.L_list, self.evals_list, self.evecs_list, self.edges_list, \
            self.disnorm_list, self.labels_list = torch.load(load_cache)
            self.disnorm_list=compute_disnorm_list(self.verts_list,self.faces_list)
            # self.edges_list=compute_edges_list_batch(self.faces_list)
            mesh_max=0
            mesh_min=1e9
            for vert in self.verts_list:
                mesh_max=max(mesh_max,vert.shape[0])
                mesh_min=min(mesh_min,vert.shape[0])
            print(f"mesh_max={mesh_max}, mesh_min={mesh_min}")
            return
        
        raise ValueError(f"No data under {self.dataset_path}! Run prepare_data first! ")
        
    def __len__(self):
        if self.divide:
            return len(self.divide_list)
        return len(self.verts_list)
    
    def split_val(self, index_list):
        self.divide=True
        self.divide_list=index_list

    def __getitem__(self, idx):   
        if self.divide:
            idx=self.divide_list[idx]    
        return {"vertices":self.verts_list[idx],'faces':self.faces_list[idx],\
                "evals":self.evals_list[idx], "evecs":self.evecs_list[idx], \
               "labels":self.labels_list[idx],"geo_feat":self.disnorm_list[idx],"mass":self.massvec_list[idx],\
               'edges':self.edges_list[idx]}

if __name__ == '__main__':
    dataset = RNAMeshDataset(mode='train', dataset_path='../data/rna')
    # dataset = RNAMeshDataset(mode='val', dataset_path='../data/rna') 
    dataset = RNAMeshDataset(mode='test', dataset_path='../data/rna')
    # dataset=HumanMeshDataset(mode='train',dataset_path='../data/human')
    # dataset=HumanMeshDataset(mode='test',dataset_path='../data/human')
    



class ShrecMeshDataset(Dataset):
    def __init__(self, mode='train', dataset_path=None, **kwargs):
        print("!!!!!")
        self.mode = mode
        self.dataset_path = dataset_path
        print(dataset_path)

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-vertex 
        self.edges_list = []
        self.curv_list = []
        self.divide=False
        self.num_freq=32
        self.device=torch.device('cuda:2')

        train_cache = os.path.join(self.dataset_path, "train_simplified.pt")
        # val_cache=os.path.join(self.dataset_path, "val_simplified.pt")
        test_cache = os.path.join(self.dataset_path, "test_simplified.pt")
        print(train_cache,test_cache)
        if mode=='train':
            load_cache = train_cache
        elif mode=='val':
            load_cache = train_cache
        elif mode=='test':
            load_cache = test_cache
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        
        if os.path.exists(load_cache):
            print(f"  --> loading dataset from cache {load_cache}")
            self.verts_list, self.faces_list, self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.labels_list,self.edges_list, self.curv_list = torch.load(load_cache)
            mesh_max=0
            mesh_min=1e9
            for vert in self.verts_list:
                mesh_max=max(mesh_max,vert.shape[0])
                mesh_min=min(mesh_min,vert.shape[0])
            print(f"mesh_max={mesh_max}, mesh_min={mesh_min}")
            
            return
            self.disnorm_list=compute_disnorm_list(self.verts_list,self.faces_list)
            self.edges_list=compute_edges_list_batch(self.faces_list)
            self.graph_L=[]
            for i in tqdm(range(len(self.verts_list))):
                self.graph_L.append(self.compute_graph_laplacian_eigenvectors(self.edges_list[i], self.verts_list[i].shape[0]))
            saving_dict={'verts_list':self.verts_list, 'faces_list':self.faces_list, 'massvec_list':self.massvec_list, 'labels_list':self.labels_list, 'edges_list':self.edges_list, 'disnorm_list':self.curv_list, 'graph_L':self.graph_L}
            torch.save(saving_dict, load_cache.split('.pt')[0]+'_sp2gno.pt')
            return
        
        raise ValueError(f"No data under {self.dataset_path}! Run prepare_data first! ")
        
    def __len__(self):
        if self.divide:
            return len(self.divide_list)
        return len(self.verts_list)
    
    def split_val(self, index_list):
        self.divide=True
        self.divide_list=index_list

    def compute_graph_laplacian_eigenvectors(self, edge_index, num_nodes):
        """使用SciPy稀疏特征值计算"""
        try:
            # 将边索引转换为numpy数组
            edge_index_np = edge_index.cpu().numpy()
            
            # 构建稀疏邻接矩阵
            data = np.ones(edge_index_np.shape[1])
            adj = sp.csr_matrix((data, (edge_index_np[0], edge_index_np[1])), 
                            shape=(num_nodes, num_nodes))
            
            # 对称化
            adj = adj + adj.T
            adj.data = np.ones_like(adj.data)  # 二值化
            
            # 计算度矩阵
            degree = np.array(adj.sum(axis=1)).flatten()
            degree[degree == 0] = 1  # 避免除零
            
            # 构建归一化拉普拉斯矩阵
            D_inv_sqrt = sp.diags(degree ** -0.5)
            L = sp.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt
            
            # 使用ARPACK计算最小的几个特征值和特征向量
            # 注意：我们计算最小的特征值对应的特征向量
            eigenvalues, eigenvectors = spla.eigsh(L, k=self.num_freq, which='SM')
            
            # 转换为PyTorch张量并返回
            U = torch.from_numpy(eigenvectors).float().to(self.device)
            
            return U
            
        except Exception as e:
            print(f"Warning: SciPy computation failed: {e}, using random vectors")
            return torch.randn(num_nodes, self.num_freq, device=self.device)

    def __getitem__(self, idx):   
        if self.divide:
            idx=self.divide_list[idx]    
        return {"vertices":self.verts_list[idx],'faces':self.faces_list[idx],\
               "mass":self.massvec_list[idx], \
               "labels":self.labels_list[idx],\
               'edges':self.edges_list[idx]}
    