import os

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from utils import read_sparse_L,sparse_np_to_torch
from tqdm import tqdm
from typing import Dict, Union, Optional, List, Tuple

class DatasetSpliter():
    def __init__(self,
                 length: int,
                 splits: Optional[Dict[str, Union[int, float]]] = None,
                 n_folds: Optional[int] = None,
                 fold: int = 0,
                 shuffle: bool = True,
                 random_seed: Optional[int] = 42):
        """
        支持两种划分模式：
        1. 固定比例/数量划分（train/val/test）：提供 splits 字典。
        2. n‑fold 交叉验证划分：提供 n_folds 和 fold。

        参数:
            length:     数据集总样本数。
            splits:     字典，包含 'train', 'val', 'test' 三个键，值为整数（绝对数量）或浮点数（比例）。
                        与 n_folds 互斥，二者必须指定其一。
            n_folds:    折数，用于 n‑fold 交叉验证。与 splits 互斥。
            fold:       当前使用的折索引（0‑based），该折将作为验证集，其余折合并为训练集。
            shuffle:    是否在划分前随机打乱索引顺序。默认为 True。
            random_seed:随机种子，当 shuffle=True 时有效。
        """
        # 参数互斥检查
        if splits is None and n_folds is None:
            raise ValueError("必须指定 splits 或 n_folds 之一")
        if splits is not None and n_folds is not None:
            raise ValueError("splits 和 n_folds 不能同时指定，请选择一种模式")

        self.length = length
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # 生成全量索引（可选择是否打乱）
        indices = np.arange(length)
        if shuffle:
            self.rng.shuffle(indices)
        self.all_indices = indices

        if splits is not None:
            # ---------- 原有模式：固定 train/val/test 划分 ----------
            self._init_splits_mode(splits, indices)
        else:
            # ---------- 新增模式：n‑fold 交叉验证 ----------
            if not isinstance(n_folds, int) or n_folds <= 1:
                raise ValueError("n_folds 必须是大于 1 的整数")
            if not (0 <= fold < n_folds):
                raise ValueError(f"fold 必须在 [0, {n_folds-1}] 范围内")
            self._init_nfold_mode(n_folds, fold, indices)

    def _init_splits_mode(self, splits: Dict, indices: np.ndarray):
        """处理固定比例/数量划分（原逻辑）"""
        if set(splits.keys()) != {'train', 'val', 'test'}:
            raise ValueError("splits 必须包含且仅包含 'train', 'val', 'test' 三个键")

        nums = {}
        remainders = {}
        for name, val in splits.items():
            if isinstance(val, float):
                exact = val * self.length
                num = int(exact)
                remainder = exact - num
            elif isinstance(val, int):
                num = val
                remainder = 0.0
            else:
                raise TypeError(f"键 {name} 的值必须是 int 或 float，收到 {type(val)}")

            if num < 0 or num > self.length:
                raise ValueError(f"{name} 的取值 {val} 转换后得到 {num}，超出范围 [0, {self.length}]")
            nums[name] = num
            remainders[name] = remainder

        total = sum(nums.values())
        if total > self.length:
            raise ValueError(f"各子集整数部分之和 ({total}) 超过总样本数 ({self.length})")

        remaining = self.length - total
        if remaining > 0:
            sorted_names = sorted(remainders.keys(), key=lambda x: remainders[x], reverse=True)
            for i in range(remaining):
                name = sorted_names[i % len(sorted_names)]
                nums[name] += 1

        self.nums = nums
        self.result = {}
        start = 0
        for name in ['train', 'val', 'test']:
            count = self.nums[name]
            self.result[name] = indices[start:start+count].tolist()
            start += count

    def _init_nfold_mode(self, n_folds: int, fold: int, indices: np.ndarray):
        """处理 n‑fold 交叉验证划分"""
        # 将索引尽可能均匀地分成 n_folds 份
        folds = np.array_split(indices, n_folds)   # 返回列表，每个元素是一个子数组
        self.folds = [fold.tolist() for fold in folds]   # 转为 Python list

        # 当前验证集 = 第 fold 折
        val_indices = self.folds[fold]
        # 训练集 = 其余所有折的合并
        train_indices = []
        for i, f in enumerate(self.folds):
            if i != fold:
                train_indices.extend(f)

        # 为了与原有接口兼容，仍然将结果存入 self.result（仅使用 'train' 和 'val'，test 留空）
        self.result = {
            'train': train_indices,
            'val': val_indices,
            'test': []      # n‑fold 模式下不单独定义测试集
        }
        # 同时记录折信息，便于后续切换 fold
        self.n_folds = n_folds
        self.current_fold = fold

    def get_fold(self, fold: int) -> Tuple[List[int], List[int]]:
        """
        切换并获取另一折的划分（仅 n‑fold 模式下有效）。
        返回 (train_indices, val_indices)
        """
        if not hasattr(self, 'n_folds'):
            raise RuntimeError("当前处于 splits 模式，无法切换折。请使用 n_folds 模式。")
        if not (0 <= fold < self.n_folds):
            raise ValueError(f"fold 必须在 [0, {self.n_folds-1}] 范围内")
        if fold == self.current_fold:
            return self.result['train'], self.result['val']
        # 重新合并训练集和验证集
        val_indices = self.folds[fold]
        train_indices = []
        for i, f in enumerate(self.folds):
            if i != fold:
                train_indices.extend(f)
        return train_indices, val_indices

    # ---------- 保留原有的接口方法 ----------
    def get_len(self, mode: str) -> int:
        """返回指定子集的样本数量（mode: 'train'/'val'/'test'）"""
        if mode not in self.result:
            raise KeyError(f"mode 必须是 'train', 'val', 'test' 之一，收到 {mode}")
        return len(self.result[mode])

    def get_index(self, mode: str, index: int) -> int:
        """返回指定子集中第 index 个样本的原始索引"""
        return self.result[mode][index]


def enhance_curvature_values(curvature, method='adaptive_gamma'):
    """
    使用 NumPy 增强曲率值的对比度。

    参数：
        curvature (np.ndarray): 输入曲率数组（可以是任意形状，但通常是一维或二维）
        method (str): 增强方法，可选：
            - 'adaptive_gamma': 基于数据分布的自适应伽马校正
            - 'log_enhancement': 对数增强
            - 其他字符串：直接返回原始 curvature

    返回：
        np.ndarray: 增强后的曲率数组，形状与输入相同
    """
    if method == 'adaptive_gamma':
        # 计算均值和标准差
        mean_val = np.mean(curvature)
        std_val = np.std(curvature)
        # 计算偏度（三阶标准矩）
        # 避免除以零，加上小常数
        skewness = np.mean((curvature - mean_val) ** 3) / (std_val ** 3 + 1e-10)

        # 根据偏度选择伽马值
        if skewness > 1.0:      # 正偏态分布，拉伸低值区域
            gamma = 0.3
        elif skewness < -1.0:    # 负偏态分布，压缩高值区域
            gamma = 1.1
        else:                    # 近似对称，不调整
            gamma = 1.0

        # 符号保留，绝对值进行伽马校正
        enhanced = np.sign(curvature) * np.abs(curvature) ** gamma

    elif method == 'log_enhancement':
        # 对数增强：对绝对值取对数，保留符号
        epsilon = 1e-6
        sign = np.sign(curvature)
        abs_curv = np.abs(curvature)
        # 防止 abs_curv 为 0 导致 log(1) = 0，但这里 log(1+abs/epsilon) 当 abs=0 时 log(1)=0，没问题
        enhanced = sign * np.log(1 + abs_curv / epsilon)

    else:
        # 未知方法，直接返回原值
        enhanced = curvature

    return enhanced

class CortexDataset(Dataset): ### normal version
    def __init__(self, mode='train', dataset_path=None, preload=False, features=None, device='cpu', **kwargs):
        self.mode = mode
        self.dataset_path = dataset_path
        self.device=device
        self.preload=preload
        self.features=features

        if mode not in ['train','val','test']:
            raise ValueError("Invalid mode: {}".format(mode))
        
        with h5py.File(self.dataset_path, 'r') as f:
            # 获取所有样本组名（假设元数据组为 'metadata'，其余都是样本组）
            # 或者从 metadata/sample_names 读取，这里采用简单规则：排除 'metadata'
            self.sample_groups = [name for name in f.keys() if name.startswith('sample_')]
            self.num_samples = len(self.sample_groups)

            if self.num_samples == 0:
                raise RuntimeError(f"No sample groups found in {self.dataset_path}")

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
        self.Spliter=DatasetSpliter(self.num_samples,{'train':0.8,'val':0.1,'test':0.1})
        # print(f"train:{self.Spliter.get_len('train')} val:{self.Spliter.get_len('val')} test:{self.Spliter.get_len('test')}")
        # 预加载数据
        if self.preload:
            self.data = self._preload_all()
            # mesh_max=0
            # mesh_min=1e9
            # for vert in self.verts_list:
            #     mesh_max=max(mesh_max,vert.shape[0])
            #     mesh_min=min(mesh_min,vert.shape[0])
            # print(f"mesh_max={mesh_max}, mesh_min={mesh_min}")
        else:
            self.data = None 
        
    def __len__(self):
        return self.Spliter.get_len(self.mode)

    def _preload_all(self):
        """将所有样本的指定特征加载到内存中，返回一个列表，每个元素是一个样本字典"""
        all_samples = []
        with h5py.File(self.dataset_path, 'r') as f:
            for name in tqdm(self.sample_groups):
                grp = f[name]
                sample = {}
                for feat in self.feature_list:
                    if feat == 'L' and self.has_sparse_L:
                        pass
                        # 重建稀疏矩阵 L
                        # L = self._read_sparse_L(grp, return_format='torch', device=self.device)
                        # sample['L'] = L
                    else:
                        # 读取密集数组并转为 torch 张量
                        arr = grp[feat][:]
                        # 如果是字符串或对象数组，保持原样（如标签可能为字符串）
                        if arr.dtype.kind in {'U', 'S', 'O'}:
                            sample[feat] = arr  # 保持 numpy 数组，不转 tensor
                        else:
                            sample[feat] = torch.from_numpy(arr).to(self.device)
                            if sample[feat].dtype==torch.double:
                                # print(":!",feat)
                                sample[feat]=sample[feat].to(torch.float)
                            elif sample[feat].dtype==torch.int32:
                                sample[feat]=sample[feat].to(torch.int64)
                all_samples.append(sample)
        return all_samples

    def __getitem__(self, idx):   
        idx=self.Spliter.get_index(self.mode,idx)
        if self.preload:
            sample = self.data[idx].copy()  # 复制防止意外修改缓存
        else:
            # 按需读取
            with h5py.File(self.dataset_path, 'r') as f:
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
                # print(torch.max(sample['sulc']),torch.min(sample['sulc']))
                # print(torch.max(sample['eval']))
                if 'cortex' in self.dataset_path:
                    sample['label']=sample['label']+1
                return {"vertices":sample['vertice'],'faces':sample['face'],\
                        "evals":sample['eval'].view(-1), "evecs":sample['evec'], \
                    "labels":sample['label'],"geo_feat":sample['curv'],"mass":sample['mass'],\
                    }
                    


class IntraDataset(Dataset): ### normal version
    def __init__(self, mode='train', dataset_path=None, preload=False, features=None, device='cpu', **kwargs):
        self.mode = mode
        self.dataset_path = dataset_path
        self.device=device
        self.preload=preload
        self.features=features
        if mode not in ['train','val','test']:
            raise ValueError("Invalid mode: {}".format(mode))
        
        with h5py.File(self.dataset_path, 'r') as f:
            # 获取所有样本组名（假设元数据组为 'metadata'，其余都是样本组）
            # 或者从 metadata/sample_names 读取，这里采用简单规则：排除 'metadata'
            self.sample_groups = [name for name in f.keys() if name.startswith('sample_')]
            self.num_samples = len(self.sample_groups)

            if self.num_samples == 0:
                raise RuntimeError(f"No sample groups found in {self.dataset_path}")

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
        self.Spliter=DatasetSpliter(self.num_samples,{'train':0.6,'val':0.2,'test':0.2})
        # print(f"train:{self.Spliter.get_len('train')} val:{self.Spliter.get_len('val')} test:{self.Spliter.get_len('test')}")
        # 预加载数据
        
    def __len__(self):
        return self.Spliter.get_len(self.mode)

    def __getitem__(self, idx):   
        idx=self.Spliter.get_index(self.mode,idx)
        if self.preload:
            pass
            # sample = self.data[idx].copy()  # 复制防止意外修改缓存
        else:
            # 按需读取
            with h5py.File(self.dataset_path, 'r') as f:
                grp = f[self.sample_groups[idx]]
                sample = {}
                for feat in self.feature_list:
                    if feat == 'L' and self.has_sparse_L:
                        continue
                        # L = sparse_np_to_torch(read_sparse_L(grp))
                        # sample['L'] = L
                    else:
                        if grp[feat].shape == ():
                            arr = grp[feat][()]
                        else:
                            arr = grp[feat][:]
                        if arr.dtype.kind in {'U', 'S', 'O'}:
                            sample[feat] = arr
                        else:
                            if feat=='curv':
                                arr=enhance_curvature_values(arr)
                            if feat!='label':
                                sample[feat] = torch.from_numpy(arr).to(self.device)
                            else:
                                sample[feat] = torch.tensor(arr,dtype=torch.int64).to(self.device)
                            if sample[feat].dtype==torch.double:
                                # print(":!",feat)
                                sample[feat]=sample[feat].to(torch.float)
                            elif sample[feat].dtype==torch.int32:
                                sample[feat]=sample[feat].to(torch.int64)
                # print(torch.max(sample['eval']))
                sample['label']=sample['label'].to(torch.int64)
                # sample['label']=torch.where(sample['label']==2,torch.ones_like(sample['label']),sample['label'])
                # sample['label']=torch.tensor(sample['label'].item(),dtype=torch.int64)
                
                # print(sample['label'])
                # print(torch.max(sample['distance']))
                # print(torch.max(sample['label']),torch.min(sample['label']))
                return {"vertices":sample['vertice'],'faces':sample['face'],\
                        "evals":sample['eval'].view(-1), "evecs":sample['evec'], \
                    "labels":sample['label'],"geo_feat":sample['distance']/torch.norm(sample['distance']),"mass":sample['mass'],\
                    }
                    


class PoissonUnstrucDataset(Dataset): ### normal version
    def __init__(self, mode='train', dataset_path=None, preload=False, features=None, device='cpu', **kwargs):
        self.mode = mode
        self.dataset_path = dataset_path
        if self.mode=='test':
            self.dataset_path=self.dataset_path.replace('data.h5','unstruc.h5')
        else:   
            self.dataset_path=self.dataset_path.replace('data.h5','struc.h5')
        self.device=device
        self.preload=preload
        self.features=features

        if mode not in ['train','val','test']:
            raise ValueError("Invalid mode: {}".format(mode))
        
        with h5py.File(self.dataset_path, 'r') as f:
            # 获取所有样本组名（假设元数据组为 'metadata'，其余都是样本组）
            # 或者从 metadata/sample_names 读取，这里采用简单规则：排除 'metadata'
            self.sample_groups = [name for name in f.keys() if name.startswith('sample_')]
            self.num_samples = len(self.sample_groups)

            if self.num_samples == 0:
                raise RuntimeError(f"No sample groups found in {self.dataset_path}")

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
        self.Spliter=DatasetSpliter(self.num_samples,{'train':0.6,'val':0.2,'test':0.2})
        # print(f"train:{self.Spliter.get_len('train')} val:{self.Spliter.get_len('val')} test:{self.Spliter.get_len('test')}")
        # 预加载数据
        if self.preload:
            self.data = self._preload_all()
            mesh_max=0
            mesh_min=1e9
            for vert in self.verts_list:
                mesh_max=max(mesh_max,vert.shape[0])
                mesh_min=min(mesh_min,vert.shape[0])
            print(f"mesh_max={mesh_max}, mesh_min={mesh_min}")
        else:
            self.data = None 
        
    def __len__(self):
        return self.Spliter.get_len(self.mode)

    def _preload_all(self):
        """将所有样本的指定特征加载到内存中，返回一个列表，每个元素是一个样本字典"""
        all_samples = []
        with h5py.File(self.h5_path, 'r') as f:
            for name in self.sample_names:
                grp = f[name]
                sample = {}
                for feat in self.feature_list:
                    if feat == 'L' and self.has_sparse_L:
                        # 重建稀疏矩阵 L
                        L = self._read_sparse_L(grp, return_format='torch', device=self.device)
                        sample['L'] = L
                    else:
                        # 读取密集数组并转为 torch 张量
                        arr = grp[feat][:]
                        # 如果是字符串或对象数组，保持原样（如标签可能为字符串）
                        if arr.dtype.kind in {'U', 'S', 'O'}:
                            sample[feat] = arr  # 保持 numpy 数组，不转 tensor
                        else:
                            sample[feat] = torch.from_numpy(arr).to(self.device)
                            if sample[feat].dtype==torch.double:
                                print(":!",feat)
                                sample[feat]=sample[feat].to(torch.float)
                all_samples.append(sample)
        return all_samples

    def __getitem__(self, idx):   
        idx=self.Spliter.get_index(self.mode,idx)
        if self.preload:
            sample = self.data[idx].copy()  # 复制防止意外修改缓存
        else:
            # 按需读取
            with h5py.File(self.dataset_path, 'r') as f:
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
                # print(torch.max(sample['sulc']),torch.min(sample['sulc']))
                # print(torch.max(sample['eval']))
                # if 'cortex' in self.dataset_path:
                #     sample['label']=sample['label']+1
                # print(self.mode)
                # print(torch.max(sample['distance']),torch.min(sample['distance']))
                # print(torch.max(sample['eval']), torch.min(sample['eval']))
                # print(torch.max(sample['distance']), torch.min(sample['distance']))
                if 'label' not in sample:
                    sample['label']=sample['output']
                return {"vertices":sample['vertice'],'input': torch.ones_like(sample['label'].unsqueeze(-1)),'faces':sample['face'],\
                        "evals":sample['eval'].view(-1), "evecs":sample['evec'], \
                    "labels":sample['label'].unsqueeze(-1),"geo_feat": 
                    sample['distance'],
                    # sample['vertice'][...,:2],
                    # torch.cat([sample['distance'].unsqueeze(-1),],dim=-1),
                    "mass":sample['mass'],\
                    }


if __name__ == '__main__':
    dataset = CortexDataset(mode='train', dataset_path='../data/cortex/processed/data.h5')
    # dataset = RNAMeshDataset(mode='val', dataset_path='../data/rna') 
    # dataset = RNAMeshDataset(mode='test', dataset_path='../data/rna')
    # dataset=HumanMeshDataset(mode='train',dataset_path='../data/human')
    # dataset=HumanMeshDataset(mode='test',dataset_path='../data/human')