import scipy
import numpy as np
import torch
from ..utils import toNP
import h5py
from tqdm import tqdm
from ..utils import read_sparse_L
def get_curvs(h5_path, overwrite=False):
    """
    compute and store curvs in h5
    """
    with h5py.File(h5_path, 'a') as f:
        sample_groups = [name for name in f.keys() if name.startswith('sample_')]
        print(f"Curvature: Found {len(sample_groups)} samples in {h5_path}")
        
        for group_name in tqdm(sample_groups, desc="Computing curvatures"):
            grp = f[group_name]
            
            if not overwrite and 'curv' in grp:
                continue
            
            if not ('mass' in grp and 'L_data' in grp):
                raise ValueError("Run get_operator first")
            
            vert = grp['vertice'][:]
            # face = grp['face'][:]
            # dtype=vert.dtype
            
            curv,_ = compute_mean_curvature_fast(vert, grp['mass'][:], read_sparse_L(grp))

            if 'curv' in grp:
                del grp['curv']  # 覆盖前先删除
            grp.create_dataset('curv', data=curv)

        f.attrs['curvs'] = "computed"
    print("Curvature computation completed.")

def enhance_curvature_values(curvature, method='adaptive_gamma'):
    """
    增强曲率值的对比度
    """
    if method == 'adaptive_gamma':
        # 基于数据分布的自适应伽马校正
        mean_val = curvature.mean()
        std_val = curvature.std()
        skewness = ((curvature - mean_val) ** 3).mean() / (std_val ** 3 + 1e-10)
        
        # 根据偏度选择伽马值
        if skewness > 1.0:  # 正偏态分布
            gamma = 0.5
        elif skewness < -1.0:  # 负偏态分布
            gamma = 1.1
        else:
            gamma = 1.0
            
        enhanced = torch.sign(curvature) * torch.abs(curvature).pow(gamma)
        
    elif method == 'log_enhancement':
        # 对数增强
        epsilon = 1e-6
        sign = torch.sign(curvature)
        abs_curv = torch.abs(curvature)
        enhanced = sign * torch.log(1 + abs_curv / epsilon)
        
    else:
        enhanced = curvature
        
    return enhanced

def compute_mean_curvature_fast(vertices, M, L):
    """
    使用预计算的M和L矩阵快速计算平均曲率

    参数:
        vertices: 顶点坐标，形状为 [n_vertices, 3] (numpy数组，稠密)
        M: 质量矩阵对角元，形状为 [n_vertices] (numpy数组，稠密一维)
        L: 拉普拉斯矩阵 (scipy.sparse矩阵，稀疏)

    返回:
        mean_curvature: 平均曲率，形状为 [n_vertices] (numpy数组)
        mean_curvature_vectors: 平均曲率向量，形状为 [n_vertices, 3] (numpy数组)
    """
    # 确保输入为NumPy数组（M可能需要从稠密矩阵提取对角元，但说明中为稠密一维）
    vertices = np.asarray(vertices)
    M = np.asarray(M).flatten()  # 强制为一维，若M原本是二维对角阵，可考虑使用np.diag(M)
    n_vertices = vertices.shape[0]

    # 计算 L @ vertices
    # 若L是稀疏矩阵，直接用 dot 或 @ 运算符
    L_x = L.dot(vertices)   # 或 L @ vertices

    # 计算 M 的倒数（添加极小值避免除零）
    M_inv_diag = 1.0 / (M + 1e-8)

    # 平均曲率向量 H = 0.5 * M^{-1} * (L x)
    # 利用广播将一维倒数乘到每个坐标分量上
    mean_curvature_vectors = 0.5 * L_x * M_inv_diag[:, np.newaxis]

    # 计算向量模长得到平均曲率大小
    mean_curvature = np.linalg.norm(mean_curvature_vectors, axis=1)

    return mean_curvature, mean_curvature_vectors


def compute_gaussian_curvature(X, faces, M_values):
    """
    # 假设我们有:
    # faces: 三角形面索引, shape: (m, 3)
    # X: 顶点坐标, shape: (n, 3)
    # M: 质量矩阵 (存储了顶点面积A_i) shape: (n)
    输出是每个顶点的高斯曲率K，shape: (n)
    """
    if faces.shape[-1] != 3 or X.shape[-1]!= 3:
        raise ValueError("faces must have 3 columns and X must have 3 columns")
    n_vertices = X.size(0)
    # 初始化高斯曲率和角度总和
    # K = torch.zeros(n_vertices, device=X.device)
    angle_sum = torch.zeros(n_vertices, device=X.device)


    # 遍历每个三角形
    for face in faces:
        i, j, k = face
        # 计算向量
        e_ij = X[j] - X[i]
        e_ik = X[k] - X[i]
        e_ji = X[i] - X[j]
        e_jk = X[k] - X[j]
        e_ki = X[i] - X[k]
        e_kj = X[j] - X[k]

        # 计算角度 (使用余弦定理和反三角函数)
        # 在顶点 i 的角
        cos_alpha = torch.dot(e_ij, e_ik) / (torch.norm(e_ij) * torch.norm(e_ik) + 1e-10)
        alpha = torch.acos(torch.clamp(cos_alpha, -1.0, 1.0))
        # 在顶点 j 的角
        cos_beta = torch.dot(e_ji, e_jk) / (torch.norm(e_ji) * torch.norm(e_jk) + 1e-10)
        beta = torch.acos(torch.clamp(cos_beta, -1.0, 1.0))
        # 在顶点 k 的角
        cos_gamma = torch.dot(e_ki, e_kj) / (torch.norm(e_ki) * torch.norm(e_kj) + 1e-10)
        gamma = torch.acos(torch.clamp(cos_gamma, -1.0, 1.0))

        # 将角度累加到对应的顶点上
        angle_sum[i] += alpha
        angle_sum[j] += beta
        angle_sum[k] += gamma

    # 计算角缺陷：2π - 总角度
    angle_defect = 2 * torch.tensor(torch.pi, device=X.device) - angle_sum
    # 高斯曲率 K = angle_defect / A_i
    K = angle_defect / (M_values + 1e-10)  # M_values 是质量矩阵的对角线值，即顶点面积

    return K

def compute_curv(verts,mass,L):
    curv, _=compute_mean_curvature_fast(verts,mass,L)
    min_curv = torch.min(curv)
    max_curv = torch.max(curv)
    normalized_curv = (curv - min_curv) / (max_curv - min_curv + 1e-8)
    return torch.exp(-normalized_curv) 
