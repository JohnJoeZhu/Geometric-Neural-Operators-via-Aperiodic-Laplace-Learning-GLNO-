import numpy as np
import torch
from utils.geometry.utils import cross, face_coords
from ..utils import toNP
import h5py
from tqdm import tqdm

def get_normals(h5_path, overwrite=False):
    """
    compute and store operators in h5
    """
    with h5py.File(h5_path, 'a') as f:
        sample_groups = [name for name in f.keys() if name.startswith('sample_')]
        print(f"Normal: Found {len(sample_groups)} samples in {h5_path}")
        
        for group_name in tqdm(sample_groups, desc="Computing normals"):
            grp = f[group_name]
            
            if not overwrite and 'normal' in grp:
                continue
            
            vert = grp['vertice'][:]
            face = grp['face'][:]
            dtype=vert.dtype
            
            normals = compute_vertex_normals(vert, face)

            if 'normal' in grp:
                del grp['normal']  # 覆盖前先删除
            grp.create_dataset('normal', data=normals)

        f.attrs['normals'] = "computed"
    print("Normal computation completed.")

def robust_vertex_normals(verts, faces, n_iter=2):
    """
    更稳健的顶点法向估计
    """
    # 计算面法向
    face_n = face_normals(verts, faces, normalized=True)
    
    # 迭代平滑法向
    for _ in range(n_iter):
        # 构建邻接矩阵
        adj_matrix = build_adjacency_matrix(faces, verts.shape[0])
        
        # 平滑法向
        smoothed_normals = torch.sparse.mm(adj_matrix, face_n)
        face_n = smoothed_normals / (torch.norm(smoothed_normals, dim=1, keepdim=True) + 1e-10)
    
    # 顶点法向为相邻面法向的平均
    vertex_normals = torch.zeros(verts.shape, device=verts.device, dtype=verts.dtype)
    for i in range(3):
        vertex_normals.index_add_(0, faces[:, i], face_n)
    
    return vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-10)

def build_adjacency_matrix(faces, n_vertices):
    """
    构建邻接矩阵用于法向平滑
    """
    # 创建边列表
    edges = torch.cat([
        faces[:, [0, 1]], faces[:, [0, 2]], 
        faces[:, [1, 0]], faces[:, [1, 2]],
        faces[:, [2, 0]], faces[:, [2, 1]]
    ], dim=0)
    
    # 去除重复边
    edges = torch.unique(edges, dim=0)
    
    # 创建稀疏邻接矩阵
    indices = edges.t()
    values = torch.ones(edges.shape[0], device=faces.device, dtype=torch.float32)
    
    return torch.sparse_coo_tensor(indices, values, (n_vertices, n_vertices))   
    
def face_normals(verts, faces, normalized=True):
    coords = face_coords(verts, faces)
    # print(coords.shape)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal

def compute_vertex_normals(verts, faces):
    ## TODO 顶点法向根据每个面的面积加权平均rather than 直接平均（cope with very ugly vertex
    # numpy in / out
    face_n = toNP(face_normals(torch.tensor(verts), torch.tensor(faces))) # ugly torch <---> numpy
    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:,i], face_n)

    norms = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    
    # 处理长度为零的法线
    zero_mask = norms == 0
    if np.any(zero_mask):
        # 设置一个小的 epsilon 值避免除以零
        norms[zero_mask] = 1e-5
        # 或者使用默认法线方向
        # vertex_normals[zero_mask] = [0, 0, 1]
    
    # 归一化法线
    vertex_normals = vertex_normals / norms

    # vertex_normals = vertex_normals / np.linalg.norm(vertex_normals,axis=-1,keepdims=True)

    return vertex_normals


def vertex_normals(verts, faces, n_neighbors_cloud=30):
    verts_np = toNP(verts)

    if faces.numel() == 0: # point cloud
    
        _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
        neigh_points = verts_np[neigh_inds,:]
        neigh_points = neigh_points - verts_np[:,np.newaxis,:]
        normals = neighborhood_normal(neigh_points)

    else: # mesh

        normals = mesh_vertex_normals(verts_np, toNP(faces))

        # if any are NaN, wiggle slightly and recompute
        bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_normals_mask.any():
            bbox = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5) * scale
            wiggle_verts = verts_np + bad_normals_mask * wiggle
            normals = mesh_vertex_normals(wiggle_verts, toNP(faces))

        # if still NaN assign random normals (probably means unreferenced verts in mesh)
        bad_normals_mask = np.isnan(normals).any(axis=1)
        if bad_normals_mask.any():
            normals[bad_normals_mask,:] = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5)[bad_normals_mask,:]
            normals = normals / np.linalg.norm(normals, axis=-1)[:,np.newaxis]
            

    normals = torch.from_numpy(normals).to(device=verts.device, dtype=verts.dtype)
        
    if torch.any(torch.isnan(normals)): raise ValueError("NaN normals :(")

    return normals


def neighborhood_normal(points):
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    # numpy in, numpy out
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:,2,:]
    return normal / np.linalg.norm(normal,axis=-1, keepdims=True)

def edge_tangent_vectors(verts, frames, edges):
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = torch.stack((compX, compY), dim=-1)

    return edge_tangent