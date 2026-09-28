import scipy
import numpy as np
import torch
from ..utils import toNP
import h5py
from tqdm import tqdm
from .frame import build_tangent_frames
from .utils import read_sparse_L

def get_grads(h5_path, overwrite=False):
    """
    compute and store operators in h5
    """
    with h5py.File(h5_path, 'a') as f:
        sample_groups = [name for name in f.keys() if name.startswith('sample_')]
        print(f"Gradient: Found {len(sample_groups)} samples in {h5_path}")
        
        for group_name in tqdm(sample_groups, desc="Computing operators"):
            grp = f[group_name]
            
            if not overwrite and all(k in grp for k in ['gradX', 'gradY']):
                continue
            
            vert = grp['vertice'][:]
            face = grp['face'][:]
            dtype=vert.dtype
            
            gradX, gradY = build_gradient_mats(vert, face, frames=grp['frame'][:], L=read_sparse_L(grp))

            for name, arr in [('gradX', gradX),('gradY', gradY)]:
                if name in grp:
                    del grp[name]  # 覆盖前先删除
                grp.create_dataset(name, data=arr)

        f.attrs['gradients'] = "computed"
    print("Gradient computation completed.")

def build_gradient_mats(verts, faces, frames, L):
    # == Build gradient matrices
    device = verts.device
    dtype = verts.dtype

    eps = 1e-5

    frames = build_tangent_frames(verts, faces, normals=normals)
    frames_np = toNP(frames)
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # For meshes, we use the same edges as were used to build the Laplacian. For point clouds, use a whole local neighborhood
    if faces.numel() == 0:
        grad_mat_np = build_grad_point_cloud(verts, frames)
    else:
        edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype)
        edge_vecs = edge_tangent_vectors(verts, frames, edges)
        grad_mat_np = build_grad(verts, edges, edge_vecs)


    # Split complex gradient in to two real sparse mats (torch doesn't like complex sparse matrices)
    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)
    gradX = utils.sparse_np_to_torch(gradX_np).to(device=device, dtype=dtype)
    gradY = utils.sparse_np_to_torch(gradY_np).to(device=device, dtype=dtype)
    return gradX, gradY

def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at vertices, produces a complex (vector value) at vertices giving the gradient. All values pointwise.
    - edges: (2, E)
    """
    
    edges_np = toNP(edges)
    edge_tangent_vectors_np = toNP(edge_tangent_vectors)

    # TODO find a way to do this in pure numpy?

    # Build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges_np.shape[1]):
        tail_ind = edges_np[0, iE]
        tip_ind = edges_np[1, iE]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(iE)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5
    for iV in range(N):
        n_neigh = len(vert_edge_outgoing[iV])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iV]
        for i_neigh in range(n_neigh):
            iE = vert_edge_outgoing[iV][i_neigh]
            jV = edges_np[1, iE]
            ind_lookup.append(jV)
    
            edge_vec = edge_tangent_vectors[iE][:]
            w_e = 1.

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iV)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(
            N, N)).tocsc()

    return mat


def build_grad_point_cloud(verts, frames, n_neighbors_cloud=30):
    verts_np = toNP(verts)
    frames_np = toNP(frames)

    _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
    neigh_points = verts_np[neigh_inds,:]
    neigh_vecs = neigh_points - verts_np[:,np.newaxis,:]

    # TODO this could easily be way faster. For instance we could avoid the weird edges format and the corresponding pure-python loop via some numpy broadcasting of the same logic. The way it works right now is just to share code with the mesh version. But its low priority since its preprocessing code.

    edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_neighbors_cloud)
    edges = np.stack((edge_inds_from, neigh_inds.flatten()))
    edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)
   
    return build_grad(verts_np, torch.tensor(edges), edge_tangent_vecs)
    