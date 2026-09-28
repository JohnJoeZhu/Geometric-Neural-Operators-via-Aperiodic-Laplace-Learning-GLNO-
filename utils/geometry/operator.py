import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import os.path

import numpy as np
import torch

import robust_laplacian
import potpourri3d as pp3d
from ..utils import toNP

import h5py
from tqdm import tqdm

def compute_operator(verts, faces, k=128):
    """
    Laplacian decomposition
    Args:
        verts (np.ndarray): vertices [V, 3].
        faces (np.ndarray): faces [F, 3]
        k (int, optional): number of eigenvalues/vectors to compute. Default 120.

    Returns:
        - evals: (k) list of eigenvalues of the Laplacian matrix.
        - evecs: (V, k) list of eigenvectors of the Laplacian.
        - evecs_trans: (k, V) list of pseudo inverse of eigenvectors of the Laplacian.
    """
    assert k >= 0, f'Number of eigenvalues/vectors should be non-negative, bug get {k}'
    if k>verts.shape[0]:
        k=verts.shape[0]-1
    is_cloud = (faces is None)
    eps = 1e-8

    # Build Laplacian matrix
    if is_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(verts)
        massvec = M.diagonal()
    else:
        L = pp3d.cotan_laplacian(verts, faces, denom_eps=1e-10)
        massvec = pp3d.vertex_areas(verts, faces)
        massvec += eps * np.mean(massvec)

    if np.isnan(L.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec).any():
        raise RuntimeError("NaN mass matrix")

    # Compute the eigenbasis
    # Prepare matrices
    L_eigsh = (L + eps * scipy.sparse.identity(L.shape[0])).tocsc()
    massvec_eigsh = massvec
    Mmat = scipy.sparse.diags(massvec_eigsh)
    eigs_sigma = eps

    fail_cnt = 0
    while True:
        try:
            evals, evecs = sla.eigsh(L_eigsh, k=k, M=Mmat, sigma=eigs_sigma)
            # Clip off any eigenvalues that end up slightly negative due to numerical error
            evals = np.clip(evals, a_min=0., a_max=float('inf'))
            evals = evals.reshape(-1, 1)
            break
        except Exception as e:
            print(e)
            if fail_cnt > 3:
                raise ValueError('Failed to compute eigen-decomposition')
            fail_cnt += 1
            print("--- decomp failed; adding eps ===> count: " + str(fail_cnt))
            L_eigsh = L + (eps * 10 ** fail_cnt) * scipy.sparse.identity(L.shape[0])

    evecs = np.array(evecs, ndmin=2)
    # evecs_trans = evecs.T @ Mmat

    # sqrt_area = np.sqrt(Mmat.diagonal().sum())
    # evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
    # evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
    # massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    # L = utils.sparse_np_to_torch(L).to(device=device, dtype=dtype)
    return massvec, L, evals, evecs #, evecs_trans, sqrt_area

def get_operators(h5_path, k_eig=128, overwrite=False, compute_size=False):
    """
    compute and store operators in h5
    """
    total_verts = []
    with h5py.File(h5_path, 'r') as f:
        sample_groups = [name for name in f.keys() if name.startswith('sample_')]
        print(f"Operator: Found {len(sample_groups)} samples in {h5_path}")
        
        for group_name in tqdm(sample_groups, desc="Computing operators"):
            grp = f[group_name]
            
            if compute_size:
                vert = grp['vertice'][:]
                total_verts.append(vert.shape[0])
                # face = grp['face'][:]
                mass, L, eval, evec = compute_operator(vert, None, k_eig)
                continue

            if not overwrite and all(k in grp for k in ['mass', 'L_data', 'eval', 'evec']):
                continue

            vert = grp['vertice'][:]
            if 'face' in grp:
               face = grp['face'][:]
            else:
                face = None
            dtype=vert.dtype
            
            mass, L, eval, evec = compute_operator(vert, face, k_eig)
            
            # 处理稀疏矩阵 L：转换为 CSC 格式并存储分量
            # 假设 L 是 torch 稀疏张量，先转为 numpy scipy 稀疏矩阵
            if scipy.sparse.issparse(L):
                L_np = L.tocsc().astype(dtype)
            else:
                raise ValueError("Unsupported type for L: " + str(type(L)))
            
            if 'L_data' in grp:
                del grp['L_data']
                del grp['L_indices']
                del grp['L_indptr']
                del grp['L_shape']
                del grp['mass']
                del grp['eval']
                del grp['evec']
            grp.create_dataset('L_data', data=L_np.data)
            grp.create_dataset('L_indices', data=L_np.indices)
            grp.create_dataset('L_indptr', data=L_np.indptr)
            grp.create_dataset('L_shape', data=L_np.shape)
            
            for name, arr in [('mass', mass),('eval', eval), ('evec', evec)]:
                if name in grp:
                    del grp[name]  # 覆盖前先删除
                grp.create_dataset(name, data=arr)
            
            grp.attrs['k_eig'] = k_eig

        # f.attrs['operators'] = "computed"
    print("Operator computation completed.")
    print(f"mean vertices: {np.mean(total_verts)}, max vertices: {np.max(total_verts)}, min vertices: {np.min(total_verts)}")



'''
def get_operators_original(verts, faces, k_eig=128, op_cache_dir=None, normals=None, overwrite_cache=False):
    """
    See documentation for compute_operators(). This essentailly just wraps a call to compute_operators, using a cache if possible.
    All arrays are always computed using double precision for stability, then truncated to single precision floats to store on disk, and finally returned as a tensor with dtype/device matching the `verts` input.
    """

    device = verts.device
    dtype = verts.dtype
    verts_np = toNP(verts)
    faces_np = toNP(faces)
    is_cloud = faces.numel() == 0

    if(np.isnan(verts_np).any()):
        raise RuntimeError("tried to construct operators from NaN verts")

    if op_cache_dir is not None:
        utils.ensure_dir_exists(op_cache_dir)
        hash_key_str = str(utils.hash_arrays((verts_np, faces_np)))
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                op_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")
            
            try:
                # print('loading path: ' + str(search_path))
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]
                cache_k_eig = npzfile["k_eig"].item()

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts, cache_verts)) or (not np.array_equal(faces, cache_faces)):
                    i_cache_search += 1
                    print("hash collision! searching next.")
                    break
                    continue

                # print("  cache hit!")

                # If we're overwriting, or there aren't enough eigenvalues, just delete it; we'll create a new
                # entry below more eigenvalues
                if overwrite_cache: 
                    print("  overwriting cache by request")
                    os.remove(search_path)
                    break
                
                if cache_k_eig < k_eig:
                    print("  overwriting cache --- not enough eigenvalues")
                    os.remove(search_path)
                    break
                
                if "L_data" not in npzfile:
                    print("  overwriting cache --- entries are absent")
                    os.remove(search_path)
                    break


                def read_sp_mat(prefix):
                    data = npzfile[prefix + "_data"]
                    indices = npzfile[prefix + "_indices"]
                    indptr = npzfile[prefix + "_indptr"]
                    shape = npzfile[prefix + "_shape"]
                    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
                    return mat

                # This entry matches! Return it.
                frames = npzfile["frames"]
                mass = npzfile["mass"]
                L = read_sp_mat("L")
                evals = npzfile["evals"][:k_eig]
                evecs = npzfile["evecs"][:,:k_eig]

                frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
                mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
                L = utils.sparse_np_to_torch(L).to(device=device, dtype=dtype)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
                edges = torch.from_numpy(npzfile["edges"]).to(device=device, dtype=dtype)
                dis_norm = torch.from_numpy(npzfile["dis_norm"]).to(device=device, dtype=dtype)
                
                found = True
                
                break

            except FileNotFoundError:
                print("  cache miss -- constructing operators")
                break
            
            except Exception as E:
                print("unexpected error loading file: " + str(E))
                print("-- constructing operators")
                break

    if not found:
        
        # No matching entry found; recompute.
        frames, mass, L, evals, evecs = compute_operators(verts, faces, k_eig, normals=normals)
        dis_norm= compute_disnorm(verts,faces)
        dtype_np = np.float32

        # Store it in the cache
        if op_cache_dir is not None:

            L_np = utils.sparse_torch_to_np(L).astype(dtype_np)

            np.savez(search_path,
                     verts=verts_np.astype(dtype_np),
                     frames=toNP(frames).astype(dtype_np),
                     faces=faces_np,
                     k_eig=k_eig,
                     mass=toNP(mass).astype(dtype_np),
                     L_data = L_np.data.astype(dtype_np),
                     L_indices = L_np.indices,
                     L_indptr = L_np.indptr,
                     L_shape = L_np.shape,
                     evals=toNP(evals).astype(dtype_np),
                     evecs=toNP(evecs).astype(dtype_np),
                     dis_norm=toNP(dis_norm).astype(dtype_np)
                     )

    return frames, mass, L, evals, evecs, dis_norm
'''