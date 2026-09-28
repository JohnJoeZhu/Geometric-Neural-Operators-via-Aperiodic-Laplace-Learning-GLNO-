from torch_geometric.nn import knn_graph
import h5py
import numpy as np
from tqdm import tqdm

def get_edges(h5_path, overwrite=False):
    """
    compute and store edges in h5
    """
    with h5py.File(h5_path, 'a') as f:
        sample_groups = [name for name in f.keys() if name.startswith('sample_')]
        print(f"Edge: Found {len(sample_groups)} samples in {h5_path}")
        
        for group_name in tqdm(sample_groups, desc="Computing edge"):
            grp = f[group_name]
            
            if not overwrite and 'edge' in grp:
                continue
            
            faces = grp['face'][:]
            edge = compute_edges(faces)

            if 'edge' in grp:
                del grp['edge']  # 覆盖前先删除
            grp.create_dataset('edge', data=edge)

        f.attrs['edges'] = "computed"
    print("Edge computation completed.")

def compute_edges(faces):
    """
    向量化方式计算边
    """
    # 每个面的三条边
    edges1 = np.stack([faces[:, 0], faces[:, 1]], axis=1)  # [F, 2]
    edges2 = np.stack([faces[:, 1], faces[:, 2]], axis=1)  # [F, 2]
    edges3 = np.stack([faces[:, 2], faces[:, 0]], axis=1)  # [F, 2]
    
    # 合并所有边
    all_edges = np.vstack([edges1, edges2, edges3])  # [3*F, 2]
    
    # 标准化边表示（小索引在前）
    sorted_edges = np.sort(all_edges, axis=1)
    
    # 去重
    unique_edges = np.unique(sorted_edges, axis=0)  # [E, 2]
    
    return unique_edges.T  # 返回 [2, E] 格式

# Finds the k nearest neighbors of source on target.
# Return is two tensors (distances, indices). Returned points will be sorted in increasing order of distance.
def find_knn(points_source, points_target, k, largest=False, omit_diagonal=False, method='brute'):

    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError("omit_diagonal can only be used when source and target are same shape")

    if method != 'cpu_kd' and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = 'cpu_kd'
        print("switching to cpu_kd knn")

    if method == 'brute':

        # Expand so both are NxMx3 tensor
        points_source_expand = points_source.unsqueeze(1)
        points_source_expand = points_source_expand.expand(-1, points_target.shape[0], -1)
        points_target_expand = points_target.unsqueeze(0)
        points_target_expand = points_target_expand.expand(points_source.shape[0], -1, -1)

        diff_mat = points_source_expand - points_target_expand
        dist_mat = norm(diff_mat)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float('inf')

        result = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return result
    
    elif method == 'cpu_kd':

        if largest:
            raise ValueError("can't do largest with cpu_kd")

        points_source_np = toNP(points_source)
        points_target_np = toNP(points_target)

        # Build the tree
        kd_tree = sklearn.neighbors.KDTree(points_target_np)

        k_search = k+1 if omit_diagonal else k 
        _, neighbors = kd_tree.query(points_source_np, k=k_search)
        
        if omit_diagonal: 
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape((neighbors.shape[0], neighbors.shape[1]-1))

        inds = torch.tensor(neighbors, device=points_source.device, dtype=torch.int64)
        dists = norm(points_source.unsqueeze(1).expand(-1, k, -1) - points_target[inds])

        return dists, inds
    
    else:
        raise ValueError("unrecognized method")
