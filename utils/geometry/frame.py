import numpy as np
import torch
from ..utils import toNP
import h5py
from tqdm import tqdm
from ..utils import sparse_np_to_torch
from .utils import read_sparse_L, dot, cross, project_to_tangent, normalize, face_coords, face_area

def get_dists(h5_path, overwrite=False):
    """
    compute and store edges in h5
    """
    with h5py.File(h5_path, 'a') as f:
        sample_groups = [name for name in f.keys() if name.startswith('sample_')]
        print(f"Frame: Found {len(sample_groups)} samples in {h5_path}")
        
        for group_name in tqdm(sample_groups, desc="Computing frames"):
            grp = f[group_name]
            
            if not overwrite and 'frame' in grp:
                continue
            
            vert = grp['vert'][:]
            face = grp['face'][:]
            norm = grp['norm'][:]
            frame = build_tangent_frames(vert,face,norm)

            if 'frame' in grp:
                del grp['frame']  # 覆盖前先删除
            grp.create_dataset('frame', data=frame)

        f.attrs['frames'] = "computed"
    print("Frame computation completed.")

def build_tangent_frames(verts, faces, normals):

    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device
    vert_normals = normals 

    # = find an orthogonal basis

    basis_cand1 = torch.tensor([1, 0, 0]).to(device=device, dtype=dtype).expand(V, -1)
    basis_cand2 = torch.tensor([0, 1, 0]).to(device=device, dtype=dtype).expand(V, -1)
    
    basisX = torch.where((torch.abs(dot(vert_normals, basis_cand1))
                          < 0.9).unsqueeze(-1), basis_cand1, basis_cand2)
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)
    frames = torch.stack((basisX, basisY, vert_normals), dim=-2)
    
    if torch.any(torch.isnan(frames)):
        raise ValueError("NaN coordinate frame! Must be very degenerate")

    return frames