import numpy as np
import trimesh
def compute_centroid(points, faces):
    if np.any(np.isnan(points)):
        raise ValueError("points contains NaN")
    if np.any(np.isinf(points)):
        raise ValueError("points contains inf")
    if np.max(faces)>=points.shape[0]:
        raise ValueError("faces contains invalid index")
    mesh = trimesh.Trimesh(vertices=points, faces=faces)
    if not np.any(np.isnan(mesh.center_mass)):
        return mesh.center_mass
    areas = mesh.area_faces                     # 形状 (n_faces,)
    centroids = mesh.triangles_center           # 形状 (n_faces, 3)
    # 或者用顶点坐标手动计算
    # centroids = mesh.vertices[mesh.faces].mean(axis=1)

    # 计算面积加权质心
    weighted_centroid = np.average(centroids, weights=areas, axis=0)
    # print(weighted_centroid)
    return weighted_centroid

def normalize_positions(pos, faces=None, method='mean', scale_method='max_rad'):
    # center and unit-scale positions

    if method == 'mean':
        # center using the average point position
        pos = (pos - torch.mean(pos, dim=-2, keepdim=True))
    elif method == 'bbox': 
        # center via the middle of the axis-aligned bounding box
        bbox_min = torch.min(pos, dim=-2).values
        bbox_max = torch.max(pos, dim=-2).values
        center = (bbox_max + bbox_min) / 2.
        pos -= center.unsqueeze(-2)
    else:
        raise ValueError("unrecognized method")

    if scale_method == 'max_rad':
        scale = torch.max(norm(pos), dim=-1, keepdim=True).values.unsqueeze(-1)
        pos = pos / scale
    elif scale_method == 'area': 
        if faces is None:
            raise ValueError("must pass faces for area normalization")
        coords = pos[faces]
        vec_A = coords[:, 1, :] - coords[:, 0, :]
        vec_B = coords[:, 2, :] - coords[:, 0, :]
        face_areas = torch.norm(torch.cross(vec_A, vec_B, dim=-1), dim=1) * 0.5
        total_area = torch.sum(face_areas)
        scale = (1. / torch.sqrt(total_area))
        pos = pos * scale
    else:
        raise ValueError("unrecognized scale method")
    return pos
