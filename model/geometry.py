import torch
import utils
import numpy as np
def to_basis(values, basis, massvec):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (B,S,V,C)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,S,K,C) transformed values
    """
    basisT = basis.transpose(-2, -1).unsqueeze(1).repeat(1, values.shape[1], 1, 1)
    # print("basisT", basisT.shape)
    # print((values * massvec.unsqueeze(1).unsqueeze(-1)).shape)
    return torch.matmul(basisT, (values * massvec.unsqueeze(1).unsqueeze(-1)))

def to_basis2(values, basis, massvec):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (B,S,g,V,C)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,S,g,K,C) transformed values
    """
    basisT = basis.transpose(-2, -1).unsqueeze(1).unsqueeze(1).repeat(1, values.shape[1],values.shape[2], 1, 1)
    # print("basisT", basisT.shape)
    # print((values * massvec.unsqueeze(1).unsqueeze(-1)).shape)
    return torch.matmul(basisT, (values * massvec.unsqueeze(1).unsqueeze(1).unsqueeze(-1)))

def from_basis(values, basis):
    """
    Transform data out of an orthonormal basis
        Inputs:
        - values: (B,...,K,C_in)
        - basis: (B,V,K)
        Outputs:
        - (B,...,V,C) reconstructed values
    """
    # basis = basis.unsqueeze(1).repeat(1, values.shape[1], 1, 1)
    if values.is_complex() or basis.is_complex():
        raise ValueError("from basis unsupport complex matmul")
        # return utils.cmatmul(utils.ensure_complex(basis), utils.ensure_complex(values))
    else:
        return torch.einsum('b...kc,bvk->b...vc', values, basis)
        # return torch.matmul(basis, values)
    

# Randomly rotate points.
# Torch in, torch out
# Note fornow, builds rotation matrix on CPU. 
def rotate(pts, randgen=None):
    R = random_rotation_matrix(randgen) 
    R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
    return torch.matmul(pts, R) 

def rotate_y(pts):
    angles = torch.rand(1, device=pts.device, dtype=pts.dtype) * (2. * np.pi)
    rot_mats = torch.zeros(3, 3, device=pts.device, dtype=pts.dtype)
    rot_mats[0,0] = torch.cos(angles)
    rot_mats[0,2] = torch.sin(angles)
    rot_mats[2,0] = -torch.sin(angles)
    rot_mats[2,2] = torch.cos(angles)
    rot_mats[1,1] = 1.

    pts = torch.matmul(pts, rot_mats)
    return pts

def random_rotation_matrix(randgen=None):
    """
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    """
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randgen is None:
        randgen = np.random.RandomState()
        
    theta, phi, z = tuple(randgen.rand(3).tolist())
    
    theta = theta * 2.0*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0 # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def compute_dis(verts, central):
    # verts: [number_of_points, 1, 3]
    # central: [1, number_of_central, 3]
    verts_expanded = verts.unsqueeze(-2)  # [n_points, 1, 3]
    central_expanded = central.unsqueeze(-3)  # [1, n_central, 3]
    
    distances = torch.sqrt(torch.sum((verts_expanded - central_expanded) ** 2, dim=-1))
    
    return distances