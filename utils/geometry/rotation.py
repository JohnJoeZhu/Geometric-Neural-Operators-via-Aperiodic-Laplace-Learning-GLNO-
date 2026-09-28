
# Randomly rotate points.
# Torch in, torch out
# Note fornow, builds rotation matrix on CPU. 
def random_rotate_points(pts, randgen=None):
    R = random_rotation_matrix(randgen) 
    R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
    return torch.matmul(pts, R) 

def random_rotate_points_y(pts):
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
