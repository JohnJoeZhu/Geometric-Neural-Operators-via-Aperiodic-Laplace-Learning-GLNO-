import numpy as np
import torch
from ..utils import toNP
import h5py
from tqdm import tqdm

def get_dists(h5_path, overwrite=False):
    """
    compute and store edges in h5
    """
    with h5py.File(h5_path, 'a') as f:
        sample_groups = [name for name in f.keys() if name.startswith('sample_')]
        print(f"Distance: Found {len(sample_groups)} samples in {h5_path}")
        
        for group_name in tqdm(sample_groups, desc="Computing distance"):
            grp = f[group_name]
            
            if not overwrite and 'distance' in grp:
                continue
            
            vert = grp['vertice'][:]
            if 'face' in grp:
                face = grp['face'][:]
            else:
                face=None
            dist = compute_dists(vert,face)

            if 'distance' in grp:
                del grp['distance']  # 覆盖前先删除
            grp.create_dataset('distance', data=dist)

        f.attrs['distance'] = "computed"
    print("Distance computation completed.")

from utils.geometry.position import compute_centroid
def compute_dists(verts,faces):
    if faces is None:
        centroid=np.mean(verts,axis=0)
    else:
        centroid=compute_centroid(verts,faces)
    # print(centroid)
    norm=np.linalg.norm(verts-centroid[np.newaxis,:],axis=-1)
    return norm

# thisis uesd in sampling invariance task

# This function and the helper class below are to support parallel computation of all-pairs geodesic distance
def all_pairs_geodesic_worker(verts, faces, i):
    import igl

    N = verts.shape[0]

    # TODO: this re-does a ton of work, since it is called independently each time. Some custom C++ code could surely make it faster.
    sources = np.array([i])[:,np.newaxis]
    targets = np.arange(N)[:,np.newaxis]
    dist_vec = igl.exact_geodesic(verts, faces, sources, targets)
    
    return dist_vec
        
class AllPairsGeodesicEngine(object):
    def __init__(self, verts, faces):
        self.verts = verts 
        self.faces = faces 
    def __call__(self, i):
        return all_pairs_geodesic_worker(self.verts, self.faces, i)

def geodesic_label_errors(target_verts, target_faces, pred_labels, gt_labels, normalization='diameter', geodesic_cache_dir=None):
    """
    Return a vector of distances between predicted and ground-truth lables (normalized by geodesic diameter or area)

    This method is SLOW when it needs to recompute geodesic distances.
    """

    # move all to numpy cpu
    target_verts = toNP(target_verts) 
    target_faces = toNP(target_faces) 

    pred_labels = toNP(pred_labels) 
    gt_labels = toNP(gt_labels) 

    dists = get_all_pairs_geodesic_distance(target_verts, target_faces, geodesic_cache_dir) 

    result_dists = dists[pred_labels, gt_labels]

    if normalization == 'diameter':
        geodesic_diameter = np.max(dists)
        normalized_result_dists = result_dists / geodesic_diameter
    elif normalization == 'area':
        total_area = torch.sum(face_area(torch.tensor(target_verts), torch.tensor(target_faces)))
        normalized_result_dists = result_dists / torch.sqrt(total_area)
    else:
        raise ValueError('unrecognized normalization')

    return normalized_result_dists


def get_all_pairs_geodesic_distance(verts_np, faces_np, geodesic_cache_dir=None):
    """
    Return a gigantic VxV dense matrix containing the all-pairs geodesic distance matrix. Internally caches, recomputing only if necessary.

    (numpy in, numpy out)
    """

    # need libigl for geodesic call
    try:
        import igl
    except ImportError as e:
        raise ImportError("Must have python libigl installed for all-pairs geodesics. `conda install -c conda-forge igl`")

    # Check the cache
    found = False 
    if geodesic_cache_dir is not None:
        utils.ensure_dir_exists(geodesic_cache_dir)
        hash_key_str = str(utils.hash_arrays((verts_np, faces_np)))
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                geodesic_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")

            try:
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts_np, cache_verts)) or (not np.array_equal(faces_np, cache_faces)):
                    i_cache_search += 1
                    continue

                # This entry matches! Return it.
                found = True
                result_dists = npzfile["dist"]
                break

            except FileNotFoundError:
                break

    if not found:
                
        print("Computing all-pairs geodesic distance (warning: SLOW!)")

        # Not found, compute from scratch
        # warning: slowwwwwww

        N = verts_np.shape[0]

        try:
            pool = Pool(None) # on 8 processors
            engine = AllPairsGeodesicEngine(verts_np, faces_np)
            outputs = pool.map(engine, range(N))
        finally: # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        result_dists = np.array(outputs)

        # replace any failed values with nan
        result_dists = np.nan_to_num(result_dists, nan=np.nan, posinf=np.nan, neginf=np.nan)

        # we expect that this should be a symmetric matrix, but it might not be. Take the min of the symmetric values to make it symmetric
        result_dists = np.fmin(result_dists, np.transpose(result_dists))

        # on rare occaisions MMP fails, yielding nan/inf; set it to the largest non-failed value if so
        max_dist = np.nanmax(result_dists)
        result_dists = np.nan_to_num(result_dists, nan=max_dist, posinf=max_dist, neginf=max_dist)

        print("...finished computing all-pairs geodesic distance")

        # put it in the cache if possible
        if geodesic_cache_dir is not None:

            print("saving geodesic distances to cache: " + str(geodesic_cache_dir))

            # TODO we're potentially saving a double precision but only using a single
            # precision here; could save storage by always saving as floats
            np.savez(search_path,
                     verts=verts_np,
                     faces=faces_np,
                     dist=result_dists
                     )

    return result_dists


