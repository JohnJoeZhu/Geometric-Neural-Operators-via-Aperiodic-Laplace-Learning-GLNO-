import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import os.path
import sys
import random
from multiprocessing import Pool

import numpy as np
import scipy.spatial
import torch
from torch.distributions.categorical import Categorical
import sklearn.neighbors

import robust_laplacian
import potpourri3d as pp3d
from ..utils import toNP

# import diffusion_net.utils as utils
# from ..utils.utils import toNP

from utils import sparse_np_to_torch

def norm(x, highdim=False):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return torch.norm(x, dim=len(x.shape) - 1)


def norm2(x, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return dot(x, x)

def normalize(x, divide_eps=1e-6, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if(len(x.shape) == 1):
        raise ValueError("called normalize() on single vector of dim " +
                         str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called normalize() with large last dimension " +
                         str(x.shape) + " are you sure?")
    return x / (norm(x, highdim=highdim) + divide_eps).unsqueeze(-1)

def face_coords(verts, faces):
    coords = verts[faces]
    return coords

def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)

def dot(vec_A, vec_B):
    return torch.sum(vec_A * vec_B, dim=-1)

# Given (..., 3) vectors and normals, projects out any components of vecs
# which lies in the direction of normals. Normals are assumed to be unit.

def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots.unsqueeze(-1)

def face_area(verts, faces):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)
    return 0.5 * norm(raw_normal)

def face_normals(verts, faces, normalized=True):
    coords = face_coords(verts, faces)
    # print(coords.shape)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal

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


def from_basis(values, basis):
    """
    Transform data out of an orthonormal basis
    Inputs:
      - values: (B,S,K,C)
      - basis: (B,V,K)
    Outputs:
      - (B,S,V,C) reconstructed values
    """
    basis = basis.unsqueeze(1).repeat(1, values.shape[1], 1, 1)
    if values.is_complex() or basis.is_complex():
        return utils.cmatmul(utils.ensure_complex(basis), utils.ensure_complex(values))
    else:
        return torch.matmul(basis, values)

