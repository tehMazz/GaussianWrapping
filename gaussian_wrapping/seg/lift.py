"""Seed lift: Gaussian centres inside a 2D mask at the rendered surface, largest 3D connected component."""
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def project(xyz, world_view, fx, fy, cx, cy):
    """(P, 3) world points -> (P, 2) pixel coords and (P,) camera z. world_view: GW row-vector w2c (4, 4)."""
    p = torch.cat([xyz, torch.ones_like(xyz[:, :1])], -1) @ world_view
    z = p[:, 2]
    zs = z.clamp_min(1e-6)
    return torch.stack([p[:, 0] / zs * fx + cx, p[:, 1] / zs * fy + cy], -1), z


def seed_from_mask(xyz, world_view, intrinsics, mask, depth, depth_tol=0.05):
    """bool [P]: centre projects into `mask` (H, W) where `depth` (H, W) > 0 and |z - depth| <= depth_tol * depth."""
    H, W = mask.shape
    uv, z = project(xyz, world_view, *intrinsics)
    ij = uv.round().long()
    inside = (z > 0) & (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
    u, v = ij[:, 0].clamp(0, W - 1), ij[:, 1].clamp(0, H - 1)
    d = depth[v, u]
    return inside & mask[v, u] & (d > 0) & ((z - d).abs() <= depth_tol * d)


def knn_spacing(xyz, k=8):
    """Median distance to the k-th nearest neighbour of (N, 3) points."""
    k = min(k, len(xyz) - 1)
    if k < 1:
        return 0.0
    return float(np.median(cKDTree(xyz).query(xyz, k=k + 1)[0][:, k]))


def largest_component(xyz, voxel):
    """bool [N]: points of the 26-connected voxel component (edge `voxel`) holding the most points."""
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool)
    ijk = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64) + 1  # 1-voxel pad: neighbour keys never wrap
    dims = ijk.max(0) + 2
    pack = lambda a: (a[..., 0] * dims[1] + a[..., 1]) * dims[2] + a[..., 2]
    keys, inv = np.unique(pack(ijk), return_inverse=True)
    inv = inv.reshape(-1)
    offsets = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3)
    rows, cols = [], []
    for d in offsets[np.abs(offsets).sum(1) > 0]:
        nk = keys + pack(d)
        pos = np.minimum(np.searchsorted(keys, nk), len(keys) - 1)
        hit = keys[pos] == nk
        rows.append(np.flatnonzero(hit))
        cols.append(pos[hit])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(keys), len(keys)))
    _, comp = connected_components(graph, directed=False)
    point_comp = comp[inv]
    return point_comp == np.bincount(point_comp).argmax()
