"""Mesh occupancy for the BEA-GS losses: the labelled regions of the model's own multi-label mesh replace the voxels.

The regions are read from the labelled Delaunay that multilabel_mesh_extraction.py --save_tets writes (tets.npz; the
label_<k>.ply faces are its label interfaces), so no watertight surface is needed. Each label's tets are rasterized
once into an inside mask on its own fine grid (a large background must not coarsen a small object) over the AABB of
that label's tets + margin; the Euclidean distance transform of the mask gives the distance to the region (0 inside),
cached as <mesh_dir>/dist_<k>.npz, looked up trilinearly on the GPU. Duck-types LabeledVoxels for occupancy_loss /
pivot_loss: contains() = distance <= band, distance(), tolerance() = band. Outside its grid, label 0 counts as inside
(the mesh crop does not cover the far background) and an object label as outside.
"""
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def rasterize_tets(tet_verts, origin, cell, shape, budget=20_000_000, eps=1e-6):
    """(nx, ny, nz) bool: grid nodes origin + ijk * cell inside one of the tets (T, 4, 3) (same device)."""
    device = tet_verts.device
    dims = torch.tensor(shape, device=device)
    inside = torch.zeros(shape, dtype=torch.bool, device=device)
    lo = torch.ceil((tet_verts.amin(dim=1) - origin) / cell).long().clamp_min(0)
    hi = torch.minimum(torch.floor((tet_verts.amax(dim=1) - origin) / cell).long(), dims - 1)
    size = (hi - lo + 1).clamp_min(0)
    count = size.prod(dim=1)
    edges = (tet_verts[:, 1:] - tet_verts[:, :1]).double()
    keep = (count > 0) & (torch.linalg.det(edges).abs() > 1e-12 * cell ** 3)  # drop empty bboxes and flat tets
    tet_verts, lo, size, count = tet_verts[keep], lo[keep], size[keep], count[keep]
    if not count.numel():
        return inside
    v0 = tet_verts[:, 0]
    inv = torch.linalg.inv(edges[keep].transpose(1, 2)).float()  # (T, 3, 3) world -> barycentric (1..3)
    ends = count.cumsum(0)
    starts = ends - count
    total = int(ends[-1])
    for s in range(0, total, budget):
        q = torch.arange(s, min(s + budget, total), device=device)
        t = torch.searchsorted(ends, q, right=True)
        r = q - starts[t]
        sz = size[t]
        k = r % sz[:, 2]
        j = (r // sz[:, 2]) % sz[:, 1]
        i = r // (sz[:, 2] * sz[:, 1])
        ijk = lo[t] + torch.stack([i, j, k], dim=1)
        lam = torch.einsum("nij,nj->ni", inv[t], origin + ijk * cell - v0[t])
        hit = (lam >= -eps).all(dim=1) & (lam.sum(dim=1) <= 1 + eps)
        ijk = ijk[hit]
        inside[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return inside


def distance_grid(inside, cell):
    """(nx, ny, nz) float32 Euclidean distance from each node to the nearest inside node (0 inside)."""
    from scipy.ndimage import distance_transform_edt
    mask = inside.cpu().numpy()
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return (distance_transform_edt(~mask) * cell).astype(np.float32)


@dataclass
class LabelGrid:
    values: torch.Tensor   # (nz, ny, nx) distance, grid_sample layout
    origin: torch.Tensor   # (3,)
    extent: torch.Tensor   # (3,) cell * (n - 1)
    band: float            # tolerance outside the region
    outside: float         # distance outside the grid


class MeshField:
    def __init__(self, grids):
        self.grids = grids

    @classmethod
    def build(cls, mesh_dir, label_ids, cell=None, band=None, max_nodes=200_000_000, max_ring=3, device="cuda",
              cache=True):
        """Grids for the tets of labels label_ids in <mesh_dir>/tets.npz. Per label, default cell = (AABB volume /
        max_nodes)^(1/3) and band = 2 cells (cell / band given: the same for all labels); margin (max_ring + 1) * band."""
        path = os.path.join(mesh_dir, "tets.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path}: run slurm/gw_mesh.sh (SAVE_TETS=1) on this model")
        with np.load(path) as data:
            points = data["points"].astype(np.float32)
            simplices = data["simplices"].astype(np.int64)
            tet_labels = data["tet_labels"].astype(np.int64)
        mtime = os.path.getmtime(path)
        grids = {}
        for k in label_ids:
            verts = points[np.unique(simplices[tet_labels == k])]
            if not verts.shape[0]:
                raise ValueError(f"{path}: no tet of label {k}")
            lo, hi = verts.min(0), verts.max(0)
            size = np.maximum(hi - lo, 1e-3 * (hi - lo).max())
            c = float(cell) if cell is not None else float((np.prod(size.astype(np.float64)) / max_nodes) ** (1 / 3))
            h = float(band) if band is not None else 2.0 * c
            margin = (max_ring + 1) * h
            origin = (lo - margin).astype(np.float64)
            shape = tuple(int(n) for n in np.ceil((hi + margin - origin) / c).astype(int) + 1)
            key = np.array([c, *origin, *shape, mtime], dtype=np.float64)
            cache_path = os.path.join(mesh_dir, f"dist_{k}.npz")
            t0 = time.time()
            values = None
            if cache and os.path.isfile(cache_path):
                with np.load(cache_path) as data:
                    if data["key"].shape == key.shape and np.allclose(data["key"], key, rtol=0, atol=1e-9):
                        values = data["dist"]
            how = "cached"
            if values is None:
                tets = torch.from_numpy(points[simplices[tet_labels == k]]).to(device)
                inside = rasterize_tets(tets, torch.tensor(origin, dtype=torch.float32, device=device), c, shape)
                values = distance_grid(inside, c)
                if cache:
                    np.savez(cache_path, dist=values, key=key)
                how = f"built ({int((values == 0).sum())} inside nodes)"
            print(f"[beags] mesh label {k}: {int((tet_labels == k).sum())} tets, grid {shape}, cell {c:.5f}, "
                  f"band {h:.5f}, {how} in {time.time() - t0:.1f}s")
            grids[k] = LabelGrid(
                torch.from_numpy(np.ascontiguousarray(values.transpose(2, 1, 0))).to(device),
                torch.tensor(origin, dtype=torch.float32, device=device),
                torch.tensor(c * (np.array(shape) - 1), dtype=torch.float32, device=device),
                h, 0.0 if k == 0 else float("inf"))
        return cls(grids)

    def to(self, device):
        return MeshField({k: LabelGrid(g.values.to(device), g.origin.to(device), g.extent.to(device), g.band, g.outside)
                          for k, g in self.grids.items()})

    def tolerance(self, labels):
        """(N,) band of each point's label (the largest band for a label without a grid)."""
        out = torch.full(labels.shape, max(g.band for g in self.grids.values()), device=labels.device)
        for k, g in self.grids.items():
            out[labels == k] = g.band
        return out

    @torch.no_grad()
    def lookup(self, points, labels, chunk=2_000_000):
        """(N,) distance to the region of each point's label, 0 inside (+inf for a label without a grid)."""
        out = torch.full((points.shape[0],), float("inf"), device=points.device)
        for k, g in self.grids.items():
            idx = (labels == k).nonzero().squeeze(1)
            for s in range(0, idx.numel(), chunk):
                sel = idx[s:s + chunk]
                u = (points[sel] - g.origin) / g.extent * 2 - 1
                v = F.grid_sample(g.values[None, None], u[None, None, None], mode="bilinear",
                                  align_corners=True).reshape(-1)
                out[sel] = torch.where((u.abs() <= 1).all(dim=1), v, torch.full_like(v, g.outside))
        return out

    def contains(self, points, labels, chunk=2_000_000):
        """(N,) bool: inside or within its label's band of the region of each point's label."""
        return self.lookup(points, labels, chunk) <= self.tolerance(labels)

    def distance(self, points, labels, max_ring=3, chunk=2_000_000):
        """(N,) distance to the region of each point's label, 0 inside, at most (max_ring + 1) * its band."""
        return torch.minimum(self.lookup(points, labels, chunk), (max_ring + 1) * self.tolerance(labels))
