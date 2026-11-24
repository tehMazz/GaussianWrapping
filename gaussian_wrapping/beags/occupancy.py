"""Labelled-voxel occupancy loss (BEA-GS `get_sampling_loss` + utils/occ_utils.py).

Samples drawn inside each object Gaussian (+-3 sigma, 3D-filtered scale) are penalised by filtered opacity * kernel
value when no voxel of the same label is within the 27-voxel neighbourhood. Only voxels whose
points all carry one label ("pure" voxels) count.
"""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from utils.general_utils import build_rotation

_BIAS = 1 << 20  # keys pack 3 x 21 bits: each axis index must be in [-2^20, 2^20)


def _pack(ijk):
    ijk = ijk + _BIAS
    return (ijk[..., 0] << 42) | (ijk[..., 1] << 21) | ijk[..., 2]


def _pack_offsets(d):
    """_pack(ijk + d) == _pack(ijk) + _pack_offsets(d) while every axis stays in range."""
    return d[:, 0] * (1 << 42) + d[:, 1] * (1 << 21) + d[:, 2]


def neighbour_offsets(radius=1, shell=False, device="cuda"):
    """(K, 3) voxel offsets with Chebyshev norm <= radius (== radius with shell)."""
    o = torch.arange(-radius, radius + 1, device=device)
    d = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)
    return d[d.abs().amax(dim=1) == radius] if shell else d


def voxel_size(points, k=25, target=25, trim=0.05, max_queries=200_000, seed=0):
    """Cube edge expected to hold `target` points, from the median k-NN density (5% trimmed)."""
    pts = points.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    queries = pts[rng.choice(len(pts), min(len(pts), max_queries), replace=False)]
    radius = cKDTree(pts).query(queries, k=k, workers=-1)[0][:, -1]
    lo, hi = np.quantile(radius, [trim, 1 - trim])
    r = np.median(radius[(radius >= lo) & (radius <= hi)])
    density = k / ((4.0 / 3.0) * np.pi * r ** 3 + 1e-12)
    return float((target / (density + 1e-12)) ** (1 / 3))


@dataclass
class LabeledVoxels:
    keys: torch.Tensor    # (M,) int64 sorted keys of pure voxels
    labels: torch.Tensor  # (M,) int64 label of each voxel
    origin: torch.Tensor  # (1, 3)
    size: float

    def to(self, device):
        return LabeledVoxels(self.keys.to(device), self.labels.to(device), self.origin.to(device), self.size)

    @classmethod
    def build(cls, points, labels, size, chunk_size=1_000_000):
        origin = points.min(dim=0).values[None]
        chunk_keys, chunk_lows, chunk_highs = [], [], []
        for start in range(0, points.shape[0], chunk_size):
            stop = min(start + chunk_size, points.shape[0])
            keys = _pack(torch.floor((points[start:stop] - origin) / size).long())
            keys, order = torch.sort(keys)
            chunk_labels = labels[start:stop][order].long()
            uniq, inverse = torch.unique_consecutive(keys, return_inverse=True)
            lo = torch.full_like(uniq, torch.iinfo(torch.int64).max).scatter_reduce(
                0, inverse, chunk_labels, "amin")
            hi = torch.full_like(uniq, torch.iinfo(torch.int64).min).scatter_reduce(
                0, inverse, chunk_labels, "amax")
            chunk_keys.append(uniq)
            chunk_lows.append(lo)
            chunk_highs.append(hi)

        keys = torch.cat(chunk_keys)
        lows = torch.cat(chunk_lows)
        highs = torch.cat(chunk_highs)
        keys, order = torch.sort(keys)
        lows, highs = lows[order], highs[order]
        uniq, inverse = torch.unique_consecutive(keys, return_inverse=True)
        lo = torch.full_like(uniq, torch.iinfo(torch.int64).max).scatter_reduce(0, inverse, lows, "amin")
        hi = torch.full_like(uniq, torch.iinfo(torch.int64).min).scatter_reduce(0, inverse, highs, "amax")
        pure = lo == hi
        if not pure.any():
            raise ValueError("no single-label voxel: point cloud empty or voxel size too large")
        return cls(uniq[pure], lo[pure], origin, float(size))

    def any_hit(self, points, labels, offsets, chunk=2_000_000):
        """(N,) bool: a pure voxel with the same label at one of `offsets` (K, 3) from each point's voxel."""
        packed = _pack_offsets(offsets)
        out = torch.empty(points.shape[0], dtype=torch.bool, device=points.device)
        for s in range(0, points.shape[0], chunk):
            keys = _pack(torch.floor((points[s:s + chunk] - self.origin) / self.size).long())[:, None] + packed
            pos = torch.searchsorted(self.keys, keys).clamp_max(self.keys.numel() - 1)
            hit = (self.keys[pos] == keys) & (self.labels[pos] == labels[s:s + chunk, None].long())
            out[s:s + chunk] = hit.any(dim=1)
        return out

    def contains(self, points, labels, chunk=2_000_000):
        """(N,) bool: a pure voxel with the same label lies in the 27-neighbourhood of each point."""
        return self.any_hit(points, labels, neighbour_offsets(device=points.device), chunk)

    def tolerance(self, labels):
        """Distance from a label's region that L_piv forgives: one voxel edge."""
        return self.size

    @torch.no_grad()
    def distance(self, points, labels, max_ring=3, chunk=2_000_000):
        """(N,) 0 within the 27-neighbourhood of a same-label voxel, else size * (first ring <= max_ring with one),
        else size * (max_ring + 1)."""
        d = torch.full((points.shape[0],), (max_ring + 1) * self.size, device=points.device)
        todo = torch.arange(points.shape[0], device=points.device)
        for ring in range(1, max_ring + 1):
            offsets = neighbour_offsets(ring, shell=ring > 1, device=points.device)
            hit = self.any_hit(points[todo], labels[todo], offsets, max(1, chunk * 27 // offsets.shape[0]))
            d[todo[hit]] = 0.0 if ring == 1 else ring * self.size
            todo = todo[~hit]
            if not todo.numel():
                break
        return d


def occupancy_loss(gaussians, gaussian_labels, voxels, samples_per_gaussian=20, limit=3.0, include_background=False,
                   max_gaussians=200_000, min_weight=1 / 255, chunk=2_000_000):
    """mean(o' * G(x) * [x outside same-label voxels]) over samples of object Gaussians
    (all labelled Gaussians with include_background; the voxels must then hold label-0 points too).
    Each call draws a random subset of max_gaussians Gaussians (None: all); samples with o'G < min_weight
    skip the voxel test and count as inside."""
    obj = gaussian_labels >= 0 if include_background else gaussian_labels > 0
    idx = obj.nonzero().squeeze(1)
    if not idx.numel():
        return gaussians.get_xyz.sum() * 0
    if max_gaussians is not None and idx.numel() > max_gaussians:
        idx = idx[torch.randperm(idx.numel(), device=idx.device)[:max_gaussians]]
    mu = gaussians.get_xyz[idx]
    rot = build_rotation(gaussians.get_rotation[idx])
    scale = gaussians.get_scaling_with_3D_filter[idx]
    opacity = gaussians.get_opacity_with_3D_filter[idx, 0]
    gid = torch.arange(mu.shape[0], device=mu.device).repeat_interleave(samples_per_gaussian)
    with torch.no_grad():
        u = (torch.rand(gid.numel(), 3, device=mu.device) * 2 - 1) * limit
        x = mu[gid] + torch.einsum("nij,nj->ni", rot[gid], u * scale[gid])
        test = opacity[gid] * torch.exp(-0.5 * (u ** 2).sum(dim=1)) >= min_weight  # G(x) = exp(-|u|^2 / 2)
        outside = torch.zeros_like(test)
        outside[test] = ~voxels.contains(x[test], gaussian_labels[idx][gid[test]], chunk)
        sel = outside.nonzero().squeeze(1)
    g = gid[sel]
    q = torch.einsum("nji,nj->ni", rot[g], x[sel] - mu[g]) / scale[g]           # R^T (x - mu) / s
    kernel = torch.exp(-0.5 * (q ** 2).sum(dim=1))
    return (opacity[g] * kernel).sum() / gid.numel()
