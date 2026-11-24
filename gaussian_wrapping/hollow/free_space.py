"""Free-space prior (H2) and opacity binarisation (H3).

The prior is "unseen space is empty" (Optimize the Unseen, Segre & Avidan 2024): density is
penalised at 3D samples and the photometric loss alone decides what survives. A sample's
density is its own Gaussian's o' G(x) with filtered opacity and scale (beags/occupancy.py),
so no neighbour search is needed. The free grid marks voxels that the training views see
through, in front of their median-depth surface.
"""
from dataclasses import dataclass

import torch

from utils.graphics_utils import fov2focal

_BIAS = 1 << 20  # keys pack 3 x 21 bits: each axis index must be in [-2^20, 2^20)
_MASK = (1 << 21) - 1


def _pack(ijk):
    ijk = ijk + _BIAS
    return (ijk[..., 0] << 42) | (ijk[..., 1] << 21) | ijk[..., 2]


def _unpack(keys):
    return torch.stack([(keys >> 42) & _MASK, (keys >> 21) & _MASK, keys & _MASK], -1) - _BIAS


def _member(sorted_keys, keys):
    if sorted_keys.numel() == 0:
        return torch.zeros_like(keys, dtype=torch.bool)
    pos = torch.searchsorted(sorted_keys, keys).clamp_max(sorted_keys.numel() - 1)
    return sorted_keys[pos] == keys


def _quat_to_rot(q):
    """build_rotation on any device."""
    r, x, y, z = torch.nn.functional.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)


@torch.no_grad()
def scene_bounds(points, cameras=(), pad=0.05, max_points=1_000_000):
    """Robust AABB: 1st-99th percentile of the centres per axis, grown to hold the camera centres
    (the space the views look through), padded by 5% of the extent."""
    if points.shape[0] > max_points:
        points = points[torch.randperm(points.shape[0], device=points.device)[:max_points]]
    lo = torch.quantile(points, 0.01, dim=0)
    hi = torch.quantile(points, 0.99, dim=0)
    if cameras:
        centres = torch.stack([c.camera_center.to(points.device, points.dtype) for c in cameras])
        lo, hi = torch.minimum(lo, centres.min(dim=0).values), torch.maximum(hi, centres.max(dim=0).values)
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def default_voxel(lo, hi, resolution=256):
    return float((hi - lo).max()) / resolution


def _params(gaussians):
    return (gaussians.get_xyz, gaussians.get_rotation, gaussians.get_scaling_with_3D_filter,
            gaussians.get_opacity_with_3D_filter[:, 0])


def _own_density(mu, rot, scale, opacity, gid, limit):
    """x uniform in the +-limit sigma box of Gaussian gid (no grad) and o'_gid G_gid(x)."""
    m, R, s = mu[gid], _quat_to_rot(rot[gid]), scale[gid]
    with torch.no_grad():
        u = (torch.rand(gid.numel(), 3, device=mu.device) * 2 - 1) * limit
        x = m + torch.einsum("nij,nj->ni", R, u * s)
    q = torch.einsum("nji,nj->ni", R, x - m) / s                        # R^T (x - mu) / s
    return x, opacity[gid] * torch.exp(-0.5 * (q ** 2).sum(dim=1))


def _inside(x, lo, hi):
    return ((x >= lo) & (x <= hi)).all(dim=-1)


@torch.no_grad()
def _uniform_pairs(mu, rot, scale, lo, hi, n, limit):
    """Uniform AABB samples drawn as (Gaussian, point in its box) pairs.

    Gaussian i is drawn with probability w_i = box_i * frac_i, frac_i = share of its world-aligned
    box inside the AABB; with weight W / frac_i (W = sum w), mean(weight * o'G(x) * [x in A]) is an
    unbiased estimate of the integral of the density over A (any A inside the AABB). Returns
    (gid, weight) or (None, None) when no Gaussian overlaps the AABB.
    """
    R = _quat_to_rot(rot)
    half = limit * torch.einsum("njk,nk->nj", R.abs(), scale)
    overlap = (torch.minimum(mu + half, hi) - torch.maximum(mu - half, lo)).clamp_min(0)
    frac = (overlap / (2 * half).clamp_min(1e-12)).prod(dim=1)
    w = (2 * limit) ** 3 * scale.prod(dim=1) * frac
    total = w.sum()
    if not total > 0:
        return None, None
    gid = torch.multinomial(w, n, replacement=True)
    return gid, total / frac[gid]


@torch.no_grad()
def mean_density(gaussians, lo, hi, contains=None, volume=None, n=2 ** 22, chunk=2 ** 20, limit=3.0):
    """Mean density over the AABB, or over the region `contains` of volume `volume` inside it."""
    mu, rot, scale, opacity = _params(gaussians)
    gid, weight = _uniform_pairs(mu, rot, scale, lo, hi, n, limit)
    if gid is None:
        return 0.0
    total = 0.0
    for s in range(0, n, chunk):
        x, dens = _own_density(mu, rot, scale, opacity, gid[s:s + chunk], limit)
        keep = _inside(x, lo, hi)
        if contains is not None:
            keep &= contains(x)
        total += float((weight[s:s + chunk] * dens * keep).sum())
    return total / n / (volume if volume is not None else float((hi - lo).prod()))


@dataclass
class FreeGrid:
    keys: torch.Tensor    # (M,) sorted packed indices of free voxels
    origin: torch.Tensor  # (3,) AABB lower corner
    size: float

    def contains(self, x):
        ijk = torch.floor((x - self.origin) / self.size).long().clamp(-1, _BIAS - 1)
        return _member(self.keys, _pack(ijk))

    @property
    def volume(self):
        return self.keys.numel() * self.size ** 3


def _voxel_ijk(dims, start, stop, device):
    """(N, 3) grid indices of flat voxel ids [start, stop), i-major (the order of _pack)."""
    idx = torch.arange(start, stop, device=device)
    return torch.stack([idx // (dims[1] * dims[2]), (idx // dims[2]) % dims[1], idx % dims[2]], -1)


@torch.no_grad()
def march_free_space(cameras, depths, lo, hi, voxel, min_views=2, margin_voxels=2, chunk=1 << 21):
    """FreeGrid of the AABB voxels that >= min_views views see through.

    depths: per-camera (H, W) z-depth, 0 = no surface (the pixel's ray is free to infinity).
    A view sees through a voxel when its centre projects into the image in front of the camera
    and more than margin_voxels voxels in front of the depth at that pixel; a voxel within
    +-margin_voxels of the depth in any view is never free. Every voxel is tested in every view,
    so far voxels are not missed between sparse rays.
    """
    dims = torch.ceil((hi - lo) / voxel).long()
    n = int(dims.prod())
    count = torch.zeros(n, dtype=torch.int32, device=lo.device)
    near = torch.zeros(n, dtype=torch.bool, device=lo.device)
    margin = margin_voxels * voxel
    for cam, depth in zip(cameras, depths):
        depth = depth.to(device=lo.device, dtype=torch.float32)
        H, W = depth.shape
        view = cam.world_view_transform.to(lo.device)
        fx, fy = fov2focal(cam.FoVx, W), fov2focal(cam.FoVy, H)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            p = (lo + (_voxel_ijk(dims, s, e, lo.device) + 0.5) * voxel) @ view[:3, :3] + view[3, :3]
            z = p[:, 2].clamp_min(1e-6)
            u = torch.floor(p[:, 0] / z * fx + W / 2).long()
            v = torch.floor(p[:, 1] / z * fy + H / 2).long()
            inside = (p[:, 2] > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            d = torch.zeros_like(z)
            d[inside] = depth[v[inside], u[inside]]
            surface = inside & (d > 0)
            count[s:e] += (inside & (~surface | (z < d - margin))).int()
            near[s:e] |= surface & ((z - d).abs() <= margin)
    free = torch.nonzero((count >= min_views) & ~near).squeeze(1)
    return FreeGrid(_pack(_voxel_ijk(dims, 0, n, lo.device)[free]) if free.numel() else free, lo, float(voxel))


@torch.no_grad()
def build_free_grid(cameras, gaussians, pipe, voxel=None, min_views=2, margin_voxels=2, bounds=None):
    """Free grid from the median depth of every training view (beags_api.render_depth_alpha)."""
    from gaussian_renderer.beags_api import render_depth_alpha  # needs the rasterizer; the rest of the module does not

    lo, hi = bounds if bounds is not None else scene_bounds(gaussians.get_xyz, cameras)
    voxel = voxel if voxel is not None else default_voxel(lo, hi)
    depths = (render_depth_alpha(cam, gaussians, pipe)[0] for cam in cameras)
    return march_free_space(cameras, depths, lo, hi, voxel, min_views, margin_voxels)


class FreeSpacePrior:
    """L = mean_g(o'G) + mean density over the AABB + lambda_free * mean_g(o'G [x in free grid]).

    g: samples_per_gaussian points in the +-3 sigma box of up to max_gaussians random Gaussians.
    u: n_random uniform points in the scene AABB, as (Gaussian, point) pairs (_uniform_pairs),
       i.e. the mean density over the AABB; this spends samples on large Gaussians (fog, floaters),
       which uniform points hit most.
    The schedule scales L by lambda_fs (1e-3): the photometric loss must dominate wherever it has
    a gradient, while Gaussians with none (unseen) are moved by the prior alone at the full Adam
    step. With SparseGaussianAdam, OR `touched` into the visibility mask or they never update.
    """

    def __init__(self, cameras, gaussians, n_random=2 ** 16, samples_per_gaussian=4, voxel=None,
                 max_gaussians=100_000, limit=3.0, lambda_free=1.0):
        self.cameras = cameras
        self.lo, self.hi = scene_bounds(gaussians.get_xyz, cameras)
        self.voxel = voxel if voxel is not None else default_voxel(self.lo, self.hi)
        self.n_random = n_random
        self.samples_per_gaussian = samples_per_gaussian
        self.max_gaussians = max_gaussians
        self.limit = limit
        self.lambda_free = lambda_free
        self.grid = None
        self.touched = None

    def update_grid(self, gaussians, pipe, min_views=2, margin_voxels=2):
        self.grid = build_free_grid(self.cameras, gaussians, pipe, self.voxel, min_views, margin_voxels,
                                    bounds=(self.lo, self.hi))
        return self.grid

    def loss(self, gaussians):
        mu, rot, scale, opacity = _params(gaussians)
        P = mu.shape[0]
        if P == 0:
            return mu.sum() * 0
        own = torch.randperm(P, device=mu.device)[:self.max_gaussians].repeat_interleave(self.samples_per_gaussian)
        pairs, weight = _uniform_pairs(mu.detach(), rot.detach(), scale.detach(), self.lo, self.hi,
                                       self.n_random, self.limit)
        gid = own if pairs is None else torch.cat([own, pairs])
        x, dens = _own_density(mu, rot, scale, opacity, gid, self.limit)
        n = own.numel()
        loss = dens[:n].mean()
        if pairs is not None:
            loss = loss + (weight * dens[n:] * _inside(x[n:], self.lo, self.hi)).mean() / (self.hi - self.lo).prod()
        if self.grid is not None and self.lambda_free > 0:
            loss = loss + self.lambda_free * (dens[:n] * self.grid.contains(x[:n])).mean()
        self.touched = torch.zeros(P, dtype=torch.bool, device=mu.device)
        self.touched[gid] = True
        return loss


def opacity_entropy_loss(gaussians):
    """mean binary entropy (nats) of the filtered opacity: pushes each Gaussian to 0 or 1."""
    o = gaussians.get_opacity_with_3D_filter.clamp(1e-6, 1 - 1e-6)
    return -(o * torch.log(o) + (1 - o) * torch.log(1 - o)).mean()


@torch.no_grad()
def apply_opacity_decay(gaussians, factor):
    """opacity *= factor (CoMe decay_opacity), written in place so the Adam moments are kept."""
    o = (gaussians.get_opacity * factor).clamp_min(1e-7)
    gaussians._opacity.data.copy_(gaussians.inverse_opacity_activation(o))
