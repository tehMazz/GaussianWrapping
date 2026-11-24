"""GAVIS anisotropic visibility field (AVF) of a GW model (Xu et al., CVPR 2026, arXiv 2605.30342).

Per Gaussian i and training camera p (Supp. Alg. 2), PhiT_p = sum of the transmittance T in front of i over the
pixels it covers / its footprint area (render_ours vis=True): 1 = fully in view and unoccluded, 0 = outside the
frustum or behind something opaque. The single-view visibility along a query direction d is
    V_p(d) = PhiT_p zeta exp(kappa d.d_p),  zeta = exp(-kappa),  d_p = normalize(x_i - campos_p),
and its sum over the training views, Vtilde(d) = sum_p V_p(d), is stored as degree-2 SH (9 coefficients, eq. 8):
    gamma_lm = 4 pi zeta i_l(kappa) sum_p PhiT_p Y_lm(d_p).
The query is the AM-GM lower bound of 1 - prod_p (1 - V_p(d)):
    v(d) = 1 - (1 - Vtilde(d) / N)^N,  N = number of training views.
Y_lm is eval_sh's real basis (utils/sh_utils.py, with its signs), so projection and evaluation use the same functions.
"""
import math

import torch

from utils.sh_utils import C0, C1, C2, eval_sh

DEGREE = 2


def sh_basis(dirs):
    """(N, 3) unit directions -> (N, 9) eval_sh's degree-2 basis values."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    return torch.stack([torch.full_like(x, C0), -C1 * y, C1 * z, -C1 * x,
                        C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * z * z - x * x - y * y), C2[3] * x * z,
                        C2[4] * (x * x - y * y)], dim=1)


def bessel_i(l, kappa):
    """Modified spherical Bessel function of the first kind i_l(kappa), l = 0..2 (series below 1e-3)."""
    k = float(kappa)
    if k < 1e-3:
        return [1.0 + k * k / 6, k / 3, k * k / 15][l]
    s, c = math.sinh(k), math.cosh(k)
    return [s / k, (k * c - s) / k ** 2, ((k * k + 3) * s - 3 * k * c) / k ** 3][l]


def coefficient_scale(kappa, device=None):
    """(9,) 4 pi zeta i_l(kappa) for each SH coefficient (l = 0, 1 x3, 2 x5)."""
    zeta = math.exp(-kappa)
    per_l = [4 * math.pi * zeta * bessel_i(l, kappa) for l in range(DEGREE + 1)]
    return torch.tensor([per_l[0]] + [per_l[1]] * 3 + [per_l[2]] * 5, dtype=torch.float32, device=device)


def phi_t(vis):
    """(P, 3) render_ours vis -> (P,) PhiT in [0, 1]: sum T / max(pixel count, footprint area). Pixels the ray never
    reached (T < 1e-4 stop) or outside the image count as 0."""
    denom = torch.maximum(vis[:, 1], vis[:, 2])
    return torch.where(denom > 0, vis[:, 0] / denom.clamp_min(1e-12), torch.zeros_like(denom)).clamp(0, 1)


def directions(xyz, campos):
    """(P, 3) unit directions from the camera centre to each Gaussian."""
    return torch.nn.functional.normalize(xyz - campos.reshape(1, 3), dim=1)


@torch.no_grad()
def build_avf(gaussians, cameras, pipe, kappa=1.0, mask=None):
    """{gamma (P, 9), n_views, kappa} from the training cameras (one render_ours per camera)."""
    from gaussian_renderer.beags_api import transmittance
    xyz = gaussians.get_xyz.detach()
    gamma = torch.zeros(xyz.shape[0], 9, device=xyz.device)
    n = 0
    for cam in cameras:
        pt = phi_t(transmittance(cam, gaussians, pipe, mask=mask))
        hit = pt > 0
        if hit.any():
            gamma[hit] += pt[hit, None] * sh_basis(directions(xyz[hit], cam.camera_center))
        n += 1
    return {"gamma": gamma * coefficient_scale(kappa, xyz.device), "n_views": n, "kappa": float(kappa)}


def vtilde(avf, xyz, campos):
    """(P,) sum of the single-view visibilities along the direction campos -> x (eval_sh, clamped to [0, N])."""
    d = directions(xyz, campos)
    return eval_sh(DEGREE, avf["gamma"][:, None, :], d)[:, 0].clamp(0, max(avf["n_views"], 1))


def query_avf(avf, xyz, campos):
    """(P,) visibility v in [0, 1] along campos -> x: AM-GM bound 1 - (1 - Vtilde / N)^N."""
    n = max(avf["n_views"], 1)
    vt = vtilde(avf, xyz, campos)
    return 1 - torch.exp(n * torch.log1p(-(vt / n).clamp(max=1 - 1e-7)))


def save_avf(path, avf, **meta):
    torch.save({**avf, "gamma": avf["gamma"].cpu(), **meta}, path)


def load_avf(path, device="cuda", **expect):
    """The AVF at path, or None when missing or when a stored field differs from expect (stale cache)."""
    import os
    if not os.path.exists(path):
        return None
    avf = torch.load(path, map_location="cpu")
    if any(avf.get(k) != v for k, v in expect.items()):
        return None
    avf["gamma"] = avf["gamma"].to(device)
    return avf
