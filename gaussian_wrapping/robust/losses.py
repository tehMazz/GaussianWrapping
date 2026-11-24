"""Photometric loss for non-clean data (CoMe confidence + SSIM-decoupled appearance) and CoMe's variance losses."""
import functools
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.sh_utils import eval_sh

try:
    from fused_ssim import fused_ssim, FusedSSIMMap
except ImportError:
    fused_ssim = FusedSSIMMap = None
try:
    from decoupled_fused_ssim import fused_ssim as decoupled_fused_ssim
except ImportError:
    decoupled_fused_ssim = None

C1 = 0.01 ** 2
C2 = 0.03 ** 2
CONF_MIN, CONF_MAX = 1e-3, 5.0


@functools.lru_cache(maxsize=None)
def _window(device, dtype):
    g = torch.exp(-(torch.arange(11, dtype=torch.float64) - 5) ** 2 / (2 * 1.5 ** 2))
    g = (g / g.sum()).float()
    return (g[:, None] @ g[None, :]).expand(3, 1, 11, 11).contiguous().to(device=device, dtype=dtype)


def _blur(x):
    return F.conv2d(x, _window(x.device, x.dtype), padding=5, groups=3)


def _moments(a, b):
    mu_a, mu_b = _blur(a), _blur(b)
    var_a = _blur(a * a) - mu_a ** 2
    var_b = _blur(b * b) - mu_b ** 2
    cov = _blur(a * b) - mu_a * mu_b
    return mu_a, mu_b, var_a, var_b, cov


def ssim_map(image, gt):
    """(3, H, W) SSIM map, zero ('same') padding; fused_ssim on CUDA."""
    if image.is_cuda and FusedSSIMMap is not None:
        return FusedSSIMMap.apply(C1, C2, image[None], gt[None], "same", True)[0]
    mu1, mu2, s1, s2, s12 = _moments(image[None], gt[None])
    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)))[0]


def decoupled_ssim_map(raw, compensated, gt):
    """CoMe SSIM-decoupled map: luminance(gt, compensated) * contrast-structure(gt, raw), (3, H, W)."""
    if raw.is_cuda and decoupled_fused_ssim is not None:
        lum, cs = decoupled_fused_ssim(gt[None], raw[None], compensated[None])
        return (lum * cs)[0]
    gt_, raw_ = gt[None], raw[None]
    mu_gt, mu_c, mu_raw = _blur(gt_), _blur(compensated[None]), _blur(raw_)
    s_gt = _blur(gt_ * gt_) - mu_gt ** 2
    s_raw = _blur(raw_ * raw_) - mu_raw ** 2
    s12 = _blur(gt_ * raw_) - mu_gt * mu_raw
    lum = (2 * mu_gt * mu_c + C1) / (mu_gt ** 2 + mu_c ** 2 + C1)
    cs = (2 * s12 + C2) / (s_gt.clamp_min(1e-12) + s_raw.clamp_min(1e-12) + C2)
    return (lum * cs)[0]


def photometric_loss(image, gt, compensated=None, lambda_dssim=0.2, decoupled=False, confidence_map=None,
                     conf_beta=0.075, ignore_mask=None):
    """(1-λ) L1 + λ D-SSIM. Returns (loss, logs).

    compensated: appearance-corrected render (3, H, W). decoupled=False: L1 and SSIM on it (gsplat bilagrid);
    decoupled=True: L1 on it, SSIM = luminance(compensated) * contrast-structure(raw) (CoMe).
    confidence_map c (1, H, W): per-pixel c*L_rgb - conf_beta*log c on the combined L1+D-SSIM, c in [1e-3, 5] (CoMe).
    ignore_mask (1, H, W) bool, True = pixel excluded from the average.
    Without confidence/ignore and not decoupled this is GW's loss (fused_ssim 'valid' mean).
    """
    target = image if compensated is None else compensated
    if decoupled and compensated is None:
        raise ValueError("decoupled appearance needs a compensated image")

    if confidence_map is None and ignore_mask is None and not decoupled:
        l1 = (target - gt).abs().mean()
        if target.is_cuda and fused_ssim is not None:
            ssim = fused_ssim(target[None], gt[None], padding="valid")
        else:
            ssim = ssim_map(target, gt)[:, 5:-5, 5:-5].mean()
        loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim)
        return loss, {"l1": l1.detach(), "dssim": (1.0 - ssim).detach()}

    l1_map = (target - gt).abs()
    s_map = decoupled_ssim_map(image, compensated, gt) if decoupled else ssim_map(target, gt)
    rgb = (1.0 - lambda_dssim) * l1_map + lambda_dssim * (1.0 - s_map)
    logs = {}
    if confidence_map is not None:
        c = confidence_map.clamp(CONF_MIN, CONF_MAX)
        logs["conf"] = c.detach().mean()
        rgb = c * rgb - conf_beta * torch.log(c)
    if ignore_mask is None:
        logs.update(l1=l1_map.detach().mean(), dssim=(1.0 - s_map).detach().mean())
        return rgb.mean(), logs
    keep = (~ignore_mask).to(rgb.dtype).expand_as(rgb)
    n = keep.sum().clamp_min(1.0)
    logs.update(l1=(l1_map.detach() * keep).sum() / n, dssim=((1.0 - s_map.detach()) * keep).sum() / n)
    return (rgb * keep).sum() / n, logs


def appearance_photometric_loss(image, gt, iteration, bilagrid=None, view_idx=None, compensated=None,
                                decoupled=False, lambda_raw=0.0, confidence_map=None, **kw):
    """train.py's robust photometric loss with an optional bilateral grid. Returns (loss, logs).

    bilagrid active at `iteration` (BilateralGridModule.is_active): compensated = grid(image, view_idx), and
    loss += bilagrid.reg_loss() (TV + identity + mean pulls). Inactive (past --bilagrid_until): the raw render drives
    the loss, as without a grid. `compensated` (e.g. exposure compensation) is used when no grid is active.
    decoupled only applies when there is a compensated image.
    lambda_raw > 0 with a compensated image: + lambda_raw * photometric_loss(raw render), same ignore mask, confidence
    detached (its -beta log c term then carries no gradient), so the raw render also fits the photos.
    kw: photometric_loss's lambda_dssim, conf_beta, ignore_mask.
    """
    grid_on = bilagrid is not None and bilagrid.is_active(iteration)
    if grid_on:
        compensated = bilagrid.apply(image, view_idx)
    loss, logs = photometric_loss(image, gt, compensated=compensated, decoupled=decoupled and compensated is not None,
                                  confidence_map=confidence_map, **kw)
    if grid_on:
        reg = bilagrid.reg_loss()
        loss = loss + reg
        logs["grid_reg"] = reg.detach()
    if lambda_raw > 0 and compensated is not None:
        conf = None if confidence_map is None else confidence_map.detach()
        raw, _ = photometric_loss(image, gt, confidence_map=conf, **kw)
        loss = loss + lambda_raw * raw
        logs["raw"] = raw.detach()
    return loss, logs


@functools.lru_cache(maxsize=None)
def _ignore_mask_cpu(directory, image_name, height, width):
    stem = os.path.splitext(image_name)[0]
    for name in (image_name, stem):
        npy, png = os.path.join(directory, name + ".npy"), os.path.join(directory, name + ".png")
        if os.path.exists(npy):
            m = np.load(npy)
            m = m.any(axis=0) if m.ndim == 3 else m != 0  # (C, H, W): any channel
        elif os.path.exists(png):
            m = np.array(Image.open(png).convert("L")) != 0
        else:
            continue
        m = torch.from_numpy(np.ascontiguousarray(m)).float()[None, None]
        return F.interpolate(m, size=(height, width), mode="nearest")[0] > 0.5
    return None


def load_ignore_mask(directory, image_name, height, width, device="cuda"):
    """<directory>/<image_name>{.npy,.png} nonzero = ignore -> (1, H, W) bool, or None if missing."""
    if not directory:
        return None
    m = _ignore_mask_cpu(directory, image_name, height, width)
    return None if m is None else m.to(device)


def view_colors(cam, gaussians):
    """Per-Gaussian view-dependent RGB (P, 3), as convert_SHs_python."""
    shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
    dirs = F.normalize(gaussians.get_xyz - cam.camera_center[None], dim=1)
    return torch.clamp_min(eval_sh(gaussians.active_sh_degree, shs_view, dirs) + 0.5, 0.0)


def _default_render():
    from gaussian_renderer.ours import render_ours
    return render_ours


def render_confidence(cam, gaussians, pipe, render_fn=None):
    """Rendered exp(confidence) normalized by alpha, (1, H, W) (CoMe: / max(alpha, 1e-3))."""
    render_fn = render_fn or _default_render()
    conf = gaussians.get_confidence.repeat(1, 3)
    bg = torch.zeros(3, device=conf.device)
    pkg = render_fn(cam, gaussians, pipe, bg, colors_precomp=conf, require_depth=False)
    return pkg["render"][:1] / pkg["alpha"].clamp_min(1e-3)


def variance_losses(cam, gaussians, pipe, gt, render_fn=None, bg_color=None):
    """CoMe colour and normal variance along rays, both scalar means over pixels.

    colour: sum_i w_i ||gt - c_i||^2 = R(c^2) - 2 gt R(c) + gt^2 alpha, summed over channels;
    bg_color given -> + T ||gt - bg||^2 as CoMe. normal: alpha - ||sum_i w_i n_i||^2 (1 + T), T detached, from the
    rasterizer's per-Gaussian normals (0 if render_fn returns no normal).
    """
    render_fn = render_fn or _default_render()
    colors = view_colors(cam, gaussians)
    bg = torch.zeros(3, device=colors.device)
    first = render_fn(cam, gaussians, pipe, bg, colors_precomp=colors, require_depth=True)
    second = render_fn(cam, gaussians, pipe, bg, colors_precomp=colors * colors, require_depth=False)
    alpha = first["alpha"]
    color_var = second["render"] - 2 * gt * first["render"] + gt * gt * alpha
    color_var = color_var.sum(dim=0)
    if bg_color is not None:
        color_var = color_var + (1 - alpha[0]) * ((gt - bg_color[:, None, None]) ** 2).sum(dim=0)
    normal = first.get("normal")
    if normal is None:
        normal_var = torch.zeros((), device=colors.device)
    else:
        weighted = normal * alpha
        normal_var = (alpha - (weighted * weighted).sum(dim=0, keepdim=True) * (2 - alpha.detach())).mean()
    return color_var.mean(), normal_var
