"""Unit tests for robust/ (CPU; CUDA tests skip without a GPU). Run from gaussian_wrapping/: python -m pytest -q tests/test_robust.py"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robust
from robust.bilagrid import BilateralGridModule
from robust.losses import (appearance_photometric_loss, decoupled_ssim_map, load_ignore_mask, photometric_loss,
                           render_confidence, ssim_map, variance_losses)
from utils.sh_utils import RGB2SH

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _images(h=24, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(3, h, w, generator=g), torch.rand(3, h, w, generator=g)


def _ssim_ref(a, b):
    """Plain 3DGS SSIM map (11x11 Gaussian window, sigma 1.5, zero padding)."""
    x = torch.arange(11, dtype=torch.float32) - 5
    g = torch.exp(-x ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    win = (g[:, None] * g[None, :]).expand(3, 1, 11, 11)
    conv = lambda t: F.conv2d(t[None], win, padding=5, groups=3)[0]
    mu1, mu2 = conv(a), conv(b)
    s1, s2, s12 = conv(a * a) - mu1 ** 2, conv(b * b) - mu2 ** 2, conv(a * b) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))


def test_bilagrid_identity_init_returns_input():
    image, _ = _images()
    grid = BilateralGridModule(4, device="cpu")
    for v in (0, 3):
        assert torch.allclose(grid.apply(image, v), image, atol=1e-6)


def test_bilagrid_tv_loss():
    grid = BilateralGridModule(2, device="cpu")
    assert grid.tv_loss().item() == pytest.approx(0.0, abs=1e-12)
    with torch.no_grad():
        grid.grid.grids.add_(0.1 * torch.randn_like(grid.grid.grids))
    tv = grid.tv_loss()
    assert tv.item() > 0


def test_bilagrid_learns_view_gain_only_on_its_view():
    image, _ = _images()
    grid = BilateralGridModule(2, lr=1e-2, device="cpu")
    for _ in range(50):
        loss = (grid.apply(image, 1) - 0.5 * image).abs().mean()
        loss.backward()
        grid.step()
        grid.zero_grad()
    assert (grid.apply(image, 1) - 0.5 * image).abs().mean() < (image - 0.5 * image).abs().mean() * 0.5
    assert torch.allclose(grid.apply(image, 0), image, atol=1e-6)


def test_bilagrid_lr_schedule_and_state():
    grid = BilateralGridModule(2, lr=2e-3, total_iterations=30000, device="cpu")
    assert grid.lr_at(0) == pytest.approx(2e-5)
    assert grid.lr_at(1000) == pytest.approx(2e-3 * 0.01 ** (1000 / 30000))
    assert grid.lr_at(30000) == pytest.approx(2e-5)
    assert grid.update_lr(500) == grid.optimizer.param_groups[0]["lr"]
    with torch.no_grad():
        grid.grid.grids.mul_(2)
    other = BilateralGridModule(2, device="cpu")
    other.load_state_dict(grid.state_dict())
    assert torch.equal(other.grid.grids, grid.grid.grids)


def test_bilagrid_identity_and_mean_losses():
    grid = BilateralGridModule(3, identity_weight=2.0, mean_weight=3.0, device="cpu")
    assert grid.identity_loss().item() == 0.0 and grid.mean_loss().item() == 0.0 and grid.reg_loss().item() == 0.0
    assert grid.drift_stats() == {"identity_rms": 0.0, "view_spread_rms": 0.0}
    with torch.no_grad():                                      # same non-identity grid on every view
        grid.grid.grids.add_(0.1)
    assert grid.identity_loss().item() == pytest.approx(0.01, rel=1e-5)
    assert grid.mean_loss().item() == pytest.approx(0.0, abs=1e-12)
    assert grid.reg_loss().item() == pytest.approx(2.0 * 0.01, rel=1e-5)   # TV of a constant offset is 0
    with torch.no_grad():
        grid.grid.grids[1].add_(0.3)
    g = grid.grid.grids
    assert grid.mean_loss().item() == pytest.approx((g - g.mean(0, keepdim=True)).pow(2).mean().item(), rel=1e-6)
    assert grid.reg_loss().item() == pytest.approx(
        (grid.tv_loss() + 2.0 * grid.identity_loss() + 3.0 * grid.mean_loss()).item(), rel=1e-6)
    assert grid.drift_stats()["view_spread_rms"] > 0
    # gradient descent on the identity pull alone brings the grids back to identity
    before = grid.identity_loss().item()
    for _ in range(20):
        grid.identity_loss().backward()
        grid.step()
        grid.zero_grad()
    assert grid.identity_loss().item() < before
    # the off weights leave reg_loss == tv_loss (current default)
    plain = BilateralGridModule(2, device="cpu")
    with torch.no_grad():
        plain.grid.grids.add_(0.1 * torch.randn_like(plain.grid.grids))
    assert torch.equal(plain.reg_loss(), plain.tv_loss())


def test_bilagrid_until_switch():
    assert all(BilateralGridModule(1, device="cpu").is_active(it) for it in (1, 15000, 30000))
    grid = BilateralGridModule(1, until=100, device="cpu")
    assert grid.is_active(1) and grid.is_active(100) and not grid.is_active(101)


def _gained_grid(n=2):
    grid = BilateralGridModule(n, identity_weight=1.0, device="cpu")
    with torch.no_grad():
        grid.grid.grids.mul_(0.5)                              # every view: 0.5 * rgb
    return grid


def test_appearance_loss_grid_then_raw_after_until():
    image, gt = _images()
    grid = _gained_grid()
    grid.until = 10
    on, logs = appearance_photometric_loss(image, gt, 10, bilagrid=grid, view_idx=1)
    ref = photometric_loss(0.5 * image, gt)[0] + grid.reg_loss()
    assert on.item() == pytest.approx(ref.item(), rel=1e-5)
    assert logs["grid_reg"].item() > 0
    raw = image.clone().requires_grad_(True)
    off, logs = appearance_photometric_loss(raw, gt, 11, bilagrid=grid, view_idx=1, decoupled=True)
    assert off.item() == pytest.approx(photometric_loss(image, gt)[0].item(), rel=1e-6)
    assert "grid_reg" not in logs
    off.backward()
    assert grid.grid.grids.grad is None and raw.grad.abs().sum() > 0
    # no grid at all: unchanged photometric_loss
    plain, _ = appearance_photometric_loss(image, gt, 5)
    assert plain.item() == pytest.approx(photometric_loss(image, gt)[0].item(), rel=1e-6)


def test_appearance_loss_lambda_raw():
    image, gt = _images()
    grid = _gained_grid()
    base, _ = appearance_photometric_loss(image, gt, 1, bilagrid=grid, view_idx=0)
    with_raw, logs = appearance_photometric_loss(image, gt, 1, bilagrid=grid, view_idx=0, lambda_raw=0.5)
    raw = photometric_loss(image, gt)[0]
    assert with_raw.item() == pytest.approx(base.item() + 0.5 * raw.item(), rel=1e-5)
    assert logs["raw"].item() == pytest.approx(raw.item(), rel=1e-6)
    # the raw term also applies to an exposure-compensated image, never without a compensated one
    comp = 0.9 * image + 0.05
    exp, _ = appearance_photometric_loss(image, gt, 1, compensated=comp, decoupled=True, lambda_raw=0.5)
    ref = photometric_loss(image, gt, compensated=comp, decoupled=True)[0] + 0.5 * raw
    assert exp.item() == pytest.approx(ref.item(), rel=1e-5)
    grid.until = 0
    off, logs = appearance_photometric_loss(image, gt, 1, bilagrid=grid, view_idx=0, lambda_raw=0.5)
    assert off.item() == pytest.approx(raw.item(), rel=1e-6) and "raw" not in logs
    # confidence: the raw term sees it detached
    conf = torch.full((1, *image.shape[1:]), 2.0, requires_grad=True)
    loss, _ = appearance_photometric_loss(image, gt, 0, bilagrid=_gained_grid(), view_idx=0, lambda_raw=1.0,
                                          confidence_map=conf, conf_beta=0.1)
    loss.backward()
    grad_one = conf.grad.clone()
    conf.grad = None
    photometric_loss(0.5 * image, gt, confidence_map=conf, conf_beta=0.1)[0].backward()
    assert torch.allclose(grad_one, conf.grad, atol=1e-7)


def _robust_args(*argv):
    import argparse
    parser = argparse.ArgumentParser()
    robust.add_args(parser)
    return parser.parse_args(list(argv))


def test_robust_args_defaults_and_checks():
    args = _robust_args()
    assert (args.bilagrid_tv_weight, args.bilagrid_identity_weight, args.bilagrid_mean_weight, args.bilagrid_until,
            args.lambda_raw) == (10.0, 0.0, 0.0, -1, 0.0)
    assert BilateralGridModule(1, device="cpu").tv_weight == args.bilagrid_tv_weight
    robust.check_args(args)                                    # appearance none, nothing set
    with pytest.raises(ValueError, match="bilagrid_until"):
        robust.check_args(_robust_args("--bilagrid_until", "20000"))
    with pytest.raises(ValueError, match="lambda_raw"):
        robust.check_args(_robust_args("--lambda_raw", "0.1", "--appearance", "exposure"))
    robust.check_args(_robust_args("--lambda_raw", "0.1", "--appearance", "exposure", "--confidence"))
    args = _robust_args("--appearance", "bilagrid+decoupled", "--bilagrid_identity_weight", "1", "--bilagrid_until",
                        "27000", "--lambda_raw", "0.2")
    robust.check_args(args)
    assert robust.bilagrid_config(args) == {"tv_weight": 10.0, "identity_weight": 1.0, "mean_weight": 0.0,
                                            "until": 27000, "lambda_raw": 0.2}


def test_ssim_map_matches_reference():
    a, b = _images()
    assert torch.allclose(ssim_map(a, b), _ssim_ref(a, b), atol=1e-5)
    assert torch.allclose(ssim_map(a, a), torch.ones_like(a), atol=1e-5)


def test_photometric_loss_plain_is_gw_loss():
    image, gt = _images()
    loss, logs = photometric_loss(image, gt)
    ref = 0.8 * (image - gt).abs().mean() + 0.2 * (1 - _ssim_ref(image, gt)[:, 5:-5, 5:-5].mean())
    assert loss.item() == pytest.approx(ref.item(), rel=1e-5)
    assert logs["l1"].item() == pytest.approx((image - gt).abs().mean().item())


def test_photometric_loss_confidence_one_is_l1_dssim():
    image, gt = _images()
    loss, _ = photometric_loss(image, gt, confidence_map=torch.ones(1, *image.shape[1:]))
    ref = 0.8 * (image - gt).abs().mean() + 0.2 * (1 - _ssim_ref(image, gt).mean())
    assert loss.item() == pytest.approx(ref.item(), rel=1e-5)


def test_photometric_loss_confidence_scales_and_clamps():
    image, gt = _images()
    base, _ = photometric_loss(image, gt, confidence_map=torch.ones(1, *image.shape[1:]))
    c = 2.0
    loss, logs = photometric_loss(image, gt, confidence_map=torch.full((1, *image.shape[1:]), c), conf_beta=0.1)
    assert loss.item() == pytest.approx(c * base.item() - 0.1 * math.log(c), rel=1e-5)
    loss, logs = photometric_loss(image, gt, confidence_map=torch.full((1, *image.shape[1:]), 100.0), conf_beta=0.1)
    assert loss.item() == pytest.approx(5.0 * base.item() - 0.1 * math.log(5.0), rel=1e-5)
    assert logs["conf"].item() == pytest.approx(5.0)


def test_photometric_loss_ignore_mask_excludes_pixels():
    image, gt = _images()
    ignore = torch.zeros(1, *image.shape[1:], dtype=torch.bool)
    ignore[:, :, 16:] = True
    loss, _ = photometric_loss(image, gt, ignore_mask=ignore)
    per_pixel = 0.8 * (image - gt).abs() + 0.2 * (1 - _ssim_ref(image, gt))
    assert loss.item() == pytest.approx(per_pixel[:, :, :16].mean().item(), rel=1e-5)
    image2 = image.clone()
    image2[:, :, 22:] = 0.0                                    # beyond the SSIM window of kept pixels
    loss2, _ = photometric_loss(image2, gt, ignore_mask=ignore)
    assert loss2.item() == pytest.approx(loss.item(), rel=1e-6)


def test_photometric_loss_compensated_and_decoupled():
    image, gt = _images()
    comp = 0.9 * image + 0.05
    loss, _ = photometric_loss(image, gt, compensated=comp)
    assert loss.item() == pytest.approx(photometric_loss(comp, gt)[0].item(), rel=1e-6)
    # decoupled with compensated == raw is the ordinary per-pixel loss
    same, _ = photometric_loss(image, gt, compensated=image, decoupled=True)
    assert same.item() == pytest.approx(photometric_loss(image, gt, confidence_map=torch.ones(1, 24, 32))[0].item(),
                                        rel=1e-4)
    # L1 sees only the compensated image, contrast-structure only the raw one
    raw = image.clone().requires_grad_(True)
    comp = comp.clone().requires_grad_(True)
    loss, _ = photometric_loss(raw, gt, compensated=comp, decoupled=True, lambda_dssim=0.0)
    loss.backward()
    assert raw.grad is None or raw.grad.abs().max() == 0
    assert comp.grad.abs().max() > 0
    shifted = decoupled_ssim_map(image, image + 0.2, gt)
    plain = decoupled_ssim_map(image + 0.2, image + 0.2, gt)
    assert not torch.allclose(shifted, plain)


def test_load_ignore_mask(tmp_path):
    m = np.zeros((10, 20), dtype=np.uint8)
    m[:, 10:] = 255
    Image.fromarray(m).save(tmp_path / "a.png")
    np.save(tmp_path / "b.npy", (m > 0)[None])
    for name in ("a.jpg", "b"):
        out = load_ignore_mask(str(tmp_path), name, 5, 10, device="cpu")
        assert out.shape == (1, 5, 10) and out.dtype == torch.bool
        assert out[:, :, 5:].all() and not out[:, :, :5].any()
    assert load_ignore_mask(str(tmp_path), "missing", 5, 10, device="cpu") is None
    assert load_ignore_mask("", "a", 5, 10, device="cpu") is None


class _MockRender:
    """Per-pixel blending with fixed weights w (P, H, W): R = sum_i w_i x_i + T bg, alpha = sum_i w_i."""

    def __init__(self, weights, normals=None):
        self.w, self.n = weights, normals

    def __call__(self, cam, gaussians, pipe, bg, colors_precomp=None, require_depth=True):
        alpha = self.w.sum(0, keepdim=True)
        out = {"render": torch.einsum("phw,pc->chw", self.w, colors_precomp) + (1 - alpha) * bg[:, None, None],
               "alpha": alpha}
        if self.n is not None and require_depth:
            out["normal"] = torch.einsum("phw,pc->chw", self.w, self.n) / alpha.clamp_min(1e-12)
        return out


def _scene(p=5, h=4, w=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    rgb = torch.rand(p, 3, generator=g)
    gaussians = SimpleNamespace(get_features=RGB2SH(rgb)[:, None],
                                get_xyz=torch.randn(p, 3, generator=g), max_sh_degree=0, active_sh_degree=0,
                                get_confidence=torch.rand(p, 1, generator=g) + 0.5)
    cam = SimpleNamespace(camera_center=torch.zeros(3))
    alpha_i = torch.rand(p, h, w, generator=g) * 0.5
    trans = torch.cumprod(torch.cat([torch.ones(1, h, w), 1 - alpha_i[:-1]]), 0)
    normals = F.normalize(torch.randn(p, 3, generator=g), dim=1)
    return rgb, gaussians, cam, alpha_i * trans, normals, torch.rand(3, h, w, generator=g)


def test_variance_losses_match_brute_force():
    rgb, gaussians, cam, w, normals, gt = _scene()
    color_var, normal_var = variance_losses(cam, gaussians, None, gt, render_fn=_MockRender(w, normals))
    ref = (w[:, None] * (gt[None] - rgb[:, :, None, None]) ** 2).sum((0, 1)).mean()
    assert color_var.item() == pytest.approx(ref.item(), rel=1e-5)

    bg = torch.tensor([0.2, 0.4, 0.6])
    color_var_bg, _ = variance_losses(cam, gaussians, None, gt, render_fn=_MockRender(w, normals), bg_color=bg)
    T = 1 - w.sum(0)
    ref_bg = ref + (T * ((gt - bg[:, None, None]) ** 2).sum(0)).mean()
    assert color_var_bg.item() == pytest.approx(ref_bg.item(), rel=1e-5)

    n_sum = torch.einsum("phw,pc->chw", w, normals)
    ref_n = (w.sum(0) - (n_sum ** 2).sum(0) * (1 + T)).mean()
    assert normal_var.item() == pytest.approx(ref_n.item(), rel=1e-5)

    _, none = variance_losses(cam, gaussians, None, gt, render_fn=_MockRender(w))
    assert none.item() == 0.0


def test_variance_zero_when_all_colors_match_gt():
    rgb, gaussians, cam, w, normals, _ = _scene()
    same = rgb[:1].expand_as(rgb).contiguous()
    gaussians.get_features = RGB2SH(same)[:, None]
    gt = same[0][:, None, None].expand(3, 4, 6)
    same_n = normals[:1].expand_as(normals)
    color_var, normal_var = variance_losses(cam, gaussians, None, gt, render_fn=_MockRender(w, same_n))
    assert abs(color_var.item()) < 1e-6
    alpha, T = w.sum(0), 1 - w.sum(0)
    assert normal_var.item() == pytest.approx((alpha - alpha ** 2 * (1 + T)).mean().item(), rel=1e-5, abs=1e-6)


def test_render_confidence_normalizes_by_alpha():
    _, gaussians, cam, w, _, _ = _scene()
    conf = render_confidence(cam, gaussians, None, render_fn=_MockRender(w))
    ref = torch.einsum("phw,p->hw", w, gaussians.get_confidence[:, 0]) / w.sum(0).clamp_min(1e-3)
    assert conf.shape == (1, 4, 6)
    assert torch.allclose(conf[0], ref, atol=1e-6)


@cuda
def test_cuda_losses_match_cpu():
    image, gt = _images(64, 80)
    ignore = torch.zeros(1, 64, 80, dtype=torch.bool)
    ignore[:, 20:40] = True
    conf = torch.rand(1, 64, 80) + 0.5
    comp = 0.9 * image + 0.05
    for kw in ({}, {"confidence_map": conf}, {"ignore_mask": ignore}, {"compensated": comp, "decoupled": True}):
        cpu, _ = photometric_loss(image, gt, **kw)
        dev = {k: v.cuda() if torch.is_tensor(v) else v for k, v in kw.items()}
        gpu, _ = photometric_loss(image.cuda(), gt.cuda(), **dev)
        assert gpu.item() == pytest.approx(cpu.item(), rel=1e-4), kw


@cuda
def test_cuda_bilagrid_identity_and_grad():
    image, _ = _images()
    image = image.cuda().requires_grad_(True)
    grid = BilateralGridModule(3)
    out = grid.apply(image, 2)
    assert torch.allclose(out, image, atol=1e-6)
    assert grid.identity_loss().item() == 0.0
    (out.sum() + grid.tv_loss()).backward()
    assert image.grad is not None and grid.grid.grids.grad[2].abs().sum() > 0
    assert grid.grid.grids.grad[0].abs().sum() == 0
