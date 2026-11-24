"""BEA-GS label extension of GW's rasterizer (diff_gaussian_rasterization_ours_beags). Run on a GPU node:
    pytest -q gaussian_wrapping/tests/test_rasterizer_labels.py
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "submodules"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
if torch.cuda.is_available():
    pytest.importorskip("diff_gaussian_rasterization_ours_beags")

PIPE = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
BG = (0.1, 0.2, 0.3)
PARAMS = ("xyz", "scaling", "rotation", "opacity", "f_dc", "f_rest")


def make_camera(width=128, height=96, fov_deg=60.0):
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2
    fovx = math.radians(fov_deg)
    fovy = 2 * math.atan(math.tan(fovx / 2) * height / width)
    view = torch.tensor(getWorld2View2(np.eye(3), np.zeros(3)), dtype=torch.float32).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    return SimpleNamespace(
        FoVx=fovx, FoVy=fovy, image_width=width, image_height=height,
        world_view_transform=view,
        full_proj_transform=(view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0),
        camera_center=view.inverse()[3, :3],
    )


def make_gaussians(n=2000, seed=0, sh_degree=3):
    """Random Gaussians in front of the camera; all tensors are leaf params with grad."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rand = lambda *s: torch.rand(*s, generator=g, device="cuda")
    xyz = torch.stack([(rand(n) - 0.5) * 2.0, (rand(n) - 0.5) * 1.5, 2.0 + 2.0 * rand(n)], dim=1)
    params = dict(
        xyz=xyz,
        scaling=torch.log(0.02 + 0.08 * rand(n, 3)),
        rotation=torch.nn.functional.normalize(rand(n, 4) - 0.5, dim=1),
        opacity=torch.logit(0.2 + 0.6 * rand(n, 1)),
        f_dc=(rand(n, 1, 3) - 0.5),
        f_rest=(rand(n, (sh_degree + 1) ** 2 - 1, 3) - 0.5) * 0.1,
    )
    return {k: v.clone().requires_grad_(True) for k, v in params.items()}, sh_degree


def as_model(p, sh_degree):
    """Duck-typed GaussianModel for render_ours (no 3D filter)."""
    return SimpleNamespace(
        get_xyz=p["xyz"], _xyz=p["xyz"], use_unbounded_opacity=False,
        get_scaling_n_opacity_with_3D_filter=(torch.exp(p["scaling"]), torch.sigmoid(p["opacity"])),
        get_rotation=torch.nn.functional.normalize(p["rotation"], dim=1),
        get_features=torch.cat([p["f_dc"], p["f_rest"]], dim=1),
        active_sh_degree=sh_degree,
    )


def _render(params, deg, cam, **kw):
    from gaussian_renderer.ours import render_ours
    return render_ours(cam, as_model(params, deg), PIPE, torch.tensor(BG, device="cuda"), **kw)


def _split_labels(cam, n_gauss, num_labels=3, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gl = torch.randint(0, num_labels, (n_gauss,), generator=g, device="cuda", dtype=torch.int32)
    pl = torch.randint(-1, num_labels, (cam.image_height, cam.image_width), generator=g, device="cuda",
                       dtype=torch.int32)
    return gl, pl


def _close_to_scale(a, b, rel=1e-4, name=""):
    """|a - b| <= rel * max|b|: gradients summed with float atomics differ in summation order."""
    err, scale = (a - b).abs().max().item(), b.abs().max().item()
    assert err <= rel * scale + 1e-12, f"{name}: max abs diff {err:.3e} vs scale {scale:.3e}"


def _zero_grads(params):
    for p in params.values():
        p.grad = None


def _all_outputs_and_grads(params, deg, cam, **kw):
    """Forward outputs and grads of a random-weighted sum of image, median depth, alpha and normal."""
    _zero_grads(params)
    out = _render(params, deg, cam, **kw)
    g = torch.Generator(device="cuda").manual_seed(7)
    loss = 0
    for k in ("render", "median_depth", "alpha", "normal"):
        loss = loss + (out[k] * torch.rand(out[k].shape, generator=g, device="cuda")).sum()
    loss.backward()
    fwd = {k: out[k].detach().clone() for k in ("render", "median_depth", "alpha", "normal", "radii")}
    grads = {k: params[k].grad.clone() for k in PARAMS}
    grads["viewspace"] = out["viewspace_points"].grad.clone()
    return fwd, grads


def test_labels_off_matches_stock(monkeypatch):
    import gaussian_renderer.ours as ours
    cam = make_camera()
    params, deg = make_gaussians()
    fwd, grads = _all_outputs_and_grads(params, deg, cam)
    monkeypatch.setattr(ours, "LabelRasterizer", None)  # falls back to stock diff_gaussian_rasterization_ours
    fwd_ref, grads_ref = _all_outputs_and_grads(params, deg, cam)
    for k in fwd:  # separate builds: allow last-bit differences from the compiler
        torch.testing.assert_close(fwd[k], fwd_ref[k], rtol=1e-6, atol=1e-6, msg=k)
    assert fwd["alpha"].max() > 0.5
    for k in grads:  # float atomics: order-dependent
        _close_to_scale(grads[k], grads_ref[k], name=k)


def test_labels_do_not_change_outputs_or_grads():
    cam = make_camera()
    params, deg = make_gaussians()
    gl, pl = _split_labels(cam, params["xyz"].shape[0])
    fwd, grads = _all_outputs_and_grads(params, deg, cam)
    fwd_l, grads_l = _all_outputs_and_grads(params, deg, cam, gaussian_labels=gl, pixel_labels=pl, num_labels=3)
    for k in fwd:
        assert torch.equal(fwd[k], fwd_l[k]), k
    for k in grads:
        _close_to_scale(grads_l[k], grads[k], name=k)


def test_no_labels_returns_none():
    cam = make_camera()
    params, deg = make_gaussians()
    out = _render(params, deg, cam)
    assert out["sag"] is None and out["votes"] is None
    assert out["alpha"].shape == (1, cam.image_height, cam.image_width)


def test_votes_match_colors_precomp_gradient():
    """votes[g, l] = sum_{p: label(p) = l} alpha_g T_g(p) = d/dc_{g,l} sum_p [label(p) = l] render_l(p)."""
    cam = make_camera()
    params, deg = make_gaussians()
    n = params["xyz"].shape[0]
    gl, pl = _split_labels(cam, n)
    colors = torch.rand(n, 3, device="cuda", requires_grad=True)
    out = _render(params, deg, cam, colors_precomp=colors, gaussian_labels=gl, pixel_labels=pl, num_labels=3)
    onehot = torch.stack([(pl == l).float() for l in range(3)])
    (out["render"] * onehot).sum().backward()
    votes = out["votes"]
    assert votes.shape == (n, 3) and not votes.requires_grad
    assert votes.sum() > 1.0
    torch.testing.assert_close(votes, colors.grad, rtol=1e-3, atol=1e-5)
    for l in range(3):
        torch.testing.assert_close(votes[:, l].sum(), out["alpha"][0][pl == l].sum(), rtol=1e-4, atol=1e-3)


def test_sag_matches_brute_force():
    """SAG(p) = sum_g [label_g != label(p)] alpha_g T_g(p), with alpha_g T_g(p) from one-hot colour renders."""
    cam = make_camera(width=24, height=16)
    params, deg = make_gaussians(n=8, seed=5)
    with torch.no_grad():
        params["xyz"][:, :2] *= 0.3
        params["scaling"].fill_(math.log(0.15))
    n = params["xyz"].shape[0]
    gl, pl = _split_labels(cam, n, num_labels=3, seed=2)
    with torch.no_grad():
        sag = _render(params, deg, cam, gaussian_labels=gl, pixel_labels=pl, num_labels=3)["sag"][0]
        bg = torch.zeros(3, device="cuda")
        from gaussian_renderer.ours import render_ours
        ref = torch.zeros_like(sag)
        for g in range(n):
            onehot = torch.zeros(n, 3, device="cuda")
            onehot[g, 0] = 1
            w = render_ours(cam, as_model(params, deg), PIPE, bg, colors_precomp=onehot)["render"][0]
            ref += w * ((pl >= 0) & (pl != gl[g]))
    assert ref.max() > 0.1
    torch.testing.assert_close(sag, ref, rtol=1e-5, atol=1e-6)
    assert torch.all(sag[pl < 0] == 0)


def test_sag_gradient_mismatch_only():
    """All pixels have label 1. Label-1 Gaussians are matched (zero SAG grad);
    for label-0 Gaussians the mismatch-only gradient is the exact derivative."""
    cam = make_camera(width=32, height=32)
    params, deg = make_gaussians(n=6, seed=3)
    with torch.no_grad():
        params["xyz"][:, :2] *= 0.15  # stack them in view
        params["scaling"].fill_(math.log(0.15))
    gl = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32, device="cuda")
    pl = torch.ones(32, 32, dtype=torch.int32, device="cuda")
    kw = dict(gaussian_labels=gl, pixel_labels=pl, num_labels=2)

    def sag_sum():
        return _render(params, deg, cam, **kw)["sag"].double().sum()

    with torch.no_grad():
        assert torch.all(_render(params, deg, cam, **kw)["votes"][:, 1] > 0)  # every Gaussian is visible
    _zero_grads(params)
    sag_sum().backward()
    analytic = params["opacity"].grad[:, 0].clone()
    assert torch.all(analytic[gl == 1] == 0)
    assert torch.all(params["xyz"].grad[gl == 1] == 0)
    assert torch.all(analytic[gl == 0] != 0)

    eps = 1e-3  # 1e-2 crosses the 1/255 alpha cutoff: stock RGB/alpha FD is off by ~15% there too
    for i in torch.nonzero(gl == 0).flatten().tolist():
        with torch.no_grad():
            params["opacity"][i] += eps
            up = sag_sum()
            params["opacity"][i] -= 2 * eps
            down = sag_sum()
            params["opacity"][i] += eps
        numeric = ((up - down) / (2 * eps)).float()
        torch.testing.assert_close(analytic[i], numeric, rtol=2e-2, atol=1e-2)


def test_mask_matches_subset_model():
    cam = make_camera()
    params, deg = make_gaussians()
    n = params["xyz"].shape[0]
    mask = torch.rand(n, device="cuda", generator=torch.Generator(device="cuda").manual_seed(4)) < 0.4
    gl, pl = _split_labels(cam, n)
    kw = dict(gaussian_labels=gl, pixel_labels=pl, num_labels=3)

    out = _render(params, deg, cam, mask=mask, **kw)
    out["render"].sum().backward()
    sub = {k: v[mask].detach().clone().requires_grad_(True) for k, v in params.items()}
    ref = _render(sub, deg, cam, gaussian_labels=gl[mask], pixel_labels=pl, num_labels=3)
    ref["render"].sum().backward()

    for k in ("render", "median_depth", "alpha", "normal", "sag"):
        assert torch.equal(out[k], ref[k]), k
    assert out["radii"].shape == (n,) and out["votes"].shape == (n, 3)
    assert torch.equal(out["radii"][mask], ref["radii"]) and torch.all(out["radii"][~mask] == 0)
    assert torch.equal(out["visibility_filter"][mask], ref["visibility_filter"])
    assert not out["visibility_filter"][~mask].any()
    torch.testing.assert_close(out["votes"][mask], ref["votes"], rtol=1e-5, atol=1e-7)  # float atomics
    assert torch.all(out["votes"][~mask] == 0)
    vs = out["viewspace_points"].grad
    assert vs.shape == (n, 3) and torch.all(vs[~mask] == 0)
    _close_to_scale(vs[mask], ref["viewspace_points"].grad, name="viewspace")
    torch.testing.assert_close(params["opacity"].grad[mask], sub["opacity"].grad, rtol=1e-5, atol=1e-7)


def test_empty_mask_with_labels():
    cam = make_camera()
    params, deg = make_gaussians()
    n = params["xyz"].shape[0]
    gl, pl = _split_labels(cam, n)
    mask = torch.zeros(n, dtype=torch.bool, device="cuda")
    out = _render(params, deg, cam, mask=mask, gaussian_labels=gl, pixel_labels=pl, num_labels=3)
    assert out["sag"].shape == (1, cam.image_height, cam.image_width) and torch.all(out["sag"] == 0)
    assert out["votes"].shape == (n, 3) and torch.all(out["votes"] == 0)
    assert out["viewspace_points"].shape == (n, 3) and not out["visibility_filter"].any()


def test_beags_api():
    from gaussian_renderer.beags_api import contribution, render_depth_alpha, render_labels
    cam = make_camera()
    params, deg = make_gaussians()
    model = as_model(params, deg)
    with torch.no_grad():
        depth, alpha = render_depth_alpha(cam, model, PIPE)
        out = render_labels(cam, model, PIPE, torch.zeros(3, device="cuda"))
        c = contribution(cam, model, PIPE)
    assert depth.shape == alpha.shape == (cam.image_height, cam.image_width)
    torch.testing.assert_close(alpha, out["alpha"][0])
    assert torch.all(depth[alpha < 0.5] == 0) and (depth > 0).any()
    assert torch.all(depth[depth > 0] > 1.5)  # the cloud lies at z in [2, 4]
    assert c.shape == (params["xyz"].shape[0],)
    torch.testing.assert_close(c.sum(), alpha.sum(), rtol=1e-4, atol=1e-3)
