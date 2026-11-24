"""GAVIS visibility field (beags/visibility.py), unseen maps (beags/unseen_map.py) and the rasterizer's vis output.

    python -m pytest -q gaussian_wrapping/tests/test_visibility.py     (CPU tests; the GPU ones skip without CUDA)
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "tests"))

from beags import unseen_map as umap  # noqa: E402
from beags import visibility as vis  # noqa: E402
from utils.sh_utils import eval_sh  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def unit(n, seed=0):
    return torch.nn.functional.normalize(torch.randn(n, 3, generator=torch.Generator().manual_seed(seed)), dim=1)


# ---------------------------------------------------------------- SH projection / query (CPU)
def test_sh_basis_matches_eval_sh():
    d = unit(200)
    basis = vis.sh_basis(d)
    for k in range(9):
        e = torch.zeros(200, 1, 9)
        e[..., k] = 1
        torch.testing.assert_close(eval_sh(2, e, d)[:, 0], basis[:, k], rtol=0, atol=1e-6)


def test_bessel_closed_forms():
    assert vis.bessel_i(0, 1.0) == pytest.approx(math.sinh(1.0))
    assert vis.bessel_i(1, 1.0) == pytest.approx(math.exp(-1.0))          # cosh 1 - sinh 1
    assert vis.bessel_i(2, 2.0) == pytest.approx((7 * math.sinh(2) - 6 * math.cosh(2)) / 8)
    for l in range(3):                                                     # the series meets the closed form
        assert vis.bessel_i(l, 0.999e-3) == pytest.approx(vis.bessel_i(l, 1.001e-3), rel=1e-2, abs=1e-9)


@pytest.mark.parametrize("kappa,tol", [(0.0, 1e-6), (0.3, 2e-3), (1.0, 0.035)])
def test_projection_reconstructs_vmf_lobe(kappa, tol):
    """sum_lm 4 pi zeta i_l Y_lm(d_p) Y_lm(d) = zeta exp(kappa d.d_p) up to the degree-2 truncation."""
    d, dp = unit(500), unit(1, seed=1)
    gamma = vis.sh_basis(dp)[0] * vis.coefficient_scale(kappa)
    approx = (vis.sh_basis(d) * gamma).sum(1)
    exact = math.exp(-kappa) * torch.exp(kappa * (d @ dp[0]))
    assert (approx - exact).abs().max() <= tol
    assert exact.max() <= 1 + 1e-6                                         # 1 along the training direction


def test_phi_t():
    v = torch.tensor([[50.0, 50, 48.0],     # unoccluded, footprint counted: 1
                      [10.0, 20, 18.0],     # half transmittance: 0.5
                      [10.0, 10, 40.0],     # 3/4 of the footprint never reached (early stop / off-screen): 0.25
                      [0.0, 0, 0.0]])       # culled
    torch.testing.assert_close(vis.phi_t(v), torch.tensor([1.0, 0.5, 0.25, 0.0]))


def avf_from_views(xyz, cams, phit, kappa):
    """Build the AVF as build_avf does, from given PhiT (V, P) and camera centres (V, 3)."""
    gamma = torch.zeros(xyz.shape[0], 9)
    for c, pt in zip(cams, phit):
        gamma += pt[:, None] * vis.sh_basis(vis.directions(xyz, c))
    return {"gamma": gamma * vis.coefficient_scale(kappa), "n_views": len(cams), "kappa": kappa}


def test_query_is_the_am_gm_bound():
    g = torch.Generator().manual_seed(3)
    xyz = torch.randn(300, 3, generator=g) * 0.3
    cams = unit(12, seed=4) * 4
    phit = torch.rand(12, 300, generator=g)
    q = unit(1, seed=5)[0] * 4
    # kappa 0: no truncation, V_p = PhiT_p in every direction; the bound is <= the exact union probability
    avf = avf_from_views(xyz, cams, phit, 0.0)
    v = vis.query_avf(avf, xyz, q)
    exact = 1 - torch.prod(1 - phit, dim=0)
    bound = 1 - (1 - phit.mean(0)) ** 12
    torch.testing.assert_close(v, bound, rtol=0, atol=1e-5)
    assert (v <= exact + 1e-6).all() and v.min() >= 0 and v.max() <= 1
    # kappa 1: within the truncation error of the bound computed from the exact lobes
    avf = avf_from_views(xyz, cams, phit, 1.0)
    dq = vis.directions(xyz, q)
    lobes = torch.stack([pt * math.exp(-1) * torch.exp((dq * vis.directions(xyz, c)).sum(1)) for c, pt in zip(cams, phit)])
    ref = 1 - (1 - lobes.mean(0)) ** 12
    assert (vis.query_avf(avf, xyz, q) - ref).abs().max() < 0.05


def test_query_is_anisotropic():
    """Seen once from -z: visible along that direction, less from the side (zeta e^0), least from behind (zeta e^-1)."""
    xyz = torch.zeros(1, 3)
    avf = avf_from_views(xyz, torch.tensor([[0.0, 0, -3]]), torch.ones(1, 1), 1.0)
    front, side, back = (float(vis.query_avf(avf, xyz, torch.tensor(c))) for c in ([0.0, 0, -3], [3.0, 0, 0], [0.0, 0, 3]))
    assert front > 0.95 and abs(side - math.exp(-1)) < 0.035 and abs(back - math.exp(-2)) < 0.035


def test_save_load_avf(tmp_path):
    avf = {"gamma": torch.randn(4, 9), "n_views": 3, "kappa": 1.0}
    path = str(tmp_path / "avf.pt")
    vis.save_avf(path, avf, source="a.ply", iteration=7)
    back = vis.load_avf(path, device="cpu", source="a.ply", iteration=7)
    torch.testing.assert_close(back["gamma"], avf["gamma"])
    assert back["n_views"] == 3 and back["source"] == "a.ply"
    assert vis.load_avf(path, device="cpu", source="b.ply") is None
    assert vis.load_avf(str(tmp_path / "missing.pt")) is None


# ---------------------------------------------------------------- unseen maps (CPU, fake rasterizer)
def test_encode_decode_and_keep_mask():
    png = umap.encode(np.array([[0.0, 0.5, 1.0], [2.0, np.nan, -1.0]]))
    assert png.dtype == np.uint8 and png.tolist() == [[0, 128, 255], [255, 255, 0]]
    assert np.allclose(umap.decode(png), png / 255.0)
    labels = torch.tensor([0, 1, 2, -1, 1])
    assert umap.keep_mask(labels, 1).tolist() == [True, False, True, False, False]


class FakeCam:
    def __init__(self, name, alpha, centre=(0.0, 0.0, -3.0)):
        self.image_name, self.alpha = name, torch.as_tensor(alpha, dtype=torch.float32)
        self.camera_center = torch.tensor(centre)
        self.original_image = torch.full((3,) + self.alpha.shape, 0.5)


def fake_render(cam, gaussians, pipe, bg, colors_precomp=None, mask=None, require_depth=True):
    """render_ours' contract: every pixel sees the mean colour of the kept Gaussians with weight alpha."""
    assert mask is not None and not require_depth
    feat = colors_precomp if colors_precomp is not None else torch.full((mask.numel(), 3), 0.25)
    alpha = cam.alpha[None]
    return {"render": feat[mask].mean(0)[:, None, None] * alpha, "alpha": alpha}


def test_write_maps(tmp_path):
    """U = sum w (1 - v) + (1 - alpha): never-seen Gaussians -> 1 everywhere, well-seen -> 1 - alpha."""
    xyz = torch.zeros(3, 3)
    gaussians = SimpleNamespace(get_xyz=xyz)
    keep = torch.tensor([True, True, False])
    alpha = np.where(np.arange(8)[None, :] < 4, 1.0, 0.25).repeat(6, 0)
    cams = [FakeCam("a", alpha), FakeCam("b", alpha)]
    never = {"gamma": torch.zeros(3, 9), "n_views": 4, "kappa": 1.0}
    stats = umap.write_maps(cams, gaussians, None, never, keep, str(tmp_path), fake_render, vis_names={"b"})
    assert (np.asarray(Image.open(tmp_path / "a.png")) == 255).all() and stats["a"] == {"mean": 1.0, "high_fraction": 1.0}
    assert np.asarray(Image.open(tmp_path / "vis" / "b.jpg")).shape == (6, 24, 3)
    seen = avf_from_views(xyz, torch.tensor([[0.0, 0, -3]]) + torch.zeros(30, 3), torch.ones(30, 3), 1.0)
    umap.write_maps(cams, gaussians, None, seen, keep, str(tmp_path), fake_render)
    u = umap.decode(np.asarray(Image.open(tmp_path / "a.png")))
    assert (u[:, :4] < 0.01).all() and np.allclose(u[:, 4:], 0.75, atol=0.01)
    info = umap.write_info(str(tmp_path), {"a": {"mean": 0.4, "high_fraction": 0.5}}, source="x.ply", label=1)
    assert info["cameras"] == 1 and info["high_fraction_mean"] == 0.5 and info["encoding"] == umap.ENCODING


def test_unseen_map_script_runs_as_a_file():
    import subprocess
    script = os.path.join(BASE_DIR, "beags", "unseen_map.py")
    out = subprocess.run([sys.executable, script, "-h"], capture_output=True, text=True, cwd=os.path.dirname(BASE_DIR))
    if "No module named" in out.stderr and ("diff_gaussian" in out.stderr or "simple_knn" in out.stderr):
        pytest.skip("GW CUDA extensions are not importable here")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "--remove" in out.stdout and "--kappa" in out.stdout


# ---------------------------------------------------------------- rasterizer vis output (GPU)
def _gpu_helpers():
    pytest.importorskip("diff_gaussian_rasterization_ours_beags")
    import test_rasterizer_labels as trl
    return trl


def _single(trl, xyz, scale, opacity):
    """Isotropic Gaussians at xyz (n, 3), as test_rasterizer_labels' params."""
    n = len(xyz)
    t = lambda v: torch.as_tensor(v, dtype=torch.float32, device="cuda")
    params = dict(xyz=t(xyz), scaling=torch.log(t(scale)).reshape(n, 1).repeat(1, 3),
                  rotation=t([[1.0, 0, 0, 0]] * n), opacity=torch.logit(t(opacity)).reshape(n, 1),
                  f_dc=torch.zeros(n, 1, 3, device="cuda"), f_rest=torch.zeros(n, 15, 3, device="cuda"))
    return trl.as_model(params, 3)


@needs_cuda
@torch.no_grad()
def test_vis_does_not_change_the_render():
    trl = _gpu_helpers()
    from gaussian_renderer.ours import render_ours
    cam = trl.make_camera()
    params, deg = trl.make_gaussians()
    model, bg = trl.as_model(params, deg), torch.tensor(trl.BG, device="cuda")
    a = render_ours(cam, model, trl.PIPE, bg)
    b = render_ours(cam, model, trl.PIPE, bg, vis=True)
    for k in ("render", "alpha", "median_depth", "normal", "radii"):
        assert torch.equal(a[k], b[k]), k
    assert a["vis"] is None and b["vis"].shape == (params["xyz"].shape[0], 5)
    v = b["vis"]
    assert (v[:, 0] <= v[:, 1] + 1e-3).all() and ((v[:, 2] > 0) == (b["radii"] > 0)).all()


@needs_cuda
@torch.no_grad()
def test_vis_unoccluded_and_occluded():
    trl = _gpu_helpers()
    from gaussian_renderer.beags_api import transmittance
    cam = trl.make_camera()
    # 0: alone (image x -0.45); 1: behind a stack of opaque walls (image x 0.3); 2: off-screen
    walls = [[0.6, 0.0, 2.0 + 0.01 * i] for i in range(4)]
    model = _single(trl, [[-1.35, 0.0, 3.0], [0.9, 0.0, 3.0], [50.0, 0.0, 3.0]] + walls,
                    [0.05, 0.05, 0.05] + [0.3] * 4, [0.8, 0.8, 0.8] + [0.99] * 4)
    v = transmittance(cam, model, trl.PIPE)
    pt = vis.phi_t(v).cpu()
    v = v.cpu()
    assert v[0, 1] > 20 and v[0, 0] == pytest.approx(float(v[0, 1]), rel=1e-5)   # T = 1 on every covered pixel
    assert 0.75 <= v[0, 1] / v[0, 2] <= 1.15                                   # pixel count ~ footprint ellipse
    assert pt[0] > 0.85 and pt[1] < 0.01 and pt[2] == 0


@needs_cuda
@torch.no_grad()
def test_vis_mask_matches_subset():
    trl = _gpu_helpers()
    from gaussian_renderer.ours import render_ours
    cam = trl.make_camera()
    params, deg = trl.make_gaussians(n=500)
    mask = torch.arange(500, device="cuda") % 3 != 0
    bg = torch.zeros(3, device="cuda")
    full = render_ours(cam, trl.as_model(params, deg), trl.PIPE, bg, mask=mask, vis=True)["vis"]
    sub = render_ours(cam, trl.as_model({k: p[mask] for k, p in params.items()}, deg), trl.PIPE, bg, vis=True)["vis"]
    assert (full[~mask] == 0).all()
    torch.testing.assert_close(full[mask], sub, rtol=1e-5, atol=1e-4)                # float atomics: order


@needs_cuda
@torch.no_grad()
def test_build_avf_on_gpu():
    trl = _gpu_helpers()
    cam = trl.make_camera()
    model = _single(trl, [[0.0, 0.0, 3.0], [50.0, 0.0, 3.0]], [0.05, 0.05], [0.8, 0.8])
    avf = vis.build_avf(model, [cam, cam], trl.PIPE, kappa=1.0)
    v = vis.query_avf(avf, model.get_xyz, cam.camera_center)
    assert avf["n_views"] == 2 and v[0] > 0.8 and v[1] == 0
