"""Per-Gaussian extras (labels), confidence and the 3DGS export of scene/gaussian_model.py.

    python -m pytest -q gaussian_wrapping/tests/test_extras.py      (gaussian_wrapping env, after
    "module load profile/deeplrn cuda/12.2 gcc/12.2.0"; the CPU tests need no GPU, the *_cuda tests skip without one)

Every Gaussian carries an identity code in f_dc (code i for Gaussian i). Clones, split children and the optimizer
state copy f_dc from their parent, so after any prune/densify the extras are aligned iff they match the code.
"""
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

GW = Path(__file__).resolve().parents[1]
ROOT = GW.parent
sys.path.insert(0, str(GW))
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "submodules"))
try:
    from scene.gaussian_model import GaussianModel
    from arguments import OptimizationParams
except ImportError as e:  # simple_knn needs the cuda/gcc modules for its libstdc++
    pytest.skip(f"GaussianModel is not importable: {e}", allow_module_level=True)
from plyfile import PlyData  # noqa: E402

import export_3dgs_ply as export  # noqa: E402

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")
N, SH_DEGREE, N_VIEWS = 12, 1, 2


@pytest.fixture(autouse=True)
def device_agnostic_build_rotation(monkeypatch):
    """utils.general_utils.build_rotation allocates on cuda; the RaDe-GS clone/split use it on CPU tensors here."""
    import scene.gaussian_model as gaussian_model

    def build_rotation(r):
        r, x, y, z = torch.nn.functional.normalize(r, dim=-1).unbind(-1)
        return torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
            2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
            2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
        ], dim=-1).view(-1, 3, 3)

    monkeypatch.setattr(gaussian_model, "build_rotation", build_rotation)


def opt_args():
    return OptimizationParams(ArgumentParser())


def make_model(n=N, device="cpu", setup=True):
    """n Gaussians; even ones small (clone), odd ones large (split) for percent_dense * extent = 0.01."""
    g = GaussianModel(SH_DEGREE, use_mip_filter=True, use_radegs_densification=True)
    gen = torch.Generator().manual_seed(0)
    code = torch.arange(n, dtype=torch.float32)
    scaling = torch.where((torch.arange(n) % 2 == 0)[:, None], torch.full((n, 3), math.log(1e-3)), torch.full((n, 3), math.log(1e-1)))
    scaling = scaling + 0.1 * torch.randn(n, 3, generator=gen)
    rotation = torch.nn.functional.normalize(torch.randn(n, 4, generator=gen), dim=-1)
    g._xyz = nn.Parameter(torch.randn(n, 3, generator=gen).to(device))
    g._features_dc = nn.Parameter(code.view(n, 1, 1).repeat(1, 1, 3).to(device))
    g._features_rest = nn.Parameter(torch.randn(n, (SH_DEGREE + 1) ** 2 - 1, 3, generator=gen).to(device))
    g._opacity = nn.Parameter(torch.randn(n, 1, generator=gen).to(device))
    g._scaling = nn.Parameter(scaling.to(device))
    g._rotation = nn.Parameter(rotation.to(device))
    g.max_radii2D = torch.zeros(n, device=device)
    g.filter_3D = (0.01 + 0.05 * torch.rand(n, 1, generator=gen)).to(device)
    g._culling = torch.zeros(n, N_VIEWS, dtype=torch.bool, device=device)
    g.factor_culling = torch.ones(n, 1, device=device)
    g.spatial_lr_scale = 1.0
    if setup:
        g.training_setup(opt_args())
    return g


def code_of(g):
    return g._features_dc[:, 0, 0].detach().round().long()


def fill_optimizer_state(g):
    """Adam state whose rows hold the identity code of their Gaussian."""
    code = g._features_dc[:, 0, 0].detach()
    for group in g.optimizer.param_groups:
        param = group["params"][0]
        rows = code.view(-1, *([1] * (param.dim() - 1))).expand_as(param).clone()
        g.optimizer.state[param] = {"step": torch.tensor(1.0), "exp_avg": rows, "exp_avg_sq": rows.clone()}


def add_extras(g):
    n = g._xyz.shape[0]
    g.labels = torch.arange(n, device=g._xyz.device)
    g.register_extra("weight", torch.arange(n, device=g._xyz.device).float()[:, None].repeat(1, 2) + 1, default_for_new="zero")
    g.enable_confidence()
    g._confidence.data[:] = torch.arange(n, device=g._xyz.device).float()[:, None] / 10.  # >= 0: threshold factor 1


def assert_aligned(g, filter_3D_ref):
    """Every per-Gaussian tensor matches the identity code of its row."""
    n = g._xyz.shape[0]
    code = code_of(g)
    assert g.labels.dtype == torch.int32
    for tensor in (g.labels, g._confidence, g.filter_3D, g._culling, g.factor_culling, g.xyz_gradient_accum, g.denom, g.max_radii2D):
        assert tensor.shape[0] == n
    assert torch.equal(g.labels.long(), code)
    assert torch.allclose(g._confidence[:, 0], code.float() / 10.)
    assert torch.equal(g.filter_3D, filter_3D_ref[code])
    for group in g.optimizer.param_groups:
        assert group["params"][0].shape[0] == n, group["name"]
    assert g.optimizer.param_groups[-1]["name"] == "confidence"
    assert g.optimizer.param_groups[-1]["params"][0] is g._confidence


# ---------------------------------------------------------------- extras through prune / densify
def test_register_extra_validates():
    g = make_model()
    with pytest.raises(ValueError):
        g.register_extra("bad", torch.zeros(N + 1))
    with pytest.raises(ValueError):
        g.register_extra("bad", torch.zeros(N), default_for_new="mean")
    assert g.labels is None
    g.labels = torch.zeros(N, dtype=torch.long)
    assert g.labels.dtype == torch.int32 and g.extras["labels"] is g.labels
    g.labels = None
    assert g.labels is None and "labels" not in g.extras


def test_prune_keeps_alignment():
    g = make_model()
    add_extras(g)
    fill_optimizer_state(g)
    filter_3D = g.filter_3D.clone()
    prune = torch.arange(N) % 3 == 0
    g.prune_points(prune)
    assert torch.equal(code_of(g), torch.arange(N)[~prune])
    assert_aligned(g, filter_3D)
    assert torch.equal(g.extras["weight"][:, 0], code_of(g).float() + 1)
    state = g.optimizer.state[g._confidence]
    assert torch.equal(state["exp_avg"][:, 0], code_of(g).float())


def test_clone_split_children_inherit():
    g = make_model()
    add_extras(g)
    fill_optimizer_state(g)
    filter_3D = g.filter_3D.clone()
    grads = torch.ones(N, 1)
    g.densify_and_clone_radegs(grads, 0.5, grads, 0.5, scene_extent=1.0, use_abs_grad=False)
    n_small = (N + 1) // 2
    assert g._xyz.shape[0] == N + n_small
    g.densify_and_split_radegs(grads, 0.5, grads, 0.5, scene_extent=1.0, use_abs_grad=False)

    code = code_of(g)
    assert g._xyz.shape[0] == 2 * N  # every large parent is replaced by its two children
    assert torch.equal(torch.bincount(code, minlength=N), torch.full((N,), 2))
    assert_aligned(g, filter_3D)
    # "zero" extras: the surviving originals (the small ones) keep their value, clones and children get 0
    weight = g.extras["weight"][:, 0]
    original = weight != 0
    assert torch.equal(code[original], torch.arange(0, N, 2))
    assert torch.equal(weight[original], code[original].float() + 1)
    # Optimizer state: originals keep theirs, new rows start at 0 (cat_tensors_to_optimizer)
    exp_avg = g.optimizer.state[g._confidence]["exp_avg"][:, 0]
    assert torch.equal(exp_avg[original], code[original].float())
    assert (exp_avg[~original] == 0).all()


def test_clone_from_mask_inherits():
    g = make_model()
    add_extras(g)
    filter_3D = g.filter_3D.clone()
    mask = torch.arange(N) % 4 == 1
    new_xyz = g._xyz[mask].detach() + 1.0
    g.densify_and_clone_from_mask(selected_pts_mask=mask, new_xyz=new_xyz)  # normal-field spoke clones
    assert g._xyz.shape[0] == N + int(mask.sum())
    assert torch.equal(code_of(g)[N:], torch.arange(N)[mask])
    assert_aligned(g, filter_3D)


def test_densify_requires_parent_idx():
    g = make_model()
    g.labels = torch.zeros(N, dtype=torch.int32)
    with pytest.raises(ValueError):
        g.densification_postfix(g._xyz[:1].detach(), g._features_dc[:1].detach(), g._features_rest[:1].detach(),
                                g._opacity[:1].detach(), g._scaling[:1].detach(), g._rotation[:1].detach())
    g.labels = None
    g.register_extra("z", torch.ones(N), default_for_new="zero")
    g.densification_postfix(g._xyz[:1].detach(), g._features_dc[:1].detach(), g._features_rest[:1].detach(),
                            g._opacity[:1].detach(), g._scaling[:1].detach(), g._rotation[:1].detach())
    assert torch.equal(g.extras["z"], torch.cat([torch.ones(N), torch.zeros(1)]))


# ---------------------------------------------------------------- confidence
def test_confidence_scales_densify_threshold():
    g = make_model()
    g.enable_confidence()
    small = torch.arange(N) % 2 == 0
    g._confidence.data[0] = math.log(0.5)  # threshold doubled for Gaussian 0
    g._confidence.data[2] = math.log(4.0)  # clamped to 1: threshold unchanged
    grads = torch.full((N, 1), 1.5)
    g.densify_and_clone_radegs(grads, 1.0, grads, 1.0, scene_extent=1.0, use_abs_grad=False)
    cloned = code_of(g)[N:]
    assert torch.equal(cloned, torch.arange(N)[small][1:])
    assert torch.equal(g.get_confidence[:, 0], torch.exp(g._confidence[:, 0]))


def test_enable_confidence_before_and_after_setup():
    for before in (True, False):
        g = make_model(setup=False)
        if before:
            g.enable_confidence(lr=1e-3)
        g.training_setup(opt_args())
        if not before:
            g.enable_confidence(lr=1e-3)
        g.enable_confidence(lr=1e-3)  # idempotent
        groups = [group for group in g.optimizer.param_groups if group["name"] == "confidence"]
        assert len(groups) == 1 and groups[0]["lr"] == 1e-3 and groups[0]["params"][0] is g._confidence
        assert g._confidence.shape == (N, 1) and (g._confidence == 0).all()
        assert g.get_confidence_scaled_threshold(2.0).shape == (N,)


def test_confidence_survives_capture_restore():
    g = make_model()
    add_extras(g)
    fill_optimizer_state(g)
    g.prune_points(torch.arange(N) == 3)
    model_args = g.capture()

    restored = GaussianModel(SH_DEGREE, use_mip_filter=True, use_radegs_densification=True)
    restored.restore(model_args, opt_args())
    assert restored.use_confidence
    assert torch.equal(restored._confidence, g._confidence)
    assert torch.equal(restored.labels, g.labels)
    assert restored._extras_default == {"labels": "parent", "weight": "zero"}
    names = [group["name"] for group in restored.optimizer.param_groups]
    assert names.count("confidence") == 1
    state = restored.optimizer.state[restored._confidence]
    assert torch.equal(state["exp_avg"][:, 0], code_of(g).float())

    old = GaussianModel(SH_DEGREE, use_mip_filter=True, use_radegs_densification=True)
    old_args = make_model().capture()[:-1]  # checkpoint written before extras existed
    old.restore(old_args, opt_args())
    assert not old.use_confidence and old.labels is None


# ---------------------------------------------------------------- save / export
def test_save_ply_writes_labels_and_confidence(tmp_path):
    g = make_model()
    add_extras(g)
    path = tmp_path / "point_cloud" / "iteration_7" / "point_cloud.ply"
    g.save_ply(str(path))
    labels = torch.load(path.parent / "labels.pt")
    assert labels.dtype == torch.int32 and torch.equal(labels, g.labels)
    element = PlyData.read(str(path)).elements[0]
    assert "confidence" in [p.name for p in element.properties]
    assert np.allclose(np.asarray(element["confidence"]), g._confidence[:, 0].detach().numpy())


def test_bake_matches_get_with_3D_filter():
    g = make_model()
    g._opacity.data[0] = 30.  # sigmoid saturates in float32
    scaling, opacity = export.bake_3d_filter(g._scaling.detach(), g._opacity.detach(), g.filter_3D)
    assert torch.isfinite(opacity).all()
    assert torch.allclose(torch.exp(scaling), g.get_scaling_with_3D_filter.detach(), rtol=1e-5, atol=0)
    assert torch.allclose(torch.sigmoid(opacity), g.get_opacity_with_3D_filter.detach(), rtol=1e-5, atol=1e-7)
    scales, opacities = g.get_scaling_n_opacity_with_3D_filter
    assert torch.allclose(torch.exp(scaling), scales.detach(), rtol=1e-5)
    assert torch.allclose(torch.sigmoid(opacity), opacities.detach(), rtol=1e-5, atol=1e-7)


def test_export_round_trip(tmp_path):
    g = make_model()
    add_extras(g)
    gw_ply = tmp_path / "model" / "point_cloud" / "iteration_7" / "point_cloud.ply"
    g.save_ply(str(gw_ply))
    out = tmp_path / "export" / "point_cloud.ply"
    info = export.export_3dgs_ply(str(gw_ply), str(out))
    assert info["n"] == N and info["sh_degree"] == SH_DEGREE and info["baked"] and info["filter_3D"]

    element = PlyData.read(str(out)).elements[0]
    k = (SH_DEGREE + 1) ** 2
    expected = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(3 * (k - 1))]
                + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    assert [p.name for p in element.properties] == expected
    assert b"binary_little_endian" in out.read_bytes()[:200]

    loaded = export.load_3dgs_ply_for_gsplat(str(out), device="cpu")
    assert torch.allclose(loaded["scales"], g.get_scaling_with_3D_filter.detach(), rtol=1e-5)
    assert torch.allclose(loaded["opacities"], g.get_opacity_with_3D_filter.detach()[:, 0], rtol=1e-5, atol=1e-7)
    assert torch.allclose(loaded["means"], g._xyz.detach())
    assert torch.allclose(loaded["quats"], g.get_rotation.detach(), atol=1e-6)
    # SH layout (N, K, 3), DC first, as artifixer_gs.render.read_ply and gsplat expect
    assert torch.allclose(loaded["colors"], g.get_features.detach())
    assert loaded["sh_degree"] == SH_DEGREE


def test_export_without_filter_copies(tmp_path):
    g = make_model()
    g.use_mip_filter = False
    gw_ply = tmp_path / "point_cloud.ply"
    g.save_ply(str(gw_ply))
    info = export.export_3dgs_ply(str(gw_ply), str(tmp_path / "out" / "point_cloud.ply"))
    assert not info["filter_3D"]
    loaded = export.load_3dgs_ply_for_gsplat(str(tmp_path / "out" / "point_cloud.ply"), device="cpu")
    assert torch.allclose(loaded["scales"], g.get_scaling.detach())
    assert torch.allclose(loaded["opacities"], g.get_opacity.detach()[:, 0])


def test_export_matches_artifixer_read_ply(tmp_path):
    artifixer = ROOT / "submodules" / "artifixer_gs"
    if not (artifixer / "artifixer_gs" / "render.py").exists():
        pytest.skip("submodules/artifixer_gs is not checked out")
    sys.path.insert(0, str(artifixer))
    render = pytest.importorskip("artifixer_gs.render")
    g = make_model()
    gw_ply = tmp_path / "point_cloud.ply"
    g.save_ply(str(gw_ply))
    out = tmp_path / "out" / "point_cloud.ply"
    export.export_3dgs_ply(str(gw_ply), str(out))
    theirs = render.read_ply(out)
    ours = export.load_3dgs_ply_for_gsplat(str(out), device="cpu")
    assert int(theirs["sh_degree"]) == ours["sh_degree"]
    for key_theirs, key_ours in (("means", "means"), ("quats", "quats"), ("scales", "scales"), ("opacities", "opacities"), ("sh", "colors")):
        assert np.allclose(theirs[key_theirs], ours[key_ours].numpy(), rtol=1e-5, atol=1e-6), key_theirs


# ---------------------------------------------------------------- GPU
@needs_cuda
def test_load_ply_round_trip_cuda(tmp_path):
    g = make_model(device="cuda")
    add_extras(g)
    path = tmp_path / "point_cloud" / "iteration_7" / "point_cloud.ply"
    g.save_ply(str(path))
    loaded = GaussianModel(SH_DEGREE)
    loaded.load_ply(str(path))
    assert loaded.use_mip_filter and loaded.use_confidence
    assert torch.allclose(loaded._confidence, g._confidence)
    assert loaded.labels.dtype == torch.int32 and loaded.labels.is_cuda and torch.equal(loaded.labels, g.labels)
    assert loaded.extras.keys() == {"labels"}

    plain = make_model(device="cuda")
    plain_path = tmp_path / "plain" / "point_cloud.ply"
    plain.save_ply(str(plain_path))
    loaded.load_ply(str(plain_path))  # no confidence property, no labels.pt
    assert not loaded.use_confidence and loaded.labels is None and loaded.extras == {}


@needs_cuda
def test_densify_and_prune_radegs_after_step_cuda():
    g = make_model(device="cuda")
    add_extras(g)
    filter_3D = g.filter_3D.clone()
    loss = sum((group["params"][0] ** 2).sum() for group in g.optimizer.param_groups)
    loss.backward()
    g.optimizer.step(torch.ones(N, dtype=torch.bool, device="cuda"), N)
    g.optimizer.zero_grad(set_to_none=True)
    g._confidence.data[:] = torch.arange(N, device="cuda").float()[:, None] / 10.
    g._features_dc.data[:] = torch.arange(N, device="cuda").float().view(N, 1, 1)
    g.xyz_gradient_accum[:] = 1.0
    g.xyz_gradient_accum_abs[:] = 1.0
    g.denom[:] = 1.0
    g._opacity.data[5] = -20.  # pruned by min_opacity
    g.densify_and_prune_radegs(max_grad=0.5, min_opacity=0.005, extent=1.0, max_screen_size=None)
    assert 5 not in code_of(g).tolist()
    assert_aligned(g, filter_3D)
    for group in g.optimizer.param_groups:
        state = g.optimizer.state.get(group["params"][0], None)
        assert state is None or state["exp_avg"].shape == group["params"][0].shape, group["name"]
