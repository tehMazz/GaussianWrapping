"""Unit tests for beags/ helpers. CPU, except consensus and assign_labels (CUDA). Run:
    pytest -q gaussian_wrapping/tests/test_beags_utils.py
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

import beags.occupancy as occupancy
from beags.masks import boundary_weight, soft_to_labels
from beags.occupancy import LabeledVoxels, _pack, _pack_offsets, neighbour_offsets, occupancy_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rotation(q):
    """utils.general_utils.build_rotation without its hard-coded CUDA device."""
    r, x, y, z = torch.nn.functional.normalize(q, dim=1).unbind(1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], dim=1).view(-1, 3, 3)


@pytest.fixture(autouse=True)
def _cpu_rotations(monkeypatch):
    if DEVICE == "cpu":
        monkeypatch.setattr(occupancy, "build_rotation", _rotation)


def test_soft_to_labels():
    soft = torch.zeros(3, 2, 2)
    soft[1, 0, 0] = 0.9
    soft[2, 0, 0] = 0.6          # lower than channel 1 -> label 1
    soft[2, 1, 1] = 0.4          # below 0.5 -> background
    assert soft_to_labels(soft).tolist() == [[1, 0], [0, 0]]


def test_boundary_weight_marks_pixels_next_to_other_label():
    labels = torch.zeros(5, 5, dtype=torch.int32)
    labels[:, 3:] = 1
    w = boundary_weight(labels, 0.1)[0]
    assert torch.all(w[:, 2] == 0.1) and torch.all(w[:, 3] == 0.1)
    assert torch.all(w[:, :2] == 1) and torch.all(w[:, 4] == 1)


def test_labels_from_votes_never_hit_is_minus_one():
    from beags.labels import labels_from_votes
    votes = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.3, 0.2], [2.0, 0.0, 0.0], [0.0, 0.0, 1e-6]])
    labels = labels_from_votes(votes)
    assert labels.dtype == torch.int32 and labels.tolist() == [-1, 1, 0, 2]


def _voxels():
    pts = torch.rand(2000, 3, device=DEVICE)                       # unit cube, label 1
    mixed = torch.tensor([[5.05, 5.05, 5.05], [5.06, 5.06, 5.06]], device=DEVICE)
    pts = torch.cat([pts, mixed])
    labels = torch.cat([torch.ones(2000, dtype=torch.int32), torch.tensor([1, 2], dtype=torch.int32)]).to(DEVICE)
    return LabeledVoxels.build(pts, labels, 0.1)


def test_voxel_membership():
    vox = _voxels()
    q = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [3.0, 3.0, 3.0], [5.05, 5.05, 5.05]], device=DEVICE)
    lab = torch.tensor([1, 2, 1, 1], device=DEVICE)
    # inside+same label, inside+other label, far away, only an impure (mixed-label) voxel nearby
    assert vox.contains(q, lab).tolist() == [True, False, False, False]


def test_labeled_voxels_can_be_built_on_cpu():
    points = torch.tensor([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.11, 0.01, 0.01]])
    labels = torch.tensor([1, 1, 2], dtype=torch.int32)
    voxels = LabeledVoxels.build(points, labels, 0.1, chunk_size=1)
    assert voxels.keys.device.type == "cpu"
    assert voxels.labels.device.type == "cpu"
    assert voxels.labels.tolist() == [1, 2]


def test_packed_offsets_match_pack():
    g = torch.Generator().manual_seed(0)
    ijk = torch.randint(-1000, 1000, (500, 3), generator=g).to(DEVICE)
    off = neighbour_offsets(2, device=DEVICE)
    assert off.shape == (125, 3) and neighbour_offsets(2, shell=True, device=DEVICE).shape == (98, 3)
    assert torch.equal(_pack(ijk[:, None] + off[None]), _pack(ijk)[:, None] + _pack_offsets(off)[None])


def _reference_contains(vox, points, labels):
    """BEA-GS_FSplat LabeledVoxels.contains (unchunked, explicit (N, 27, 3) offsets)."""
    ijk = torch.floor((points - vox.origin) / vox.size).long()
    keys = _pack(ijk[:, None, :] + neighbour_offsets(device=points.device)[None])
    pos = torch.searchsorted(vox.keys, keys).clamp_max(vox.keys.numel() - 1)
    hit = (vox.keys[pos] == keys) & (vox.labels[pos] == labels[:, None].long())
    return hit.any(dim=1)


def _scene(n=300, seed=0):
    """Voxels with labels 0 (top), 1 (x < 0.5), 2 (x >= 0.5) + random Gaussians around them."""
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(4000, 3, generator=g)
    lab = torch.where(pts[:, 2] > 0.8, 0, torch.where(pts[:, 0] < 0.5, 1, 2)).int()
    voxels = LabeledVoxels.build(pts, lab, 0.1).to(DEVICE)
    params = dict(
        xyz=torch.rand(n, 3, generator=g) * 1.4 - 0.2,
        rotation=torch.rand(n, 4, generator=g) - 0.5,
        scaling=0.01 + 0.09 * torch.rand(n, 3, generator=g),
        opacity=0.1 + 0.8 * torch.rand(n, 1, generator=g),
    )
    labels = torch.randint(0, 3, (n,), generator=g).int().to(DEVICE)
    return voxels, {k: v.to(DEVICE).requires_grad_(True) for k, v in params.items()}, labels


def _model(p):
    return SimpleNamespace(get_xyz=p["xyz"], get_rotation=torch.nn.functional.normalize(p["rotation"], dim=1),
                           get_scaling_with_3D_filter=p["scaling"], get_opacity_with_3D_filter=p["opacity"])


def _reference_occupancy(gaussians, gaussian_labels, voxels, samples_per_gaussian=20, limit=3.0,
                         include_background=False):
    """BEA-GS_FSplat occupancy_loss on the 3D-filtered scale / opacity."""
    obj = torch.ones_like(gaussian_labels, dtype=torch.bool) if include_background else gaussian_labels != 0
    mu = gaussians.get_xyz[obj]
    rot = occupancy.build_rotation(gaussians.get_rotation[obj])
    scale = gaussians.get_scaling_with_3D_filter[obj]
    gid = torch.arange(mu.shape[0], device=mu.device).repeat_interleave(samples_per_gaussian)
    with torch.no_grad():
        u = (torch.rand(gid.numel(), 3, device=mu.device) * 2 - 1) * limit
        x = mu[gid] + torch.einsum("nij,nj->ni", rot[gid], u * scale[gid])
        outside = ~_reference_contains(voxels, x, gaussian_labels[obj][gid])
    q = torch.einsum("nji,nj->ni", rot[gid], x - mu[gid]) / scale[gid]
    kernel = torch.exp(-0.5 * (q ** 2).sum(dim=1))
    return (gaussians.get_opacity_with_3D_filter[obj][gid, 0] * kernel * outside).mean()


def test_contains_chunked_equals_unchunked():
    voxels, p, labels = _scene()
    x = torch.rand(5000, 3, device=DEVICE) * 1.6 - 0.3
    lab = torch.randint(0, 3, (5000,), device=DEVICE)
    ref = _reference_contains(voxels, x, lab)
    assert 0 < ref.sum() < ref.numel()
    assert torch.equal(voxels.contains(x, lab), ref)
    assert torch.equal(voxels.contains(x, lab, chunk=37), ref)


def _loss_and_grad(fn, p, *args, **kwargs):
    torch.manual_seed(0)
    loss = fn(_model(p), *args, **kwargs)
    grads = torch.autograd.grad(loss, [p["xyz"], p["opacity"], p["scaling"]])
    return loss.detach(), grads


@pytest.mark.parametrize("include_background", [False, True])
def test_occupancy_loss_matches_unchunked_reference(include_background):
    voxels, p, labels = _scene()
    ref, ref_grads = _loss_and_grad(_reference_occupancy, p, labels, voxels, include_background=include_background)
    assert ref > 0
    for chunk in (2_000_000, 101):
        loss, grads = _loss_and_grad(occupancy_loss, p, labels, voxels, include_background=include_background,
                                     max_gaussians=None, min_weight=0.0, chunk=chunk)
        torch.testing.assert_close(loss, ref, rtol=1e-5, atol=1e-7)
        for a, b in zip(grads, ref_grads):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_occupancy_loss_chunking_and_weight_cut():
    voxels, p, labels = _scene()
    full, _ = _loss_and_grad(occupancy_loss, p, labels, voxels, max_gaussians=None, min_weight=0.0)
    cut, _ = _loss_and_grad(occupancy_loss, p, labels, voxels, max_gaussians=None)
    cut_chunked, _ = _loss_and_grad(occupancy_loss, p, labels, voxels, max_gaussians=None, chunk=53)
    assert 0 < cut <= full and full - cut < 1 / 255
    torch.testing.assert_close(cut_chunked, cut)
    sub, _ = _loss_and_grad(occupancy_loss, p, labels, voxels, max_gaussians=50)
    sub_chunked, _ = _loss_and_grad(occupancy_loss, p, labels, voxels, max_gaussians=50, chunk=53)
    torch.testing.assert_close(sub_chunked, sub)


def _gaussians(center, scale, opacity=0.8):
    n = center.shape[0]
    return SimpleNamespace(
        get_xyz=center.to(DEVICE), get_scaling_with_3D_filter=torch.full((n, 3), scale, device=DEVICE),
        get_rotation=torch.tensor([[1.0, 0, 0, 0]], device=DEVICE).repeat(n, 1),
        get_opacity_with_3D_filter=torch.full((n, 1), opacity, device=DEVICE))


def test_occupancy_loss_zero_inside_positive_outside():
    vox = _voxels()
    labels = torch.ones(1, dtype=torch.int32, device=DEVICE)
    inside = occupancy_loss(_gaussians(torch.tensor([[0.5, 0.5, 0.5]]), 0.01), labels, vox)
    outside = occupancy_loss(_gaussians(torch.tensor([[3.0, 3.0, 3.0]]), 0.01), labels, vox)
    background = occupancy_loss(_gaussians(torch.tensor([[3.0, 3.0, 3.0]]), 0.01), labels * 0, vox)
    never_hit = occupancy_loss(_gaussians(torch.tensor([[3.0, 3.0, 3.0]]), 0.01), -labels, vox,
                               include_background=True)
    assert inside.item() == 0 and outside.item() > 0 and background.item() == 0 and never_hit.item() == 0


def test_label_distance_rings():
    from beags.pivots import label_distance
    h = 0.1
    vox = LabeledVoxels.build(torch.zeros(1, 3, device=DEVICE), torch.ones(1, dtype=torch.int32, device=DEVICE), h)
    k = torch.tensor([0, 1, 2, 3, 5, -2], device=DEVICE)
    points = torch.stack([(k + 0.5) * h, torch.full_like(k, 0.5 * h, dtype=torch.float32),
                          torch.full_like(k, -0.5 * h, dtype=torch.float32)], dim=1)
    labels = torch.ones(6, dtype=torch.int32, device=DEVICE)
    d = label_distance(vox, points, labels, max_ring=3, chunk=2)
    torch.testing.assert_close(d, torch.tensor([0, 0, 2, 3, 4, 2], device=DEVICE, dtype=torch.float32) * h)
    assert torch.all(label_distance(vox, points, labels * 2) == 4 * h)    # no voxel of label 2


def _pivot_gaussians(center, scale_z, opacity=0.8):
    n = center.shape[0]
    scale = torch.full((n, 3), 0.02, device=DEVICE)
    scale[:, 2] = scale_z
    normals = torch.tensor([[0.0, 0.0, 1.0]], device=DEVICE).repeat(n, 1)
    return SimpleNamespace(
        get_xyz=center.to(DEVICE), get_scaling_with_3D_filter=scale,
        _rotation=torch.tensor([[1.0, 0, 0, 0]], device=DEVICE).repeat(n, 1),
        get_opacity_with_3D_filter=torch.full((n, 1), opacity, device=DEVICE),
        use_gaussian_features=True, convert_features_to_normals=lambda normalize=True: normals)


def test_pivot_loss_zero_when_pivots_inside_voxels(monkeypatch):
    import regularization.normal_field as normal_field
    from beags.pivots import front_pivots, pivot_loss
    if DEVICE == "cpu":
        monkeypatch.setattr(normal_field, "build_scaling_rotation", lambda s, r: _rotation(r) @ torch.diag_embed(s))
    vox = _voxels()
    labels = torch.ones(2, dtype=torch.int32, device=DEVICE)
    center = torch.tensor([[0.5, 0.5, 0.5], [0.3, 0.6, 0.4]])
    flat = _pivot_gaussians(center, 0.02)
    torch.testing.assert_close(front_pivots(flat, torch.arange(2, device=DEVICE)),
                               center.to(DEVICE) + torch.tensor([0.0, 0.0, 0.06], device=DEVICE))
    assert pivot_loss(flat, labels, vox).item() == 0
    tall = _pivot_gaussians(center, 0.5)                                 # pivot at z ~ 2: > 3 rings away
    torch.testing.assert_close(pivot_loss(tall, labels, vox), torch.tensor(0.8 * 3, device=DEVICE))
    assert pivot_loss(tall, -labels, vox).item() == 0                   # never-hit Gaussians are skipped


@needs_cuda
def test_backproject_project_roundtrip():
    from beags.consensus import backproject, project
    cam = _camera()
    depth = torch.full((cam.image_height, cam.image_width), 3.0, device="cuda")
    uv, z = project(backproject(depth, cam), cam)
    ys, xs = torch.meshgrid(torch.arange(cam.image_height), torch.arange(cam.image_width), indexing="ij")
    torch.testing.assert_close(uv[..., 0].cpu(), xs.float(), atol=1e-3, rtol=0)
    torch.testing.assert_close(uv[..., 1].cpu(), ys.float(), atol=1e-3, rtol=0)
    torch.testing.assert_close(z, depth)


@needs_cuda
def test_consensus_of_single_view_is_identity():
    from beags.consensus import reproject_masks
    cam = _camera()
    H, W = cam.image_height, cam.image_width
    depth = torch.full((1, H, W), 3.0, device="cuda")
    masks = torch.zeros(1, 2, H, W, device="cuda")
    masks[0, 1, 20:50, 30:90] = 1
    masks[0, 0] = 1 - masks[0, 1]
    out = reproject_masks([cam], depth, masks, torch.ones(1, 2, dtype=torch.bool))
    # the outermost pixel ring counts as out of bounds (same as BEA-GS)
    torch.testing.assert_close(out[..., 1:-1, 1:-1], masks.cpu()[..., 1:-1, 1:-1])


@needs_cuda
def test_consensus_without_valid_view_keeps_own_mask():
    """Depth holes (rendered depth 0) fail the check in every view, the target included: the default keeps the
    view's own mask there; fallback="zero" (the old behaviour) turns the object into background."""
    from beags.consensus import reproject_masks
    cam = _camera()
    H, W = cam.image_height, cam.image_width
    depth = torch.full((1, H, W), 3.0, device="cuda")
    depth[0, 25:45, 40:80] = 0                                     # hole inside the object
    masks = torch.zeros(1, 2, H, W, device="cuda")
    masks[0, 1, 20:50, 30:90] = 1
    masks[0, 0] = 1 - masks[0, 1]
    present = torch.ones(1, 2, dtype=torch.bool)
    own = reproject_masks([cam], depth, masks, present)
    torch.testing.assert_close(own[..., 1:-1, 1:-1], masks.cpu()[..., 1:-1, 1:-1])
    zero = reproject_masks([cam], depth, masks, present, fallback="zero")
    assert float(zero[0, :, 25:45, 40:80].abs().sum()) == 0.0
    torch.testing.assert_close(zero[0, 1, 20:25, 30:90], masks.cpu()[0, 1, 20:25, 30:90])


@needs_cuda
def test_assign_labels_never_hit_gets_minus_one():
    pytest.importorskip("diff_gaussian_rasterization_ours_beags")
    from beags.labels import assign_labels
    cam = _camera()
    cam.masks = SimpleNamespace(labels=torch.ones((cam.image_height, cam.image_width), dtype=torch.int32,
                                                  device="cuda"))
    xyz = torch.tensor([[0.0, 0.0, 3.0], [0.3, 0.1, 3.5], [0.0, 0.0, -3.0]], device="cuda")  # last: behind camera
    gaussians = SimpleNamespace(
        get_xyz=xyz, _xyz=xyz, use_unbounded_opacity=False,
        get_scaling_n_opacity_with_3D_filter=(torch.full((3, 3), 0.2, device="cuda"),
                                              torch.full((3, 1), 0.8, device="cuda")),
        get_rotation=torch.tensor([[1.0, 0, 0, 0]], device="cuda").repeat(3, 1),
        get_features=torch.zeros((3, 16, 3), device="cuda"), active_sh_degree=3)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    assert assign_labels(gaussians, [cam], pipe, num_labels=2).tolist() == [1, 1, -1]


def _camera(width=128, height=96, fov_deg=60.0):
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
