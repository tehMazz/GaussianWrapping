"""Hollowing tests. CPU: free-grid marching, sample loss, entropy, decay, density estimator, floaters (AABB and
world grid), legacy hole keys. GPU (skipped without CUDA / beags_api): seen ledger, unseen pruning, carving, free
grid, schedule, hole metrics."""
import json
import math
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import pytest
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "submodules"))

from hollow.free_space import (FreeGrid, FreeSpacePrior, _pack, _unpack, apply_opacity_decay, march_free_space,
                               mean_density, opacity_entropy_loss)
from utils.general_utils import inverse_sigmoid
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


def _camera(center=(0.0, 0.0, 0.0), width=32, height=32, fov_deg=60.0, device="cpu"):
    """Camera at `center` looking down +z (identity rotation)."""
    fov = math.radians(fov_deg)
    view = torch.tensor(getWorld2View2(np.eye(3), -np.asarray(center)), dtype=torch.float32).transpose(0, 1).to(device)
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov).transpose(0, 1).to(device)
    return SimpleNamespace(FoVx=fov, FoVy=fov, image_width=width, image_height=height, world_view_transform=view,
                           full_proj_transform=view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0),
                           camera_center=view.inverse()[3, :3])


class _Duck:
    """GaussianModel stand-in on CPU with leaf raw parameters."""

    def __init__(self, xyz, scale, opacity, rotation=None):
        n = xyz.shape[0]
        self._xyz = xyz.clone().requires_grad_(True)
        self._scaling = torch.log(scale).clone().requires_grad_(True)
        self._opacity = inverse_sigmoid(opacity).reshape(n, 1).clone().requires_grad_(True)
        rotation = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1) if rotation is None else rotation
        self._rotation = rotation.clone().requires_grad_(True)
        self.inverse_opacity_activation = inverse_sigmoid

    get_xyz = property(lambda self: self._xyz)
    get_rotation = property(lambda self: torch.nn.functional.normalize(self._rotation, dim=1))
    get_scaling_with_3D_filter = property(lambda self: torch.exp(self._scaling))
    get_opacity = property(lambda self: torch.sigmoid(self._opacity))
    get_opacity_with_3D_filter = property(lambda self: torch.sigmoid(self._opacity))


def _random_duck(n=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    return _Duck(torch.rand(n, 3, generator=g) * 2 - 1, 0.05 + 0.05 * torch.rand(n, 3, generator=g),
                 0.2 + 0.6 * torch.rand(n, generator=g), torch.rand(n, 4, generator=g) - 0.5)


LO, HI = torch.tensor([-2.0, -2.0, -0.5]), torch.tensor([2.0, 2.0, 4.0])


def _free(grid, *points):
    return grid.contains(torch.tensor(points, dtype=torch.float32)).tolist()


def test_pack_roundtrip():
    ijk = torch.tensor([[0, 0, 0], [-5, 7, 123], [2 ** 20 - 1, -(2 ** 20), 3]])
    assert torch.equal(_unpack(_pack(ijk)), ijk)


def test_march_free_space_plane():
    cam = _camera()
    depth = torch.full((32, 32), 2.0)
    grid = march_free_space([cam], [depth], LO, HI, 0.1, min_views=1, margin_voxels=2)
    # in front of the wall, inside the margin, behind, outside the frustum
    assert _free(grid, (0, 0, 1.0), (0, 0, 1.85), (0, 0, 3.0), (1.0, 0, 0.5)) == [True, False, False, False]
    assert _free(grid, (1e9, 0, 0), (-1e9, -1e9, -1e9)) == [False, False]


def test_march_free_space_min_views():
    depth = torch.full((32, 32), 2.0)
    one = march_free_space([_camera()], [depth], LO, HI, 0.1, min_views=2)
    assert one.keys.numel() == 0
    two = march_free_space([_camera(), _camera((0.05, 0, 0))], [depth, depth], LO, HI, 0.1, min_views=2)
    assert _free(two, (0, 0, 1.0)) == [True]


def test_march_free_space_no_surface_is_free_to_aabb_exit():
    grid = march_free_space([_camera()], [torch.zeros(32, 32)], LO, HI, 0.1, min_views=1)
    assert _free(grid, (0, 0, 1.0), (0, 0, 3.5), (0, 0, 4.5)) == [True, True, False]


def test_free_space_prior_gradient_sign():
    duck = _random_duck()
    prior = FreeSpacePrior(None, duck, n_random=4096, samples_per_gaussian=8)
    torch.manual_seed(0)
    loss = prior.loss(duck)
    loss.backward()
    assert loss.item() > 0
    assert prior.touched.all()
    assert (duck._opacity.grad > 0).all()      # descent lowers opacity
    assert (duck._scaling.grad > 0).all()      # and shrinks the Gaussians at fixed samples


def test_free_space_prior_free_grid_term():
    duck = _random_duck()
    prior = FreeSpacePrior(None, duck, n_random=1024, samples_per_gaussian=4)
    torch.manual_seed(0)
    base = prior.loss(duck).item()
    lo = prior.lo - 1
    dims = torch.ceil((prior.hi + 1 - lo) / 0.25).long().tolist()
    ijk = torch.stack(torch.meshgrid(*[torch.arange(d) for d in dims], indexing="ij"), -1).reshape(-1, 3)
    prior.grid = FreeGrid(torch.empty(0, dtype=torch.long), lo, 0.25)
    torch.manual_seed(0)
    assert prior.loss(duck).item() == pytest.approx(base)
    prior.grid = FreeGrid(torch.sort(_pack(ijk)).values, lo, 0.25)  # every sample is in free space
    torch.manual_seed(0)
    assert prior.loss(duck).item() > base + 0.01


def test_mean_density_single_gaussian():
    s, o = 0.1, 0.5
    integral = o * (2 * math.pi) ** 1.5 * s ** 3 * math.erf(3 / math.sqrt(2)) ** 3
    lo, hi = -torch.ones(3), torch.ones(3)
    torch.manual_seed(0)
    centred = _Duck(torch.zeros(1, 3), torch.full((1, 3), s), torch.tensor([o]))
    assert mean_density(centred, lo, hi, n=2 ** 18) == pytest.approx(integral / 8, rel=0.03)
    half = lambda x: x[:, 0] > 0
    assert mean_density(centred, lo, hi, half, 4.0, n=2 ** 18) == pytest.approx(integral / 8, rel=0.03)
    on_face = _Duck(torch.tensor([[1.0, 0, 0]]), torch.full((1, 3), s), torch.tensor([o]))
    assert mean_density(on_face, lo, hi, n=2 ** 18) == pytest.approx(integral / 2 / 8, rel=0.03)


def test_opacity_entropy_loss():
    duck = _Duck(torch.zeros(3, 3), torch.ones(3, 3), torch.tensor([0.5, 0.3, 0.8]))
    loss = opacity_entropy_loss(duck)
    h = lambda p: -(p * math.log(p) + (1 - p) * math.log(1 - p))
    assert loss.item() == pytest.approx((h(0.5) + h(0.3) + h(0.8)) / 3, rel=1e-4)
    loss.backward()
    g = duck._opacity.grad[:, 0]
    assert abs(g[0]) < 1e-6 and g[1] > 0 and g[2] < 0  # 0.3 -> 0, 0.8 -> 1
    binary = _Duck(torch.zeros(2, 3), torch.ones(2, 3), torch.tensor([1e-4, 1 - 1e-4]))
    assert opacity_entropy_loss(binary).item() < 1e-2


def test_apply_opacity_decay():
    duck = _random_duck(50)
    before = duck.get_opacity.detach().clone()
    apply_opacity_decay(duck, 0.9)
    assert torch.allclose(duck.get_opacity.detach(), before * 0.9, atol=1e-6)


def test_count_floaters():
    hm = pytest.importorskip("hollow_metrics", exc_type=ImportError)
    g = torch.Generator().manual_seed(0)
    main = torch.rand(2000, 3, generator=g) * torch.tensor([1.0, 1.0, 0.05])        # dense slab
    blobs = torch.cat([torch.full((5, 3), 0.5) + torch.tensor([0, 0, 0.5]), torch.full((3, 3), 0.9) + torch.tensor([0, 0, 0.4])])
    xyz = torch.cat([main, blobs])
    opacity = torch.ones(xyz.shape[0])
    opacity[-1] = 0.1                                                                   # transparent: not occupied
    n, count = hm.count_floaters(xyz, opacity, torch.zeros(3) - 0.1, torch.ones(3) + 0.5, 0.05)
    assert (n, count) == (2, 7)


def test_count_floaters_world_grid():
    """AABB-relative cells move with lo (and so does the count); snapped cells do not."""
    hm = pytest.importorskip("hollow_metrics", exc_type=ImportError)
    x = torch.tensor([0.012, 0.102, 0.192, 0.282, 0.372])                              # 0.09 apart, voxel 0.05
    xyz = torch.stack([x, torch.full_like(x, 0.5), torch.full_like(x, 0.5)], 1)
    ones, hi = torch.ones(5), torch.ones(3)
    lo = lambda l: torch.tensor([l, 0.0, 0.0])
    assert hm.count_floaters(xyz, ones, lo(-0.03), hi, 0.05) != hm.count_floaters(xyz, ones, lo(-0.01), hi, 0.05)
    world = hm.count_floaters(xyz, ones, lo(-0.05), hi, 0.05)                           # origin on the world grid
    for l in (-0.049, -0.03, -0.01, -0.001):
        assert hm.count_floaters(xyz, ones, lo(l), hi, 0.05, snap=True) == world
    shifted = xyz + torch.tensor([0.25, 0.0, 0.0])                                      # 5 cells
    assert hm.count_floaters(shifted, ones, lo(0.21), hi + 1, 0.05, snap=True) == world


def test_load_metrics_legacy_keys(tmp_path):
    hm = pytest.importorskip("hollow_metrics", exc_type=ImportError)
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"hole_views": 3, "hole_emptiness": 0.99, "hole_emptiness_8x8": 0.995,
                               "hole_unseen_alpha": 2e-4, "floaters": 7}))
    assert hm.load_metrics(old) == {"hole_views": 3, "hole_alpha": 0.99, "hole_alpha_8x8": 0.995,
                                    "hole_alpha_unseen": 2e-4, "floaters": 7}
    both = tmp_path / "both.json"
    both.write_text(json.dumps({"hole_alpha": 0.5, "hole_emptiness": 0.9}))
    assert hm.load_metrics(both) == {"hole_alpha": 0.5}
    assert set(hm.LEGACY_KEYS.values()) == {"hole_alpha", "hole_alpha_8x8", "hole_alpha_unseen"}


# ---------------- GPU ----------------

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _beags_api():
    return pytest.importorskip("gaussian_renderer.beags_api", exc_type=ImportError)


def _gpu_camera(center=(0.0, 0.0, 0.0), size=64):
    from scene.cameras import Camera

    fov = math.radians(60.0)
    return Camera(colmap_id=0, R=np.eye(3), T=-np.asarray(center, dtype=np.float64), FoVx=fov, FoVy=fov,
                  image=torch.zeros(3, size, size), gt_alpha_mask=None, image_name=f"v{center}", uid=0)


def _gpu_model(xyz, opacity, scale, cameras):
    from arguments import OptimizationParams
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud

    g = GaussianModel(0, use_mip_filter=True)
    pts = xyz.cpu().numpy()
    g.create_from_pcd(BasicPointCloud(points=pts, colors=np.full_like(pts, 0.5), normals=np.zeros_like(pts)), 1.0)
    g._scaling.data[:] = torch.log(torch.as_tensor(scale, dtype=torch.float32, device="cuda")).expand_as(g._scaling)
    g._opacity.data[:] = inverse_sigmoid(torch.as_tensor(opacity, dtype=torch.float32, device="cuda")).reshape(-1, 1)
    g.training_setup(OptimizationParams(ArgumentParser()))
    g.init_culling(len(cameras))
    g.compute_3D_filter(cameras=cameras)
    return g


def _wall_scene(cameras):
    """Opaque wall at z = 4 (o 0.95), faint fog at z = 2 (o 0.03), and Gaussians behind the cameras."""
    s = torch.linspace(-2.5, 2.5, 60)
    wall = torch.stack([*torch.meshgrid(s, s, indexing="ij"), torch.full((60, 60), 4.0)], -1).reshape(-1, 3)
    f = torch.linspace(-0.4, 0.4, 5)
    fog = torch.stack([*torch.meshgrid(f, f, indexing="ij"), torch.full((5, 5), 2.0)], -1).reshape(-1, 3)
    behind = torch.tensor([[0.0, 0.0, -3.0], [0.5, 0.5, -2.0]])
    xyz = torch.cat([wall, fog, behind])
    opacity = torch.cat([torch.full((wall.shape[0],), 0.95), torch.full((fog.shape[0],), 0.03), torch.full((2,), 0.9)])
    g = _gpu_model(xyz, opacity, 0.06, cameras)
    n_wall, n_fog = wall.shape[0], fog.shape[0]
    return g, slice(0, n_wall), slice(n_wall, n_wall + n_fog), slice(n_wall + n_fog, None)


PIPE = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)


@gpu
def test_seen_and_prune_unseen():
    _beags_api()
    from hollow.ledger import compute_seen, prune_unseen

    cams = [_gpu_camera()]
    g, wall, fog, behind = _wall_scene(cams)
    seen = compute_seen(cams, g, PIPE)
    assert (seen[behind] == 0).all() and seen[wall].sum() > 100 and (seen[fog] > 0).all()
    n0 = g.get_xyz.shape[0]
    tau = 1e-6
    expected = int((seen < tau).sum())
    assert prune_unseen(g, seen, tau) == expected >= 2
    assert g.get_xyz.shape[0] == g.filter_3D.shape[0] == n0 - expected


@gpu
def test_carve_see_through():
    _beags_api()
    from hollow.carve import carve_see_through

    cams = [_gpu_camera(), _gpu_camera((0.1, 0.0, 0.0))]
    g, wall, fog, behind = _wall_scene(cams)
    carve = carve_see_through(g, cams, PIPE, threshold=0.9)
    assert carve[fog].float().mean() > 0.9
    assert not carve[wall].any() and not carve[behind].any()


@gpu
def test_build_free_grid():
    _beags_api()
    from hollow.free_space import build_free_grid

    cams = [_gpu_camera(), _gpu_camera((0.1, 0.0, 0.0))]
    g, *_ = _wall_scene(cams)
    lo, hi = torch.tensor([-3.0, -3.0, -0.5], device="cuda"), torch.tensor([3.0, 3.0, 5.0], device="cuda")
    grid = build_free_grid(cams, g, PIPE, 0.1, bounds=(lo, hi))
    pts = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0], [0.0, 0.0, 4.6]], device="cuda")
    assert grid.contains(pts).tolist() == [True, True, False, False]


@gpu
def test_schedule_loss_and_prune():
    _beags_api()
    from hollow.schedule import HollowSchedule

    cams = [_gpu_camera(), _gpu_camera((0.1, 0.0, 0.0))]
    g, *_ = _wall_scene(cams)
    parser = ArgumentParser()
    HollowSchedule.add_args(parser)
    args = parser.parse_args(["--hollow", "--hollow_from", "0", "--lambda_entropy", "0", "--prune_unseen_every", "1",
                              "--carve_every", "1", "--seen_tau", "1e-6"])
    args.iterations = 10
    hollow = HollowSchedule(args, SimpleNamespace(getTrainCameras=lambda: cams), g, PIPE)
    loss = hollow.loss(1)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0
    assert hollow.prior.grid is not None and hollow.prior.grid.keys.numel() > 0
    assert hollow.touched is not None and hollow.touched.shape[0] == g.get_xyz.shape[0]
    assert g._opacity.grad is not None and (g._opacity.grad[hollow.touched] > 0).all()
    n0 = g.get_xyz.shape[0]
    assert hollow.after_step(1)                    # behind-camera (unseen) and fog (see-through) Gaussians
    assert g.get_xyz.shape[0] < n0 and hollow.touched is None
    assert not HollowSchedule(parser.parse_args([]), SimpleNamespace(getTrainCameras=lambda: cams), g, PIPE).loss(1)


@gpu
def test_hole_metrics():
    _beags_api()
    hm = pytest.importorskip("hollow_metrics", exc_type=ImportError)
    from hollow.ledger import compute_seen

    cams = [_gpu_camera()]
    g, wall, fog, behind = _wall_scene(cams)
    xyz = g.get_xyz.detach()
    obj = torch.zeros(xyz.shape[0], dtype=torch.bool, device="cuda")
    obj[wall] = (xyz[wall, :2].abs() < 0.6).all(dim=1)                 # a patch of the wall is the object
    seen = compute_seen(cams, g, PIPE)
    out = hm.hole_metrics(cams, g, PIPE, obj, seen, 1e-6)
    assert set(out) == {"hole_views", "hole_alpha", "hole_alpha_8x8", "hole_alpha_unseen"}
    assert out["hole_views"] == 1
    assert 0 <= out["hole_alpha"] < 0.5                                 # nothing behind the patch
    assert 0 <= out["hole_alpha_8x8"] <= 1
    assert out["hole_alpha_unseen"] < 1e-4                              # unseen = behind the camera
