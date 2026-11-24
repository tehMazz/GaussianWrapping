"""render_ply_frames.py: ArtiFixer-GS's render_ply backend (RENDER_CMD) with GW's render_ours.

    python -m pytest -q gaussian_wrapping/tests/test_render_ply_frames.py   (GPU test skips without CUDA)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

GW = Path(__file__).resolve().parents[1]
ROOT = GW.parent
AFX = ROOT / "submodules" / "artifixer_gs"
sys.path.insert(0, str(GW))
sys.path.append(str(ROOT))
sys.path.append(str(AFX))

import render_ply_frames as rpf  # noqa: E402
import export_3dgs_ply as export  # noqa: E402

needs_artifixer = pytest.mark.skipif(not (AFX / "artifixer_gs" / "render.py").is_file(),
                                     reason="submodules/artifixer_gs is not checked out")
W, H = 64, 48


def write_transforms(path, w2cs, fx=50.0, fy=52.0, cx=W / 2, cy=H / 2):
    """transforms.json (nerfstudio, OpenGL c2w) of OpenCV world-to-camera matrices."""
    frames = []
    for i, w2c in enumerate(w2cs):
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1  # OpenCV -> OpenGL
        frames.append({"file_path": f"images/{i}.png", "transform_matrix": c2w.tolist()})
    path.write_text(json.dumps({"camera_model": "OPENCV", "w": W, "h": H, "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
                                "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "frames": frames}))
    return path


def look_at_w2c(eye, target=(0., 0., 0.)):
    """OpenCV w2c (x right, y down, z forward) looking from eye at target, world up = -y."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0., -1., 0.])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    w2c = np.eye(4)
    w2c[:3, :3] = np.stack([x, y, z])
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    return w2c


def test_principal_point_offset():
    Ks = np.array([[[50, 0, W / 2], [0, 50, H / 2], [0, 0, 1]], [[50, 0, W / 2 + 1.5], [0, 50, H / 2 - 0.25], [0, 0, 1]]])
    assert rpf.principal_point_offset(Ks[:1], W, H) == 0.0
    assert rpf.principal_point_offset(Ks, W, H) == pytest.approx(1.5)


def test_png_quantization_matches_artifixer():
    rgb = torch.tensor([0.0, 0.5, 1.0, 1.2, -0.1, 0.999]).reshape(3, 1, 2)
    assert rpf.to_png_rgb(rgb).tolist() == [[[0, 255, 0], [128, 255, 255]]]
    assert rpf.to_png_alpha(torch.tensor([[0.0, 0.5, 1.0]])).tolist() == [[0, 128, 255]]


@needs_artifixer
def test_empty_ply_writes_background_frames(tmp_path):
    """No Gaussians (e.g. an empty removal subset): black renders, zero opacity and depth, without the GPU."""
    ply = tmp_path / "empty.ply"
    ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nend_header\n")
    traj = write_transforms(tmp_path / "transforms.json", [look_at_w2c([0, 0, -3]), look_at_w2c([1, 0, -3])])
    n = rpf.render_frames(ply, traj, tmp_path / "out", pipe=None, depth=True)
    assert n == 2 and rpf.ply_vertex_count(ply) == 0
    for i in range(2):
        assert np.asarray(Image.open(tmp_path / "out" / "renders" / f"{i:05d}.png")).max() == 0
        assert np.asarray(Image.open(tmp_path / "out" / "opacity" / f"{i:05d}.png")).shape == (H, W)
        assert not np.load(tmp_path / "out" / "depth" / f"{i:05d}.npy").any()


@needs_artifixer
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.no_grad()
def test_baked_ply_renders_like_native_cuda(tmp_path):
    """GW model with a 3D filter rendered natively == its export_3dgs_ply.py PLY through render_frames (the
    transforms.json camera path ArtiFixer-GS uses): same pixels up to 8-bit quantization, same alpha and depth."""
    from argparse import ArgumentParser
    from arguments import PipelineParams
    from gaussian_renderer.ours import render_ours
    from scene import GaussianModel

    parser = ArgumentParser()
    params = PipelineParams(parser)
    pipe = params.extract(parser.parse_args([]))
    rng = np.random.default_rng(0)
    n = 400
    g = GaussianModel(1, use_mip_filter=True)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device="cuda")
    g._xyz = t(rng.uniform(-0.6, 0.6, (n, 3)))
    g._features_dc = t(rng.normal(0, 0.5, (n, 1, 3)))
    g._features_rest = t(rng.normal(0, 0.1, (n, 3, 3)))
    g._opacity = t(rng.normal(1.0, 1.0, (n, 1)))
    g._scaling = t(np.log(rng.uniform(0.02, 0.08, (n, 3))))
    g._rotation = t(rng.normal(size=(n, 4)))
    g.filter_3D = t(rng.uniform(0.0, 0.03, (n, 1)))
    g.active_sh_degree = 1
    gw_ply = tmp_path / "gw" / "point_cloud.ply"
    gw_ply.parent.mkdir()
    g.save_ply(str(gw_ply))
    ply = tmp_path / "export" / "point_cloud.ply"
    export.export_3dgs_ply(str(gw_ply), str(ply))

    w2cs = [look_at_w2c([0.3, -0.4, -2.5]), look_at_w2c([-1.5, -0.2, -2.0])]
    traj = write_transforms(tmp_path / "transforms.json", w2cs)
    rpf.render_frames(ply, traj, tmp_path / "out", pipe, depth=True)

    from artifixer_gs.render import trajectory_cameras
    viewmats, Ks, _, _ = trajectory_cameras(traj)
    bg = torch.zeros(3, device="cuda")
    for i in range(len(w2cs)):
        cam = rpf.gw_camera(viewmats[i], Ks[i], W, H, i)
        native = render_ours(cam, g, pipe, bg, require_depth=True)
        rgb = np.asarray(Image.open(tmp_path / "out" / "renders" / f"{i:05d}.png")).astype(int)
        alpha = np.asarray(Image.open(tmp_path / "out" / "opacity" / f"{i:05d}.png")).astype(int)
        assert np.abs(rgb - rpf.to_png_rgb(native["render"]).astype(int)).max() <= 1
        assert np.abs(alpha - rpf.to_png_alpha(native["alpha"][0]).astype(int)).max() <= 1
        assert alpha.max() > 200  # the cloud is in view
        depth = np.load(tmp_path / "out" / "depth" / f"{i:05d}.npy")
        a = native["alpha"][0].cpu().numpy()
        med = native["median_depth"][0].cpu().numpy()
        assert not depth[a < 0.5].any() and (depth[a >= 0.5] > 0).all()   # every opaque pixel has a depth
        assert np.allclose(depth[(a >= 0.5) & (med > 0)], med[(a >= 0.5) & (med > 0)], rtol=1e-4, atol=1e-5)
        z = (np.asarray(viewmats[i]) @ np.r_[np.zeros(3), 1.0])[2]  # depth of the world origin, the cloud centre
        assert abs(np.median(depth[depth > 0]) - z) < 0.7


def test_surface_depth_falls_back_to_expected_depth():
    """GW's median depth is 0 on some opaque pixels: the expected depth fills them; transparent pixels stay 0."""
    median = torch.tensor([[2.0, 0.0, 0.0, 3.0]])
    alpha = torch.tensor([[0.9, 0.52, 0.3, 0.6]])
    expected = torch.tensor([[2.1, 4.0, 5.0, 3.2]])
    assert rpf.surface_depth(median, alpha, expected).tolist() == [[2.0, 4.0, 0.0, 3.0]]


@needs_artifixer
def test_center_principal_point(tmp_path, monkeypatch):
    from artifixer_gs.render import trajectory_cameras
    traj = write_transforms(tmp_path / "t.json", [look_at_w2c([0, 0, -3])], cx=W / 2 + 3, cy=H / 2 - 2)
    _, Ks, _, _ = trajectory_cameras(traj)
    assert rpf.principal_point_offset(Ks, W, H) == pytest.approx(3)
    monkeypatch.setenv("CENTER_PRINCIPAL_POINT", "1")
    _, Ks, _, _ = trajectory_cameras(traj)
    assert rpf.principal_point_offset(Ks, W, H) == 0 and Ks[0, 0, 0] == pytest.approx(50.0)
