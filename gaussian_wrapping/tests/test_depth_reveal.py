"""beags/depth_reveal.py: camera geometry, multi-view consistency, classes and score (CPU)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from beags import depth_reveal as dr

H, W, F = 40, 60, 50.0


def look_at(center, target, up=(0.0, -1.0, 0.0)):
    """PinholeView at center looking at target (COLMAP-style camera: x right, y down, z forward)."""
    c, t = np.asarray(center, float), np.asarray(target, float)
    z = (t - c) / np.linalg.norm(t - c)
    x = np.cross(np.asarray(up, float), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 0)                     # world -> camera rows
    w2v = np.eye(4)
    w2v[:3, :3] = R.T                              # row-vector convention: p_view = p @ R.T + t
    w2v[3, :3] = -c @ R.T
    return dr.PinholeView(torch.tensor(w2v, dtype=torch.float32), F, F, (W - 1) / 2, (H - 1) / 2, H, W)


def plane_depth(view, height=0.0):
    """(H, W) z-depth of the plane y_world = height seen by view (0 where the ray misses it)."""
    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    d_cam = torch.stack([(j.float() - view.cx) / F, (i.float() - view.cy) / F, torch.ones(H, W)], -1).reshape(-1, 3)
    d_world = d_cam @ view.rot_inv                  # direction at z = 1
    t = (height - view.center[1]) / d_world[:, 1]
    return torch.where(t > 0, t, torch.zeros_like(t)).reshape(H, W)


def test_project_backproject_round_trip():
    view = look_at([0.3, -2.0, -3.0], [0.0, 0.0, 0.0])
    rows, cols = torch.tensor([0, 5, 39, 20]), torch.tensor([0, 17, 59, 30])
    z = torch.tensor([1.0, 2.5, 4.0, 7.0])
    xyz = view.backproject(rows, cols, z)
    u, v, zz = view.project(xyz)
    torch.testing.assert_close(u, cols.float(), atol=1e-4, rtol=0)
    torch.testing.assert_close(v, rows.float(), atol=1e-4, rtol=0)
    torch.testing.assert_close(zz, z, atol=1e-5, rtol=0)
    torch.testing.assert_close(view.center, torch.tensor([0.3, -2.0, -3.0]), atol=1e-5, rtol=0)


def test_principal_point_is_image_centre():
    view = look_at([0.0, 0.0, -5.0], [0.0, 0.0, 0.0])
    u, v, z = view.project(torch.tensor([[0.0, 0.0, 0.0]]))
    assert float(u) == pytest.approx((W - 1) / 2) and float(v) == pytest.approx((H - 1) / 2) and float(z) == pytest.approx(5)


def test_check_views_seen_occluded_behind_outside():
    """X on the floor (y = 0). View 0 sees the floor there; view 1 has a wall in front; view 2 sees past X
    (no floor: depth 0); view 3 looks away (X outside the image)."""
    x = torch.tensor([[0.0, 0.0, 0.0]])
    views = [look_at([0.0, -3.0, -3.0], [0.0, 0.0, 0.0]), look_at([2.0, -3.0, -2.0], [0.0, 0.0, 0.0]),
             look_at([-2.0, -3.0, -2.0], [0.0, 0.0, 0.0]), look_at([0.0, -3.0, -3.0], [0.0, -3.0, -10.0])]
    depths = torch.stack([plane_depth(views[0]), plane_depth(views[1]) * 0.5, torch.zeros(H, W), plane_depth(views[3])])
    ray = torch.nn.functional.normalize(x - torch.tensor([[0.0, -3.0, 3.0]]), dim=1)
    res = dr.check_views(x, ray, views, depths, [0.02, 0.6], radius=1)
    assert res["could"].item() == 3                       # view 3 cannot see X
    assert res["seen"][0].item() == 1                     # only view 0 at 2%
    assert res["seen"][1].item() == 2                     # the wall at half depth passes a 60% tolerance
    assert res["occluded"].item() == 1 and res["behind"].item() == 1
    d0 = torch.nn.functional.normalize(x - views[0].center, dim=1)
    assert res["best_cos"].item() == pytest.approx(float((d0 * ray).sum()), abs=1e-5)


def test_check_views_skip_and_never_seen():
    x = torch.tensor([[0.0, 0.0, 0.0]])
    views = [look_at([0.0, -3.0, -3.0], [0.0, 0.0, 0.0])] * 2
    depths = torch.stack([plane_depth(views[0])] * 2)
    ray = torch.nn.functional.normalize(x - views[0].center, dim=1)
    res = dr.check_views(x, ray, views, depths, [0.02], skip=0)
    assert res["could"].item() == 1 and res["seen"][0].item() == 1
    res = dr.check_views(x, ray, views, depths * 0.5, [0.02])
    assert res["seen"][0].item() == 0 and res["occluded"].item() == 2 and res["best_cos"].item() == -1


def test_classify_and_score():
    d_with = torch.tensor([[2.0, 2.0, 2.0, 0.0]])
    a_with = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    d_wo = torch.tensor([[2.01, 3.0, 0.0, 0.0]])
    a_wo = torch.tensor([[1.0, 1.0, 0.1, 0.0]])
    cls, rel = dr.classify(d_with, a_with, d_wo, a_wo, reveal_tol=0.02, min_alpha=0.5)
    assert cls.tolist() == [[dr.UNCHANGED, dr.REVEALED, dr.EMPTIED, dr.UNCHANGED]]
    assert rel[0, 1].item() == pytest.approx(0.5)
    seen = torch.tensor([[0, 1, 0, 0]])
    s = dr.score(cls, seen, min_seen=4)
    assert s.tolist() == [[0.0, pytest.approx(0.75), 1.0, 0.0]]
    assert dr.score(cls, seen * 0 + 9, min_seen=3)[0, 1].item() == 0.0   # seen often: S 0 whatever the share
    assert dr.score(cls, seen * 0, min_seen=3)[0, 1].item() == 1.0       # never seen


def test_object_pixels_never_count_as_seen():
    """X just behind the object's surface in q: within tol of the object's depth, but q only saw the object."""
    x = torch.tensor([[0.0, 0.0, 0.0]])
    views = [look_at([0.0, -3.0, -3.0], [0.0, 0.0, 0.0])]
    depth = (plane_depth(views[0]) * 0.99)[None]                          # object surface 1% in front of the floor
    ray = torch.nn.functional.normalize(x - views[0].center, dim=1)
    assert dr.check_views(x, ray, views, depth, [0.02])["seen"][0].item() == 1
    obj = torch.ones_like(depth, dtype=torch.bool)
    res = dr.check_views(x, ray, views, depth, [0.02], ref_object=obj)
    assert res["seen"][0].item() == 0 and res["occluded"].item() == 0 and res["behind"].item() == 1
    res = dr.check_views(x, ray, views, depth * 0.5, [0.02], ref_object=obj)
    assert res["occluded"].item() == 1 and res["occluded_obj"].item() == 1


def test_figures_and_stats_cpu(tmp_path):
    rng = np.random.default_rng(0)
    h, w = 20, 30
    cls = rng.integers(0, 3, (h, w)).astype(np.uint8)
    seen = rng.integers(0, 50, (h, w)).astype(np.int32)
    could = seen + rng.integers(0, 50, (h, w)).astype(np.int32)
    share = np.where(could > 0, seen / np.maximum(could, 1), 0).astype(np.float32)
    img = rng.random((h, w, 3)).astype(np.float32)
    angle = np.where(seen > 0, rng.random((h, w)) * 90, np.nan).astype(np.float32)
    dr.view_figure(str(tmp_path / "v.jpg"), "v", img, img, img, rng.random((h, w)), cls, rng.random((h, w)), seen,
                   seen // 2, share, angle, 0.02, 244)
    row = dr.grid_row(img, img, rng.random((h, w)), cls, rng.random((h, w)).astype(np.float32), width=60)
    assert row.shape[1] == 4 * 60 and dr.grid_header(60).shape[1] == row.shape[1]
    dark = rng.random(100) < 0.3
    dr.hist_figure(str(tmp_path / "h.png"), seen.ravel()[:100], share.ravel()[:100], seen.ravel()[:100] // 2, dark)
    st = dr.seen_stats(seen.ravel()[:100], could.ravel()[:100], dark)
    json.dumps(st)
    assert st["n_all"] == 100 and 0 <= st["never_seen_lt3_dark"] <= 1
    assert dr.seen_stats(np.zeros(0), np.zeros(0), np.zeros(0, bool)) == {"n": 0}
    assert (tmp_path / "v.jpg").stat().st_size > 0 and (tmp_path / "h.png").stat().st_size > 0


def test_script_help_runs_as_a_file():
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "beags", "depth_reveal.py")
    out = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ""})
    assert out.returncode == 0, out.stderr[-2000:]
    assert "--reveal_tol" in out.stdout
