"""Unit tests for seed2gs.py / seg/ (CPU; CUDA tests skip without a GPU). Run from gaussian_wrapping/: python -m pytest -q tests/test_seed2gs.py"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seg.lift import knn_spacing, largest_component, project, seed_from_mask
from seg.orbits import elevation, flat_orbit, orbit_pose, spiral, world_up
from seg.votes import flashsplat_labels

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _look_at_cv(pos, target, up):
    """OpenCV c2w (x right, y down, z forward) at pos looking at target."""
    z = target - pos
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    c2w[:3, 3] = pos
    return c2w


def _ref(anchor=np.array([0.3, -0.2, 0.5]), up=np.array([0.0, 0.0, 1.0])):
    pos = anchor + np.array([2.0, 1.0, 0.8])
    c2w = _look_at_cv(pos, anchor + np.array([0.1, 0.05, 0.0]), up)  # anchor slightly off-centre
    return c2w, anchor, up


def _cam_dir(c2w, p):
    q = np.linalg.inv(c2w) @ np.append(p, 1.0)
    return q[:2] / q[2]


def test_world_up_opencv():
    up = np.array([0.0, 0.0, 1.0])
    c2ws = [_look_at_cv(np.array([np.cos(a), np.sin(a), 0.5]) * 3, np.zeros(3), up) for a in np.linspace(0, 6, 7)]
    assert np.allclose(world_up(c2ws) @ up, 1.0, atol=0.1)
    level = [_look_at_cv(np.array([np.cos(a), np.sin(a), 0.0]) * 3, np.zeros(3), up) for a in np.linspace(0, 6, 7)]
    assert np.allclose(world_up(level), up, atol=1e-6)


def test_flat_orbit_geometry():
    c2w, anchor, up = _ref()
    views = flat_orbit(c2w, anchor, up, n=24)
    assert len(views) == 24
    assert np.allclose(views[0], c2w)
    r0 = np.linalg.norm(c2w[:3, 3] - anchor)
    e0 = elevation(c2w, anchor, up)
    d0 = _cam_dir(c2w, anchor)
    for v in views:
        assert np.allclose(v[:3, :3] @ v[:3, :3].T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.norm(v[:3, 3] - anchor), r0)          # radius = reference distance
        assert np.isclose(elevation(v, anchor, up), e0)                   # flat
        assert np.allclose(_cam_dir(v, anchor), d0, atol=1e-9)            # anchor keeps its image position
    # one full turn, evenly spaced azimuths
    flat = [(v[:3, 3] - anchor) - ((v[:3, 3] - anchor) @ up) * up for v in views]
    angles = [np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)) for a, b in zip(flat, flat[1:])]
    assert np.allclose(angles, 2 * np.pi / 24)


def test_spiral_climbs_and_caps():
    c2w, anchor, up = _ref()
    e0 = elevation(c2w, anchor, up)
    views = spiral(c2w, anchor, up, n=24, climb_deg=30.0)
    assert np.allclose(views[0], c2w)
    elev = np.array([elevation(v, anchor, up) for v in views])
    assert np.all(np.diff(elev) > 0)
    assert np.isclose(elev[-1] - e0, np.radians(30.0))
    r0 = np.linalg.norm(c2w[:3, 3] - anchor)
    assert all(np.isclose(np.linalg.norm(v[:3, 3] - anchor), r0) for v in views)
    capped = spiral(c2w, anchor, up, n=24, climb_deg=170.0, max_elev_deg=60.0)
    assert max(elevation(v, anchor, up) for v in capped) <= np.radians(60.0) + 1e-9


def test_orbit_pose_keeps_level_camera_level():
    up = np.array([0.0, 0.0, 1.0])
    c2w = _look_at_cv(np.array([3.0, 0.0, 0.0]), np.zeros(3), up)
    for yaw, pitch in ((0.7, 0.0), (1.3, 0.4), (-2.0, -0.3)):
        v = orbit_pose(c2w, np.zeros(3), yaw, pitch, up)
        assert abs(v[:3, 0] @ up) < 1e-9  # camera x axis stays horizontal


def _synthetic_scene(seed=0):
    """Identity camera (looks along +z), 64x64, f = 64. Object: dense cube at z~2 (big) and a small cube at z~2 far
    to the side, both inside the mask; background plane at z = 4 everywhere; clutter outside the mask at z = 2."""
    g = np.random.default_rng(seed)
    big = g.uniform([-0.3, -0.3, 1.95], [0.1, 0.1, 2.05], (800, 3))
    small = g.uniform([0.6, 0.6, 1.95], [0.7, 0.7, 2.05], (60, 3))
    back = np.column_stack([g.uniform(-2, 2, (500, 2)), np.full(500, 4.0)])
    clutter = g.uniform([-0.9, 0.3, 1.95], [-0.6, 0.6, 2.05], (100, 3))
    xyz = torch.tensor(np.concatenate([big, small, back, clutter]), dtype=torch.float32)
    H = W = 64
    fx = fy = 64.0
    cx = cy = (W - 1) / 2
    uv, z = project(xyz, torch.eye(4), fx, fy, cx, cy)
    mask = torch.zeros(H, W, dtype=torch.bool)
    depth = torch.full((H, W), 4.0)
    ij = uv.round().long()
    for lo, hi in ((0, 800), (800, 860)):  # mask = footprint of both object cubes, surface depth 2
        u, v = ij[lo:hi, 0].clamp(0, W - 1), ij[lo:hi, 1].clamp(0, H - 1)
        mask[v.min():v.max() + 1, u.min():u.max() + 1] = True
    depth[mask] = 1.95
    groups = np.repeat(np.arange(4), [800, 60, 500, 100])
    return xyz, mask, depth, (fx, fy, cx, cy), groups


def test_project_identity():
    xyz = torch.tensor([[0.0, 0.0, 2.0], [1.0, -0.5, 2.0]])
    uv, z = project(xyz, torch.eye(4), 10.0, 10.0, 5.0, 5.0)
    assert torch.allclose(uv, torch.tensor([[5.0, 5.0], [10.0, 2.5]]))
    assert torch.allclose(z, torch.tensor([2.0, 2.0]))


def test_seed_lift_synthetic():
    xyz, mask, depth, intr, groups = _synthetic_scene()
    seed = seed_from_mask(xyz, torch.eye(4), intr, mask, depth, depth_tol=0.05).numpy()
    assert seed[groups == 0].mean() > 0.95 and seed[groups == 1].mean() > 0.95
    assert not seed[groups == 2].any()   # behind the surface
    assert not seed[groups == 3].any()   # outside the mask
    points = xyz.numpy()[seed].astype(np.float64)
    voxel = knn_spacing(points, 8)
    keep = largest_component(points, voxel)
    kept_groups = groups[seed][keep]
    assert np.all(kept_groups == 0) and keep.sum() >= 0.95 * 800


def test_seed_lift_rejects_behind_camera_and_outside():
    xyz = torch.tensor([[0.0, 0.0, -2.0], [100.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    mask = torch.ones(8, 8, dtype=torch.bool)
    depth = torch.full((8, 8), 2.0)
    seed = seed_from_mask(xyz, torch.eye(4), (8.0, 8.0, 3.5, 3.5), mask, depth)
    assert seed.tolist() == [False, False, True]


def test_largest_component_and_spacing():
    grid = np.stack(np.meshgrid(*[np.arange(5) * 0.1] * 3, indexing="ij"), -1).reshape(-1, 3)
    assert np.isclose(knn_spacing(grid, 1), 0.1)
    other = grid[:8] + 10.0
    pts = np.concatenate([other, grid])
    keep = largest_component(pts, 0.1)
    assert keep.tolist() == [False] * 8 + [True] * len(grid)
    # diagonal neighbours are connected (26-connectivity)
    diag = np.array([[0.0, 0.0, 0.0], [0.12, 0.12, 0.12], [0.24, 0.24, 0.24], [5.0, 5.0, 5.0]])
    assert largest_component(diag, 0.1).tolist() == [True, True, True, False]
    assert largest_component(np.zeros((0, 3)), 0.1).shape == (0,)


def test_flashsplat_rule():
    votes = torch.tensor([[3.0, 1.0], [1.0, 3.0], [0.0, 0.0], [1.0, 1.1]])
    assert flashsplat_labels(votes).tolist() == [0, 1, 0, 1]
    assert flashsplat_labels(votes, gamma=0.1).tolist() == [0, 1, 0, 0]   # background bias flips the tie-ish one
    assert flashsplat_labels(votes, gamma=-0.7).tolist() == [1, 1, 0, 1]  # unvoted stays background
    assert flashsplat_labels(votes).dtype == torch.int32
    # scale invariance: normalisation makes the rule depend on the vote ratio only
    assert flashsplat_labels(votes * 1000, gamma=0.1).tolist() == [0, 1, 0, 0]


def test_flashsplat_matches_reference():
    """Same as FlashSplat multi_instance_opt for one object (row 1) of a (L=2, P) contribution matrix."""
    g = torch.Generator().manual_seed(0)
    votes = torch.rand(200, 2, generator=g) * (torch.rand(200, 1, generator=g) > 0.1)
    for gamma in (0.0, 0.2, -0.2):
        contrib = votes.T
        ref = torch.stack([contrib.sum(0) - contrib[1], contrib[1]], 0)
        ref = torch.nn.functional.normalize(ref, dim=0)
        ref[0] += gamma
        expected = (ref.argmax(0) == 1) & (votes.sum(1) > 0)
        assert torch.equal(flashsplat_labels(votes, gamma).bool(), expected)


@cuda
def test_virtual_camera_roundtrip():
    from types import SimpleNamespace
    from seed2gs import cam_to_world, virtual_camera
    c2w, anchor, up = _ref()
    ref = SimpleNamespace(image_height=48, image_width=64, FoVx=1.0, FoVy=0.8)
    for i, v in enumerate(flat_orbit(c2w, anchor, up, n=6)):
        cam = virtual_camera(ref, v, f"v{i}", i)
        assert np.allclose(cam_to_world(cam), v, atol=1e-5)
        assert (cam.image_height, cam.image_width) == (48, 64)
        small = virtual_camera(ref, v, f"s{i}", i, size=(24, 32))  # --track_side: same pose and FoV
        assert (small.image_height, small.image_width) == (24, 32) and small.FoVx == ref.FoVx
        assert np.allclose(cam_to_world(small), v, atol=1e-5)


# --- SAM3 worker protocol (sam3_runner.py --fake, CPU) ---------------------------------------------------------------

def _jpg(path, w, h):
    from PIL import Image
    Image.fromarray(np.full((h, w, 3), 128, np.uint8)).save(path)


def test_sam3_worker_protocol_fake(tmp_path):
    from seg.sam3 import Sam3Worker
    worker = Sam3Worker(python=sys.executable, runner_args=["--fake"])
    try:
        ready = worker.wait_ready()
        assert ready["ready"] and ready["fake"] and set(ready["seconds"]) == {"import", "build", "total"}
        assert ready["waited"] >= 0 and ready["since_start"] >= ready["waited"]
        _jpg(tmp_path / "ref.jpg", 40, 20)
        det = worker.detect(str(tmp_path / "ref.jpg"), "a thing", str(tmp_path / "ref.npy"))
        assert det["found"] and det["score"] == pytest.approx(0.9) and "seconds" in det
        ref = np.load(tmp_path / "ref.npy")
        assert ref.shape == (20, 40) and ref.dtype == bool and ref[10, 20] and not ref[0, 0]
        assert not worker.detect(str(tmp_path / "ref.jpg"), "nothing", str(tmp_path / "none.npy"))["found"]
        frames = tmp_path / "frames"
        frames.mkdir()
        for i in range(3):
            _jpg(frames / f"{i}.jpg", 20, 10)  # capped frames: half the reference size
        res = worker.track(str(frames), str(tmp_path / "ref.npy"), str(tmp_path / "out"))
        assert res["frames"] == 3 and res["nonempty"] == 3 and "seconds" in res
        for i in range(3):
            m = np.load(tmp_path / "out" / f"{i}.npy")
            assert m.shape == (10, 20) and m[5, 10] and not m[0, 0]
        with pytest.raises(RuntimeError, match="no frames"):  # error answer, worker keeps serving
            worker.track(str(tmp_path), str(tmp_path / "ref.npy"), str(tmp_path / "bad"))
        assert worker.detect(str(tmp_path / "ref.jpg"), "again", str(tmp_path / "again.npy"))["found"]
    finally:
        worker.close()
    assert worker.proc.returncode == 0


def test_sam3_worker_dead_process_raises():
    from seg.sam3 import Sam3Worker
    worker = Sam3Worker(python=sys.executable, runner_args=["--no_such_flag"])  # argparse exits before ready
    with pytest.raises(RuntimeError, match="exited with code"):
        worker.wait_ready()
    worker.kill()


def test_runner_serve_loop():
    from seg.sam3_runner import serve
    out = []
    lines = ['{"op": "echo", "x": 1}', "", '{"op": "boom"}', "not json", '{"op": "nope"}', '{"op": "quit"}',
             '{"op": "echo", "x": 2}']

    def boom():
        raise ValueError("bad")
    serve(lines, out.append, dict(echo=lambda x: {"x": x}, boom=boom))
    assert out[0] == {"x": 1}
    assert out[1]["error"].startswith("ValueError: bad")
    assert "error" in out[2] and "unknown op nope" in out[3]["error"]
    assert out[4] == {"ok": True} and len(out) == 5  # nothing after quit


def test_skip_init_restores_and_skips():
    from seg.sam3_runner import skip_init
    original = torch.nn.init.kaiming_uniform_
    with skip_init(True):
        assert torch.nn.init.kaiming_uniform_ is not original
        w = torch.full((4, 4), 7.0)
        torch.nn.init.trunc_normal_(w)
        torch.nn.init.zeros_(w)
        assert torch.all(w == 7.0)
        lin = torch.nn.Linear(3, 2)
        lin.load_state_dict({"weight": torch.ones(2, 3), "bias": torch.zeros(2)}, strict=True)
        assert torch.all(lin.weight == 1)
    assert torch.nn.init.kaiming_uniform_ is original
    with skip_init(False):
        assert torch.nn.init.kaiming_uniform_ is original


def test_worker_args_from_argv(monkeypatch):
    import seg.sam3 as sam3
    seen = {}
    monkeypatch.setattr(sam3, "Sam3Worker", lambda *a: seen.setdefault("args", a))
    sam3.worker_from_argv(["-m", "x", "--prompt", "p", "--sam3_python", "/py", "--sam3_args=--fake --no_prefetch"])
    assert seen["args"][0] == "/py" and seen["args"][3] == ["--fake", "--no_prefetch"]
    # unquoted shell expansion (slurm SEED2GS_ARGS) keeps literal quotes; commas separate flags
    for value in ("--sam3_args='--fake'", "--sam3_args=--fake,--no_prefetch", "--sam3_args='--fake --no_prefetch'"):
        seen.clear()
        sam3.worker_from_argv(["-m", "x", value])
        assert seen["args"][3] == ["--fake"] + (["--no_prefetch"] if "prefetch" in value else [])


def test_sam3_worker_close_before_ready_kills(tmp_path):
    import time
    from seg.sam3 import Sam3Worker
    slow = tmp_path / "slow_python"
    slow.write_text("#!/bin/sh\nexec sleep 60\n")  # a worker still importing / loading the checkpoint
    slow.chmod(0o755)
    worker = Sam3Worker(python=str(slow))
    t = time.time()
    worker.close()  # the GW side failed early: no wait for ready, no worker error in place of ours
    assert time.time() - t < 10 and worker.proc.returncode is not None and worker.ready is None


# --- mask IO (seg/masks.py) vs the BEA-GS loader (beags/masks.py) -------------------------------------------------

def test_beags_mask_roundtrip(tmp_path):
    from beags.masks import load_npy_mask, make_view_masks, soft_to_labels
    from seg.masks import beags_mask, object_channel, save_beags_mask
    g = np.random.default_rng(0)
    obj = g.random((30, 40)) > 0.6
    m = beags_mask(obj)
    assert m.shape == (2, 30, 40) and m.dtype == np.float16
    save_beags_mask(str(tmp_path), "IMG_0001", obj)
    soft = load_npy_mask(str(tmp_path / "IMG_0001.npy"), 30, 40)  # what beags_prepare / attach_masks reads
    assert torch.equal(soft_to_labels(soft), torch.from_numpy(obj.astype(np.int32)))
    vm = make_view_masks(soft, 30, 40, 2, 1.0, device="cpu")
    assert vm.present.tolist() == [True, True]
    assert np.array_equal(object_channel(np.load(tmp_path / "IMG_0001.npy")), obj)
    assert np.array_equal(object_channel(obj), obj)
    empty = beags_mask(np.zeros((5, 5), bool))  # object not visible: channel 1 absent, as SAM2 masks
    assert (soft_to_labels(torch.from_numpy(empty.astype(np.float32))) == 0).all()


def test_capped_size_and_slug():
    import subprocess
    from seg.masks import capped_size, prompt_slug
    assert capped_size(378, 504, 1008) == (378, 504)
    assert capped_size(900, 1600, 1008) == (900, 1600)      # short side under SAM3's 1008 square: untouched
    assert capped_size(1512, 2016, 1008) == (1008, 1344)    # short side capped, never below 1008 on either axis
    assert capped_size(3024, 2000, 1008) == (1524, 1008)
    assert capped_size(3024, 4032, 0) == (3024, 4032)
    common = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "slurm", "common.sh")
    fn = "".join(line for line in open(common) if line.startswith("slug()") or line.startswith("  local s; s=")) + "}\n"
    for p in ("fern plant", "Rubber duck with RED hat!", "  ", "a/b c.d"):
        out = subprocess.run(["bash", "-c", fn + 'slug "$1"', "_", p], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == prompt_slug(p)
    assert prompt_slug("  ") == "object" and prompt_slug("Green Apple") == "green_apple"


def test_link_model_keeps_source_untouched(tmp_path):
    import subprocess
    common = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "slurm", "common.sh")
    text = open(common).read()
    start = text.index("link_model() {")
    fn = text[start:text.index("\n}\n", start) + 3]
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "point_cloud" / "iteration_7").mkdir(parents=True)
    (src / "beags").mkdir()
    for f, txt in (("cfg_args", "Namespace()"), ("cameras.json", "[]"), ("results.json", "{}"),
                   ("point_cloud/iteration_7/point_cloud.ply", "ply"), ("point_cloud/iteration_7/labels.pt", "L"),
                   ("beags/labeled_points.npz", "N")):
        (src / f).write_text(txt)
    subprocess.run(["bash", "-c", fn + 'link_model "$1" "$2" 7', "_", str(src), str(dst)], check=True)
    it = dst / "point_cloud" / "iteration_7"
    assert (dst / "cfg_args").is_symlink() and (it / "point_cloud.ply").is_symlink()
    assert (it / "point_cloud.ply").read_text() == "ply"
    assert not (it / "labels.pt").exists() and not (dst / "beags").exists() and not (dst / "results.json").exists()
    (it / "labels.pt").write_text("seed")  # what beags_prepare writes into the view
    assert (src / "point_cloud" / "iteration_7" / "labels.pt").read_text() == "L"
    subprocess.run(["bash", "-c", fn + 'link_model "$1" "$2" 7', "_", str(src), str(dst)], check=True)  # rerun
    assert (it / "labels.pt").read_text() == "seed" and not (it / "labels.pt").is_symlink()


# --- seed2gs_eval.py -------------------------------------------------------------------------------------------------

def _box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def _seed_dir(root, name, views, prompt, labels=None):
    from seg.masks import save_beags_mask
    d = root / name
    (d / "masks").mkdir(parents=True)
    for view, obj in views.items():
        save_beags_mask(str(d / "masks"), view, obj)
    (d / "seed2gs.json").write_text('{"prompt": "%s"}' % prompt)
    if labels is not None:
        torch.save(torch.tensor(labels, dtype=torch.int32), d / "labels.pt")
    return str(d)


def test_view_metrics_and_matching():
    from seed2gs_eval import match_channels, view_metrics
    gt = _box(10, 10, 0, 5, 0, 10)
    r = view_metrics(gt.copy(), gt)
    assert r["IoU"] == 1 and r["bg_IoU"] == 1 and r["mIoU2"] == 1 and r["acc"] == 1 and r["BIoU"] == 1
    half = _box(10, 10, 0, 5, 0, 5)
    r = view_metrics(half, gt)
    assert r["IoU"] == pytest.approx(0.5) and r["acc"] == pytest.approx(0.75)
    assert r["bg_IoU"] == pytest.approx(50 / 75)
    # prompt 0 overlaps GT 2 best, prompt 1 overlaps GT 1 and GT 2, Hungarian keeps them distinct
    assert match_channels(np.array([[0, 0.1, 0.9], [0, 0.6, 0.8]])) == [2, 1]
    assert match_channels(np.array([[0, 0.3], [0, 0.4]])) == [None, 1]
    assert match_channels(np.zeros((1, 1))) == [None]


def test_label_agreement():
    from seed2gs_eval import label_agreement
    provided = np.array([-1, 0, 0, 1, 1, 1, 2, 2])
    seed = np.array([1, 0, 1, 1, 1, 0, 0, 0])
    r = label_agreement(seed, provided)
    assert r["provided_label"] == 1 and r["n_never_hit"] == 1
    assert r["IoU"] == pytest.approx(2 / 4) and r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["accuracy"] == pytest.approx(5 / 7) and r["IoU_per_label"] == {1: 0.5, 2: 0.0}
    assert label_agreement(seed, provided, label=2)["IoU"] == 0.0
    with pytest.raises(ValueError, match="different models"):
        label_agreement(seed[:-1], provided)


def test_seed2gs_eval_end_to_end(tmp_path):
    from seed2gs_eval import evaluate
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    a, b = _box(40, 60, 5, 20, 5, 25), _box(40, 60, 22, 38, 30, 55)
    for v in ("test_0", "test_1"):
        gt = np.stack([~(a | b), a, b]).astype(np.uint8)  # (C, H, W) uint8 like the LERF GT
        gt = np.stack([np.kron(c, np.ones((2, 2), np.uint8)) for c in gt])  # GT at 2x the camera resolution
        np.save(gt_dir / f"{v}.npy", gt)
    provided = np.array([0, 0, 1, 1, 2, 2, -1])
    s_b = _seed_dir(tmp_path / "seed2gs", "duck", {"test_0": b, "test_1": b, "frame_1": a}, "duck",
                    labels=[0, 0, 0, 0, 1, 1, 1])
    s_a = _seed_dir(tmp_path / "seed2gs", "apple", {"test_0": a}, "apple", labels=[0, 0, 1, 1, 0, 0, 0])
    results, summary = evaluate([s_b, s_a], str(gt_dir), labels=None)
    duck, apple = results
    assert duck["gt_channel"] == 2 and apple["gt_channel"] == 1
    assert duck["n_views"] == 2 and duck["IoU"] == pytest.approx(1.0) and apple["IoU"] == pytest.approx(1.0)
    assert apple["missing_views"] == ["test_1"] and apple["n_views"] == 1
    assert summary["mIoU"] == pytest.approx(1.0) and summary["n_scored"] == 2
    assert os.path.exists(os.path.join(s_b, "seed2gs_eval.json"))
    # forced wrong channel, provided labels
    torch.save(torch.tensor(provided), tmp_path / "labels.pt")
    results, summary = evaluate([s_b, s_a], str(gt_dir), labels=str(tmp_path / "labels.pt"), gt_channels=[1, 1])
    assert results[0]["IoU"] == 0.0 and results[1]["IoU"] == pytest.approx(1.0)
    assert results[0]["gaussians"]["provided_label"] == 2 and results[0]["gaussians"]["IoU"] == pytest.approx(1.0)
    assert results[1]["gaussians"]["provided_label"] == 1 and summary["gaussian_mIoU"] == pytest.approx(1.0)


def test_seed2gs_eval_failed_prompts(tmp_path):
    import json
    from seed2gs_eval import evaluate
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    a, b = _box(40, 60, 5, 20, 5, 25), _box(40, 60, 22, 38, 30, 55)
    for v in ("test_0", "test_1"):
        np.save(gt_dir / f"{v}.npy", np.stack([~(a | b), a, b]).astype(np.uint8))
    torch.save(torch.tensor([0, 0, 1, 1, 2, 2, -1]), tmp_path / "labels.pt")
    root = tmp_path / "seed2gs"
    ok = _seed_dir(root, "duck", {"test_0": b, "test_1": b}, "duck", labels=[0, 0, 0, 0, 1, 1, 1])
    # failed prompt: stale perfect masks from an earlier run, no labels.pt, failure record
    bad = _seed_dir(root, "apple", {"test_0": a, "test_1": a}, "apple")
    os.remove(os.path.join(bad, "seed2gs.json"))
    with open(os.path.join(bad, "seed2gs_failed.json"), "w") as f:
        json.dump(dict(prompt="apple", failed="SAM3 found no 'apple'"), f)
    never = str(root / "chair")  # never ran at all
    results, summary = evaluate([ok, bad, never], str(gt_dir), labels=str(tmp_path / "labels.pt"))
    duck, apple, chair = results
    assert "failed" not in duck and duck["IoU"] == pytest.approx(1.0) and duck["gaussians"]["IoU"] == pytest.approx(1.0)
    assert apple["prompt"] == "apple" and "SAM3 found no" in apple["failed"] and apple["n_views"] == 2
    assert apple["IoU"] == 0.0 and "error" in apple["gaussians"]  # stale masks ignored, no labels.pt read
    assert "no labels.pt" in chair["failed"] and os.path.exists(os.path.join(never, "seed2gs_eval.json"))
    assert summary["n_failed"] == 2 and summary["mIoU_found"] == pytest.approx(1.0)
    assert summary["mIoU"] == pytest.approx(1.0 / summary["n_scored"])  # failed prompts count as IoU 0
    assert summary["gaussian_mIoU"] == pytest.approx(1.0)
    # old-format failure record in seed2gs.json is recognised too
    with open(os.path.join(bad, "seed2gs.json"), "w") as f:
        json.dump(dict(prompt="apple", failed="old"), f)
    torch.save(torch.zeros(7, dtype=torch.int32), os.path.join(bad, "labels.pt"))
    os.remove(os.path.join(bad, "seed2gs_failed.json"))
    assert evaluate([bad], str(gt_dir))[0][0]["failed"] == "old"


def test_launch_depends_on_other_mask_source_job(tmp_path):
    import subprocess
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "slurm", "launch_one_per_dataset.sh")
    text = open(script).read()
    start = text.index("SUFFIX=; OTHER=")
    fn = text[start:text.index("\n}\n", start) + 3]
    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "squeue").write_text('#!/bin/sh\n[ "$5" = gw_x ] && printf "11\\n12\\n"\nexit 0\n')
    (stubs / "sbatch").write_text('#!/bin/sh\necho "SBATCH $*"\n')
    for f in stubs.iterdir():
        f.chmod(0o755)
    env = dict(os.environ, PATH=f"{stubs}:{os.environ['PATH']}", USER="u")
    run = lambda src: subprocess.run(["bash", "-c", "set -euo pipefail\n" + fn + "submit gw_x p a b c d"],
                                     env=dict(env, MASK_SOURCE=src), capture_output=True, text=True, check=True).stdout
    out = run("seed2gs")  # the provided job gw_x is queued: wait for it
    assert "SBATCH -J gw_x_seed2gs --dependency=afterany:11:12 slurm/gw_scene.sh a b c d" in out
    out = run("provided")  # no gw_x_seed2gs job queued: no dependency
    assert "SBATCH -J gw_x slurm/gw_scene.sh a b c d" in out
