"""export_3dgs_ply.py --check: the gsplat-vs-native comparison and its two-process IO (gsplat_check.py).

    python -m pytest -q gaussian_wrapping/tests/test_export_check.py      (gaussian_wrapping env, after
    "module load profile/deeplrn cuda/12.2 gcc/12.2.0"; the CPU tests need no GPU nor gsplat, the *_cuda test needs both)

The worker subprocess runs with --dry_run here (the dump compared with itself), which covers the dump layout, the
manifest, the PLY loading through artifixer_gs.render.read_ply and the report, but not gsplat itself.
"""
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

GW = Path(__file__).resolve().parents[1]
ROOT = GW.parent
sys.path.insert(0, str(GW))
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "submodules"))

import export_3dgs_ply as export  # noqa: E402
import gsplat_check as check  # noqa: E402
from utils.graphics_utils import getProjectionMatrix, getWorld2View2  # noqa: E402

ARTIFIXER = (ROOT / "submodules" / "artifixer_gs" / "artifixer_gs" / "render.py").exists()
needs_artifixer = pytest.mark.skipif(not ARTIFIXER, reason="submodules/artifixer_gs is not checked out")


def rotation(rx, ry, rz):
    cx, cy, cz, sx, sy, sz = math.cos(rx), math.cos(ry), math.cos(rz), math.sin(rx), math.sin(ry), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def fake_camera(W=64, H=48, fovx=1.1, fovy=0.9, name="img"):
    """GW Camera fields on CPU, built as scene/cameras.py does (R is camera-to-world, T world-to-camera)."""
    R, T = rotation(0.1, -0.2, 0.05), np.array([0.3, -0.1, 2.0])
    world_view = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
    proj = getProjectionMatrix(znear=0.01, zfar=100., fovX=fovx, fovY=fovy).transpose(0, 1)
    return SimpleNamespace(image_width=W, image_height=H, FoVx=fovx, FoVy=fovy, image_name=name,
                           world_view_transform=world_view, full_proj_transform=world_view @ proj)


def images(H=6, W=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((H, W, 3)).astype(np.float32), rng.random((H, W)).astype(np.float32), rng.random((H, W, 3)).astype(np.float32)


def write_ply(path, n=10, sh_degree=1, seed=0):
    rng = np.random.default_rng(seed)
    k = (sh_degree + 1) ** 2
    export.write_3dgs_ply(str(path), {
        "xyz": rng.normal(size=(n, 3)).astype(np.float32),
        "f_dc": rng.normal(size=(n, 3)).astype(np.float32),
        "f_rest": rng.normal(size=(n, 3 * (k - 1))).astype(np.float32),
        "opacity": rng.normal(size=(n, 1)).astype(np.float32),
        "scaling": rng.normal(size=(n, 3)).astype(np.float32) - 3.,
        "rotation": rng.normal(size=(n, 4)).astype(np.float32),
    })


# ---------------------------------------------------------------- settings, IO
def test_raster_settings():
    assert check.raster_settings(0.) == {"eps2d": 0., "rasterize_mode": "classic", "near_plane": 0.2}
    assert check.raster_settings(0.1) == {"eps2d": 0.1, "rasterize_mode": "antialiased", "near_plane": 0.2}


def test_dump_view_round_trip(tmp_path):
    rgb, alpha, gt = images()
    rgb[0, 0] = [1.5, -0.2, 0.5]  # clamped on write
    file = check.dump_view(str(tmp_path), "train_0000", rgb, alpha, gt)
    assert file == "train_0000.npz"
    view = check.load_view(str(tmp_path), file)
    assert view["rgb"].dtype == np.float32 and view["rgb"].shape == rgb.shape and view["alpha"].shape == alpha.shape
    assert np.abs(view["rgb"] - np.clip(rgb, 0., 1.)).max() <= 0.5 / 65535 + 1e-7
    assert np.abs(view["alpha"] - alpha).max() <= 0.5 / 65535 + 1e-7
    assert np.abs(view["gt"] - gt).max() <= 0.5 / 255 + 1e-7
    check.dump_view(str(tmp_path), "test_0000", rgb, alpha)
    assert check.load_view(str(tmp_path), "test_0000.npz")["gt"] is None
    with pytest.raises(AssertionError):
        check.dump_view(str(tmp_path), "bad", rgb, alpha[:-1])


# ---------------------------------------------------------------- comparison
def test_compare_view_known_values():
    rgb, alpha, gt = images()
    row = check.compare_view(rgb, alpha, rgb.copy(), alpha.copy(), gt)
    assert row["psnr"] == pytest.approx(120.) and row["alpha_mae"] == 0. and row["gt_drop"] == pytest.approx(0.)
    native = np.full((4, 4, 3), 0.5, np.float32)
    row = check.compare_view(native, np.ones((4, 4), np.float32), native + 0.1, np.full((4, 4), 0.75, np.float32),
                             gt=np.full((4, 4, 3), 0.5, np.float32))
    assert row["psnr"] == pytest.approx(20., abs=1e-4)
    assert row["alpha_mae"] == pytest.approx(0.25)
    assert row["psnr_gt_native"] == pytest.approx(120.) and row["psnr_gt_gsplat"] == pytest.approx(20., abs=1e-4)
    assert row["gt_drop"] == pytest.approx(100., abs=1e-4)
    # out-of-range values are clamped before comparing
    row = check.compare_view(np.full((2, 2, 3), 1.2, np.float32), np.zeros((2, 2), np.float32),
                             np.ones((2, 2, 3), np.float32), np.zeros((2, 2), np.float32))
    assert row["psnr"] == pytest.approx(120.) and "gt_drop" not in row


def test_summarize_means_splits_worst_pass():
    rows = [
        {"name": "train_0000", "split": "train", "psnr": 40., "alpha_mae": 0.01, "psnr_gt_native": 30., "psnr_gt_gsplat": 29.8, "gt_drop": 0.2},
        {"name": "train_0001", "split": "train", "psnr": 30., "alpha_mae": 0.03, "psnr_gt_native": 28., "psnr_gt_gsplat": 27.6, "gt_drop": 0.4},
        {"name": "test_0000", "split": "test", "psnr": 50., "alpha_mae": 0.02, "psnr_gt_native": 25., "psnr_gt_gsplat": 25., "gt_drop": 0.},
    ]
    report = check.summarize({"ply": "x.ply", "eps2d": 0., "views": ["dropped"]}, rows, max_drop=0.5)
    assert report["ply"] == "x.ply" and report["views"] == rows and report["n_views"] == 3
    assert report["psnr"] == pytest.approx(40.) and report["alpha_mae"] == pytest.approx(0.02)
    assert report["gt_drop"] == pytest.approx(0.2) and report["pass"] is True and report["max_drop"] == 0.5
    assert list(report["splits"]) == ["train", "test"]
    assert report["splits"]["train"]["n_views"] == 2 and report["splits"]["train"]["psnr"] == pytest.approx(35.)
    assert report["splits"]["test"]["gt_drop"] == pytest.approx(0.)
    assert report["worst"]["name"] == "train_0001" and report["worst"]["psnr"] == 30.
    assert check.summarize({}, rows, max_drop=0.1)["pass"] is False
    no_gt = [{k: v for k, v in row.items() if k in ("name", "split", "psnr", "alpha_mae")} for row in rows]
    report = check.summarize({}, no_gt)
    assert report["pass"] is None and "gt_drop" not in report
    assert "PASS" not in check.format_report({**report, "backend": "x"})
    with pytest.raises(AssertionError):
        check.summarize({}, [])


# ---------------------------------------------------------------- cameras
def test_camera_matrices_match_gw_projection():
    """K @ viewmat projects like GW's full_proj_transform + ndc2Pix, shifted by gsplat's +0.5 pixel centre."""
    camera = fake_camera()
    viewmat, K, W, H = export.camera_matrices(camera)
    assert (W, H) == (64, 48) and viewmat.shape == (4, 4) and K.shape == (3, 3)
    assert np.allclose(viewmat[3], [0, 0, 0, 1])
    rng = np.random.default_rng(0)
    points = rng.normal(size=(20, 3)) * 0.3
    homogeneous = np.concatenate([points, np.ones((20, 1))], axis=1)
    # GW (row vectors): p_hom = [x, 1] @ full_proj, ndc = p_hom / w, pixel = ((ndc + 1) * S - 1) / 2
    p_hom = homogeneous @ camera.full_proj_transform.double().numpy()
    ndc = p_hom[:, :2] / p_hom[:, 3:]
    gw_pixel = ((ndc + 1.) * np.array([W, H]) - 1.) * 0.5
    # gsplat (OpenCV): camera = viewmat @ x, pixel = K @ camera / z, centres at i + 0.5
    cam = (viewmat @ homogeneous.T).T[:, :3]
    assert (cam[:, 2] > 0.2).all()
    uv = (K @ cam.T).T
    gs_pixel = uv[:, :2] / uv[:, 2:]
    assert np.allclose(gs_pixel, gw_pixel + 0.5, atol=1e-4)
    # the view-space depth GW culls on (p_view.z) is the OpenCV z
    p_view = homogeneous @ camera.world_view_transform.double().numpy()
    assert np.allclose(p_view[:, 2], cam[:, 2])


def test_dump_native_views_manifest(tmp_path):
    cameras = [fake_camera(W=5, H=6, name=f"img{i}") for i in range(3)]
    views = [(f"{split}_{i:04d}", split, camera, *images(seed=i))
             for i, (split, camera) in enumerate(zip(("train", "train", "test"), cameras))]
    meta = {"ply": "x.ply", "iteration": 7, "background": [0., 0., 0.], **check.raster_settings(0.)}
    dump = tmp_path / "dump"
    assert export.dump_native_views(iter(views), str(dump), meta) == 3
    manifest = check.read_manifest(str(dump))
    assert manifest["iteration"] == 7 and manifest["eps2d"] == 0. and manifest["rasterize_mode"] == "classic"
    assert [v["name"] for v in manifest["views"]] == ["train_0000", "train_0001", "test_0002"]
    assert [v["image_name"] for v in manifest["views"]] == ["img0", "img1", "img2"]
    entry = manifest["views"][2]
    viewmat, K, W, H = export.camera_matrices(cameras[2])
    assert (entry["width"], entry["height"]) == (W, H) == (5, 6)
    assert np.allclose(entry["viewmat"], viewmat) and np.allclose(entry["K"], K)
    loaded = check.load_view(str(dump), entry["file"])
    assert np.allclose(loaded["rgb"], views[2][3], atol=1e-4) and loaded["gt"] is not None


# ---------------------------------------------------------------- PLY + worker subprocess
@needs_artifixer
def test_load_ply_for_gsplat_matches_export_loader(tmp_path):
    ply = tmp_path / "point_cloud.ply"
    write_ply(ply)
    theirs = check.load_ply_for_gsplat(str(ply), device="cpu")
    ours = export.load_3dgs_ply_for_gsplat(str(ply), device="cpu")
    assert theirs["sh_degree"] == ours["sh_degree"] == 1
    for key_theirs, key_ours in (("means", "means"), ("quats", "quats"), ("scales", "scales"), ("opacities", "opacities"), ("sh", "colors")):
        assert theirs[key_theirs].dtype == torch.float32, key_theirs
        assert torch.allclose(theirs[key_theirs], ours[key_ours], rtol=1e-5, atol=1e-6), key_theirs


@needs_artifixer
def test_worker_subprocess_dry_run(tmp_path):
    """export_3dgs_ply.run_gsplat_worker -> gsplat_check.py in a fresh interpreter (PYTHONPATH cleared), --dry_run."""
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    ply = export_dir / "point_cloud.ply"
    write_ply(ply, n=7)
    views = [(f"train_{i:04d}", "train", fake_camera(W=5, H=6, name=f"img{i}"), *images(seed=i)) for i in range(2)]
    meta = {"ply": str(ply), "iteration": 3, "background": [1., 1., 1.], **check.raster_settings(0.)}
    dump = export_dir / "gsplat_check_views"
    export.dump_native_views(iter(views), str(dump), meta)
    out = export_dir / check.REPORT
    report = export.run_gsplat_worker(sys.executable, str(dump), str(out), 0.5, extra_args=["--dry_run"])
    assert json.loads(out.read_text()) == report
    assert report["backend"] == "dry_run" and report["n_views"] == 2 and report["pass"] is True
    assert report["psnr"] == pytest.approx(120.) and report["alpha_mae"] == 0. and report["gt_drop"] == pytest.approx(0.)
    assert report["ply"] == str(ply) and report["iteration"] == 3 and report["background"] == [1., 1., 1.]
    assert [row["image_name"] for row in report["views"]] == ["img0", "img1"]
    # default --out: <dump>/../gsplat_check.json
    os.remove(out)
    check.check_dump(str(dump), dry_run=True)
    assert out.exists()


def write_dump(dump, n=2, ply="x.ply"):
    views = [(f"train_{i:04d}", "train", fake_camera(W=5, H=6, name=f"img{i}"), *images(seed=i)) for i in range(n)]
    meta = {"ply": str(ply), "iteration": 3, "background": [0., 0., 0.], **check.raster_settings(0.)}
    return export.dump_native_views(iter(views), str(dump), meta)


def test_clear_dump_only_clears_a_dump(tmp_path):
    export.clear_dump(str(tmp_path / "missing"))  # new dir: nothing to do
    dump = tmp_path / "dump"
    write_dump(dump)
    export.clear_dump(str(dump))
    assert dump.is_dir() and not list(dump.iterdir())
    # an export dir (or any dir with other files) is refused and left untouched
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    write_ply(export_dir / "point_cloud.ply")
    write_dump(export_dir)
    before = sorted(p.name for p in export_dir.iterdir())
    with pytest.raises(ValueError, match="not a gsplat_check dump"):
        export.clear_dump(str(export_dir))
    assert sorted(p.name for p in export_dir.iterdir()) == before
    (dump / "sub").mkdir()  # a sub-directory is not a dump file either
    with pytest.raises(ValueError):
        export.clear_dump(str(dump))
    with pytest.raises(ValueError, match="not a directory"):
        export.clear_dump(str(export_dir / "point_cloud.ply"))


def test_remove_dump_keeps_other_files(tmp_path):
    dump = tmp_path / "dump"
    write_dump(dump)
    (dump / "notes.txt").write_text("mine")
    export.remove_dump(str(dump))
    assert [p.name for p in dump.iterdir()] == ["notes.txt"]
    (dump / "notes.txt").unlink()
    write_dump(dump)
    export.remove_dump(str(dump))
    assert not dump.exists()  # emptied: removed
    export.remove_dump(str(dump))  # idempotent


def fake_check_env(monkeypatch, tmp_path, fail_on=None):
    """check_export without a scene or GPU: fake scene module, native_views yields 2 random views, no gsplat here."""
    monkeypatch.setitem(sys.modules, "scene", SimpleNamespace(Scene=lambda *a, **k: object(),
                                                             GaussianModel=lambda *a, **k: object()))
    monkeypatch.setattr(check, "gsplat_available", lambda: False)

    def views(*args, **kwargs):
        for i in range(2):
            if fail_on == i:
                raise RuntimeError("CUDA out of memory (fake)")
            yield f"train_{i:04d}", "train", fake_camera(W=5, H=6, name=f"img{i}"), *images(seed=i)
    monkeypatch.setattr(export, "native_views", views)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    write_ply(export_dir / "point_cloud.ply")
    dataset = SimpleNamespace(sh_degree=1, white_background=False, kernel_size=0.)
    return dataset, export_dir


@pytest.mark.parametrize("gs_python", ["/bin/false", "/nonexistent/python"])
def test_check_export_worker_failure_keeps_export(tmp_path, monkeypatch, gs_python):
    """A failing worker (non-zero exit / missing interpreter) is reported, not raised; dump and stale report removed."""
    dataset, export_dir = fake_check_env(monkeypatch, tmp_path)
    (export_dir / check.REPORT).write_text("{}")  # stale report of an earlier run
    result = export.check_export(dataset, None, 3, str(export_dir / "point_cloud.ply"), ours_ply=False, gs_python=gs_python)
    assert set(result) == {"error", "backend", "gs_python"} and result["backend"] == "subprocess"
    assert sorted(p.name for p in export_dir.iterdir()) == ["point_cloud.ply"]


def test_check_export_failure_keep_views(tmp_path, monkeypatch):
    dataset, export_dir = fake_check_env(monkeypatch, tmp_path)
    result = export.check_export(dataset, None, 3, str(export_dir / "point_cloud.ply"), ours_ply=False, gs_python="/bin/false",
                                 keep_views=True)
    assert "error" in result
    assert len(check.read_manifest(str(export_dir / "gsplat_check_views"))["views"]) == 2


def test_check_export_native_oom_cleans_partial_dump(tmp_path, monkeypatch):
    dataset, export_dir = fake_check_env(monkeypatch, tmp_path, fail_on=1)
    result = export.check_export(dataset, None, 3, str(export_dir / "point_cloud.ply"), ours_ply=False, gs_python="/bin/false")
    assert "out of memory" in result["error"]
    assert sorted(p.name for p in export_dir.iterdir()) == ["point_cloud.ply"]


def test_check_export_refuses_export_dir_as_check_dir(tmp_path, monkeypatch):
    dataset, export_dir = fake_check_env(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="not a gsplat_check dump"):
        export.check_export(dataset, None, 3, str(export_dir / "point_cloud.ply"), ours_ply=False, gs_python="/bin/false",
                            dump_dir=str(export_dir))
    assert (export_dir / "point_cloud.ply").is_file()


@needs_artifixer
def test_check_export_relative_check_dir_other_cwd(tmp_path, monkeypatch):
    """Relative --check_dir from a cwd that is not the repo root (the worker runs with cwd = repo root)."""
    dataset, export_dir = fake_check_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    worker = export.run_gsplat_worker
    monkeypatch.setattr(export, "run_gsplat_worker",
                        lambda *args, **kwargs: worker(*args, **kwargs, extra_args=["--dry_run"]))
    result = export.check_export(dataset, None, 3, "export/point_cloud.ply", ours_ply=False, gs_python=sys.executable,
                                 dump_dir="views")
    assert "error" not in result, result
    assert result["backend"] == "dry_run" and result["n_views"] == 2 and result["pass"] is True
    assert (export_dir / check.REPORT).is_file() and not (tmp_path / "views").exists()


# ---------------------------------------------------------------- GPU + gsplat
def test_check_export_ours_ply_without_gsplat(tmp_path, monkeypatch):
    """No gsplat anywhere: the exported PLY is still rendered with render_ours and compared (ours_ply_check.json)."""
    dataset, export_dir = fake_check_env(monkeypatch, tmp_path)
    loaded = []

    class FakeModel:
        def __init__(self, sh_degree):
            self.sh_degree = sh_degree
        def load_ply(self, path):
            loaded.append((path, self.sh_degree))   # the PLY's model: SH degree from the file (None)
    monkeypatch.setitem(sys.modules, "scene", SimpleNamespace(Scene=lambda *a, **k: object(), GaussianModel=FakeModel))
    calls = []

    def fake_render(camera, model, pipe, bg, kernel_size=0.):
        calls.append((camera.image_name, kernel_size))
        rgb, alpha, _ = images(seed=int(camera.image_name[-1]))     # the native view of that camera: identical
        return {"render": torch.from_numpy(rgb).permute(2, 0, 1), "alpha": torch.from_numpy(alpha)[None]}
    monkeypatch.setitem(sys.modules, "gaussian_renderer.ours", SimpleNamespace(render_ours=fake_render))
    monkeypatch.delenv("GS_RENDER_PY", raising=False)
    ply = str(export_dir / "point_cloud.ply")
    result = export.check_export(dataset, None, 3, ply, gs_python=None)
    assert loaded == [(ply, None)] and calls == [("img0", 0.), ("img1", 0.)]
    report = json.loads((export_dir / export.OURS_PLY_REPORT).read_text())
    assert report["backend"] == "ours_ply" and report["n_views"] == 2 and report["alpha_mae"] == 0.
    assert report["gt_drop"] == pytest.approx(0.) and report["pass"] is True
    assert result["ours_ply"]["n_views"] == 2 and not (export_dir / check.REPORT).exists()


@pytest.mark.skipif(not (torch.cuda.is_available() and check.gsplat_available()), reason="needs CUDA and gsplat")
def test_gsplat_render_single_gaussian_cuda(tmp_path):
    """One opaque Gaussian on the optical axis covers the image centre; the background shows in the corners."""
    device = "cuda"
    gaussians = {
        "means": torch.tensor([[0., 0., 2.]], device=device),
        "quats": torch.tensor([[1., 0., 0., 0.]], device=device),
        "scales": torch.full((1, 3), 0.2, device=device),
        "opacities": torch.tensor([0.98], device=device),
        "sh": torch.zeros(1, 1, 3, device=device),
        "sh_degree": 0,
    }
    K = np.array([[40., 0., 16.], [0., 40., 16.], [0., 0., 1.]])
    rgb, alpha = check.gsplat_render(gaussians, np.eye(4), K, 32, 32, [1., 1., 1.], 0., "classic")
    assert rgb.shape == (32, 32, 3) and alpha.shape == (32, 32)
    assert alpha[16, 16] > 0.9 and alpha[0, 0] < 1e-3
    assert np.allclose(rgb[0, 0], 1., atol=1e-3)  # white background
    assert np.allclose(rgb[16, 16], 0.5 * alpha[16, 16] + (1. - alpha[16, 16]), atol=1e-2)  # SH 0 -> 0.5
