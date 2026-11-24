"""CPU tests of gaussian_wrapping/beags_artifixer.py (the GW -> ArtiFixer-GS glue).

    python -m pytest -q gaussian_wrapping/tests/test_beags_artifixer.py      (gaussian_wrapping env: torch, numpy,
    PIL, plyfile, pytest; no GPU)

A synthetic GW model (PLY with filter_3D and extra fields, labels.pt, cfg_args) on a synthetic COLMAP dir; the real
export_3dgs_ply.py runs (CPU) and the env.sh it feeds to artifixer_gs/scripts/run.sh is read back with bash.
Confidence maps (beags/confidence_map.py): CUDA is reported absent unless a test says otherwise, and the GPU render
subprocess is replaced by the real map writer on fake cameras and a fake rasterizer.
"""
import importlib.util
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

GW = Path(__file__).resolve().parents[1]
ROOT = GW.parent
sys.path.insert(0, str(GW))
import beags_artifixer as bax  # noqa: E402
from beags import confidence_map as cmap  # noqa: E402
from beags import unseen_map as umap  # noqa: E402

IT = "iteration_33000"


# ---------------------------------------------------------------- synthetic model / COLMAP / masks
GW_FIELDS = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(9)]
             + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)] + ["filter_3D", "confidence"])


def write_gw_ply(path, n, confidence=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.random.default_rng(0).normal(size=(n, len(GW_FIELDS))).astype(np.float32)
    values[:, GW_FIELDS.index("filter_3D")] = np.abs(values[:, GW_FIELDS.index("filter_3D")]) * 0.1
    values[:, [GW_FIELDS.index(f"scale_{i}") for i in range(3)]] -= 3.0
    fields = [k for k in GW_FIELDS if confidence or k != "confidence"]
    vertex = np.empty(n, dtype=[(k, "f4") for k in fields])
    for k in fields:
        vertex[k] = values[:, GW_FIELDS.index(k)]
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))
    return vertex


def write_images_bin(path, centres, rotation=np.eye(3)):
    """COLMAP images.bin with world-to-camera R = rotation.T, camera centres `centres` {name: (3,)}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    R = rotation.T
    w, x, y, z = _rotmat2qvec(R)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(centres)))
        for i, (name, c) in enumerate(centres.items(), 1):
            t = -R @ np.asarray(c, dtype=np.float64)
            f.write(struct.pack("<idddddddi", i, w, x, y, z, *t, 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))


def _rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0], [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                  [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0], [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return q if q[0] >= 0 else -q


def rotation(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


STEMS = [f"img_{i:03d}" for i in range(8)]
CENTRES = {f"{s}.png": np.array([np.cos(i), 0.3 * i, np.sin(i) + 0.1 * i * i]) for i, s in enumerate(STEMS)}


def make_scene(tmp, n=500, label=2, images="images_2", camera=None):
    """GW BEA-GS-like model output/gw_beags/SCN/beags (cfg_args, iteration_33000 PLY + labels) on a COLMAP dir;
    camera = "<MODEL> <params>" writes sparse/0/cameras.txt (default: no cameras file)."""
    colmap = tmp / "data" / "SCN"
    (colmap / images).mkdir(parents=True)
    for s in STEMS:
        Image.new("RGB", (8, 6)).save(colmap / images / f"{s}.png")
    write_images_bin(colmap / "sparse" / "0" / "images.bin", CENTRES)
    if camera:
        (colmap / "sparse" / "0" / "cameras.txt").write_text(f"# test camera\n1 {camera.split()[0]} 8 6 {' '.join(camera.split()[1:])}\n")
    model = tmp / "output" / "gw_beags" / "SCN" / "beags"
    write_gw_ply(model / "point_cloud" / "iteration_30000" / "point_cloud.ply", n + 7)
    vertex = write_gw_ply(model / "point_cloud" / IT / "point_cloud.ply", n)
    labels = torch.zeros(n, dtype=torch.int32)
    labels[50:120] = label
    labels[200:210] = 1
    labels[300:305] = -1
    torch.save(labels, model / "point_cloud" / IT / "labels.pt")
    (model / "cfg_args").write_text(f"Namespace(sh_degree=3, source_path='{colmap}', model_path='{model}', "
                                    f"images='{images}', resolution=-1, white_background=False, eval=True)")
    return model, colmap, labels, vertex


def chw_masks(folder, label, present=STEMS[:5]):
    folder.mkdir(parents=True)
    for s in STEMS[:6]:
        m = np.zeros((label + 1, 6, 8), np.float16)
        m[0] = 1
        if s in present:
            m[label, 2:4, 3:6] = 1
            m[0, 2:4, 3:6] = 0
        np.save(folder / f"{s}.npy", m)
    Image.new("L", (8, 6), 255).save(folder / f"{STEMS[0]}_visualized.png")   # not named after a photo: ignored
    np.save(folder / "unrelated.npy", np.ones((3, 6, 8)))
    return folder


def read_env(path):
    out = subprocess.run(["bash", "-c", f"set -a; source '{path}'; env -0"], capture_output=True, check=True,
                         env={"PATH": os.environ["PATH"]}).stdout
    return dict(kv.split("=", 1) for kv in out.decode().split("\0") if "=" in kv)


@pytest.fixture(scope="module", autouse=True)
def no_cuda():
    """CPU behaviour on any node: --confidence_maps auto skips, and nothing launches the GPU renderer."""
    def no_render(*a):
        raise AssertionError("the confidence render subprocess must not run in these tests")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bax, "cuda_available", lambda: False)
        mp.setattr(bax, "render_confidence_maps", no_render)
        yield


@pytest.fixture(scope="module")
def removal_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("removal")
    model, colmap, labels, vertex = make_scene(tmp)
    masks = chw_masks(tmp / "masks", label=2)
    out = tmp / "artifixer" / "SCN" / "normal"
    bax.main(["-m", str(model), "--label", "2", "--variant", "normal", "--masks", str(masks), "--out", str(out)])
    return {"model": model, "colmap": colmap, "labels": labels, "vertex": vertex, "masks": masks, "out": out}


# ---------------------------------------------------------------- beags_artifixer.py
def test_removal_env(removal_run):
    r = removal_run
    out, model = r["out"], r["model"]
    obj = np.load(out / "object_mask.npy")
    assert obj.dtype == bool and obj.shape == (500,) and np.array_equal(obj, r["labels"].numpy() == 2)
    env = read_env(out / "env.sh")
    assert env["PLY"] == str(out / "export" / "point_cloud.ply")
    assert env["COLMAP"] == str(r["colmap"]) and env["IMAGES"] == "images_2" and env["OUT"] == str(out / "SCN")
    assert env["OBJECT_MASK"] == str(out / "object_mask.npy") and env["MASK_DIR"] == str(r["masks"])
    assert env["MASK_LABEL"] == "2"                     # (C, H, W): channel k is object k
    assert env["MODE"] == "removal" and env["REMOVAL_VARIANT"] == "normal" and env["HOLE_MODE"] == "dilate"
    assert float(env["EPS2D"]) == 0.0 and "TRAJ_MODE" not in env and "HOLE_MESH" not in env
    assert "METRIC_SCALE" not in env and "CAPTION_H5" not in env
    assert env["FASTGS_REPO"] == str(ROOT) and env["FASTGS_ARGS"] == ""
    assert "CONFIDENCE_DIR" not in env and not (out / "confidence").exists()   # no CUDA: auto skips
    assert env["TRAIN_CMD"] == f"{sys.executable} gaussian_wrapping/train.py --rasterizer ours"
    assert env["AF3D_EXPORT"] == f"{sys.executable} gaussian_wrapping/export_3dgs_ply.py"
    info = json.loads((out / "inputs.json").read_text())
    assert info["confidence_maps"]["mode"] == "auto"
    assert info["confidence_maps"]["status"] == "skipped: no CUDA device (the maps are rendered on the GPU)"
    assert info["num_gaussians"] == 500 and info["object_gaussians"] == 70 and info["variant"] == "normal"
    assert info["iteration"] == 33000 and info["gw_ply"] == str(model / "point_cloud" / IT / "point_cloud.ply")
    assert info["mask_stats"] == {"masks": 6, "photos": 8, "object_views": 5, "layout": "chw",
                                  "object_fraction_mean": pytest.approx(5 * 6 / 48 / 6)}


def test_renderer_env(removal_run, tmp_path):
    """Default: every ArtiFixer-GS render through GW's render_ours (RENDER_CMD, run in this repo); gsplat on request."""
    env = read_env(removal_run["out"] / "env.sh")
    assert env["RENDER_CMD"] == f"{sys.executable} gaussian_wrapping/render_ply_frames.py --kernel_size 0.0"
    assert env["RENDER_CWD"] == str(ROOT) and (ROOT / "gaussian_wrapping" / "render_ply_frames.py").is_file()
    assert json.loads((removal_run["out"] / "inputs.json").read_text())["renderer"] == "gw"
    out = tmp_path / "gs"
    bax.main(["-m", str(removal_run["model"]), "--label", "2", "--variant", "normal", "--masks", str(removal_run["masks"]),
              "--out", str(out), "--renderer", "gsplat"])
    env = read_env(out / "env.sh")
    assert "RENDER_CMD" not in env and "RENDER_CWD" not in env


def test_fallback_masks_fill_missing_photos(tmp_path):
    """Consensus masks of some photos + fallback masks of all: <out>/masks links each photo once, primary first."""
    model, _, _, _ = make_scene(tmp_path)
    primary = chw_masks(tmp_path / "consensus", label=2)          # STEMS[:6]
    fallback = chw_masks(tmp_path / "sam2", label=2, present=STEMS[:2])
    for s in STEMS[6:]:
        np.save(fallback / f"{s}.npy", np.zeros((3, 6, 8), np.float16))
    out = tmp_path / "af"
    bax.main(["-m", str(model), "--label", "2", "--variant", "normal", "--masks", str(primary),
              "--fallback_masks", str(fallback), "--out", str(out)])
    merged = out / "masks"
    assert sorted(p.stem for p in merged.iterdir()) == STEMS
    assert all(os.path.realpath(merged / f"{s}.npy") == os.path.realpath(primary / f"{s}.npy") for s in STEMS[:6])
    assert all(os.path.realpath(merged / f"{s}.npy") == os.path.realpath(fallback / f"{s}.npy") for s in STEMS[6:])
    assert read_env(out / "env.sh")["MASK_DIR"] == str(merged)
    info = json.loads((out / "inputs.json").read_text())
    assert info["mask_merge"]["from_masks"] == 6 and info["mask_merge"]["from_fallback"] == 2
    assert info["mask_stats"]["masks"] == 8 and info["mask_stats"]["photos"] == 8


@pytest.mark.parametrize("camera, ignore", [("SIMPLE_RADIAL 7 4 3 0.02", True), ("PINHOLE 7 7 4 3", False)])
def test_ignore_distortion_from_camera_model(tmp_path, camera, ignore):
    """GW reads SIMPLE_RADIAL / RADIAL / OPENCV as pinhole: ArtiFixer-GS is told to drop the distortion too."""
    model, _, _, _ = make_scene(tmp_path, camera=camera)
    masks = chw_masks(tmp_path / "masks", label=2)
    out = tmp_path / "af"
    bax.main(["-m", str(model), "--label", "2", "--variant", "normal", "--masks", str(masks), "--out", str(out)])
    env = read_env(out / "env.sh")
    assert env.get("IGNORE_DISTORTION") == ("1" if ignore else None)
    assert json.loads((out / "inputs.json").read_text())["camera_models"] == [camera.split()[0]]


def test_exported_ply_is_baked_and_aligned(removal_run):
    """export_3dgs_ply.py output: standard attributes only, same order, 3D filter baked (contract [E])."""
    v = removal_run["vertex"]
    exported = PlyData.read(str(removal_run["out"] / "export" / "point_cloud.ply"))["vertex"]
    names = [p.name for p in exported.properties]
    assert names == [k for k in GW_FIELDS if k not in ("filter_3D", "confidence")]
    assert exported.count == bax.ply_vertex_count(str(removal_run["out"] / "export" / "point_cloud.ply")) == 500
    for k in ("x", "y", "z", "f_dc_0", "f_rest_8", "rot_3"):
        assert np.array_equal(exported[k], v[k])
    s2 = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float64)) ** 2
    f2 = v["filter_3D"].astype(np.float64)[:, None] ** 2
    scale = np.log(np.sqrt(s2 + f2))
    alpha = 1 / (1 + np.exp(-v["opacity"].astype(np.float64))) * np.sqrt(s2.prod(1) / (s2 + f2).prod(1))
    assert np.allclose(np.stack([exported[f"scale_{i}"] for i in range(3)], 1), scale, atol=1e-5)
    assert np.allclose(1 / (1 + np.exp(-exported["opacity"].astype(np.float64))), alpha, atol=1e-5)
    assert (np.asarray(exported["nx"]) == 0).all()


def test_export_reuse(removal_run, capsys):
    r = removal_run
    argv = ["-m", str(r["model"]), "--label", "2", "--variant", "normal", "--masks", str(r["masks"]), "--out", str(r["out"])]
    ply = r["out"] / "export" / "point_cloud.ply"
    before = ply.stat().st_mtime
    bax.main(argv)
    assert "is up to date" in capsys.readouterr().out and ply.stat().st_mtime == before
    src = r["model"] / "point_cloud" / IT / "point_cloud.ply"
    later = time.time() + 5
    os.utime(src, (later, later))                      # the GW PLY changed: exported again
    bax.main(argv)
    assert "is up to date" not in capsys.readouterr().out and ply.stat().st_mtime > before
    # Another iteration into the same out dir: re-exported, and the label count check catches the mismatch.
    with pytest.raises(AssertionError, match="labels.pt has 500 entries, the PLY 507"):
        torch.save(torch.full((500,), 2, dtype=torch.int32), r["model"] / "point_cloud" / "iteration_30000" / "labels.pt")
        bax.main(argv + ["--iteration", "30000"])
    (r["model"] / "point_cloud" / "iteration_30000" / "labels.pt").unlink()


def test_2d_masks_and_mask_label(tmp_path):
    model, _, _, _ = make_scene(tmp_path, label=1)
    binary = tmp_path / "binary"
    binary.mkdir()
    for i, s in enumerate(STEMS):
        m = np.zeros((6, 8), np.uint8)
        m[1:3, 1:3] = 255 if i % 2 else 0
        Image.fromarray(m).save(binary / f"{s}.png")
    out = tmp_path / "o1"
    bax.main(["-m", str(model), "--label", "1", "--variant", "bighull", "--masks", str(binary), "--out", str(out)])
    env = read_env(out / "env.sh")
    assert "MASK_LABEL" not in env and env["REMOVAL_VARIANT"] == "bighull"   # 2D map without --mask_label: > 0
    assert json.loads((out / "inputs.json").read_text())["mask_stats"]["object_views"] == 4
    ids = tmp_path / "ids"
    ids.mkdir()
    for s in STEMS:
        m = np.zeros((6, 8), np.int32)
        m[0, :2], m[5, :] = 3, 7
        np.save(ids / f"{s}.npy", m)
    out = tmp_path / "o2"
    bax.main(["-m", str(model), "--label", "1", "--variant", "normal", "--masks", str(ids), "--mask_label", "3",
              "--out", str(out)])
    assert read_env(out / "env.sh")["MASK_LABEL"] == "3"
    stats = json.loads((out / "inputs.json").read_text())["mask_stats"]
    assert stats["layout"] == "2d" and stats["object_fraction_mean"] == pytest.approx(2 / 48)


def test_mask_object_rule():
    chw = np.zeros((3, 2, 2), np.float16)
    chw[2, 0, 0], chw[2, 1, 1] = 0.9, 0.4
    assert bax.mask_object(chw, 2).tolist() == [[True, False], [False, False]]
    assert bax.mask_object(np.array([[0, 5], [2, 0]]), 1).tolist() == [[False, True], [True, False]]
    assert bax.mask_object(np.array([[0, 5], [2, 0]]), 1, mask_label=2).tolist() == [[False, False], [True, False]]
    with pytest.raises(AssertionError):
        bax.mask_object(chw, 3)


@pytest.mark.parametrize("variant", ["vanilla_object", "vanilla_vidsplat", "vanilla_travel"])
def test_vanilla_default_out_is_flat(tmp_path, monkeypatch, variant):
    model, _, labels, _ = make_scene(tmp_path)
    monkeypatch.setattr(bax, "ROOT_DIR", str(tmp_path / "repo"))
    bax.main(["-m", str(model), "--label", "1", "--variant", variant])
    out = tmp_path / "repo" / "output" / "artifixer" / "SCN" / variant   # <scene> of output/gw_beags/SCN/beags
    env = read_env(out / "env.sh")
    assert set(env) >= {"PLY", "COLMAP", "IMAGES", "OBJECT_MASK", "EPS2D", "TRAIN_CMD"} and "MASK_DIR" not in env
    assert env["TRAJ_MODE"] == variant[len("vanilla_"):] and "MODE" not in env and "HOLE_MODE" not in env
    assert env["OUT"] == str(out / "SCN") and env["FASTGS_REPO"] == str(tmp_path / "repo")
    assert np.load(out / "object_mask.npy").sum() == int((labels == 1).sum())
    assert not (model / "object_mask.npy").exists() and not (model / "export").exists()


def test_scene_name(tmp_path):
    assert bax.scene_name(str(tmp_path / "output" / "gw" / "bonsai" / "train")) == "bonsai"
    assert bax.scene_name(str(tmp_path / "output" / "gw_hollow" / "bonsai" / "hollow") + "/") == "bonsai"
    assert bax.scene_name(str(tmp_path / "output" / "beags" / "bonsai")) == "bonsai"


def test_hole_modes(tmp_path):
    model, _, _, _ = make_scene(tmp_path)
    masks = chw_masks(tmp_path / "masks", label=2)
    run = lambda *a: bax.main(["-m", str(model), "--label", "2", "--masks", str(masks), "--out", str(tmp_path / "h"), *a])
    with pytest.raises(AssertionError, match="no hole mesh .*hole_2.ply"):
        run("--variant", "normal", "--hole_mode", "mesh")
    run("--variant", "normal", "--hole_mode", "alpha")
    env = read_env(tmp_path / "h" / "env.sh")
    assert env["HOLE_MODE"] == "alpha" and "HOLE_MESH" not in env
    mesh = model / "mesh" / "hole_2.ply"
    mesh.parent.mkdir()
    mesh.write_bytes(b"ply\n")
    run("--variant", "normal", "--hole_mode", "mesh", "--eps2d", "0.1")
    env = read_env(tmp_path / "h" / "env.sh")
    assert env["HOLE_MODE"] == "mesh" and env["HOLE_MESH"] == str(mesh) and float(env["EPS2D"]) == 0.1
    if importlib.util.find_spec("nvdiffrast") is not None:
        assert env["MESH_PY"] == sys.executable
    run("--variant", "bighull")                         # a mesh that exists is passed on in every mode
    assert read_env(tmp_path / "h" / "env.sh")["HOLE_MESH"] == str(mesh)
    with pytest.raises(AssertionError, match="no hole mesh"):
        run("--variant", "normal", "--hole_mesh", str(tmp_path / "missing.ply"))
    run("--variant", "normal", "--train_cmd", "python gaussian_wrapping/train.py --rasterizer ours --hollow",
        "--train_args", "--lambda_lpips 0.2 --appearance bilagrid")
    env = read_env(tmp_path / "h" / "env.sh")
    assert env["TRAIN_CMD"].endswith("--hollow") and env["FASTGS_ARGS"] == "--lambda_lpips 0.2 --appearance bilagrid"


def test_errors(tmp_path):
    model, _, _, _ = make_scene(tmp_path)
    run = lambda *a: bax.main(["-m", str(model), "--out", str(tmp_path / "e"), *a])
    with pytest.raises(AssertionError, match="background"):
        run("--label", "0", "--variant", "vanilla_object")
    with pytest.raises(AssertionError, match="needs --masks"):
        run("--label", "2", "--variant", "normal")
    with pytest.raises(AssertionError, match=r"no Gaussian has label 5; per label: \{-1: 5, 0: 420, 1: 10, 2: 65\}"):
        torch.save(torch.cat([torch.full((5,), -1), torch.zeros(420), torch.ones(10), torch.full((65,), 2)]).int(),
                   model / "point_cloud" / IT / "labels.pt")
        run("--label", "5", "--variant", "vanilla_object")
    with pytest.raises(AssertionError, match="inside the model dir"):
        bax.main(["-m", str(model), "--label", "2", "--variant", "vanilla_object", "--out", str(model / "afx")])
    torch.save(torch.zeros(499, dtype=torch.int32), model / "point_cloud" / IT / "labels.pt")
    with pytest.raises(AssertionError, match="499 entries"):
        run("--label", "2", "--variant", "vanilla_object")
    masks = chw_masks(tmp_path / "m_absent", label=2, present=[])
    torch.save(torch.full((500,), 2, dtype=torch.int32), model / "point_cloud" / IT / "labels.pt")
    with pytest.raises(AssertionError, match="in none of the masks"):
        run("--label", "2", "--variant", "normal", "--masks", str(masks))
    with pytest.raises(SystemExit):
        run("--label", "2", "--variant", "vanilla")


# ---------------------------------------------------------------- metric scale / caption reuse
def test_frame_ratio():
    ours = {k[:-4]: c for k, c in CENTRES.items()}
    R, t = rotation([1, 2, 3], 40), np.array([5.0, -2, 1])
    similar = {k: 3.5 * R @ c + t for k, c in ours.items()}
    ratio, why, shared = bax.frame_ratio(ours, similar)
    assert ratio == pytest.approx(3.5) and why == "" and shared == 8
    skewed = {k: np.array([3.0, 1.0, 1.0]) * c for k, c in ours.items()}
    assert bax.frame_ratio(ours, skewed)[0] is None
    assert bax.frame_ratio(ours, {k: similar[k] for k in list(similar)[:2]})[1].startswith("2 shared photos")


def test_metric_scale_and_caption_from_artifixer_root(tmp_path):
    model, _, _, _ = make_scene(tmp_path)
    root = tmp_path / "af" / "output" / "bakerh_undis" / "SCN"
    R, t = rotation([0, 1, 1], -70), np.array([0.5, 4.0, -3.0])
    # ArtiFixer's prepared copy: same photos, its frame 4x ours (and rotated, shifted), one photo fewer.
    write_images_bin(root / "3dgrut_input" / "SCN" / "sparse" / "0" / "images.bin",
                     {k: 4.0 * R @ c + t for k, c in list(CENTRES.items())[1:]})
    (root / "metric_alignment").mkdir(parents=True)
    (root / "metric_alignment" / "scale_info.txt").write_text("Scale factor: 0.5\nother: 1\n")
    (root / "captions" / "SCN").mkdir(parents=True)
    (root / "captions" / "SCN" / "caption.h5").write_bytes(b"h5")
    (root / "split.json").write_text(json.dumps({"test": {"SCN": {"prompt_path": "captions/SCN/caption.h5"}}}))
    out = tmp_path / "o"
    argv = ["-m", str(model), "--label", "2", "--variant", "vanilla_object", "--artifixer_root", str(root), "--out", str(out)]
    bax.main(argv)
    env = read_env(out / "env.sh")
    assert float(env["METRIC_SCALE"]) == pytest.approx(2.0)   # 0.5 m per its unit, 4 of its units per ours
    assert env["CAPTION_H5"] == str((root / "captions" / "SCN" / "caption.h5").resolve())
    check = json.loads((out / "inputs.json").read_text())["metric_scale_check"]
    assert check["shared_photos"] == 7 and check["frame_ratio"] == pytest.approx(4.0)
    # A frame that is not a similarity of ours: the scale is not reused (MoGe runs).
    write_images_bin(root / "3dgrut_input" / "SCN" / "sparse" / "0" / "images.bin",
                     {k: np.array([1.0, 2.0, 1.0]) * c for k, c in CENTRES.items()})
    bax.main(argv)
    assert "METRIC_SCALE" not in read_env(out / "env.sh")
    assert json.loads((out / "inputs.json").read_text())["metric_scale_source"].startswith("none:")
    # An explicit scale wins.
    bax.main(argv + ["--metric_scale", "0.25"])
    assert read_env(out / "env.sh")["METRIC_SCALE"] == "0.25"


def test_env_matches_run_sh():
    """Every variable env.sh exports is one artifixer_gs/scripts/run.sh reads."""
    run_sh = (ROOT / "submodules" / "artifixer_gs" / "scripts" / "run.sh")
    if not run_sh.is_file():
        pytest.skip("submodules/artifixer_gs is not checked out")
    text = run_sh.read_text()
    for var in ("PLY", "COLMAP", "IMAGES", "OBJECT_MASK", "OUT", "MASK_DIR", "MASK_LABEL", "METRIC_SCALE", "MODE",
                "REMOVAL_VARIANT", "HOLE_MODE", "HOLE_MESH", "TRAJ_MODE", "EPS2D", "FASTGS_REPO", "TRAIN_CMD",
                "FASTGS_ARGS", "AF3D_EXPORT", "MESH_PY", "CONFIDENCE_DIR", "CONF_THRESH", "UNSEEN_DIR", "UNSEEN_THRESH"):
        assert "${" + var in text or "$" + var in text, var
    assert '--confidence_dir "$CONFIDENCE_DIR" --conf_threshold "$CONF_THRESH"' in text
    assert '--unseen_dir "$UNSEEN_DIR" --unseen_threshold "$UNSEEN_THRESH"' in text
    assert all(f"    {m})" in text or f"    {m}|" in text or f"|{m})" in text for m in bax.HOLE_MODES)
    assert "DROP_POINTS=${DROP_POINTS:-1}" in text and "EPS2D=${EPS2D:-0.3}" in text


# ---------------------------------------------------------------- confidence maps (beags/confidence_map.py)
class FakeCam:
    def __init__(self, name, alpha):
        self.image_name, self.alpha = name, torch.as_tensor(alpha, dtype=torch.float32)


class FakeGaussians:
    def __init__(self, log_conf):
        self._confidence = torch.as_tensor(log_conf, dtype=torch.float32).reshape(-1, 1)

    @property
    def get_confidence(self):
        return torch.exp(self._confidence)


def fake_render(cam, gaussians, pipe, bg, colors_precomp=None, require_depth=True):
    """render_ours' contract for a colors_precomp pass: every pixel sees the mean colour with weight alpha."""
    assert colors_precomp.shape == (gaussians._confidence.shape[0], 3) and not require_depth
    alpha = cam.alpha[None]
    return {"render": colors_precomp.mean(0)[:, None, None] * alpha, "alpha": alpha}


def fake_cams(alpha_of=lambda i: np.full((6, 8), 0.9)):
    return [FakeCam(s, alpha_of(i)) for i, s in enumerate(STEMS)]


def test_ply_has_confidence(tmp_path):
    write_gw_ply(tmp_path / "with.ply", 5)
    write_gw_ply(tmp_path / "without.ply", 5, confidence=False)
    assert cmap.ply_has_confidence(str(tmp_path / "with.ply"))
    assert not cmap.ply_has_confidence(str(tmp_path / "without.ply"))
    other = np.zeros(2, dtype=[("confidence", "f4")])          # a `confidence` of another element does not count
    vertex = np.zeros(2, dtype=[("x", "f4")])
    PlyData([PlyElement.describe(vertex, "vertex"), PlyElement.describe(other, "face_info")]).write(str(tmp_path / "o.ply"))
    assert not cmap.ply_has_confidence(str(tmp_path / "o.ply"))


def test_encode_decode():
    conf = np.array([[0.0, 0.5, 1.0], [3.0, np.nan, 0.25]])
    alpha = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.001]])
    png = cmap.encode(conf, alpha, min_alpha=0.01)
    assert png.dtype == np.uint8 and png.tolist() == [[0, 128, 255], [255, 255, 255]]   # c >= 1, NaN, empty: 255
    assert np.allclose(cmap.decode(png), png / 255.0)


def test_write_maps_normalizes_by_alpha(tmp_path):
    """c = R(exp(conf)) / alpha: the alpha-weighted mean confidence, whatever the coverage."""
    log_conf = np.log([0.2, 0.4])                               # mean c 0.3
    alpha = lambda i: np.where(np.arange(8)[None, :] < 4, 0.5, 0.005).repeat(6, 0)   # right half: nothing rendered
    stats = cmap.write_maps(fake_cams(alpha), FakeGaussians(log_conf), None, str(tmp_path), fake_render)
    assert sorted(stats) == STEMS and sorted(p.stem for p in tmp_path.glob("*.png")) == STEMS
    png = np.asarray(Image.open(tmp_path / f"{STEMS[3]}.png"))
    v = int(png[0, 0])
    assert png.shape == (6, 8) and abs(v - 0.3 * 255) <= 0.5 and (png[:, :4] == v).all() and (png[:, 4:] == 255).all()
    assert stats[STEMS[3]] == {"mean": pytest.approx(0.5 * v / 255 + 0.5, abs=1e-4), "low_fraction": 0.5}
    info = cmap.write_info(str(tmp_path), stats, source="x.ply")
    assert info["cameras"] == 8 and info["low_fraction_mean"] == 0.5 and info["encoding"] == cmap.ENCODING
    assert json.loads((tmp_path / "info.json").read_text())["views"][STEMS[0]]["low_fraction"] == 0.5


def gpu_free_renderer(calls, log_conf=(-1.0, -0.5)):
    """Stands in for the confidence_map.py subprocess: the real writer and info.json, fake cameras / rasterizer."""
    def run(model_path, iteration, out_dir, colmap, images):
        calls.append(out_dir)
        stats = cmap.write_maps(fake_cams(), FakeGaussians(log_conf), None, out_dir, fake_render)
        ply = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        cmap.write_info(out_dir, stats, source=ply, iteration=iteration, source_path=colmap, images=images)
    return run


def test_confidence_maps_rendered_and_reused(tmp_path, monkeypatch, capsys):
    model, colmap, _, _ = make_scene(tmp_path)
    masks = chw_masks(tmp_path / "masks", label=2)
    calls = []
    monkeypatch.setattr(bax, "cuda_available", lambda: True)
    monkeypatch.setattr(bax, "render_confidence_maps", gpu_free_renderer(calls))
    out = tmp_path / "afx"
    argv = ["-m", str(model), "--label", "2", "--variant", "normal", "--masks", str(masks), "--hole_mode", "alpha",
            "--out", str(out)]
    bax.main(argv)
    conf = out / "confidence"
    env = read_env(out / "env.sh")
    assert env["CONFIDENCE_DIR"] == str(conf) and calls == [str(conf) + ".tmp"] and not os.path.exists(calls[0])
    assert sorted(p.stem for p in conf.glob("*.png")) == STEMS
    c = (np.exp(-1.0) + np.exp(-0.5)) / 2
    v = np.asarray(Image.open(conf / f"{STEMS[0]}.png"))
    assert abs(int(v[0, 0]) - c * 255) <= 0.5 and (v == v[0, 0]).all()
    record = json.loads((out / "inputs.json").read_text())["confidence_maps"]
    assert record["status"] == "rendered" and record["dir"] == str(conf) and record["cameras"] == 8
    assert record["mean"] == pytest.approx(v[0, 0] / 255, abs=1e-4) and record["low_fraction_mean"] == 1.0   # c < 0.5
    assert json.loads((out / "inputs.json").read_text())["CONFIDENCE_DIR"] == str(conf)
    capsys.readouterr()
    bax.main(argv)                                       # same PLY and cameras: reused
    assert len(calls) == 1 and f"{conf} is up to date" in capsys.readouterr().out
    assert json.loads((out / "inputs.json").read_text())["confidence_maps"]["status"] == "reused"
    (conf / "stale.png").write_bytes(b"")
    later = time.time() + 5
    os.utime(model / "point_cloud" / IT / "point_cloud.ply", (later, later))   # the GW PLY changed: rendered again
    bax.main(argv)
    assert len(calls) == 2 and not (conf / "stale.png").exists()
    (colmap / "images").mkdir()
    for s in STEMS:
        Image.new("RGB", (16, 12)).save(colmap / "images" / f"{s}.png")
    bax.main(argv + ["--images", "images"])               # other cameras: rendered again
    assert len(calls) == 3
    bax.main(argv + ["--confidence_maps", "off"])
    assert "CONFIDENCE_DIR" not in read_env(out / "env.sh") and len(calls) == 3
    assert json.loads((out / "inputs.json").read_text())["confidence_maps"]["status"].startswith("off")


def test_confidence_maps_without_confidence(tmp_path, monkeypatch):
    model, _, _, _ = make_scene(tmp_path)
    write_gw_ply(model / "point_cloud" / IT / "point_cloud.ply", 500, confidence=False)
    monkeypatch.setattr(bax, "cuda_available", lambda: True)
    out = tmp_path / "v"
    run = lambda *a: bax.main(["-m", str(model), "--label", "2", "--variant", "vanilla_object", "--out", str(out), *a])
    run()                                                # auto: skipped with the reason, nothing rendered
    assert "CONFIDENCE_DIR" not in read_env(out / "env.sh")
    status = json.loads((out / "inputs.json").read_text())["confidence_maps"]["status"]
    assert status.startswith("skipped:") and "has no per-Gaussian confidence (train.py --confidence)" in status
    with pytest.raises(AssertionError, match="--confidence_maps on: .*has no per-Gaussian confidence"):
        run("--confidence_maps", "on")
    monkeypatch.setattr(bax, "cuda_available", lambda: False)
    write_gw_ply(model / "point_cloud" / IT / "point_cloud.ply", 500)
    with pytest.raises(AssertionError, match="--confidence_maps on: no CUDA device"):
        run("--confidence_maps", "on")


def test_confidence_maps_failed_render(tmp_path, monkeypatch, capsys):
    """auto: a failed render is skipped (env.sh / inputs.json still written), the previous maps stay; on: it stops."""
    model, colmap, _, _ = make_scene(tmp_path)
    calls = []
    monkeypatch.setattr(bax, "cuda_available", lambda: True)
    monkeypatch.setattr(bax, "render_confidence_maps", gpu_free_renderer(calls))
    out = tmp_path / "v"
    run = lambda *a: bax.main(["-m", str(model), "--label", "2", "--variant", "vanilla_object", "--out", str(out), *a])
    run()
    conf = out / "confidence"
    before = sorted(p.name for p in conf.iterdir())
    assert len(calls) == 1 and f"{STEMS[0]}.png" in before

    def failing(model_path, iteration, out_dir, colmap, images):
        os.makedirs(out_dir, exist_ok=True)
        (Path(out_dir) / "half.png").write_bytes(b"")
        raise subprocess.CalledProcessError(1, ["confidence_map.py"])
    monkeypatch.setattr(bax, "render_confidence_maps", failing)
    later = time.time() + 5
    os.utime(model / "point_cloud" / IT / "point_cloud.ply", (later, later))   # stale: rendered again, and fails
    (out / "env.sh").unlink()
    capsys.readouterr()
    run()
    assert "confidence maps skipped: beags/confidence_map.py failed (exit 1" in capsys.readouterr().out
    assert "CONFIDENCE_DIR" not in read_env(out / "env.sh")
    record = json.loads((out / "inputs.json").read_text())
    assert "CONFIDENCE_DIR" not in record and record["confidence_maps"]["status"].startswith(
        "skipped: beags/confidence_map.py failed (exit 1")
    assert sorted(p.name for p in conf.iterdir()) == before and not (out / "confidence.tmp").exists()
    with pytest.raises(subprocess.CalledProcessError):
        run("--confidence_maps", "on")
    assert sorted(p.name for p in conf.iterdir()) == before


def test_confidence_map_script_runs_as_a_file():
    """`python gaussian_wrapping/beags/confidence_map.py -h` resolves the GW imports (gaussian_wrapping/ on sys.path)."""
    script = GW / "beags" / "confidence_map.py"
    out = subprocess.run([sys.executable, str(script), "-h"], capture_output=True, text=True, cwd=str(ROOT))
    if "No module named" in out.stderr and ("diff_gaussian" in out.stderr or "simple_knn" in out.stderr):
        pytest.skip("GW CUDA extensions are not importable here")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "--min_alpha" in out.stdout and "--out" in out.stdout


# ---------------------------------------------------------------- unseen maps (beags/unseen_map.py, --hole_mode gavis)
def gpu_free_unseen(calls, value=0.8):
    """Stands in for the unseen_map.py subprocess: constant maps + the real info.json."""
    def run(model_path, iteration, out_dir, colmap, images, label, kappa):
        calls.append((out_dir, label, kappa))
        os.makedirs(out_dir, exist_ok=True)
        stats = {}
        for s in STEMS:
            Image.fromarray(umap.encode(np.full((6, 8), value))).save(os.path.join(out_dir, f"{s}.png"))
            stats[s] = {"mean": value, "high_fraction": float(value > umap.HIGH)}
        ply = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        umap.write_info(out_dir, stats, source=ply, iteration=iteration, source_path=colmap, images=images,
                        label=label, kappa=kappa)
    return run


def test_unseen_maps_rendered_and_reused(tmp_path, monkeypatch, capsys):
    model, _, _, _ = make_scene(tmp_path)
    masks = chw_masks(tmp_path / "masks", label=2)
    calls = []
    monkeypatch.setattr(bax, "cuda_available", lambda: True)
    monkeypatch.setattr(bax, "render_unseen_maps", gpu_free_unseen(calls))
    out = tmp_path / "afx"
    argv = ["-m", str(model), "--label", "2", "--variant", "normal", "--masks", str(masks), "--out", str(out),
            "--confidence_maps", "off"]
    bax.main(argv + ["--hole_mode", "alpha"])               # auto: only with gavis
    assert "UNSEEN_DIR" not in read_env(out / "env.sh") and calls == []
    assert json.loads((out / "inputs.json").read_text())["unseen_maps"]["status"].startswith("off")
    bax.main(argv + ["--hole_mode", "gavis"])
    unseen = out / "unseen"
    env = read_env(out / "env.sh")
    assert env["HOLE_MODE"] == "gavis" and env["UNSEEN_DIR"] == str(unseen)
    assert calls == [(str(unseen) + ".tmp", 2, 1.0)] and not os.path.exists(calls[0][0])
    assert sorted(p.stem for p in unseen.glob("*.png")) == STEMS
    record = json.loads((out / "inputs.json").read_text())["unseen_maps"]
    assert record["status"] == "rendered" and record["cameras"] == 8 and record["high_fraction_mean"] == 1.0
    capsys.readouterr()
    bax.main(argv + ["--hole_mode", "gavis"])               # same PLY, label and kappa: reused
    assert len(calls) == 1 and f"{unseen} is up to date" in capsys.readouterr().out
    bax.main(argv + ["--hole_mode", "gavis", "--kappa", "0.5"])   # another kappa: rendered again
    assert len(calls) == 2 and calls[-1][2] == 0.5
    later = time.time() + 5
    os.utime(model / "point_cloud" / IT / "point_cloud.ply", (later, later))   # the GW PLY changed: rendered again
    bax.main(argv + ["--hole_mode", "gavis", "--kappa", "0.5"])
    assert len(calls) == 3
    os.utime(unseen / "info.json", (later + 1, later + 1))   # as if rendered after the PLY changed, however fast the run
    bax.main(argv + ["--hole_mode", "alpha", "--unseen_maps", "on", "--kappa", "0.5"])   # on: with any hole mode
    assert read_env(out / "env.sh")["UNSEEN_DIR"] == str(unseen) and len(calls) == 3
    with pytest.raises(AssertionError, match="--hole_mode gavis needs the unseen maps"):
        bax.main(argv + ["--hole_mode", "gavis", "--unseen_maps", "off"])
    monkeypatch.setattr(bax, "cuda_available", lambda: False)
    with pytest.raises(AssertionError, match="no CUDA device"):
        bax.main(argv + ["--hole_mode", "gavis", "--kappa", "2"])
    bax.main(["-m", str(model), "--label", "2", "--variant", "vanilla_object", "--out", str(tmp_path / "v"),
              "--unseen_maps", "on", "--confidence_maps", "off"])                         # no removal: never rendered
    assert "UNSEEN_DIR" not in read_env(tmp_path / "v" / "env.sh") and len(calls) == 3
