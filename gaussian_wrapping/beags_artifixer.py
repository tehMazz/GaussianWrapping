"""Inputs for ArtiFixer diffusion (submodules/artifixer_gs, ArtiFixer-GS) on a Gaussian Wrapping (BEA-GS) model.

    python gaussian_wrapping/beags_artifixer.py -m <model> [--iteration N] --label <k> --variant <variant> \
        [--masks <2D mask dir>] [--hole_mode dilate|alpha|mesh|gavis] [--hole_mesh <ply>] [--confidence_maps auto|on|off]
        [--unseen_maps auto|on|off] [--kappa 1]

Variants (as BEA-GS_FSplat slurm/beags_artifixer.sh):
    normal | bighull      object removal + inpainting (run.sh MODE=removal, REMOVAL_VARIANT; hole = mask 12 px |
                          hull 80 px, run.sh's defaults). The removed Gaussians are labels.pt == k; the 2D masks of
                          object k (MASK_DIR, MASK_LABEL, required) shape the per-view holes, HOLE_MODE refines them.
    vanilla_object | vanilla_vidsplat | vanilla_travel
                          no removal, ArtiFixer3D+ on a novel camera path (TRAJ_MODE object|vidsplat|travel);
                          object orbits the label-k Gaussians.
The GW PLY is not standard 3DGS (3D filter, normals, extra fields): export_3dgs_ply.py bakes it into --out/export/.
Writes, into --out (default output/artifixer/<scene>/<variant>; never inside the model dir):
    export/point_cloud.ply  standard INRIA 3DGS PLY (PLY), same Gaussians in the same order as the GW PLY
    object_mask.npy   bool [P], labels.pt == k, aligned 1:1 with the exported PLY vertex order
    confidence/       per-view artifact map <image stem>.png (+ info.json) of a model trained with --confidence:
                      beags/confidence_map.py renders exp(confidence) of every camera (colors_precomp pass, on the GPU),
                      255 * min(c, 1), dark = low confidence; reused while newer than the GW PLY. --confidence_maps
                      auto (default): when the GW PLY has a `confidence` property and CUDA is available, else skipped
                      with the reason (also when the render fails: rendered into confidence.tmp/, the previous maps
                      are kept on disk but not exported); on: required, a failed render stops; off: never. run.sh's split stage reads it when CONF_THRESH is set
    unseen/           per-view unseen map <image stem>.png (+ avf.pt, info.json) after removing label k:
                      beags/unseen_map.py (GAVIS visibility field of the full model from the training cameras, rendered
                      without label k as sum w (1 - v) + (1 - alpha)), 255 * U, white = unseen; reused while newer than
                      the GW PLY with the same label and kappa. --unseen_maps auto (default): with --hole_mode gavis;
                      on: always; off: never (not with gavis). run.sh's split stage reads it with HOLE_MODE=gavis
    env.sh            the variables artifixer_gs/scripts/run.sh reads: PLY, COLMAP (cfg source_path), IMAGES (cfg
                      images), OBJECT_MASK, OUT (<out>/<scene>) [, MASK_DIR, MASK_LABEL, METRIC_SCALE, CAPTION_H5],
                      MODE, REMOVAL_VARIANT, HOLE_MODE [, HOLE_MESH, MESH_PY] | TRAJ_MODE, EPS2D (0: no 2D low-pass in GW),
                      RENDER_CMD + RENDER_CWD (--renderer gw, default: every ArtiFixer-GS PLY render goes through
                      render_ply_frames.py, GW's render_ours, instead of gsplat; ArtiFixer only does the diffusion),
                      IGNORE_DISTORTION=1 (only with non-pinhole COLMAP cameras: GW trained them as pinhole),
                      [CONFIDENCE_DIR (<out>/confidence),] [UNSEEN_DIR (<out>/unseen),] and
                      the ArtiFixer3D trainer: FASTGS_REPO (this repo), TRAIN_CMD (this repo's train.py --rasterizer
                      ours, needs --lpips_images), FASTGS_ARGS (--train_args), AF3D_EXPORT (export_3dgs_ply.py)
    inputs.json       the same plus counts (Gaussians, masks matched to photos, views showing the object) and
                      confidence_maps (mode, status, dir, cameras, encoding, mean confidence)
Mask files are <image stem>.npy or .png: a 2D id / binary map (object = > 0, or == --mask_label) or a BEA-GS
(C, H, W) array (channel 0 background, object = channel k > 0.5; MASK_LABEL = k).
HOLE_MESH defaults to output/<method>/<scene>/mesh/hole_<k>.ply (else <model>/mesh/hole_<k>.ply;
multilabel_mesh_extraction.py --remove k) when it exists; MESH_PY (the Python that runs artifixer_gs.hole_mesh) is
this one when it imports nvdiffrast (the GW env does).
--artifixer_root <ArtiFixer prepared root of the same scene> reuses what ArtiFixer already computed there:
    metric_alignment/scale_info.txt -> METRIC_SCALE, converted to this COLMAP frame through the camera centres of
        the photos both share (not reused, so MoGe runs, when the two frames are not a similarity apart)
    split.json prompt_path (captions/<id>/caption.h5) -> CAPTION_H5, copied in place of a new caption
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'submodules'))

import importlib.util
import json
import shlex
import shutil
import subprocess
from argparse import ArgumentParser, Namespace
from itertools import combinations

import numpy as np

from beags import confidence_map as cmap
from beags import unseen_map as umap

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
VARIANTS = {"normal": "removal", "bighull": "removal",
            "vanilla_object": "vanilla", "vanilla_vidsplat": "vanilla", "vanilla_travel": "vanilla"}
HOLE_MODES = ("dilate", "alpha", "mesh", "gavis")
FLAT_MODES = ("train", "hollow", "beags", "mesh")
CONFIDENCE_MODES = ("auto", "on", "off")
UNSEEN_MODES = ("auto", "on", "off")


def ply_vertex_count(path):
    """Vertex count from the PLY header (the body is not read)."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply", f"{path} is not a PLY"
        for raw in iter(f.readline, b""):
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise ValueError(f"{path}: no 'element vertex' in the PLY header")


def object_mask_from_labels(labels, num_gaussians, label):
    labels = np.asarray(labels).reshape(-1)
    assert len(labels) == num_gaussians, f"labels.pt has {len(labels)} entries, the PLY {num_gaussians} Gaussians"
    obj = labels == label
    ids, counts = np.unique(labels, return_counts=True)
    assert obj.any(), f"no Gaussian has label {label}; per label: {dict(zip(ids.tolist(), counts.tolist()))}"
    return obj


def scene_name(model_path):
    """<scene> of a flat output/<method>/<scene>/<mode> model dir, else the model dir's name."""
    model_path = os.path.normpath(os.path.abspath(model_path))
    name = os.path.basename(model_path)
    return os.path.basename(os.path.dirname(model_path)) if name in FLAT_MODES else name


def latest_iteration(model_path):
    folder = os.path.join(model_path, "point_cloud")
    its = [int(d.split("_")[-1]) for d in os.listdir(folder) if d.startswith("iteration_")]
    assert its, f"no point_cloud/iteration_* in {model_path}"
    return max(its)


def export_ply(model_path, iteration, ply_in, ply_out):
    """export_3dgs_ply.py (baked 3D filter, standard 3DGS attributes); kept when it is newer than the same GW PLY."""
    try:
        with open(os.path.join(os.path.dirname(ply_out), "export_info.json")) as f:
            fresh = json.load(f)["source"] == ply_in and os.path.getmtime(ply_out) >= os.path.getmtime(ply_in)
    except (OSError, KeyError, ValueError):
        fresh = False
    if fresh:
        print(f"[beags] {ply_out} is up to date")
        return
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "export_3dgs_ply.py"), "-m", model_path,
                    "--iteration", str(iteration), "--out", ply_out], check=True)


def cuda_available():
    import torch
    return torch.cuda.is_available()


def render_confidence_maps(model_path, iteration, out_dir, colmap, images):
    """beags/confidence_map.py in a subprocess (loads the GW scene on the GPU)."""
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "beags", "confidence_map.py"), "-m", model_path,
                    "--iteration", str(iteration), "--source_path", colmap, "--images", images, "--out", out_dir],
                   check=True)


def confidence_maps(mode, model_path, iteration, gw_ply, out_dir, colmap, images):
    """(<out_dir> or None, inputs.json record): renders the maps, reuses fresh ones, or says why there are none."""
    record = {"mode": mode, "encoding": cmap.ENCODING}
    if mode == "off":
        return None, {**record, "status": "off (--confidence_maps off)"}
    why = None
    if not cmap.ply_has_confidence(gw_ply):
        why = f"{gw_ply} has no per-Gaussian confidence (train.py --confidence)"
    elif not cuda_available():
        why = "no CUDA device (the maps are rendered on the GPU)"
    if why:
        assert mode == "auto", f"--confidence_maps on: {why}"
        return None, {**record, "status": f"skipped: {why}"}
    info_path = os.path.join(out_dir, "info.json")
    try:
        with open(info_path) as f:
            info = json.load(f)
        fresh = (info["source"] == gw_ply and info["source_path"] == colmap and info["images"] == images
                 and os.path.getmtime(info_path) >= os.path.getmtime(gw_ply))
    except (OSError, KeyError, ValueError):
        fresh = False
    if fresh:
        print(f"[beags] {out_dir} is up to date")
    else:
        # rendered into <out_dir>.tmp and swapped in on success: a failed render leaves the previous maps alone
        tmp = out_dir.rstrip(os.sep) + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            render_confidence_maps(model_path, iteration, tmp, colmap, images)
        except subprocess.CalledProcessError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            if mode == "on":
                raise
            why = f"beags/confidence_map.py failed (exit {e.returncode}, see its log above)"
            print(f"[beags] confidence maps skipped: {why}; {out_dir} is not exported")
            return None, {**record, "status": f"skipped: {why}"}
        shutil.rmtree(out_dir, ignore_errors=True)  # no map of another model / camera set survives
        os.replace(tmp, out_dir)
        with open(info_path) as f:
            info = json.load(f)
    status = "reused" if fresh else "rendered"
    return out_dir, {**record, "status": status, "dir": out_dir, "cameras": info["cameras"], "mean": info["mean"],
                     "low_fraction_mean": info["low_fraction_mean"]}


def render_unseen_maps(model_path, iteration, out_dir, colmap, images, label, kappa):
    """beags/unseen_map.py in a subprocess (loads the GW scene on the GPU)."""
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "beags", "unseen_map.py"), "-m", model_path,
                    "--iteration", str(iteration), "--source_path", colmap, "--images", images, "--out", out_dir,
                    "--remove", str(label), "--kappa", repr(float(kappa)), "--vis"], check=True)


def unseen_maps(mode, hole_mode, model_path, iteration, gw_ply, out_dir, colmap, images, label, kappa):
    """(<out_dir> or None, inputs.json record): renders the maps, reuses fresh ones, or says why there are none."""
    record = {"mode": mode, "encoding": umap.ENCODING, "kappa": float(kappa)}
    if mode == "off" or (mode == "auto" and hole_mode != "gavis"):
        assert hole_mode != "gavis", "--hole_mode gavis needs the unseen maps (--unseen_maps auto|on)"
        return None, {**record, "status": f"off (--unseen_maps {mode}, --hole_mode {hole_mode})"}
    assert cuda_available(), "no CUDA device: the unseen maps (--hole_mode gavis) are rendered on the GPU"
    info_path = os.path.join(out_dir, "info.json")
    try:
        with open(info_path) as f:
            info = json.load(f)
        fresh = (info["source"] == gw_ply and info["source_path"] == colmap and info["images"] == images
                 and info["label"] == label and info["kappa"] == float(kappa)
                 and os.path.getmtime(info_path) >= os.path.getmtime(gw_ply))
    except (OSError, KeyError, ValueError):
        fresh = False
    if fresh:
        print(f"[beags] {out_dir} is up to date")
    else:  # rendered into <out_dir>.tmp and swapped in on success, as the confidence maps
        tmp = out_dir.rstrip(os.sep) + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        render_unseen_maps(model_path, iteration, tmp, colmap, images, label, kappa)
        shutil.rmtree(out_dir, ignore_errors=True)
        os.replace(tmp, out_dir)
        with open(info_path) as f:
            info = json.load(f)
    status = "reused" if fresh else "rendered"
    return out_dir, {**record, "status": status, "dir": out_dir, "cameras": info["cameras"], "mean": info["mean"],
                     "high_fraction_mean": info["high_fraction_mean"]}


def load_mask(path):
    if path.endswith(".npy"):
        return np.load(path)
    from PIL import Image
    m = np.asarray(Image.open(path))
    return m[..., 0] if m.ndim == 3 else m


def mask_object(m, label, mask_label=None):
    """Object pixels of one mask file: 2D id / binary map, or BEA-GS (C, H, W) with object = channel `label`."""
    m = np.nan_to_num(np.asarray(m))
    if m.ndim == 2:
        return m == mask_label if mask_label is not None else m > 0
    assert m.ndim == 3, f"mask of shape {m.shape}: expected (H, W) or (C, H, W)"
    assert label < m.shape[0], f"(C, H, W) mask with C={m.shape[0]} has no channel {label}"
    return m[label].astype(np.float32) > 0.5


def image_stems(colmap, images):
    folder = os.path.join(colmap, images)
    assert os.path.isdir(folder), f"no image folder {folder}"
    return sorted(os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))


def mask_files(mask_dir, stems):
    """{stem: path} of the masks named after a photo (.npy preferred over .png); other files are ignored."""
    found = {}
    for name in sorted(os.listdir(mask_dir)):
        stem, ext = os.path.splitext(name)
        if stem in stems and ext.lower() in (".npy", ".png") and (stem not in found or ext == ".npy"):
            found[stem] = os.path.join(mask_dir, name)
    return found


def mask_stats(mask_dir, stems, label, mask_label=None):
    """Matched masks, views showing the object, the mask layout (2d | chw) and MASK_LABEL to export (or None)."""
    files = mask_files(mask_dir, set(stems))
    assert files, f"no mask in {mask_dir} is named after a photo (e.g. {stems[0]}.npy or .png)"
    layout, present, fractions = None, 0, []
    for path in files.values():
        m = load_mask(path)
        kind = "chw" if np.ndim(m) == 3 else "2d"
        assert layout in (None, kind), f"{path}: mixed mask layouts ({layout} and {kind}) in {mask_dir}"
        layout = kind
        obj = mask_object(m, label, mask_label)
        present += bool(obj.any())
        fractions.append(float(obj.mean()))
    # (C, H, W): channel k is object k, so MASK_LABEL = k; 2D maps: > 0 unless a label was asked for.
    export_label = label if layout == "chw" else mask_label
    return {"masks": len(files), "photos": len(stems), "object_views": present, "layout": layout,
            "object_fraction_mean": float(np.mean(fractions))}, export_label


def merge_masks(primary, fallback, stems, out_dir):
    """<out_dir>/<stem>.<ext> symlinks: the primary mask of each photo, else the fallback's. -> (from primary, fallback)"""
    first, second = mask_files(primary, set(stems)), mask_files(fallback, set(stems))
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if os.path.islink(os.path.join(out_dir, name)):
                os.remove(os.path.join(out_dir, name))
    os.makedirs(out_dir, exist_ok=True)
    for stem in stems:
        src = first.get(stem) or second.get(stem)
        if src:
            os.symlink(os.path.abspath(src), os.path.join(out_dir, stem + os.path.splitext(src)[1]))
    return len(first), len([s for s in stems if s not in first and s in second])


def colmap_loader():
    # By file path: importing the scene package would load the CUDA extensions.
    spec = importlib.util.spec_from_file_location("_colmap_loader", os.path.join(BASE_DIR, "scene", "colmap_loader.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sparse_dir(colmap):
    for d in (os.path.join(colmap, "sparse", "0"), os.path.join(colmap, "sparse"), colmap):
        if os.path.isfile(os.path.join(d, "images.bin")) or os.path.isfile(os.path.join(d, "images.txt")):
            return d
    raise FileNotFoundError(f"no sparse/0/images.bin (or .txt) under {colmap}")


def camera_models(sparse):
    """Set of COLMAP camera model names of a sparse model (empty without cameras.bin / .txt)."""
    loader = colmap_loader()
    if os.path.isfile(os.path.join(sparse, "cameras.bin")):
        cameras = loader.read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    elif os.path.isfile(os.path.join(sparse, "cameras.txt")):  # (GW's text reader asserts PINHOLE: parse the names)
        with open(os.path.join(sparse, "cameras.txt")) as f:
            return {line.split()[1] for line in f if line.strip() and not line.startswith("#")}
    else:
        return set()
    return {cam.model for cam in cameras.values()}


def camera_centres(sparse):
    """{image stem: camera centre} of a COLMAP model."""
    loader = colmap_loader()
    if os.path.isfile(os.path.join(sparse, "images.bin")):
        images = loader.read_extrinsics_binary(os.path.join(sparse, "images.bin"))
    else:
        images = loader.read_extrinsics_text(os.path.join(sparse, "images.txt"))
    return {os.path.splitext(os.path.basename(im.name))[0]: -loader.qvec2rotmat(im.qvec).T @ im.tvec
            for im in images.values()}


def read_scale_info(path):
    with open(path) as f:
        for line in f:
            if line.startswith("Scale factor:"):
                return float(line.split(":", 1)[1].split()[0])
    raise ValueError(f"no 'Scale factor:' line in {path}")


def frame_ratio(ours, theirs, min_cameras=3, tol=0.01, max_pairs=20000):
    """Scale of `theirs` relative to `ours` (|c_theirs| = r |c_ours|) from the cameras both have, or (None, why).

    RMS spread about the centroid and the median pairwise distance ratio must agree within tol, and pairwise
    ratios must be consistent: otherwise the two frames are not a similarity apart."""
    common = sorted(set(ours) & set(theirs))
    if len(common) < min_cameras:
        return None, f"{len(common)} shared photos (need {min_cameras})", len(common)
    a = np.array([ours[k] for k in common], dtype=np.float64)
    b = np.array([theirs[k] for k in common], dtype=np.float64)
    spread_a = np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean())
    spread_b = np.sqrt(((b - b.mean(0)) ** 2).sum(1).mean())
    if spread_a < 1e-9 or spread_b < 1e-9:
        return None, "the shared photos are all at one position", len(common)
    rms = spread_b / spread_a
    pairs = list(combinations(range(len(common)), 2))
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in np.random.default_rng(0).choice(len(pairs), max_pairs, replace=False)]
    i, j = np.array(pairs).T
    da, db = np.linalg.norm(a[i] - a[j], axis=1), np.linalg.norm(b[i] - b[j], axis=1)
    keep = da > 1e-3 * spread_a
    ratios = db[keep] / da[keep]
    median = float(np.median(ratios))
    spread = float(np.percentile(np.abs(ratios / median - 1), 90))
    if abs(median / rms - 1) > tol or spread > tol:
        return None, (f"camera frames differ by more than a similarity (median pair ratio {median:.5f}, "
                      f"rms ratio {rms:.5f}, 90th pct deviation {spread:.4f})"), len(common)
    return float(rms), "", len(common)


def metric_scale_from_root(root, colmap):
    """(metres per unit of `colmap`, source description, info) from a prepared root's scale_info.txt, or (None, ...)."""
    info_path = os.path.join(root, "metric_alignment", "scale_info.txt")
    if not os.path.isfile(info_path):
        return None, f"none: no {info_path}", {}
    scale = read_scale_info(info_path)
    candidates = [os.path.join(root, sub, scene) for sub in ("3dgrut_input", "input")
                  if os.path.isdir(os.path.join(root, sub)) for scene in sorted(os.listdir(os.path.join(root, sub)))]
    candidates.append(root)
    for cand in candidates:
        try:
            theirs = camera_centres(sparse_dir(cand))
        except FileNotFoundError:
            continue
        ratio, why, shared = frame_ratio(camera_centres(sparse_dir(colmap)), theirs)
        info = {"scale_info": info_path, "scale_info_value": scale, "reference_colmap": cand, "shared_photos": shared}
        if ratio is None:
            return None, f"none: {info_path} not reused, {why}", info
        info["frame_ratio"] = ratio
        return scale * ratio, f"{info_path} (x {ratio:.6f}: its COLMAP frame vs ours, {shared} shared photos)", info
    return None, f"none: {info_path} has no COLMAP model next to it to check the frame", {"scale_info": info_path}


def caption_from_root(root):
    """The caption.h5 a prepared root's split.json points to (None if there is none)."""
    try:
        with open(os.path.join(root, "split.json")) as f:
            entry = next(iter(json.load(f)["test"].values()))
    except (OSError, KeyError, StopIteration, ValueError):
        return None
    path = os.path.realpath(os.path.join(root, entry.get("prompt_path", "")))
    return path if entry.get("prompt_path") and os.path.isfile(path) else None


def train_has_lpips_images():
    with open(os.path.join(BASE_DIR, "train.py")) as f:
        return "--lpips_images" in f.read()


def build_parser():
    parser = ArgumentParser(description="BEA-GS (Gaussian Wrapping) -> ArtiFixer-GS inputs")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1, help="default: the latest point_cloud/iteration_*")
    parser.add_argument("--label", type=int, required=True, help="object label k (0 = background)")
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--masks", default="", help="2D object masks <image stem>.npy|.png (required for removal)")
    parser.add_argument("--fallback_masks", default="",
                        help="masks for the photos --masks lacks (e.g. the SAM2 masks behind the BEA-GS consensus masks, "
                             "which cover training views only): merged into <out>/masks as symlinks")
    parser.add_argument("--mask_label", type=int, default=None,
                        help="2D id maps: object = this id (default: > 0); (C, H, W) masks always use channel k")
    parser.add_argument("--colmap", default="", help="default: cfg_args source_path")
    parser.add_argument("--images", default="", help="default: cfg_args images")
    parser.add_argument("--metric_scale", type=float, default=None, help="metres per scene unit (skips MoGe)")
    parser.add_argument("--artifixer_root", default="",
                        help="ArtiFixer prepared root of this scene: its metric scale and caption.h5 are reused")
    parser.add_argument("--hole_mode", choices=HOLE_MODES, default="dilate",
                        help="removal hole: dilate (the dilated mask) | alpha (its see-through pixels) | mesh (alpha | HOLE_MESH in front) "
                             "| gavis (its pixels the training views never saw from that direction: unseen map > UNSEEN_THRESH)")
    parser.add_argument("--hole_mesh", default="", help="default: <model>/../mesh/hole_<k>.ply (flat layout), else <model>/mesh/hole_<k>.ply, if it exists")
    parser.add_argument("--confidence_maps", choices=CONFIDENCE_MODES, default="auto",
                        help="per-view artifact maps <out>/confidence (CONFIDENCE_DIR): auto = when the PLY has "
                             "confidence and CUDA is available, on = required, off = never")
    parser.add_argument("--unseen_maps", choices=UNSEEN_MODES, default="auto",
                        help="per-view unseen maps <out>/unseen (UNSEEN_DIR, GAVIS visibility): auto = with --hole_mode gavis, "
                             "on = always, off = never")
    parser.add_argument("--kappa", type=float, default=1.0, help="GAVIS vMF concentration (beags/unseen_map.py)")
    parser.add_argument("--eps2d", type=float, default=0.0, help="gsplat 2D low-pass (0: GW adds none)")
    parser.add_argument("--renderer", choices=("gw", "gsplat"), default="gw",
                        help="ArtiFixer-GS PLY renders: gw = GW's render_ours (render_ply_frames.py as RENDER_CMD), "
                             "gsplat = ArtiFixer-GS's own (EPS2D)")
    parser.add_argument("--train_cmd", default="",
                        help="ArtiFixer3D trainer run in this repo (default: <python> gaussian_wrapping/train.py --rasterizer ours)")
    parser.add_argument("--train_args", default="", help="extra trainer flags (FASTGS_ARGS)")
    parser.add_argument("--scene", default="", help="default: <scene> of output/<method>/<scene>/<mode>, else the model dir name")
    parser.add_argument("--out", default="", help="default: output/artifixer/<scene>/<variant>")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    mode = VARIANTS[args.variant]
    assert args.label > 0, "label 0 is the background"
    assert mode == "vanilla" or args.masks, f"--variant {args.variant} needs --masks (the 2D masks shape the holes)"

    model = os.path.abspath(args.model_path)
    with open(os.path.join(model, "cfg_args")) as f:
        cfg = eval(f.read(), {"Namespace": Namespace})
    colmap = os.path.abspath(args.colmap or cfg.source_path)
    images = args.images or cfg.images
    iteration = args.iteration if args.iteration >= 0 else latest_iteration(model)
    it_dir = os.path.join(model, "point_cloud", f"iteration_{iteration}")
    gw_ply = os.path.join(it_dir, "point_cloud.ply")
    num_gaussians = ply_vertex_count(gw_ply)
    import torch  # noqa: E402  (only here: the rest is numpy)
    labels = torch.load(os.path.join(it_dir, "labels.pt"), map_location="cpu").numpy()
    obj = object_mask_from_labels(labels, num_gaussians, args.label)

    scene = args.scene or scene_name(model)
    out = os.path.abspath(args.out or os.path.join(ROOT_DIR, "output", "artifixer", scene, args.variant))
    assert not out.startswith(model + os.sep), f"{out} is inside the model dir"
    os.makedirs(out, exist_ok=True)
    ply = os.path.join(out, "export", "point_cloud.ply")
    export_ply(model, iteration, gw_ply, ply)
    assert ply_vertex_count(ply) == num_gaussians, f"{ply} has {ply_vertex_count(ply)} Gaussians, {gw_ply} {num_gaussians}"
    env = {"PLY": ply, "COLMAP": colmap, "IMAGES": images, "OBJECT_MASK": os.path.join(out, "object_mask.npy"),
           "OUT": os.path.join(out, scene)}
    np.save(env["OBJECT_MASK"], obj)
    print(f"[beags] label {args.label}: {int(obj.sum())} of {num_gaussians} Gaussians -> {env['OBJECT_MASK']}")

    record = {"variant": args.variant, "mode": mode, "label": args.label, "iteration": iteration, "model": model,
              "gw_ply": gw_ply, "num_gaussians": num_gaussians, "object_gaussians": int(obj.sum()), "masks": args.masks}
    if args.masks:
        stems, masks = image_stems(colmap, images), args.masks
        if args.fallback_masks:
            masks = os.path.join(out, "masks")
            n_first, n_fallback = merge_masks(args.masks, args.fallback_masks, stems, masks)
            record["mask_merge"] = {"masks": os.path.abspath(args.masks), "fallback": os.path.abspath(args.fallback_masks),
                                    "from_masks": n_first, "from_fallback": n_fallback}
            print(f"[beags] masks: {n_first} from {args.masks}, {n_fallback} from {args.fallback_masks} -> {masks}")
        stats, export_label = mask_stats(masks, stems, args.label, args.mask_label)
        assert mode == "vanilla" or stats["object_views"], f"object {args.label} is in none of the masks"
        if mode == "removal" and stats["masks"] < stats["photos"]:
            # artifixer_gs.removal split needs the mask of every frame in FRAMES (default: all photos)
            print(f"[beags] WARNING: {masks} has masks for {stats['masks']} of {stats['photos']} photos: the split stage "
                  f"fails on the others unless FRAMES leaves them out (BEA-GS consensus masks cover training views "
                  f"only: add --fallback_masks <SAM2 masks>)")
        env["MASK_DIR"] = os.path.abspath(masks)
        if export_label is not None:
            env["MASK_LABEL"] = export_label
        record["mask_stats"] = stats
        print(f"[beags] masks {masks}: {stats['masks']} of {stats['photos']} photos ({stats['layout']}), "
              f"object in {stats['object_views']}")

    scale, source = args.metric_scale, "--metric_scale" if args.metric_scale is not None else "none (MoGe)"
    if scale is None and args.artifixer_root:
        scale, source, record["metric_scale_check"] = metric_scale_from_root(args.artifixer_root, colmap)
    if scale is not None:
        env["METRIC_SCALE"] = repr(float(scale))
    record.update(metric_scale=scale, metric_scale_source=source)
    print(f"[beags] metric scale: {scale} ({source})")
    caption = caption_from_root(args.artifixer_root) if args.artifixer_root else None
    if caption:
        env["CAPTION_H5"] = caption
    print(f"[beags] caption.h5: {caption or 'none (CAPTION text or the captioner)'}")

    if mode == "removal":
        env.update(MODE="removal", REMOVAL_VARIANT=args.variant, HOLE_MODE=args.hole_mode)
    else:
        env["TRAJ_MODE"] = args.variant[len("vanilla_"):]
    hole_mesh = args.hole_mesh
    if not hole_mesh:  # flat output/<method>/<scene>/{beags,mesh}, else a mesh/ inside the model dir
        flat = os.path.join(os.path.dirname(model), "mesh", f"hole_{args.label}.ply")
        nested = os.path.join(model, "mesh", f"hole_{args.label}.ply")
        hole_mesh = flat if os.path.basename(model) in FLAT_MODES and os.path.isfile(flat) else nested
    hole_mesh = os.path.abspath(hole_mesh)
    if args.hole_mesh or (mode == "removal" and args.hole_mode == "mesh"):
        assert os.path.isfile(hole_mesh), f"no hole mesh {hole_mesh} (multilabel_mesh_extraction.py --remove {args.label})"
    if os.path.isfile(hole_mesh):
        env["HOLE_MESH"] = hole_mesh
        if importlib.util.find_spec("nvdiffrast") is not None:
            env["MESH_PY"] = sys.executable
    print(f"[beags] hole: {args.hole_mode}, mesh {env.get('HOLE_MESH', 'none')}")
    conf_dir, record["confidence_maps"] = confidence_maps(args.confidence_maps, model, iteration, gw_ply,
                                                          os.path.join(out, "confidence"), colmap, images)
    if conf_dir:
        env["CONFIDENCE_DIR"] = conf_dir
    cm = record["confidence_maps"]
    print(f"[beags] confidence maps: {cm['status']}" + (f", {cm['cameras']} views, mean confidence {cm['mean']:.3f} -> "
                                                      f"{conf_dir}" if conf_dir else ""))
    if mode == "removal":
        unseen_dir, record["unseen_maps"] = unseen_maps(args.unseen_maps, args.hole_mode, model, iteration, gw_ply,
                                                        os.path.join(out, "unseen"), colmap, images, args.label, args.kappa)
    else:
        unseen_dir, record["unseen_maps"] = None, {"mode": args.unseen_maps, "status": f"off (MODE={mode})"}
    if unseen_dir:
        env["UNSEEN_DIR"] = unseen_dir
    um = record["unseen_maps"]
    print(f"[beags] unseen maps: {um['status']}" + (f", {um['cameras']} views, mean U {um['mean']:.3f} -> {unseen_dir}"
                                                    if unseen_dir else ""))
    env["EPS2D"] = repr(float(args.eps2d))
    env["CENTER_PRINCIPAL_POINT"] = "1"  # GW ignores COLMAP's cx, cy (dataset_readers.py): renders + hole_mesh likewise
    if args.renderer == "gw":  # every ArtiFixer-GS render through GW's rasterizer; ArtiFixer only does the diffusion
        # kernel_size 0: train.py renders with 0.0 whatever cfg_args says. On black whatever the model's background:
        # ArtiFixer was trained on black-composited renders
        cmd = f"{sys.executable} gaussian_wrapping/render_ply_frames.py --kernel_size 0.0"
        env.update(RENDER_CMD=cmd, RENDER_CWD=ROOT_DIR)
    record["renderer"] = args.renderer
    print(f"[beags] renderer: {args.renderer}" + (f" ({env['RENDER_CMD']})" if args.renderer == "gw" else ""))
    models = camera_models(sparse_dir(colmap))
    record["camera_models"] = sorted(models)
    if models - {"PINHOLE", "SIMPLE_PINHOLE"}:
        # scene/dataset_readers.py reads these as pinhole and drops the distortion: ArtiFixer-GS must do the same
        env["IGNORE_DISTORTION"] = "1"
        print(f"[beags] COLMAP cameras {sorted(models)}: IGNORE_DISTORTION=1 (GW trained them as pinhole)")
    env.update(FASTGS_REPO=ROOT_DIR,
               TRAIN_CMD=args.train_cmd or f"{sys.executable} gaussian_wrapping/train.py --rasterizer ours",
               FASTGS_ARGS=args.train_args,
               AF3D_EXPORT=f"{sys.executable} gaussian_wrapping/export_3dgs_ply.py")
    record["train_lpips_images"] = train_has_lpips_images()
    if not record["train_lpips_images"]:
        print("[beags] WARNING: gaussian_wrapping/train.py has no --lpips_images yet: the ArtiFixer3D train stage will fail")

    with open(os.path.join(out, "inputs.json"), "w") as f:
        json.dump({**record, **env}, f, indent=1)
    with open(os.path.join(out, "env.sh"), "w") as f:
        f.writelines(f"export {k}={shlex.quote(str(v))}\n" for k, v in env.items())
    print(f"[beags] wrote {out}/env.sh")


if __name__ == "__main__":
    main()
