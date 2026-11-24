"""Every removal benchmark scene -> the scene + mask layout the BEA-GS pipeline expects, plus one index.json.

    python gaussian_wrapping/prepare_removal_scenes.py --dataset all [--out <root>]
    python gaussian_wrapping/prepare_removal_scenes.py --dataset spinnerf --scenes 1 2

Three different problems, one per group:

* **mip360** (7) is ours and writable, so it is used in place; only its index entry is written.
* **3DOVS / LERF / LLFF** (20) live under another user's `gabri/datasets/<ds>/raw/`, which is read-only.
  `readColmapSceneInfo` calls `storePly` unconditionally (`scene/dataset_readers.py:162`) to write
  `points3D.ply` next to the model, so it fails there with PermissionError, swallowed into
  "Could not find Colmap scene info". We therefore symlink the images and **copy** the small `sparse/` dir
  (12-22 MB) into a writable scene dir. Masks are already `(C, H, W)` `.npy` and are used where they are.
* **SPIn-NeRF / 360-USID** (17) need a filtered COLMAP model:
  - SPIn-NeRF has 100 images but only the last 60 contain the object (`images_4/label/` covers exactly
    `sorted(images_4)[40:100]`); the first 40 are the object-free split, and training on all 100 would fit a
    scene that both has and lacks the object. Its model also names images `<stem>.jpg` while `images_4/` holds
    `<stem>.png`, so the reader would not find them (`dataset_readers.py:98` joins the sparse name verbatim).
  - 360-USID ships one model for train + test (carton: 222 entries) but only the 186 train images on disk, and
    `readColmapCameras` opens every listed image with no existence check, so the read fails outright.
  Their 2D masks are converted to the same `(C, H, W)` bool `.npy` the pipeline already reads (channel 0
  background, channel 1 object, so `REMOVE=1`): SPIn-NeRF `label/` is `{0,1}`, 360-USID `object_masks/` is a
  bimodal grayscale JPEG binarised at 127.

Writes `<out>/index.json`: `{"<name>": {"scene_dir", "masks", "gt", "args", "views", ...}}` with `<name>` like
`mip360_kitchen`, `3dovs_bench`, `spinnerf_1`, `usid_carton` — `slurm/launch_all_scenes.sh` reads only that file.
"""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
IP = BASE.parent.parent                                   # .../InpaintingPaper
G = Path("/leonardo_work/IscrC_EditGS/gabri/datasets")
MIP = Path("/leonardo_work/IscrC_EditGS/datasets/mip360")
SPINNERF = IP / "data/SPIn-NeRF"
USID = IP / "data/360-USID"

_spec = importlib.util.spec_from_file_location("colmap_loader", BASE / "scene" / "colmap_loader.py")
colmap = importlib.util.module_from_spec(_spec)           # imported directly: scene/__init__ needs CUDA
_spec.loader.exec_module(colmap)
MODEL_NAME = {m.model_id: m.model_name for m in colmap.CAMERA_MODELS}

# dataset -> (mask root, scene list) for the groups whose masks are already in the pipeline's format
PASSTHROUGH = {
    "mip360": (G / "mipnerf360_hd_masks/masks", MIP, ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"], ["-i", "images_4"]),
    "3dovs": (G / "3DOVS/masks", G / "3DOVS/raw", ["bed", "bench", "blue_sofa", "covered_desk", "lawn", "office_desk", "room", "snacks", "sofa", "table"], []),
    "lerf": (G / "lerf/masks", G / "lerf/raw", ["figurines", "ramen", "teatime"], []),
    "llff": (G / "nerf_llff_data_original/masks", G / "nerf_llff_data_original/raw", ["fern", "flower", "fortress", "horns", "leaves", "orchids", "trex"], []),
}


def as_pinhole(model, params):
    """(fx, fy, cx, cy) of a COLMAP camera, dropping distortion.

    `read_intrinsics_text` only accepts PINHOLE (scene/colmap_loader.py:179), while the binary reader accepts any
    model and `readColmapCameras` (scene/dataset_readers.py:86-94) then uses params[0] as both focal lengths for
    SIMPLE_PINHOLE / SIMPLE_RADIAL / RADIAL and params[0:2] for PINHOLE / OPENCV*, ignoring the distortion terms.
    Writing PINHOLE here reproduces exactly that, so nothing is lost that the pipeline would have used.
    """
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):           # f, cx, cy[, k...]
        return params[0], params[0], params[1], params[2]
    if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):  # fx, fy, cx, cy[, k...]
        return params[0], params[1], params[2], params[3]
    raise AssertionError(f"camera model {model} is not handled by scene/dataset_readers.py")


def write_cameras_txt(path, cameras):
    """COLMAP cameras.txt for {id: Camera}, every camera written as PINHOLE; returns the models converted."""
    converted = []
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid, cam in sorted(cameras.items()):
            model = cam.model if isinstance(cam.model, str) else MODEL_NAME[cam.model]
            params = " ".join(repr(float(v)) for v in as_pinhole(model, cam.params))
            if model != "PINHOLE":
                converted.append(model)
            f.write(f"{cid} PINHOLE {cam.width} {cam.height} {params}\n")
    return sorted(set(converted))


def write_images_txt(path, images, rename=None):
    """COLMAP images.txt for {id: Image}; `rename` maps the stored name to the file name to use.

    The second line of each entry (the 2D observations) is left empty: readColmapCameras uses only the pose, the
    camera id and the name, and read_extrinsics_text parses an empty line to a (0, 2) array."""
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n"
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for iid, im in sorted(images.items()):
            q = " ".join(repr(float(v)) for v in im.qvec)
            t = " ".join(repr(float(v)) for v in im.tvec)
            f.write(f"{iid} {q} {t} {im.camera_id} {rename(im.name) if rename else im.name}\n\n")


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink() if dst.is_symlink() or dst.is_file() else shutil.rmtree(dst)
    dst.symlink_to(src)


def entry(name, scene_dir, masks, gt, args, **extra):
    return {"name": name, "scene_dir": str(scene_dir), "masks": str(masks), "gt": str(gt), "args": args, **extra}


# ------------------------------------------------------------------ writable copies (3DOVS / LERF / LLFF / mip360)
def prepare_passthrough(out, dataset):
    """mip360 is writable and used in place; the gabri-owned groups get symlinked images + a copied sparse/."""
    mask_root, src_root, scenes, args = PASSTHROUGH[dataset]
    entries = []
    for scene in scenes:
        src = src_root / scene
        assert src.is_dir(), f"{dataset}/{scene}: no {src}"
        masks = mask_root / "training/sam2" / scene / "masks"
        gt = mask_root / "gt_masks" / scene / "masks"
        assert masks.is_dir(), f"{dataset}/{scene}: no masks at {masks}"
        n_masks = len(list(masks.glob("*.npy")))
        if dataset == "mip360":                                   # ours, writable: storePly can write there
            entries.append(entry(f"{dataset}_{scene}", src, masks, gt, args, masks_count=n_masks, copied_sparse=False))
            continue
        dst = out / "scenes" / dataset / scene
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.name != "sparse":
                link(item, dst / item.name)
        sparse = dst / "sparse"                                   # copied, not linked: the reader writes into it
        if sparse.exists():
            shutil.rmtree(sparse)
        shutil.copytree(src / "sparse", sparse)
        entries.append(entry(f"{dataset}_{scene}", dst, masks, gt, args, masks_count=n_masks, copied_sparse=True))
    return entries


# ------------------------------------------------------------------ filtered models (SPIn-NeRF / 360-USID)
def build_scene(out, dataset, scene, src_sparse, keep_names, image_for, masks, rename=None):
    extr = colmap.read_extrinsics_binary(str(src_sparse / "images.bin"))
    intr = colmap.read_intrinsics_binary(str(src_sparse / "cameras.bin"))
    kept = {i: im for i, im in extr.items() if im.name in keep_names}
    missing = keep_names - {im.name for im in kept.values()}
    assert not missing, f"{dataset}/{scene}: {len(missing)} view(s) not in the COLMAP model, e.g. {sorted(missing)[:3]}"
    cams = {im.camera_id for im in kept.values()}

    sdir = out / "scenes" / dataset / scene
    sparse = sdir / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    converted = write_cameras_txt(sparse / "cameras.txt", {c: intr[c] for c in cams})
    write_images_txt(sparse / "images.txt", kept, rename)
    link(src_sparse / "points3D.bin", sparse / "points3D.bin")
    for stale in ("cameras.bin", "images.bin"):               # a binary pair would win the read, so never leave one
        (sparse / stale).unlink(missing_ok=True)

    imgdir = sdir / "images"
    shutil.rmtree(imgdir, ignore_errors=True)
    sizes = set()
    for im in kept.values():
        src = image_for(im.name)
        link(src, imgdir / (rename(im.name) if rename else im.name))
        if len(sizes) < 3:
            sizes.add(Image.open(src).size)

    mdir = out / "masks" / dataset / "training/sam2" / scene / "masks"
    mdir.mkdir(parents=True, exist_ok=True)
    mask_sizes = set()
    for stem, obj in masks.items():
        np.save(mdir / f"{stem}.npy", np.stack([~obj, obj]))   # channel 0 background, channel 1 object
        mask_sizes.add(obj.shape[::-1])
    gt = out / "masks" / dataset / "gt_masks" / scene
    gt.mkdir(parents=True, exist_ok=True)
    link(mdir, gt / "masks")                                   # no independent GT for these datasets

    assert sizes == mask_sizes, f"{dataset}/{scene}: images {sizes} but masks {mask_sizes}"
    return entry(f"{'spinnerf' if dataset == 'SPIn-NeRF' else 'usid'}_{scene}", sdir, mdir, gt / "masks", [],
                 views=len(kept), cameras=len(cams), image_size=list(sorted(sizes)[0]), masks_count=len(masks),
                 object_fraction=round(float(np.mean([m.mean() for m in masks.values()])), 4),
                 converted_to_pinhole=converted)


def prepare_spinnerf(out, scenes):
    entries = []
    for scene in scenes:
        root = SPINNERF / scene
        stems = sorted(p.stem for p in (root / "images_4/label").glob("*.png"))
        assert stems, f"no labels in {root}/images_4/label"
        masks = {s: np.asarray(Image.open(root / "images_4/label" / f"{s}.png")) > 0 for s in stems}
        # the model names images <stem>.jpg; the images_4 PNG is linked under that name (PIL reads by content)
        entries.append(build_scene(out, "SPIn-NeRF", scene, root / "sparse/0", {f"{s}.jpg" for s in stems},
                                   lambda n, r=root: r / "images_4" / f"{Path(n).stem}.png", masks))
    return entries


def prepare_usid(out, scenes):
    entries = []
    for scene in scenes:
        root = USID / scene
        on_disk = {p.name for p in (root / "images").iterdir() if p.is_file()}
        masks = {p.stem: np.asarray(Image.open(p).convert("L")) > 127     # bimodal, see the module docstring
                 for p in sorted((root / "object_masks").iterdir()) if p.is_file()}
        keep = {n for n in on_disk if Path(n).stem in masks}
        assert len(keep) == len(on_disk), f"{scene}: {len(on_disk) - len(keep)} image(s) have no object mask"
        entries.append(build_scene(out, "360-USID", scene, root / "sparse/0", keep,
                                   lambda n, r=root: r / "images" / n, masks))
    return entries


BUILDERS = {
    "spinnerf": (prepare_spinnerf, lambda: sorted(p.name for p in SPINNERF.iterdir() if p.is_dir())),
    "usid": (prepare_usid, lambda: sorted(p.name for p in USID.iterdir() if p.is_dir())),
    **{d: (lambda out, _s, d=d: prepare_passthrough(out, d), lambda d=d: PASSTHROUGH[d][2]) for d in PASSTHROUGH},
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(BUILDERS) + ["all"], required=True)
    p.add_argument("--scenes", nargs="*", default=None, help="default: every scene of the dataset")
    p.add_argument("--out", type=Path, default=IP / "data" / "removal_scenes")
    args = p.parse_args(argv)

    groups = list(BUILDERS) if args.dataset == "all" else [args.dataset]
    index_path = args.out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for group in groups:
        fn, all_scenes = BUILDERS[group]
        entries = fn(args.out, args.scenes or all_scenes())
        for e in entries:
            index[e["name"]] = e
            print(f"{e['name']:22s} {e.get('views', e['masks_count']):4d} views  {e['masks_count']:4d} masks"
                  + (f"  {e['image_size']}" if "image_size" in e else "")
                  + (f"  [{'+'.join(e['converted_to_pinhole'])}->PINHOLE]" if e.get("converted_to_pinhole") else "")
                  + ("  [sparse copied]" if e.get("copied_sparse") else ""))
    args.out.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=1, sort_keys=True))
    print(f"\n{len(index)} scenes in {index_path}")


if __name__ == "__main__":
    main()
