"""Monocular depth + normal priors (MoGe-2) for a COLMAP scene, and how well they agree with the COLMAP geometry.

    python gaussian_wrapping/mono_prior_prepare.py -s <scene> [-i images] --out output/mono_prior/<scene>/moge2 \
        [--width 1008] [--ckpt .../moge-2-vitl-normal/.../model.pt]

Runs in the artifixer env (it has the `moge` package); needs a GPU. Per training image, writes in --out:
    depth/<name>.npy   float16 (H, W) metric depth along the camera z axis, 0 where MoGe's mask is off
    normal/<name>.npy  float16 (H, W, 3) camera-space normals, OpenCV axes (x right, y down, z forward)
MoGe gets the true horizontal FoV from the COLMAP intrinsics, so its depth uses the same projection as training.
Agreement with COLMAP (sparse/0 tracks of each image): one scale per image s = median(z_colmap / z_moge), then the
relative depth error |s z_moge - z_colmap| / z_colmap. stats.json: per image scale + error; summary: median error and
how stable the scale is across images (if stable, one global scale maps MoGe into the scene's units).
viz.jpg: 6 views, photo | MoGe depth | MoGe normal | COLMAP-point error, with column titles.
"""
import argparse
import json
import os
import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, rel):
    """Import one file by path: the scene/ and utils/ packages pull in CUDA extensions this env does not have."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cl = _load("colmap_loader", "scene/colmap_loader.py")
read_extrinsics_binary, read_extrinsics_text = _cl.read_extrinsics_binary, _cl.read_extrinsics_text
read_intrinsics_binary, read_intrinsics_text, qvec2rotmat = _cl.read_intrinsics_binary, _cl.read_intrinsics_text, _cl.qvec2rotmat
titled = _load("grid_titles", "utils/grid_titles.py").titled

CKPT = ("/leonardo_work/IscrC_EditGS/gabri/hf_cache/hub/models--Ruicheng--moge-2-vitl-normal/snapshots/"
        "b135031bae30b5ac2ae141a0e68717795ce38340/model.pt")


def read_model(sparse):
    b = os.path.exists(os.path.join(sparse, "images.bin"))
    ex = read_extrinsics_binary(os.path.join(sparse, "images.bin")) if b else \
        read_extrinsics_text(os.path.join(sparse, "images.txt"))
    cb = os.path.exists(os.path.join(sparse, "cameras.bin"))
    cams = read_intrinsics_binary(os.path.join(sparse, "cameras.bin")) if cb else \
        read_intrinsics_text(os.path.join(sparse, "cameras.txt"))
    return ex, cams


def read_points(sparse):
    """({point3D_id: xyz}, {image_id: [point3D_id, ...]}) from points3D.bin or .txt (tracks, not images.txt 2D
    points: some scenes, e.g. SPIn-NeRF, ship images.txt without them)."""
    pb = os.path.join(sparse, "points3D.bin")
    pts, seen = {}, {}
    if os.path.exists(pb):
        import struct
        with open(pb, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                pid, x, y, z = struct.unpack("<Qddd", f.read(32))
                f.read(11)                                   # rgb (3 B) + error (8 B)
                L = struct.unpack("<Q", f.read(8))[0]
                tr = struct.unpack("<" + "ii" * L, f.read(8 * L))
                pts[pid] = np.array([x, y, z])
                for iid in set(tr[0::2]):
                    seen.setdefault(iid, []).append(pid)
    else:
        for line in open(os.path.join(sparse, "points3D.txt")):
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            pid = int(v[0])
            pts[pid] = np.array([float(v[1]), float(v[2]), float(v[3])])
            for iid in set(int(i) for i in v[8::2]):
                seen.setdefault(iid, []).append(pid)
    return pts, seen


def turbo(x, lo, hi, valid):
    import matplotlib.cm as cm
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (cm.turbo(1 - t)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = (255, 0, 255)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", required=True)
    ap.add_argument("-i", "--images", default="images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=0, help="resize images to this width (0 = keep)")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--n_viz", type=int, default=6)
    a = ap.parse_args()
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained(a.ckpt).cuda().eval()
    sparse = os.path.join(a.source, "sparse", "0")
    ex, cams = read_model(sparse)
    pts, seen = read_points(sparse)
    for sub in ("depth", "normal"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    rows, keep = [], {}
    names = sorted(ex.values(), key=lambda e: e.name)
    viz_ids = set(np.unique(np.linspace(0, len(names) - 1, a.n_viz).round().astype(int)).tolist())
    for k, e in enumerate(names):
        cam = cams[e.camera_id]
        im = Image.open(os.path.join(a.source, a.images, e.name)).convert("RGB")
        if a.width and im.width != a.width:
            im = im.resize((a.width, round(im.height * a.width / im.width)), Image.LANCZOS)
        W, H = im.size
        fx = cam.params[0]
        fov_x = float(np.degrees(2 * np.arctan(cam.width / (2 * fx))))
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().div(255).cuda()
        with torch.no_grad():
            o = model.infer(x, fov_x=fov_x)
        depth = o["depth"].float().cpu().numpy()
        mask = o["mask"].bool().cpu().numpy() & np.isfinite(depth)
        depth = np.where(mask, depth, 0)
        normal = np.nan_to_num(o["normal"].float().cpu().numpy())
        stem = os.path.splitext(e.name)[0]
        np.save(os.path.join(a.out, "depth", stem + ".npy"), depth.astype(np.float16))
        np.save(os.path.join(a.out, "normal", stem + ".npy"), normal.astype(np.float16))
        # agreement with the COLMAP points this image sees (its tracks), projected with the COLMAP intrinsics
        R, t = qvec2rotmat(e.qvec), e.tvec
        ids = [p for p in seen.get(e.id, []) if p in pts]
        X = np.stack([pts[p] for p in ids]) if ids else np.zeros((0, 3))
        Xc = X @ R.T + t
        zc = Xc[:, 2]
        if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            fx_ = fy_ = cam.params[0]
            cx_, cy_ = cam.params[1], cam.params[2]
        else:                                                # PINHOLE, OPENCV, FULL_OPENCV, ...: fx fy cx cy
            fx_, fy_, cx_, cy_ = cam.params[:4]
        sx = W / cam.width
        u = (fx_ * Xc[:, 0] / np.maximum(zc, 1e-9) + cx_) * sx
        v = (fy_ * Xc[:, 1] / np.maximum(zc, 1e-9) + cy_) * sx
        inside = (zc > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        zc, u, v = zc[inside], u[inside], v[inside]
        ui, vi = u.astype(int), v.astype(int)
        zm = depth[vi, ui]
        ok = (zm > 0) & (zc > 0)
        s = float(np.median(zc[ok] / zm[ok])) if ok.sum() > 10 else float("nan")
        rel = np.abs(s * zm[ok] - zc[ok]) / zc[ok]
        row = {"image": e.name, "fov_x": fov_x, "n_points": int(ok.sum()), "scale": s,
               "rel_err_median": float(np.median(rel)) if ok.any() else None,
               "rel_err_p90": float(np.percentile(rel, 90)) if ok.any() else None, "mask_share": float(mask.mean())}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if k in viz_ids:
            keep[k] = (np.asarray(im), depth, mask, normal, ui[ok], vi[ok], rel)
    scales = np.array([r["scale"] for r in rows if np.isfinite(r["scale"])])
    errs = np.array([r["rel_err_median"] for r in rows if r["rel_err_median"] is not None])
    summary = {"images": len(rows), "rel_err_median_over_images": float(np.median(errs)),
               "rel_err_p90_over_images": float(np.percentile(errs, 90)),
               "scale_median": float(np.median(scales)), "scale_cv": float(scales.std() / scales.mean())}
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump({"summary": summary, "per_image": rows}, f, indent=1)
    print("[mono] summary", json.dumps(summary))
    # viz: photo | depth | normal | error at COLMAP points (green < 2 %, yellow < 10 %, red above)
    tiles = []
    for k in sorted(keep):
        rgb, depth, mask, normal, ui, vi, rel = keep[k]
        lo, hi = np.percentile(depth[mask], [2, 98]) if mask.any() else (0, 1)
        nrm = ((normal * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        err = (rgb * 0.35).astype(np.uint8)
        col = np.where(rel[:, None] < 0.02, [40, 220, 60], np.where(rel[:, None] < 0.10, [250, 210, 40], [240, 40, 40]))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                err[np.clip(vi + dy, 0, err.shape[0] - 1), np.clip(ui + dx, 0, err.shape[1] - 1)] = col
        tiles.append(np.concatenate([rgb, turbo(depth, lo, hi, mask), nrm, err], axis=1))
    w = min(t.shape[1] for t in tiles)
    grid = Image.fromarray(np.concatenate([t[:, :w] for t in tiles], axis=0))
    grid = grid.resize((grid.width // 2, grid.height // 2), Image.LANCZOS)
    titled(grid, ["photo", "MoGe-2 depth (red = near)", "MoGe-2 normal",
                  "error vs COLMAP points (green < 2%, yellow < 10%, red)"]).save(
        os.path.join(a.out, "viz.jpg"), quality=88)
    print(f"[mono] wrote {a.out}")


if __name__ == "__main__":
    main()
