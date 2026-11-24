"""
gsplat-vs-native render equivalence of an exported 3DGS PLY (plan metric 6): the gsplat side and the shared pieces.

export_3dgs_ply.py --check renders the views natively (render_ours, at the exported iteration). When gsplat is
importable there it compares in-process; otherwise (the GW env has no gsplat) it dumps the native views and runs
this script in a Python with torch + gsplat (--gs_python, default $GS_RENDER_PY):

    $GS_RENDER_PY gaussian_wrapping/gsplat_check.py --views <dump dir> [--out <json>] [--max_drop 0.5]

Dump layout (dump_view / write_manifest; default dump dir <export dir>/gsplat_check_views, removed afterwards):
    manifest.json   ply, iteration, eps2d, rasterize_mode, near_plane, background [3], views: [{name, split,
                    image_name, file, width, height, viewmat (4x4 OpenCV world-to-camera), K (3x3)}]
    <name>.npz      native rgb (H, W, 3) and alpha (H, W) as uint16 (1/65535 steps), photo gt (H, W, 3) as uint8
The PLY is read with artifixer_gs.render.read_ply (submodules/artifixer_gs) and rasterized as artifixer_gs.render
does, with GW's settings: eps2d = kernel_size (0 for GW, whose 3D filter is baked into the PLY; antialiased when
> 0, which matches GW's opacity compensation), near plane 0.2 (GW's near culling), the native background.
Per view:
    psnr                            gsplat vs native RGB, both clamped to [0, 1] (capped at 120 dB)
    alpha_mae                       mean |alpha_gsplat - alpha_native|
    psnr_gt_native, psnr_gt_gsplat  each against the photo; gt_drop = psnr_gt_native - psnr_gt_gsplat
Report (--out, default <dump dir>/../gsplat_check.json): means overall and per split, worst view, per-view rows,
and pass = mean gt_drop <= --max_drop (plan target: gsplat within 0.5 dB of native).
--dry_run compares the dump with itself (no gsplat, CPU): it tests the dump IO and the PLY loading only.
Module-level imports are numpy and the standard library only, so the GW env imports this file too.
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
ARTIFIXER_DIR = os.path.join(ROOT_DIR, 'submodules', 'artifixer_gs')

import json
import math
from argparse import ArgumentParser
from typing import Dict, Iterable, List, Optional

import numpy as np

MANIFEST = "manifest.json"
REPORT = "gsplat_check.json"
GW_NEAR_PLANE = 0.2  # GW culls p_view.z <= 0.2 (cuda_rasterizer/auxiliary.h)
PSNR_CAP_MSE = 1e-12  # identical renders -> 120 dB
DEFAULT_MAX_DROP = 0.5


# ---------------------------------------------------------------- settings and IO
def raster_settings(kernel_size: float) -> Dict:
    """gsplat settings matching render_ours: GW adds kernel_size to the 2D covariance with opacity compensation."""
    kernel_size = float(kernel_size)
    return {
        "eps2d": kernel_size,
        "rasterize_mode": "antialiased" if kernel_size > 0. else "classic",
        "near_plane": GW_NEAR_PLANE,
    }


def to_uint16(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x, 0., 1.) * 65535.).astype(np.uint16)


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x, 0., 1.) * 255.).astype(np.uint8)


def dump_view(dump_dir: str, name: str, rgb: np.ndarray, alpha: np.ndarray, gt: Optional[np.ndarray] = None) -> str:
    """Write one native view; rgb (H, W, 3), alpha (H, W), gt (H, W, 3) in [0, 1]. Returns the file name."""
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and alpha.shape == rgb.shape[:2], (rgb.shape, alpha.shape)
    arrays = {"rgb": to_uint16(rgb), "alpha": to_uint16(alpha)}
    if gt is not None:
        assert gt.shape == rgb.shape, (gt.shape, rgb.shape)
        arrays["gt"] = to_uint8(gt)
    file = f"{name}.npz"
    np.savez_compressed(os.path.join(dump_dir, file), **arrays)
    return file


def load_view(dump_dir: str, file: str) -> Dict[str, np.ndarray]:
    """float32 rgb (H, W, 3), alpha (H, W) and gt (H, W, 3) or None."""
    with np.load(os.path.join(dump_dir, file)) as data:
        view = {"rgb": data["rgb"].astype(np.float32) / 65535., "alpha": data["alpha"].astype(np.float32) / 65535.}
        view["gt"] = data["gt"].astype(np.float32) / 255. if "gt" in data.files else None
    return view


def view_entry(name: str, split: str, image_name: str, file: str, width: int, height: int,
               viewmat: np.ndarray, K: np.ndarray) -> Dict:
    return {
        "name": name, "split": split, "image_name": image_name, "file": file, "width": int(width), "height": int(height),
        "viewmat": np.asarray(viewmat, dtype=np.float64).reshape(4, 4).tolist(),
        "K": np.asarray(K, dtype=np.float64).reshape(3, 3).tolist(),
    }


def write_manifest(dump_dir: str, meta: Dict, views: List[Dict]):
    with open(os.path.join(dump_dir, MANIFEST), "w") as f:
        json.dump({**meta, "views": views}, f, indent=1)


def read_manifest(dump_dir: str) -> Dict:
    with open(os.path.join(dump_dir, MANIFEST)) as f:
        return json.load(f)


# ---------------------------------------------------------------- comparison
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10. * math.log10(1. / max(mse, PSNR_CAP_MSE))


RADIAL_BINS = 4


def radial_mse(a: np.ndarray, b: np.ndarray, bins: int = RADIAL_BINS) -> List[float]:
    """MSE of (H, W, 3) a vs b in `bins` rings of distance from the image centre (0 = centre, 1 = corner).
    Depth-sort differences (GW sorts by distance to the camera, gsplat by view z) grow towards the border."""
    h, w = a.shape[:2]
    y, x = np.mgrid[:h, :w]
    r = np.hypot(x - (w - 1) / 2, y - (h - 1) / 2) / np.hypot((w - 1) / 2, (h - 1) / 2)
    ring = np.minimum((r * bins).astype(int), bins - 1)
    err = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean(-1)
    return [float(err[ring == k].mean()) for k in range(bins)]


def compare_view(native_rgb: np.ndarray, native_alpha: np.ndarray, gsplat_rgb: np.ndarray, gsplat_alpha: np.ndarray,
                 gt: Optional[np.ndarray] = None) -> Dict:
    """Metrics of one view; rgb (H, W, 3) and alpha (H, W), clamped to [0, 1] here."""
    assert native_rgb.shape == gsplat_rgb.shape and native_alpha.shape == gsplat_alpha.shape, \
        (native_rgb.shape, gsplat_rgb.shape, native_alpha.shape, gsplat_alpha.shape)
    native_rgb, gsplat_rgb = np.clip(native_rgb, 0., 1.), np.clip(gsplat_rgb, 0., 1.)
    row = {
        "psnr": psnr(native_rgb, gsplat_rgb),
        "alpha_mae": float(np.mean(np.abs(native_alpha.astype(np.float64) - gsplat_alpha.astype(np.float64)))),
        "mse_radial": radial_mse(native_rgb, gsplat_rgb),
    }
    if gt is not None:
        row["psnr_gt_native"] = psnr(native_rgb, gt)
        row["psnr_gt_gsplat"] = psnr(gsplat_rgb, gt)
        row["gt_drop"] = row["psnr_gt_native"] - row["psnr_gt_gsplat"]
    return row


METRICS = ("psnr", "alpha_mae", "psnr_gt_native", "psnr_gt_gsplat", "gt_drop")


def _means(rows: List[Dict]) -> Dict:
    out = {"n_views": len(rows)}
    for key in METRICS:
        values = [row[key] for row in rows if key in row]
        if len(values) > 0:
            out[key] = float(np.mean(values))
    radial = [row["mse_radial"] for row in rows if "mse_radial" in row]
    if radial:
        out["mse_radial"] = np.mean(radial, axis=0).tolist()  # gsplat vs native, centre -> corner rings
    return out


def summarize(meta: Dict, rows: List[Dict], max_drop: float = DEFAULT_MAX_DROP) -> Dict:
    """Report: meta, overall and per-split means, worst view (lowest psnr), pass, and the per-view rows."""
    assert len(rows) > 0, "no views to compare"
    report = {key: value for key, value in meta.items() if key != "views"}
    report.update(_means(rows))
    report["splits"] = {split: _means([row for row in rows if row.get("split") == split])
                        for split in dict.fromkeys(row.get("split") for row in rows)}
    worst = min(rows, key=lambda row: row["psnr"])
    report["worst"] = {key: worst[key] for key in ("name", "split", "image_name", "psnr", "alpha_mae") if key in worst}
    report["max_drop"] = float(max_drop)
    report["pass"] = bool(report["gt_drop"] <= max_drop) if "gt_drop" in report else None
    report["views"] = rows
    return report


def write_report(path: str, report: Dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def format_report(report: Dict) -> str:
    what = "exported PLY (render_ours)" if report.get("backend") == "ours_ply" else "gsplat"
    text = (f"{what} vs render_ours on {report['n_views']} views ({report.get('backend', '?')}): "
            f"PSNR {report['psnr']:.2f} dB, alpha MAE {report['alpha_mae']:.5f}")
    if "gt_drop" in report:
        text += (f"; vs photos native {report['psnr_gt_native']:.2f} / {what.split()[0]} {report['psnr_gt_gsplat']:.2f} dB "
                 f"(drop {report['gt_drop']:+.3f}, target <= {report['max_drop']}: {'PASS' if report['pass'] else 'FAIL'})")
    return text


# ---------------------------------------------------------------- gsplat (torch + gsplat env)
def gsplat_available() -> bool:
    try:
        import gsplat  # noqa: F401
    except ImportError:
        return False
    return True


def load_ply_for_gsplat(path: str, device: str = "cuda") -> Dict:
    """artifixer_gs.render.read_ply as torch tensors on device (keys means, quats, scales, opacities, sh) + sh_degree."""
    import torch
    if ARTIFIXER_DIR not in sys.path:
        sys.path.append(ARTIFIXER_DIR)
    from artifixer_gs.render import read_ply
    from pathlib import Path

    gaussians = read_ply(Path(path))
    sh_degree = int(gaussians.pop("sh_degree"))
    out = {key: torch.from_numpy(np.ascontiguousarray(value)).to(device) for key, value in gaussians.items()}
    out["sh_degree"] = sh_degree
    return out


def gsplat_render(gaussians: Dict, viewmat: np.ndarray, K: np.ndarray, width: int, height: int, background,
                  eps2d: float, rasterize_mode: str, near_plane: float = GW_NEAR_PLANE):
    """One view with gsplat (as artifixer_gs.render.render_frames). Returns float32 rgb (H, W, 3), alpha (H, W)."""
    import torch
    from gsplat import rasterization

    device = gaussians["means"].device
    with torch.no_grad():
        colors, alphas, _ = rasterization(
            means=gaussians["means"],
            quats=gaussians["quats"],
            scales=gaussians["scales"],
            opacities=gaussians["opacities"],
            colors=gaussians["sh"],
            viewmats=torch.as_tensor(np.asarray(viewmat), dtype=torch.float32, device=device).reshape(1, 4, 4),
            Ks=torch.as_tensor(np.asarray(K), dtype=torch.float32, device=device).reshape(1, 3, 3),
            width=int(width),
            height=int(height),
            sh_degree=int(gaussians["sh_degree"]),
            near_plane=float(near_plane),
            eps2d=float(eps2d),
            rasterize_mode=rasterize_mode,
            backgrounds=torch.as_tensor(background, dtype=torch.float32, device=device).reshape(1, 3),
        )
    return colors[0, ..., :3].clamp(0., 1.).float().cpu().numpy(), alphas[0, ..., 0].float().cpu().numpy()


def check_dump(dump_dir: str, out: Optional[str] = None, max_drop: float = DEFAULT_MAX_DROP, device: str = "cuda",
               dry_run: bool = False) -> Dict:
    """Render every view of a dump with gsplat, compare with the native renders, write and return the report."""
    manifest = read_manifest(dump_dir)
    out = out or os.path.join(os.path.dirname(os.path.abspath(dump_dir)), REPORT)
    gaussians = load_ply_for_gsplat(manifest["ply"], device="cpu" if dry_run else device)
    print(f"[INFO] {manifest['ply']}: {gaussians['means'].shape[0]} Gaussians, SH {gaussians['sh_degree']}, "
          f"eps2d {manifest['eps2d']} ({manifest['rasterize_mode']}), {len(manifest['views'])} views", flush=True)
    rows = []
    for view in manifest["views"]:
        native = load_view(dump_dir, view["file"])
        if dry_run:
            rgb, alpha = native["rgb"], native["alpha"]
        else:
            rgb, alpha = gsplat_render(gaussians, view["viewmat"], view["K"], view["width"], view["height"],
                                       manifest["background"], manifest["eps2d"], manifest["rasterize_mode"],
                                       manifest.get("near_plane", GW_NEAR_PLANE))
        rows.append({"name": view["name"], "split": view["split"], "image_name": view.get("image_name"),
                     **compare_view(native["rgb"], native["alpha"], rgb, alpha, native["gt"])})
    report = summarize({**manifest, "backend": "dry_run" if dry_run else "subprocess"}, rows, max_drop)
    write_report(out, report)
    print(f"[INFO] {format_report(report)}\n[INFO] Saved {out}", flush=True)
    return report


if __name__ == "__main__":
    # Keep gaussian_wrapping/ (scene/, utils/, ...) off sys.path: this runs in another env.
    sys.path = [p for p in sys.path if os.path.abspath(p or os.curdir) != BASE_DIR]
    parser = ArgumentParser(description="gsplat side of export_3dgs_ply.py --check (run in a torch + gsplat env)")
    parser.add_argument("--views", required=True, type=str, help="Dump dir written by export_3dgs_ply.py --check")
    parser.add_argument("--out", default=None, type=str, help=f"Report JSON. Default: <views>/../{REPORT}")
    parser.add_argument("--max_drop", default=DEFAULT_MAX_DROP, type=float,
                        help="Pass if mean PSNR vs photos drops by at most this (dB) from native to gsplat")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dry_run", action="store_true", help="Compare the dump with itself (no gsplat; IO test)")
    args = parser.parse_args()
    check_dump(args.views, args.out, args.max_drop, args.device, args.dry_run)
