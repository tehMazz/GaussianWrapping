"""Per-view "unseen" maps after removing object k: the ArtiFixer hole of HOLE_MODE=gavis (GAVIS visibility field).

    python gaussian_wrapping/beags/unseen_map.py -m <model> [--iteration N] --remove <k> --out <dir> [--kappa 1]
        [--vis] [--source_path <COLMAP>] [--images <folder>]

The AVF (beags/visibility.py) is built on the FULL model (object present) from the training cameras, so whatever the
object hid gets a low visibility. Every camera (train + test: every photo ArtiFixer may see) is then rendered without
label k (labels != k and >= 0), with colors_precomp = 1 - v(d) along that camera's view direction, on black:
    U = sum_i w_i (1 - v_i(d)) + (1 - alpha)
the colour mass that comes from Gaussians the training views never saw from this direction, plus empty space (the
hollowing leaves unseen space empty, so 1 - alpha stands in for GAVIS's virtual particles).
Writes into --out (beags_artifixer.py: <out>/unseen; never inside the model dir):
    <image stem>.png  uint8 L, 255 * clip(U, 0, 1): white = unseen = inpaint, black = seen
    avf.pt            the AVF (gamma (P, 9), n_views, kappa, source, iteration), reused while those match
    info.json         source PLY, iteration, label, kappa, cameras, encoding, per-view mean U and fraction of U > 0.5
    vis/              with --vis: photo | removal render | U heatmap for up to 8 views
Only numpy / PIL at import (beags_artifixer.py reads the encoding without torch or CUDA).
"""
import json
import os
import sys

if __name__ == "__main__":  # run as a file: gaussian_wrapping/ on sys.path (not beags/), as the other scripts
    _GW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[0] = _GW
    sys.path.append(os.path.dirname(_GW))
    sys.path.append(os.path.join(os.path.dirname(_GW), "submodules"))

import numpy as np

ENCODING = "png8: 255 * clip(U, 0, 1), U = rendered (1 - v) + (1 - alpha) without the removed label; high = unseen"
HIGH = 0.5  # info.json: fraction of pixels above this


def encode(unseen):
    """(H, W) unseen score -> uint8 (H, W), see ENCODING."""
    u = np.clip(np.nan_to_num(np.asarray(unseen, dtype=np.float32), nan=1.0), 0.0, 1.0)
    return np.round(u * 255).astype(np.uint8)


def decode(png):
    """uint8 map -> unseen score in [0, 1]."""
    return np.asarray(png, dtype=np.float32) / 255.0


def keep_mask(labels, remove):
    """(P,) bool: the Gaussians left after removing label `remove` (label -1 = never hit, also dropped)."""
    return (labels != remove) & (labels >= 0)


def render_view(cam, gaussians, pipe, avf, keep, render_fn):
    """(unseen, rgb, alpha) of one camera without the removed Gaussians: unseen and alpha (H, W), rgb (3, H, W)."""
    import torch
    from beags.visibility import query_avf
    v = query_avf(avf, gaussians.get_xyz, cam.camera_center)
    feat = torch.stack([1 - v, torch.zeros_like(v), torch.zeros_like(v)], dim=1)
    bg = torch.zeros(3, device=v.device)
    pkg = render_fn(cam, gaussians, pipe, bg, colors_precomp=feat, mask=keep, require_depth=False)
    alpha = pkg["alpha"][0]
    rgb = render_fn(cam, gaussians, pipe, bg, mask=keep, require_depth=False)["render"]
    return pkg["render"][0] + (1 - alpha), rgb, alpha


def heatmap(u):
    """(H, W) in [0, 1] -> (H, W, 3) uint8, black -> red -> yellow -> white."""
    u = np.clip(u, 0, 1)[..., None]
    rgb = np.concatenate([np.clip(3 * u, 0, 1), np.clip(3 * u - 1, 0, 1), np.clip(3 * u - 2, 0, 1)], axis=-1)
    return np.round(rgb * 255).astype(np.uint8)


def write_maps(cams, gaussians, pipe, avf, keep, out_dir, render_fn, vis_names=()):
    """<out_dir>/<image_name>.png for every camera (+ vis/ rows for vis_names); stats {image_name: {mean, high_fraction}}."""
    import torch
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    with torch.no_grad():
        for cam in cams:
            unseen, rgb, _ = render_view(cam, gaussians, pipe, avf, keep, render_fn)
            png = encode(unseen.float().cpu().numpy())
            Image.fromarray(png).save(os.path.join(out_dir, f"{cam.image_name}.png"))
            u = decode(png)
            stats[cam.image_name] = {"mean": round(float(u.mean()), 4), "high_fraction": round(float((u > HIGH).mean()), 4)}
            if cam.image_name in vis_names:
                os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)
                to8 = lambda t: np.round(t.clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                row = [to8(cam.original_image[:3]), to8(rgb), heatmap(u)]
                Image.fromarray(np.concatenate(row, axis=1)).save(os.path.join(out_dir, "vis", f"{cam.image_name}.jpg"))
    return stats


def write_info(out_dir, stats, **fields):
    """<out_dir>/info.json: fields (source, iteration, label, kappa, ...) + summary of write_maps' stats."""
    info = {**fields, "cameras": len(stats), "encoding": ENCODING, "high_threshold": HIGH,
            "mean": float(np.mean([s["mean"] for s in stats.values()])),
            "high_fraction_mean": float(np.mean([s["high_fraction"] for s in stats.values()])), "views": stats}
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def add_args(parser):
    parser.add_argument("--out", required=True, help="output dir: <image stem>.png + avf.pt + info.json")
    parser.add_argument("--remove", type=int, required=True, help="label k of the removed object")
    parser.add_argument("--kappa", type=float, default=1.0, help="vMF concentration around each training direction")
    parser.add_argument("--vis", action="store_true", help="also write vis/ (photo | removal | U) for 8 views")


def main():
    from beags.common import load_model, parse_args
    from beags.visibility import build_avf, load_avf, save_avf
    from gaussian_renderer.ours import render_ours

    dataset, pipe, _, args = parse_args("GW per-view unseen maps after a removal (GAVIS visibility)", add_args)
    out_dir = os.path.abspath(args.out)
    model = os.path.normpath(os.path.abspath(dataset.model_path))
    assert not out_dir.startswith(model + os.sep), f"{out_dir} is inside the model dir"
    scene, gaussians = load_model(dataset, args.iteration, with_labels=True)
    ply = os.path.join(model, "point_cloud", f"iteration_{scene.loaded_iter}", "point_cloud.ply")
    labels = gaussians.labels.reshape(-1)
    keep = keep_mask(labels, args.remove)
    assert (labels == args.remove).any(), f"no Gaussian has label {args.remove}"
    train = list(scene.getTrainCameras())
    cams = train + list(scene.getTestCameras())
    os.makedirs(out_dir, exist_ok=True)

    avf_path = os.path.join(out_dir, "avf.pt")
    expect = dict(source=ply, iteration=scene.loaded_iter, kappa=float(args.kappa), n_views=len(train),
                  num_gaussians=int(labels.numel()))
    avf = load_avf(avf_path, **expect)
    if avf is not None and os.path.getmtime(avf_path) < os.path.getmtime(ply):
        avf = None
    if avf is None:
        print(f"[INFO] AVF of {labels.numel()} Gaussians from {len(train)} training cameras (kappa {args.kappa})")
        avf = build_avf(gaussians, train, pipe, kappa=args.kappa)
        save_avf(avf_path, avf, **expect)
    else:
        print(f"[INFO] reusing {avf_path}")

    vis_names = {c.image_name for c in cams[:: max(1, len(cams) // 8)][:8]} if args.vis else set()
    print(f"[INFO] unseen maps of {len(cams)} cameras without label {args.remove} "
          f"({int((~keep).sum())} of {keep.numel()} Gaussians removed) -> {out_dir}")
    stats = write_maps(cams, gaussians, pipe, avf, keep, out_dir, render_ours, vis_names)
    info = write_info(out_dir, stats, source=ply, iteration=scene.loaded_iter, label=args.remove, kappa=args.kappa,
                      source_path=dataset.source_path, images=dataset.images, train_cameras=len(train))
    print(f"[INFO] {len(stats)} maps: mean U {info['mean']:.3f}, "
          f"{100 * info['high_fraction_mean']:.2f}% of pixels above {HIGH}")


if __name__ == "__main__":
    main()
