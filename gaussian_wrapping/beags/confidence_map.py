"""Per-view confidence maps of a GW model trained with --confidence: the ArtiFixer artifact map (plan 1b).

    python gaussian_wrapping/beags/confidence_map.py -m <model> [--iteration N] --out <dir> [--min_alpha 0.01]
        [--source_path <COLMAP>] [--images <folder>]

The per-Gaussian log-confidence is the GW PLY's `confidence` property (gaussian_model.save_ply, train.py --confidence).
Every camera of the scene (train + test: every photo ArtiFixer may see) is rendered with exp(confidence) as
colors_precomp (no CUDA change) and normalized by alpha, as the training loss reads it (robust/losses.py
render_confidence). Writes into --out (beags_artifixer.py: <out>/confidence; never inside the model dir):
    <image stem>.png  uint8 L, 255 * min(c, 1): c starts at 1 and the loss c L - beta log c pushes it below 1 where the
                      photos disagree with the model, so dark = low confidence = artifact, white = confident; pixels
                      with alpha < --min_alpha (nothing rendered, see-through: ArtiFixer's alpha hole) are 255
    info.json         source PLY, iteration, cameras, encoding, per-view mean c and fraction of pixels with c < 0.5
Only numpy / PIL at import (beags_artifixer.py reads ply_has_confidence and the encoding without torch or CUDA).
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

ENCODING = "png8: 255 * min(c, 1), c = rendered exp(confidence) / alpha; alpha < min_alpha -> 255"
LOW = 0.5  # info.json: fraction of pixels below this confidence


def ply_has_confidence(path):
    """True when the PLY's vertex element has a `confidence` property (header only, the body is not read)."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply", f"{path} is not a PLY"
        in_vertex = False
        for raw in iter(f.readline, b""):
            parts = raw.decode("ascii", "replace").split()
            if parts[:1] == ["end_header"]:
                break
            if parts[:1] == ["element"]:
                in_vertex = parts[1:2] == ["vertex"]
            elif in_vertex and parts[:1] == ["property"] and parts[-1] == "confidence":
                return True
    return False


def encode(conf, alpha, min_alpha=0.01):
    """(H, W) confidence and alpha -> uint8 (H, W), see ENCODING."""
    c = np.clip(np.nan_to_num(np.asarray(conf, dtype=np.float32), nan=1.0), 0.0, 1.0)
    c = np.where(np.asarray(alpha) < min_alpha, 1.0, c)
    return np.round(c * 255).astype(np.uint8)


def decode(png):
    """uint8 map -> confidence in [0, 1]."""
    return np.asarray(png, dtype=np.float32) / 255.0


def render_view(cam, gaussians, pipe, render_fn):
    """(confidence, alpha), both (H, W) tensors: R(exp(conf)) / max(alpha, 1e-3) and alpha (render_confidence's rule)."""
    import torch
    conf = gaussians.get_confidence.repeat(1, 3)
    bg = torch.zeros(3, device=conf.device)
    pkg = render_fn(cam, gaussians, pipe, bg, colors_precomp=conf, require_depth=False)
    alpha = pkg["alpha"][0]
    return pkg["render"][0] / alpha.clamp_min(1e-3), alpha


def write_maps(cams, gaussians, pipe, out_dir, render_fn, min_alpha=0.01):
    """<out_dir>/<image_name>.png for every camera; per-view stats {image_name: {mean, low_fraction}}."""
    import torch
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    with torch.no_grad():
        for cam in cams:
            conf, alpha = render_view(cam, gaussians, pipe, render_fn)
            png = encode(conf.float().cpu().numpy(), alpha.float().cpu().numpy(), min_alpha)
            Image.fromarray(png).save(os.path.join(out_dir, f"{cam.image_name}.png"))
            c = decode(png)
            stats[cam.image_name] = {"mean": round(float(c.mean()), 4), "low_fraction": round(float((c < LOW).mean()), 4)}
    return stats


def write_info(out_dir, stats, **fields):
    """<out_dir>/info.json: fields (source, iteration, source_path, images, ...) + summary of write_maps' stats."""
    info = {**fields, "cameras": len(stats), "encoding": ENCODING, "low_threshold": LOW,
            "mean": float(np.mean([s["mean"] for s in stats.values()])),
            "low_fraction_mean": float(np.mean([s["low_fraction"] for s in stats.values()])), "views": stats}
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def add_args(parser):
    parser.add_argument("--out", required=True, help="output dir: <image stem>.png + info.json")
    parser.add_argument("--min_alpha", type=float, default=0.01, help="below this alpha a pixel is written as confident (255)")


def main():
    from beags.common import load_model, parse_args
    from gaussian_renderer.ours import render_ours

    dataset, pipe, _, args = parse_args("GW per-view confidence maps (ArtiFixer artifact map)", add_args)
    out_dir = os.path.abspath(args.out)
    model = os.path.normpath(os.path.abspath(dataset.model_path))
    assert not out_dir.startswith(model + os.sep), f"{out_dir} is inside the model dir"
    scene, gaussians = load_model(dataset, args.iteration, with_labels=False)
    ply = os.path.join(model, "point_cloud", f"iteration_{scene.loaded_iter}", "point_cloud.ply")
    assert gaussians.use_confidence, f"{ply} has no per-Gaussian confidence (train.py --confidence)"
    cams = list(scene.getTrainCameras()) + list(scene.getTestCameras())
    assert cams, f"no cameras in {dataset.source_path}"
    print(f"[INFO] confidence maps of {len(cams)} cameras -> {out_dir}")
    stats = write_maps(cams, gaussians, pipe, out_dir, render_ours, args.min_alpha)
    info = write_info(out_dir, stats, source=ply, iteration=scene.loaded_iter, source_path=dataset.source_path,
                      images=dataset.images, train_cameras=len(scene.getTrainCameras()), min_alpha=args.min_alpha)
    print(f"[INFO] {len(stats)} maps: mean confidence {info['mean']:.3f}, "
          f"{100 * info['low_fraction_mean']:.2f}% of pixels below {LOW}")


if __name__ == "__main__":
    main()
