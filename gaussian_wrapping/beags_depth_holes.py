"""Where does render_depth_alpha return depth 0 inside an object mask, and why (alpha < 0.5, or median depth 0)?

    python gaussian_wrapping/beags_depth_holes.py -m <model> --iteration N --masks <SAM2 dir> [--label 1] [--n_views 6]
           [--out <dir>]

The mask consensus (beags/consensus.py) drops every pixel where the target view's own depth is 0, so these holes
decide how much of the object the consensus keeps. Prints, per view and overall, inside the label's input mask:
    alpha_lt_0p5   rendered alpha < 0.5           (render_depth_alpha zeroes the depth there)
    median_zero    raw median_depth <= 0 or NaN    (the rasterizer found no median surface)
    hole           depth 0 after render_depth_alpha
and saves <out>/<view>_{alpha,depth,hole}.png for the first views (hole: red = alpha, blue = median, magenta = both).
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'submodules'))

import numpy as np
import torch
import torchvision

from beags.common import load_model, parse_args
from beags.masks import attach_masks
from gaussian_renderer.ours import render_ours


def add_args(parser):
    parser.add_argument("--masks", required=True)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--n_views", type=int, default=6)
    parser.add_argument("--save_views", type=int, default=2)
    parser.add_argument("--out", default="")
    parser.add_argument("--save_depth", action="store_true", help="also <out>/depth.npz: raw median depth per view")


@torch.no_grad()
def main():
    dataset, pipe, opt, args = parse_args("depth holes", add_args)
    scene, gaussians = load_model(dataset, args.iteration, with_labels=False)
    cams = list(scene.getTrainCameras())
    attach_masks(cams, args.masks, 1.0, device="cpu")
    cams = [c for c in cams if c.masks.labels is not None and bool((c.masks.labels == args.label).any())]
    cams = [cams[i] for i in np.unique(np.linspace(0, len(cams) - 1, min(args.n_views, len(cams))).round().astype(int))]
    out = args.out or os.path.join(os.path.dirname(os.path.normpath(dataset.model_path)), "depth_holes")
    os.makedirs(out, exist_ok=True)
    tot = {"pixels": 0, "alpha_lt_0p5": 0, "median_zero": 0, "hole": 0}
    rows, depths = [], {}
    for v, cam in enumerate(cams):
        pkg = render_ours(cam, gaussians, pipe, torch.zeros(3, device="cuda"), require_depth=True)
        depth, alpha = pkg["median_depth"][0], pkg["alpha"][0]
        obj = (cam.masks.labels == args.label).cuda()
        a = alpha < 0.5
        m = ~(depth > 0)                      # <= 0 or NaN
        hole = a | m
        if args.save_depth:
            depths[cam.image_name] = torch.nan_to_num(depth, 0.0).cpu().numpy().astype(np.float32)
        row = {"view": cam.image_name, "pixels": int(obj.sum())}
        for k, t in (("alpha_lt_0p5", a), ("median_zero", m), ("hole", hole)):
            row[k] = int((t & obj).sum())
        row["alpha_in_obj_pct10_50"] = [float(x) for x in torch.quantile(alpha[obj].float(), torch.tensor(
            [0.1, 0.5], device="cuda"))]
        rows.append(row)
        for k in tot:
            tot[k] += row[k]
        print(f"[holes] {json.dumps(row)}")
        if v < args.save_views:
            torchvision.utils.save_image(alpha[None], os.path.join(out, f"{cam.image_name}_alpha.png"))
            d = torch.nan_to_num(depth, 0.0)
            d = d / d[obj].max().clamp_min(1e-6) if bool(obj.any()) else d
            torchvision.utils.save_image(d.clamp(0, 1)[None], os.path.join(out, f"{cam.image_name}_depth.png"))
            vis = torch.stack([a & obj, torch.zeros_like(obj), m & obj]).float()
            vis = vis + 0.25 * obj[None].float() * (1 - vis.amax(0, keepdim=True))
            torchvision.utils.save_image(vis, os.path.join(out, f"{cam.image_name}_hole.png"))
    frac = {k: tot[k] / max(1, tot["pixels"]) for k in ("alpha_lt_0p5", "median_zero", "hole")}
    print(f"[holes] inside label {args.label} over {len(cams)} views: {json.dumps(frac)}")
    if args.save_depth:
        np.savez_compressed(os.path.join(out, "depth.npz"), **depths)
    import diff_gaussian_rasterization_ours_beags as rast
    print(f"[holes] rasterizer: {os.path.dirname(rast.__file__)}")
    with open(os.path.join(out, "holes.json"), "w") as f:
        json.dump({"overall": frac, "views": rows}, f, indent=2)


if __name__ == "__main__":
    main()
