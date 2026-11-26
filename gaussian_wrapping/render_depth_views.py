"""Depth and normals of a GW model next to the photos, to judge geometry quality by eye.

    python gaussian_wrapping/render_depth_views.py -m <model> [--iteration N] [--n_views 6] [--out <dir>]

Views: --n_views cameras spread evenly over test + train (beags_render_labels.py rule).
<out>/grid/<view>.png = photo | median depth (turbo, near = red, magenta = depth 0) | rendered normal | normal from
the depth map (finite differences, black where invalid). The two normal maps agree where the geometry is a clean
surface; noise, terracing or holes in the last column are geometry problems. <out>/stats.json: per view, depth-0
share and the median angle between the two normals.
Which rasterizer build is used follows PYTHONPATH (RASTER_PATH in slurm/common.sh).
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'submodules'))

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402

from beags.common import load_model, parse_args  # noqa: E402
from gaussian_renderer.ours import render_ours  # noqa: E402
from utils.geometry_utils import depth_to_normal_with_mask  # noqa: E402
from utils.grid_titles import add_column_titles  # noqa: E402

TITLES = ["photo", "median depth (red = near, magenta = none)", "rendered normal", "normal from depth"]


def add_args(parser):
    parser.add_argument("--n_views", default=6, type=int)
    parser.add_argument("--out", default=None)


def turbo(depth, valid):
    v = depth[valid]
    lo, hi = (torch.quantile(v, 0.02).item(), torch.quantile(v, 0.98).item()) if v.numel() else (0.0, 1.0)
    x = ((depth - lo) / max(hi - lo, 1e-6)).clamp(0, 1).cpu().numpy()
    rgb = torch.from_numpy(cm.turbo(1 - x)[..., :3]).float().permute(2, 0, 1)
    rgb[:, ~valid.cpu()] = torch.tensor([1.0, 0.0, 1.0])[:, None]
    return rgb.cuda()


@torch.no_grad()
def main():
    dataset, pipe, opt, args = parse_args("render depth views", add_args)
    scene, gaussians = load_model(dataset, args.iteration, with_labels=False)
    out = args.out or os.path.join(os.path.dirname(os.path.normpath(os.path.abspath(dataset.model_path))),
                                   f"depth_views_{scene.loaded_iter}")
    os.makedirs(os.path.join(out, "grid"), exist_ok=True)
    cams = list(scene.getTestCameras()) + list(scene.getTrainCameras())
    cams = [cams[i] for i in np.unique(np.linspace(0, len(cams) - 1, min(args.n_views, len(cams))).round().astype(int))]
    import diff_gaussian_rasterization_ours_beags as rast
    print(f"[depth] rasterizer {os.path.dirname(rast.__file__)}, {len(cams)} views -> {out}")
    rows, small = [], []
    for cam in cams:
        pkg = render_ours(cam, gaussians, pipe, torch.zeros(3, device="cuda"), require_depth=True)
        depth = torch.nan_to_num(pkg["median_depth"], 0.0)                 # (1, H, W)
        valid = depth[0] > 0
        n_rend = F.normalize(pkg["normal"], dim=0)                          # (3, H, W)
        n_depth, n_ok = depth_to_normal_with_mask(cam, depth)
        n_ok = n_ok[0]
        cos = (n_rend * n_depth).sum(0).abs().clamp(0, 1)
        angle = torch.rad2deg(torch.acos(cos[n_ok])) if bool(n_ok.any()) else torch.zeros(1, device="cuda")
        photo = cam.original_image[:3].cuda()
        grid = torch.cat([photo, turbo(depth[0], valid), n_rend * 0.5 + 0.5,
                          torch.where(n_ok[None], n_depth * 0.5 + 0.5, torch.zeros_like(n_depth))], dim=2).clamp(0, 1)
        torchvision.utils.save_image(grid, os.path.join(out, "grid", f"{cam.image_name}.png"))
        add_column_titles(os.path.join(out, "grid", f"{cam.image_name}.png"), TITLES)
        small.append(F.interpolate(grid[None], scale_factor=0.5, mode="area")[0].cpu())
        row = {"view": cam.image_name, "depth0_share": float((~valid).float().mean()),
               "normal_angle_deg_p50": float(angle.median()), "normal_angle_deg_p90": float(torch.quantile(angle, 0.9))}
        rows.append(row)
        print(f"[depth] {json.dumps(row)}")
    width = min(g.shape[2] for g in small)
    torchvision.utils.save_image(torch.cat([g[:, :, :width] for g in small], dim=1), os.path.join(out, "grid_all.jpg"))
    add_column_titles(os.path.join(out, "grid_all.jpg"), TITLES)
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
