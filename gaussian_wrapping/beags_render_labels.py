"""BEA-GS per-label renders, to check the labels by eye.

    python gaussian_wrapping/beags_render_labels.py -m <model> [--iteration N] [--n_views 8] [--out <dir>]

On --n_views cameras spread evenly over test + train, renders the full model and each label alone (white bg):
    <out>/full/<view>.png, <out>/label_<k>/<view>.png
    <out>/grid/<view>.png    GT | full | label_0 | label_1 | ...
    <out>/grid_all.jpg       every grid stacked, half resolution
Default <out>: <model>/../renders_labels_<it> (flat layout, next to the model dir).
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'submodules'))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from beags.common import load_model, parse_args
from gaussian_renderer.beags_api import render_labels


def add_args(parser):
    parser.add_argument("--n_views", default=8, type=int)
    parser.add_argument("--out", default=None, type=str)


@torch.no_grad()
def main():
    dataset, pipe, opt, args = parse_args("BEA-GS per-label renders", add_args)
    scene, gaussians = load_model(dataset, args.iteration, with_labels=True)
    out_dir = args.out or os.path.join(os.path.dirname(os.path.normpath(os.path.abspath(dataset.model_path))),
                                       f"renders_labels_{scene.loaded_iter}")
    cams = list(scene.getTestCameras()) + list(scene.getTrainCameras())
    cams = [cams[i] for i in np.unique(np.linspace(0, len(cams) - 1, min(args.n_views, len(cams))).round().astype(int))]
    num_labels = int(gaussians.labels.max()) + 1
    counts = torch.bincount(gaussians.labels[gaussians.labels >= 0].long(), minlength=num_labels).tolist()
    print(f"[INFO] {len(cams)} views, Gaussians per label: {counts}, writing to {out_dir}")

    white = torch.ones(3, device="cuda")
    masks = [("full", None)] + [(f"label_{k}", gaussians.labels == k) for k in range(num_labels)]
    for name, _ in masks + [("grid", None)]:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    grids = []
    for cam in cams:
        row = [cam.original_image[:3].cuda()]
        for name, mask in masks:
            img = render_labels(cam, gaussians, pipe, white, mask=mask)["render"].clamp(0, 1)
            torchvision.utils.save_image(img, os.path.join(out_dir, name, f"{cam.image_name}.png"))
            row.append(img)
        grid = torch.cat(row, dim=2)
        torchvision.utils.save_image(grid, os.path.join(out_dir, "grid", f"{cam.image_name}.png"))
        grids.append(F.interpolate(grid[None], scale_factor=0.5, mode="area")[0].cpu())
    width = min(g.shape[2] for g in grids)  # views can differ by a pixel after downscaling
    torchvision.utils.save_image(torch.cat([g[:, :, :width] for g in grids], dim=1), os.path.join(out_dir, "grid_all.jpg"))
    print(f"[INFO] Saved {len(cams)} views x {len(masks)} renders, grids in {os.path.join(out_dir, 'grid')} and grid_all.jpg")


if __name__ == "__main__":
    main()
