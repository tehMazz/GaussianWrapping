"""BEA-GS step 1 on a trained GW model (2dgs train_base.py @30k + create_pointcloud_multi.py).

    python gaussian_wrapping/beags_prepare.py -m <model> --masks <SAM2 mask dir>

1. multi-view mask consensus through the rendered median depth -> <model>/beags/masks
2. per-Gaussian labels by voting with the consensus masks -> point_cloud/iteration_<it>/labels.pt (-1 = never hit)
3. labelled point cloud of the objects -> <model>/beags/labeled_points.npz
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

from beags.common import beags_path, labels_path, load_model, parse_args
from beags.consensus import iter_reproject_masks
from beags.labels import assign_labels, labeled_point_cloud
from beags.masks import attach_masks, make_view_masks
from gaussian_renderer.beags_api import render_depth_alpha


def add_args(parser):
    parser.add_argument("--masks", required=True, help="dir with <image_name>.npy (C, H, W) masks")
    parser.add_argument("--reprojection_threshold", type=float, default=0.01)


@torch.no_grad()
def consensus_masks(cameras, gaussians, pipe, num_labels, threshold):
    depths = [render_depth_alpha(c, gaussians, pipe)[0].detach().cpu() for c in cameras]
    H = min(depth.shape[0] for depth in depths)
    W = min(depth.shape[1] for depth in depths)
    depths = [depth[:H, :W].contiguous() for depth in depths]
    masks = [None if c.masks.soft is None else c.masks.soft[:, :H, :W].contiguous() for c in cameras]
    present = torch.stack([c.masks.present for c in cameras])
    for camera, soft in zip(cameras, iter_reproject_masks(cameras, depths, masks, present, threshold)):
        if soft.shape[-2:] != (camera.image_height, camera.image_width):
            soft = F.interpolate(soft[None], size=(camera.image_height, camera.image_width),
                                 mode="bilinear", align_corners=False)[0]
        yield soft


def main():
    dataset, pipe, opt, args = parse_args("BEA-GS prepare", add_args)
    scene, gaussians = load_model(dataset, args.iteration, with_labels=False)
    cameras = scene.getTrainCameras()
    num_labels = attach_masks(cameras, args.masks)

    print("[beags] mask consensus")
    out_dir = beags_path(dataset.model_path, "masks")
    os.makedirs(out_dir, exist_ok=True)
    for cam, soft in zip(cameras, consensus_masks(cameras, gaussians, pipe, num_labels, args.reprojection_threshold)):
        np.save(os.path.join(out_dir, f"{cam.image_name}.npy"), soft.half().numpy())
        cam.masks = make_view_masks(soft, cam.image_height, cam.image_width, num_labels, 1.0)

    print("[beags] label assignment")
    labels = assign_labels(gaussians, cameras, pipe, num_labels)
    torch.save(labels, labels_path(dataset.model_path, scene.loaded_iter))
    print(f"[beags] Gaussians per label: {torch.bincount(labels[labels >= 0].long(), minlength=num_labels).tolist()}, "
          f"never hit: {int((labels < 0).sum())}")

    print("[beags] labelled point cloud")
    xyz, lab = labeled_point_cloud(gaussians, cameras, pipe, labels)
    np.savez(beags_path(dataset.model_path, "labeled_points.npz"), xyz=xyz.cpu().numpy(), labels=lab.cpu().numpy())
    print(f"[beags] {xyz.shape[0]} points, per label: {torch.bincount(lab.long()).tolist()}")


if __name__ == "__main__":
    main()
