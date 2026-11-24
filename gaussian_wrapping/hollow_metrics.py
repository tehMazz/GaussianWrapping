"""Hollowing metrics: how much geometry sits where none is needed.

    python gaussian_wrapping/hollow_metrics.py -m <model> --iteration N [--object_mask <npy bool P>] [--masks <dir>]
        [--voxel <size>] [--floater_voxel <size>]

Writes <model>/hollow_metrics_<it>.json (or --out) with
  unseen      Gaussians with seen < seen_tau (hollow/ledger.py): share of count and of sum o' * prod(s')
  free space  mean density over the ray-carved free grid (hollow/free_space.py) and over the AABB
  floaters    26-connected components of voxels holding a centre with o' >= 0.5, other than the largest
              component (count and Gaussians in them). Voxel: --floater_voxel, else --voxel, else AABB extent / 256.
              The AABB (scene_bounds) depends on the model, so default voxels differ between models: compare
              models only at a fixed absolute size; an absolute floater voxel is also aligned to the world
              origin (grid at multiples of the size, not at the AABB corner). Recorded: voxel, voxel_mode,
              floater_voxel, floater_voxel_mode (aabb/256 | absolute), floater_grid (aabb | world).
  fog         share of o' < 0.05 and < 0.1, mean opacity entropy (nats)
  hole        with --object_mask, over training views where the object is visible, inside the object's
              silhouette (alpha > 0.5): hole_alpha, the mean alpha of the scene without the object (an alpha:
              higher = fuller, target 0 behind unseen space); hole_alpha_8x8, the same after 8x8 max pooling
              (ArtiFixer latent cells); hole_alpha_unseen, the same rendering only Gaussians with seen < seen_tau
              (unseen geometry behind the object, target 0); with --masks, hole_tightness = silhouette area /
              mask area. Files written before the rename (hole_emptiness, hole_emptiness_8x8,
              hole_unseen_alpha) read with the new keys through load_metrics(path).
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "submodules"))

from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer.beags_api import render_labels
from hollow.free_space import (_member, _pack, _unpack, build_free_grid, default_voxel, mean_density,
                               opacity_entropy_loss, scene_bounds)
from hollow.ledger import compute_seen
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


# Keys before the rename (hole_emptiness was an alpha: higher = fuller) -> current keys.
LEGACY_KEYS = {"hole_emptiness": "hole_alpha", "hole_emptiness_8x8": "hole_alpha_8x8",
               "hole_unseen_alpha": "hole_alpha_unseen"}
FLOATER_MIN_OPACITY = 0.5


def load_metrics(path):
    """hollow_metrics json with the legacy hole keys renamed (a current key wins over its legacy one)."""
    with open(path) as f:
        metrics = json.load(f)
    for old, new in LEGACY_KEYS.items():
        if old in metrics:
            value = metrics.pop(old)
            metrics.setdefault(new, value)
    return metrics


@torch.no_grad()
def count_floaters(xyz, opacity, lo, hi, voxel, min_opacity=FLOATER_MIN_OPACITY, snap=False):
    """(components - 1, Gaussians outside the largest component) of occupied voxels of the AABB [lo, hi].
    Voxels start at lo, or with snap at the multiple of voxel below lo (a world-aligned grid, the same cells
    for every model)."""
    keep = (opacity >= min_opacity) & ((xyz >= lo) & (xyz <= hi)).all(dim=1)
    if not keep.any():
        return 0, 0
    origin = torch.floor(lo / voxel) * voxel if snap else lo
    keys, inverse = torch.unique(_pack(torch.floor((xyz[keep] - origin) / voxel).long()), return_inverse=True)
    ijk = _unpack(keys)
    o = torch.tensor([-1, 0, 1], device=xyz.device)
    offsets = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(27, 3)[14:]  # 13 forward neighbours
    src, dst = [], []
    for off in offsets:
        nb = _pack(ijk + off)
        hit = _member(keys, nb)
        src.append(torch.nonzero(hit)[:, 0])
        dst.append(torch.searchsorted(keys, nb[hit]))
    src, dst = torch.cat(src).cpu().numpy(), torch.cat(dst).cpu().numpy()
    M = keys.numel()
    n, labels = connected_components(coo_matrix((np.ones(len(src)), (src, dst)), shape=(M, M)), directed=False)
    per_gaussian = labels[inverse.cpu().numpy()]
    largest = np.bincount(per_gaussian).argmax()
    return int(n - 1), int((per_gaussian != largest).sum())


def _alpha(cam, gaussians, pipe, background, mask):
    if not mask.any():
        return torch.zeros((int(cam.image_height), int(cam.image_width)), device="cuda")
    return render_labels(cam, gaussians, pipe, background, mask=mask)["alpha"][0]


def _mask_2d(path, channel, height, width):
    if not os.path.exists(path):
        return None
    m = torch.from_numpy(np.load(path)[channel].astype(np.float32))[None, None]
    return F.interpolate(m, size=(height, width), mode="bilinear", align_corners=False)[0, 0].cuda() > 0.5


@torch.no_grad()
def hole_metrics(cameras, gaussians, pipe, obj, seen, tau, masks_dir=None, mask_channel=1):
    background = torch.zeros(3, device="cuda")
    rest, unseen_rest = ~obj, ~obj & (seen < tau)
    alphas, pooled, unseen, tightness = [], [], [], []
    for cam in cameras:
        sil = _alpha(cam, gaussians, pipe, background, obj) > 0.5
        if not sil.any():
            continue
        alpha = _alpha(cam, gaussians, pipe, background, rest)
        alphas.append(float(alpha[sil].mean()))
        cells = F.max_pool2d(sil[None, None].float(), 8, ceil_mode=True)[0, 0] > 0
        pooled.append(float(F.max_pool2d(alpha[None, None], 8, ceil_mode=True)[0, 0][cells].mean()))
        unseen.append(float(_alpha(cam, gaussians, pipe, background, unseen_rest)[sil].mean()))
        if masks_dir is not None:
            m = _mask_2d(os.path.join(masks_dir, cam.image_name + ".npy"), mask_channel, *sil.shape)
            if m is not None and m.any():
                tightness.append(float(sil.sum() / m.sum()))
    mean = lambda v: float(np.mean(v)) if v else None
    out = {"hole_views": len(alphas), "hole_alpha": mean(alphas), "hole_alpha_8x8": mean(pooled),
           "hole_alpha_unseen": mean(unseen)}
    if masks_dir is not None:
        out["hole_tightness"] = mean(tightness)
        out["hole_tightness_views"] = len(tightness)
    return out


@torch.no_grad()
def main(args, dataset, pipe):
    gaussians = GaussianModel(dataset.sh_degree, use_mip_filter=not args.disable_mip_filter,
                              use_unbounded_opacity=dataset.use_unbounded_opacity)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = scene.getTrainCameras().copy()
    xyz = gaussians.get_xyz
    opacity = gaussians.get_opacity_with_3D_filter[:, 0]
    scale = gaussians.get_scaling_with_3D_filter
    P = xyz.shape[0]
    print(f"[INFO] {P} Gaussians, {len(cameras)} training views.")

    seen = compute_seen(cameras, gaussians, pipe)
    unseen = seen < args.seen_tau
    mass = opacity * scale.prod(dim=1)
    lo, hi = scene_bounds(xyz, cameras)
    voxel, voxel_mode = (args.voxel, "absolute") if args.voxel is not None else (default_voxel(lo, hi), "aabb/256")
    floater_voxel, floater_mode = ((args.floater_voxel, "absolute") if args.floater_voxel is not None
                                   else (voxel, voxel_mode))
    snap = floater_mode == "absolute"
    print(f"[INFO] voxel {voxel:.5g} ({voxel_mode}), floater voxel {floater_voxel:.5g} ({floater_mode}).")
    grid = build_free_grid(cameras, gaussians, pipe, voxel, bounds=(lo, hi))
    n_floaters, floater_gaussians = count_floaters(xyz, opacity, lo, hi, floater_voxel, snap=snap)

    metrics = {
        "iteration": scene.loaded_iter,
        "n_gaussians": P,
        "seen_tau": args.seen_tau,
        "unseen_count": int(unseen.sum()),
        "unseen_share": float(unseen.float().mean()),
        "unseen_mass_share": float(mass[unseen].sum() / mass.sum().clamp_min(1e-12)),
        "voxel": voxel,
        "voxel_mode": voxel_mode,
        "aabb": [lo.tolist(), hi.tolist()],
        "free_voxels": grid.keys.numel(),
        "free_share": grid.volume / float((hi - lo).prod()),
        "free_space_mass": mean_density(gaussians, lo, hi, grid.contains, grid.volume) if grid.keys.numel() else None,
        "aabb_mass": mean_density(gaussians, lo, hi),
        "floaters": n_floaters,
        "floater_gaussians": floater_gaussians,
        "floater_voxel": floater_voxel,
        "floater_voxel_mode": floater_mode,
        "floater_grid": "world" if snap else "aabb",
        "floater_min_opacity": FLOATER_MIN_OPACITY,
        "fog_share_005": float((opacity < 0.05).float().mean()),
        "fog_share_010": float((opacity < 0.1).float().mean()),
        "opacity_entropy": float(opacity_entropy_loss(gaussians)),
    }
    if args.object_mask is not None:
        obj = torch.from_numpy(np.load(args.object_mask)).bool().reshape(-1).cuda()
        assert obj.shape[0] == P, f"object mask has {obj.shape[0]} entries, model has {P} Gaussians"
        metrics.update(hole_metrics(cameras, gaussians, pipe, obj, seen, args.seen_tau, args.masks, args.mask_channel))

    out = args.out or os.path.join(dataset.model_path, f"hollow_metrics_{scene.loaded_iter}.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[INFO] Saved {out}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Hollowing metrics")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--object_mask", default=None, type=str, help="(P,) bool .npy of object Gaussians")
    parser.add_argument("--masks", default=None, type=str, help="dir of <image_name>.npy (C,H,W) masks, for hole tightness")
    parser.add_argument("--mask_channel", default=1, type=int)
    parser.add_argument("--seen_tau", default=1.0, type=float)
    parser.add_argument("--voxel", default=None, type=float,
                        help="absolute voxel size of the free grid and the floaters (default: AABB extent / 256)")
    parser.add_argument("--floater_voxel", default=None, type=float,
                        help="absolute, world-aligned floater voxel size, to compare floaters across models "
                             "(default: --voxel)")
    parser.add_argument("--disable_mip_filter", action="store_true")
    parser.add_argument("--out", default=None, type=str)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    main(args, model.extract(args), pipeline.extract(args))
