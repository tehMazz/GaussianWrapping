"""BEA-GS step 2: label-aware finetune of a prepared GW model (2dgs train_finetune_size.py).

    python gaussian_wrapping/beags_finetune.py -m <model> [--out output/gw_beags/<scene>/beags] [--hollow ...]

loss = 0.8 L1 + 0.2 D-SSIM + lambda_sag * mean(SAG * boundary_weight) + lambda_occ * L_occ + lambda_piv * L_piv
Never-hit Gaussians (label -1) are pruned first. No densification; opacity pruning every `prune_interval`;
sparse Adam steps every iteration on the visible Gaussians plus those reached by L_occ / L_piv.
Saves <out or model>/point_cloud/iteration_<loaded + ft_iterations>/{point_cloud.ply, labels.pt}.
--hollow adds the hollow.schedule.HollowSchedule hooks and its args, driven by the global iteration
loaded + 1 .. loaded + ft_iterations (its final prune runs at the last one).
"""
import os
import sys
from argparse import Namespace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'submodules'))

from random import randint

import numpy as np
import torch
from fused_ssim import fused_ssim
from tqdm import tqdm

from beags import ft_probe
from beags.common import beags_path, load_model, parse_args
from beags.labels import labeled_point_cloud
from beags.masks import attach_masks
from beags.mesh_field import MeshField
from beags.occupancy import LabeledVoxels, occupancy_loss, voxel_size
from beags.pivots import pivot_loss
from gaussian_renderer.beags_api import render_labels
from utils.loss_utils import l1_loss


def add_args(parser):
    parser.add_argument("--masks", default="", help="default: <model>/beags/masks (consensus masks)")
    parser.add_argument("--out", default="", help="model dir for the result (default: -m)")
    parser.add_argument("--ft_iterations", type=int, default=3000)
    parser.add_argument("--lambda_sag", type=float, default=0.5)
    parser.add_argument("--lambda_occ", type=float, default=10.0)
    parser.add_argument("--lambda_piv", type=float, default=1.0)
    parser.add_argument("--samples_per_gaussian", type=int, default=20)
    parser.add_argument("--max_gaussians", type=int, default=200_000,
                        help="Gaussians drawn per iteration for L_occ and L_piv")
    parser.add_argument("--pivot_max_ring", type=int, default=3)
    parser.add_argument("--voxel_vicinity", type=float, default=4.30)
    parser.add_argument("--no_occ_background", action="store_true",
                        help="L_occ / L_piv on object Gaussians only (default: also background, label 0, "
                             "with the background points added to the voxels)")
    parser.add_argument("--occ_source", choices=["points", "mesh"], default="points",
                        help="L_occ / L_piv against voxels of labeled_points.npz, or the label regions of the model's "
                             "multi-label mesh (--mesh_dir/tets.npz, baked to distance grids, beags/mesh_field.py)")
    parser.add_argument("--mesh_dir", default="",
                        help="--occ_source mesh: dir with tets.npz (slurm/gw_mesh.sh, SAVE_TETS=1)")
    parser.add_argument("--mesh_band", type=float, default=None,
                        help="tolerance outside a label's mesh region (default 2 cells of that label's grid)")
    parser.add_argument("--mesh_cell", type=float, default=None,
                        help="distance grid cell, all labels (default per label, from --mesh_max_nodes)")
    parser.add_argument("--mesh_max_nodes", type=int, default=200_000_000,
                        help="distance grid nodes per label, before the margin (4 bytes each on the GPU)")
    parser.add_argument("--lambda_alpha_obj", type=float, default=0.0,
                        help="B10: L1(rendered alpha of label k alone, its mask). ExGS uses 2.5; 0 = off.")
    parser.add_argument("--alpha_obj_labels", type=int, nargs="*", default=[1],
                        help="labels to supervise with --lambda_alpha_obj (default: the removed object, 1)")
    parser.add_argument("--dilation_weight", type=float, default=1.0)
    parser.add_argument("--prune_opacity", type=float, default=0.05)
    parser.add_argument("--prune_interval", type=int, default=100)
    ft_probe.add_args(parser)
    if "--hollow" in sys.argv:
        from hollow.schedule import HollowSchedule
        HollowSchedule.add_args(parser)  # --hollow + its hollow_* args
    else:
        parser.add_argument("--hollow", action="store_true", help="hollowing hooks (hollow/schedule.py)")


def load_voxels(model_path, vicinity, background_xyz=None):
    """Object voxels from beags_prepare; the voxel size always comes from the object points."""
    with np.load(beags_path(model_path, "labeled_points.npz")) as data:
        xyz = torch.from_numpy(data["xyz"])
        labels = torch.from_numpy(data["labels"])
    size = voxel_size(xyz) * vicinity
    print(f"[beags] voxel size {size:.5f} from {xyz.shape[0]} labelled points")
    if background_xyz is not None:
        print(f"[beags] + {background_xyz.shape[0]} background points")
        xyz = torch.cat([xyz, background_xyz.detach().cpu()])
        labels = torch.cat([labels, torch.zeros(background_xyz.shape[0], dtype=labels.dtype)])
    return LabeledVoxels.build(xyz, labels, size).to("cuda")


def save(dataset, gaussians, out, iteration):
    """point_cloud.ply + labels.pt (save_ply); a separate --out dir also gets cfg_args."""
    if os.path.abspath(out) != os.path.abspath(dataset.model_path):
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "cfg_args"), "w") as f:
            f.write(str(Namespace(**{**vars(dataset), "model_path": out})))
    gaussians.save_ply(os.path.join(out, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
    print(f"[beags] saved iteration {iteration} to {os.path.join(out, 'point_cloud')}")


def finetune(dataset, pipe, opt, args):
    scene, gaussians = load_model(dataset, args.iteration, with_labels=True)
    gaussians.training_setup(opt)
    gaussians.epoch_prune = None   # the finetune runs its own short loop; no epoch window here
    cameras = scene.getTrainCameras()
    never_hit = gaussians.labels < 0
    gaussians.prune_points(never_hit)
    print(f"[beags] pruned {int(never_hit.sum())} never-hit Gaussians, {gaussians.get_xyz.shape[0]} left")
    num_labels = attach_masks(cameras, args.masks or beags_path(dataset.model_path, "masks"),
                              args.dilation_weight, device="cpu")
    occ_background = not args.no_occ_background
    if args.occ_source == "mesh":
        if not args.mesh_dir:
            raise ValueError("--occ_source mesh needs --mesh_dir")
        present = torch.unique(gaussians.labels).tolist()
        label_ids = [k for k in present if k > 0 or (k == 0 and occ_background)]
        voxels = MeshField.build(args.mesh_dir, label_ids, args.mesh_cell, args.mesh_band, args.mesh_max_nodes,
                                 args.pivot_max_ring)
    else:
        background_xyz = None
        if occ_background:  # background cloud from the loaded (pre-finetune) model, as for the objects
            background_xyz, _ = labeled_point_cloud(gaussians, cameras, pipe, gaussians.labels,
                                                    label_ids=[0], output_device="cpu")
        voxels = load_voxels(dataset.model_path, args.voxel_vicinity, background_xyz)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    start = scene.loaded_iter
    hollow = None
    if args.hollow:
        from hollow.schedule import HollowSchedule
        args.iterations = start + args.ft_iterations
        hollow = HollowSchedule(args, scene, gaussians, pipe)

    probe = ft_probe.FinetuneProbe(args, scene) if args.snap_every > 0 else None
    if probe is not None:
        probe.snapshot(0, gaussians, pipe)

    stack = []
    progress = tqdm(range(1, args.ft_iterations + 1), desc="BEA-GS finetune")
    for it in progress:
        gaussians.update_learning_rate(start - args.ft_iterations + it)  # continue the tail of the schedule
        if not stack:
            stack = list(cameras)
        cam = stack.pop(randint(0, len(stack) - 1))

        pixel_labels = cam.masks.labels.cuda()
        boundary = cam.masks.boundary.cuda()
        out = render_labels(cam, gaussians, pipe, bg, gaussian_labels=gaussians.labels,
                            pixel_labels=pixel_labels, num_labels=num_labels)
        gt = cam.original_image.cuda()
        if cam.gt_mask is not None:
            alpha_mask = cam.gt_mask.cuda()
            gt = gt * alpha_mask + bg[:, None, None] * (1.0 - alpha_mask)
        photo = (1 - opt.lambda_dssim) * l1_loss(out["render"], gt) + \
            opt.lambda_dssim * (1 - fused_ssim(out["render"][None], gt[None], padding="valid"))
        sag = (out["sag"] * boundary).mean()
        labels = gaussians.labels
        occ = occupancy_loss(gaussians, labels, voxels, args.samples_per_gaussian,
                             include_background=occ_background, max_gaussians=args.max_gaussians)
        piv = pivot_loss(gaussians, labels if occ_background else torch.where(labels == 0, -1, labels),
                         voxels, args.pivot_max_ring, args.max_gaussians) if args.lambda_piv > 0 else occ * 0
        loss = photo + args.lambda_sag * sag + args.lambda_occ * occ + args.lambda_piv * piv
        alpha_obj = None
        # ---B10: the extracted object must render opaque inside its mask and empty outside---
        # occ (weight 10) and sag both push object opacity DOWN; measured on spinnerf_7 this erodes
        # label 1 from 17740 Gaussians at mean opacity 0.992 to 11748 at 0.743 over 3000 iterations.
        # This is the counter-force, and it is ExGS's L1(alpha_ext, mask) term (lambda 2.5 there).
        obj_radii = None
        if args.lambda_alpha_obj > 0:
            valid = (pixel_labels >= 0).float()          # maskless views contribute nothing
            if valid.sum() > 0:
                for k in args.alpha_obj_labels:
                    sel = labels == k
                    if not bool(sel.any()):
                        continue
                    obj = render_labels(cam, gaussians, pipe, bg, mask=sel)
                    target = (pixel_labels == k).float()
                    term = args.lambda_alpha_obj * (valid * (obj["alpha"][0] - target).abs()).sum() / valid.sum()
                    alpha_obj = term if alpha_obj is None else alpha_obj + term
                    loss = loss + term
                    obj_radii = obj["radii"]
        if hollow is not None:
            loss = loss + hollow.loss(start + it)
        snap = probe is not None and probe.due(it, args.ft_iterations)
        if snap:  # weighted terms, measured on this step's graph before backward
            terms = {"photo": photo, "sag": args.lambda_sag * sag, "occ": args.lambda_occ * occ,
                     "piv": args.lambda_piv * piv}
            if alpha_obj is not None:
                terms["alpha_obj"] = alpha_obj
            push = ft_probe.term_push(terms, gaussians, args.snap_label)
        loss.backward()

        with torch.no_grad():
            visible = out["radii"] > 0
            if obj_radii is not None:
                visible = visible | (obj_radii > 0)
            if gaussians._opacity.grad is not None:
                visible |= gaussians._opacity.grad[:, 0] != 0
            if hollow is not None and hollow.touched is not None:
                visible |= hollow.touched
            if type(gaussians.optimizer).__name__ in ("ViewAdam", "RowAdam"):
                gaussians.optimizer.step(visible)   # gs-atoms optimizers take no N
            else:
                gaussians.optimizer.step(visible, visible.shape[0])
            gaussians.optimizer.zero_grad(set_to_none=True)
            if it % args.prune_interval == 0:
                gaussians.prune_points((gaussians.get_opacity < args.prune_opacity).squeeze(1))
            if hollow is not None:
                hollow.after_step(start + it)
            if snap:
                probe.snapshot(it, gaussians, pipe, terms={k: v.item() for k, v in terms.items()}, push=push)
            if it % 10 == 0:
                progress.set_postfix(photo=f"{photo.item():.4f}", sag=f"{sag.item():.5f}", occ=f"{occ.item():.5f}",
                                     piv=f"{piv.item():.4f}", n=gaussians.get_xyz.shape[0])

    save(dataset, gaussians, args.out or dataset.model_path, start + args.ft_iterations)


if __name__ == "__main__":
    finetune(*parse_args("BEA-GS finetune", add_args))
