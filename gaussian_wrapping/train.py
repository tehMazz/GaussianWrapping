import os
import sys
import gc
import yaml
from functools import partial
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
SUBMODULES_DIR = os.path.join(ROOT_DIR, 'submodules')
sys.path.append(ROOT_DIR)
sys.path.append(SUBMODULES_DIR)
sys.path.append(os.path.join(SUBMODULES_DIR, 'Depth-Anything-V2'))

import torch
from random import randint
from utils.loss_utils import l1_loss, L1_loss_appearance, get_img_grad_weight
from fused_ssim import fused_ssim

from gaussian_renderer import network_gui
from gaussian_renderer import render_imp
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

import numpy as np
import time
import json

from lpipsPyTorch import lpips
import robust
from robust.bilagrid import BilateralGridModule
from robust.losses import appearance_photometric_loss, load_ignore_mask, render_confidence, variance_losses
from hollow.schedule import HollowSchedule
from utils.geometry_utils import depth_to_normal, depth_to_normal_with_mask
from utils.log_utils import log_normal_field_training_progress
from regularization.regularizer.normal_field import (
    initialize_normal_field,
    compute_normal_field_regularization,
    reset_normal_field_state_at_next_iteration,
    densify_normal_field,
    prune_non_maximal_gaussians,
)
from regularization.regularizer.mono_prior import MonoPrior
from regularization.regularizer.depth_order import (
    initialize_depth_order_supervision,
    compute_depth_order_regularization,
)
from regularization.regularizer.multiview import (
    initialize_multiview_regularization,
    compute_multiview_regularization,
)
from regularization.regularizer.mesh_in_the_loop import (
    initialize_mesh_in_the_loop_regularization,
    compute_mesh_in_the_loop_regularization,
    reset_milo_state_at_next_iteration,
)


def read_lpips_images(path):
    """Image names (with or without extension) whose loss is LPIPS only, e.g. ArtiFixer3D's generated frames."""
    if not path:
        return set()
    with open(path) as f:
        return {os.path.basename(line.strip()).split(".")[0] for line in f if line.strip()}


def resolve_appearance(args):
    """
    Maps --appearance and GW's legacy --exposure_compensation / --decoupled_appearance onto one mode.
    Without --appearance, the legacy flags decide (GW default: exposure compensation).
    With --appearance, the legacy flags must agree with it.
    """
    if args.exposure_compensation and args.decoupled_appearance:
        raise ValueError("Exposure compensation and appearance network cannot be used together")
    legacy = "exposure" if args.exposure_compensation else ("network" if args.decoupled_appearance else None)
    if args.appearance is None:
        if args.exposure_compensation is None and not args.decoupled_appearance:
            legacy = "exposure"
        args.appearance = legacy or "none"
    elif legacy not in (None, args.appearance):
        raise ValueError(
            f"--appearance {args.appearance} conflicts with --exposure_compensation / --decoupled_appearance "
            f"(pass --no-exposure_compensation)"
        )
    args.exposure_compensation = args.appearance == "exposure"
    args.decoupled_appearance = args.appearance == "network"


def training(
    dataset, opt, pipe, 
    testing_iterations, saving_iterations, 
    checkpoint_iterations, checkpoint, 
    debug_from, args, 
    depth_order_config, normal_field_config, multiview_config, milo_config,
    log_interval,
):
    # ---Prepare logger---
    run = prepare_output_and_logger(dataset, args)
    time_start = time.time()

    # ---Initialize scene and Gaussians---
    first_iter = 0
    use_mip_filter = not args.disable_mip_filter
    
    if args.use_normal_field:
        if normal_field_config["use_smallest_axis"]:
            n_gaussian_features = 1
        else:
            n_gaussian_features = 4
    else:
        n_gaussian_features = 0
    
    n_pivots_per_gaussian = milo_config["n_pivots_per_gaussian"] if args.milo else 2
    
    gaussians = GaussianModel(
        sh_degree=dataset.sh_degree, 
        use_mip_filter=use_mip_filter, 
        learn_occupancy=True if args.milo else False,
        use_appearance_network=args.decoupled_appearance,
        n_gaussian_features=n_gaussian_features,
        n_pivots_per_gaussian=n_pivots_per_gaussian,
        use_radegs_densification=True,
        use_unbounded_opacity=dataset.use_unbounded_opacity,
        use_exposure_compensation=args.exposure_compensation,
    )
    scene = Scene(dataset, gaussians)
    
    if args.exposure_compensation:
        n_cameras = len(scene.getTrainCameras().copy())
        print(f"[INFO] Using exposure compensation for {n_cameras} cameras.")
        gaussians.initialize_exposure_compensation(num_cameras=n_cameras)
    
    gaussians.training_setup(opt)
    print(f"[INFO] Using 3D Mip Filter: {gaussians.use_mip_filter}")
    print(f"[INFO] Using learnable SDF: {gaussians.learn_occupancy}")

    if args.use_normal_field:
        print(f"[INFO] Using {n_gaussian_features} learnable Gaussian features.")

    if args.dense_gaussians:
        print("[INFO] Using dense Gaussians.")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        if args.use_normal_field:
            if first_iter > normal_field_config["start_iter"]:
                normal_field_config["start_iter"] = first_iter + 1
        if args.milo:
            if first_iter > milo_config["start_iter"]:
                milo_config["start_iter"] = first_iter + 1
        if args.multiview:
            if first_iter > multiview_config["start_multiview"]:
                multiview_config["start_multiview"] = first_iter + 1

    # ---Robustness to non-clean data (robust/)---
    if args.confidence:
        print(f"[INFO] Using per-Gaussian confidence from iteration {args.conf_from_iter}.")
        gaussians.enable_confidence()
    bilagrid = None
    if args.appearance.startswith("bilagrid"):
        n_cameras = len(scene.getTrainCameras())
        print(f"[INFO] Using {args.appearance} appearance for {n_cameras} cameras.")
        bilagrid = BilateralGridModule(
            num_views=n_cameras, total_iterations=opt.iterations, tv_weight=args.bilagrid_tv_weight,
            identity_weight=args.bilagrid_identity_weight, mean_weight=args.bilagrid_mean_weight,
            until=args.bilagrid_until,
        )
        print(f"[INFO] Bilateral grid: {robust.bilagrid_config(args)}.")
        bilagrid_checkpoint = checkpoint and os.path.join(
            os.path.dirname(checkpoint), os.path.basename(checkpoint).replace("chkpnt", "bilagrid")
        )
        if bilagrid_checkpoint and os.path.exists(bilagrid_checkpoint):
            bilagrid.load_state_dict(torch.load(bilagrid_checkpoint))
    use_robust_loss = (bilagrid is not None) or args.confidence or bool(args.ignore_masks)
    if use_robust_loss and gaussians.use_appearance_network:
        raise ValueError("--confidence / --ignore_masks are not supported with the appearance network")
    use_variance_losses = (args.lambda_color_var > 0) or (args.lambda_normal_var > 0)

    # ---Labels (BEA-GS), carried through densification and pruning---
    if args.labels:
        gaussians.labels = torch.load(args.labels, map_location="cuda")
        print(f"[INFO] Loaded {gaussians.labels.shape[0]} labels from {args.labels}.")

    # ---ArtiFixer3D distillation: generated frames are supervised by LPIPS only---
    lpips_names = read_lpips_images(args.lpips_images)
    if lpips_names:
        train_names = {cam.image_name for cam in scene.getTrainCameras()}
        missing = lpips_names - train_names
        assert not missing, f"--lpips_images names not in the training cameras: {sorted(missing)[:10]}"
        print(f"[INFO] LPIPS-only supervision on {len(lpips_names)} of {len(train_names)} training images.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Initialize culling stats
    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    gaussians.init_culling(len(scene.getTrainCameras().copy()))
    
    # Initialize 3D Mip filter
    if use_mip_filter:
        gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())

    # ---Hollowing (hollow/): no geometry in unseen, free or see-through space---
    hollow = HollowSchedule(args, scene, gaussians, pipe)
    if hollow.enabled:
        print(f"[INFO] Using hollowing from iteration {args.hollow_from}.")

    # ---gs-atoms (B3/B6/B7): the gate, the gated opacity decay and the per-epoch contribution prune---
    # All three read the same per-Gaussian vis buffer, which only the "ours" rasterizer produces.
    atoms_cover_signal = epoch_prune = None
    need_vis = (opt.optimizer_type != "sparse_adam") or (opt.opa_decay < 1.0) or opt.prune_contrib
    if need_vis:
        if args.rasterizer != "ours":
            raise ValueError("gs-atoms (--optimizer_type / --opa_decay / --prune_contrib) needs --rasterizer ours: "
                             f"the per-Gaussian vis buffer does not exist under {args.rasterizer!r}.")
        from atoms import cover_signal as atoms_cover_signal
        if opt.prune_contrib:
            from atoms import EpochPrune
            n_cams = len(scene.getTrainCameras())
            window = max(n_cams, 250)   # floor: 17-view scenes would otherwise prune ~10x more often
            epoch_prune = EpochPrune(contrib_min=opt.contrib_min, untouched=opt.prune_untouched,
                                     window=window, log=print)
            epoch_prune.start(gaussians.get_xyz.shape[0], gaussians.get_xyz.device)
            gaussians.epoch_prune = epoch_prune
            print(f"[INFO] EpochPrune: window {window} steps ({n_cams} train views), "
                  f"contrib_min {opt.contrib_min}, untouched {opt.prune_untouched}.")
    epoch_prune_until = opt.epoch_prune_until_iter if opt.epoch_prune_until_iter >= 0 else opt.densify_until_iter

    # Additional variables
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    all_viewpoints = None
    postfix_dict = {}
    ema_loss_for_log = 0.0
    ema_depth_normal_loss_for_log = 0.0
    
    # ---Prepare Normal Field Optimization---
    if args.use_normal_field:
        print("[INFO] Using Normal Field.")
        normal_field_state = initialize_normal_field(
            scene=scene,
            config=normal_field_config,
        )
        if normal_field_config["reset_normals_after_densification"]:
            print(f"[INFO] Normal features will be reset after densification.")
            print(f"        > Resetting normal directions: {normal_field_config['reset_normal_directions']}.")
            print(f"        > Resetting normal signs: {normal_field_config['reset_normal_signs']}.")
    ema_normal_field_alignment_loss_for_log = 0.0
    ema_front_pivots_visibility_loss_for_log = 0.0
    ema_back_pivots_occlusion_loss_for_log = 0.0
    ema_gaussian_flattening_loss_for_log = 0.0
    ema_sdf_and_normal_field_consistency_loss_for_log = 0.0
    
    # ---Prepare Multiview Regularization---
    if args.multiview:
        print("[INFO] Using multiview regularization.")
        multiview_state = initialize_multiview_regularization(
            scene=scene,
            pipe=pipe,
            kernel_size=0.0,
            multiview_config=multiview_config,
        )
    ema_multiview_loss_for_log = 0.0
    
    # ---Prepare Mesh-In-the-Loop Regularization---
    if args.milo:
        print(f"[INFO] Using mesh-in-the-loop regularization with {n_pivots_per_gaussian} pivots per Gaussian.")
        milo_state = initialize_mesh_in_the_loop_regularization(
            scene=scene,
            gaussians=gaussians,
            milo_config=milo_config,
        )
    ema_mesh_depth_loss_for_log = 0.0
    ema_mesh_normal_loss_for_log = 0.0
    ema_occupied_centers_loss_for_log = 0.0

    # ---Prepare Depth-Order Regularization---    
    if args.depth_order:
        print("[INFO] Using depth order regularization.")
        print(f"        > Using expected depth with depth_ratio {depth_order_config['depth_ratio']} for depth order regularization.")
        if depth_order_config["deactivate_depth_order_after"] > -1:
            print(f"        > Deactivating at iteration {depth_order_config['deactivate_depth_order_after']}.")
        depth_priors = initialize_depth_order_supervision(
            scene=scene,
            config=depth_order_config,
            device='cuda',
        )
    ema_depth_order_loss_for_log = 0.0

    # ---Monocular depth/normal prior (MoGe-2), only for weakly conditioned scenes (see mono_prior.py)---
    mono_prior = MonoPrior.from_args(args, scene)
        
    # ---Log optimizable param groups---
    print(f"[INFO] Found {len(gaussians.optimizer.param_groups)} optimizable param groups:")
    n_total_params = 0
    for param in gaussians.optimizer.param_groups:
        name = param['name']
        n_params = len(param['params'])
        print(f"\n========== {name} ==========")
        print(f"Learning rate: {param['lr']}")
        print(f"Total number of param groups: {n_params}")
        for param_i in param['params']:
            print(f"   > Shape {param_i.shape}")
            n_total_params = n_total_params + param_i.numel()
    if gaussians.learn_occupancy:
        print(f"\n========== base_occupancy ==========")
        print(f"   > Not learnable")
        print(f"   > Shape {gaussians._base_occupancy.shape}")
    print(f"\nTotal number of optimizable parameters: {n_total_params}\n")
    
    # ---Start optimization loop---    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):   

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_imp(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # > Past --bilagrid_until the grid is frozen and the losses see the raw render.
        bilagrid_on = bilagrid is not None and bilagrid.is_active(iteration)
        if bilagrid_on:
            bilagrid.update_lr(iteration)
        elif bilagrid is not None and iteration == args.bilagrid_until + 1:
            print(f"\n[INFO] Bilateral grid off from iteration {iteration}: losses on the raw render.")

        # ---Update SH degree---
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ---Select random viewpoint---
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_idx_stack = list(range(len(viewpoint_stack)))
            all_viewpoints = scene.getTrainCameras().copy()

        _random_view_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_idx = viewpoint_idx_stack.pop(_random_view_idx)
        viewpoint_cam = viewpoint_stack.pop(_random_view_idx)

        # ---Render scene---
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
            
        reg_kick_on = iteration >= args.regularization_from_iter
        normal_field_kick_on = args.use_normal_field and (iteration >= normal_field_config["start_iter"])
        depth_order_kick_on = args.depth_order
        if args.depth_order and depth_order_config["deactivate_depth_order_after"] > -1:
            if iteration == depth_order_config["deactivate_depth_order_after"]:
                print(f"[INFO] Deactivating depth order regularization at iteration {iteration}.")
            depth_order_kick_on = depth_order_kick_on and (iteration < depth_order_config["deactivate_depth_order_after"])
        multiview_kick_on = args.multiview and (iteration >= multiview_config["start_multiview"])
        milo_kick_on = args.milo and (iteration >= milo_config["start_iter"])
        mono_kick_on = mono_prior is not None and mono_prior.active(iteration)
        
        render_depth_in_forward_pass = (
            mono_kick_on or
            reg_kick_on 
            or normal_field_kick_on 
            or depth_order_kick_on
            or multiview_kick_on
            or milo_kick_on
        )
        
        # If depth-normal regularization or normal field regularization are active,
        # we use the rasterizer compatible with depth and normal rendering.
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, bg,
            require_coord=False,
            require_depth=render_depth_in_forward_pass,
            vis=need_vis,
        )
        # gs-atoms gate + cover signal, read once and reused by the step, the decay and EpochPrune (B3).
        # vis[:, 0] is the summed transmittance in front of each Gaussian, so vis[:, 0] > 0 is exactly
        # "some pixel blended this row" - the same row set as contrib_max > 0, and opacity-independent
        # (gating on alpha*T instead collapses after an opacity reset; see atoms/gate.py:6-7).
        atom_gate = atom_cover = atom_contrib = None
        if need_vis:
            with torch.no_grad():
                _vis = render_pkg["vis"]
                atom_gate = _vis[:, 0] > opt.gate_threshold if opt.gate_threshold > 0 else _vis[:, 0] > 0
                atom_contrib = _vis[:, 4]
                atom_cover = atoms_cover_signal(atom_contrib, gaussians._opacity)

        # ---Compute losses---
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], render_pkg["radii"]
        )
        gt_image = viewpoint_cam.original_image.cuda()
        if viewpoint_cam.gt_mask is not None:
            alpha_mask = viewpoint_cam.gt_mask.cuda()
            gt_image = gt_image * alpha_mask + bg.unsqueeze(-1).unsqueeze(-1) * (1.0 - alpha_mask)

        # Rendering loss
        is_generated_view = viewpoint_cam.image_name in lpips_names
        if is_generated_view:
            # ArtiFixer3D distillation: generated frames only get lambda_lpips * LPIPS(vgg), without
            # appearance compensation or confidence, so their inconsistencies do not blur the model.
            loss = args.lambda_lpips * lpips(image.clamp(0, 1) * 2 - 1, gt_image * 2 - 1, net_type='vgg').mean()
        elif use_robust_loss:
            compensated = None
            if bilagrid is None and gaussians.use_exposure_compensation:
                compensated = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.uid, return_transformed_image=True)
            confidence_map = None
            if args.confidence and iteration >= args.conf_from_iter:
                confidence_map = render_confidence(viewpoint_cam, gaussians, pipe)
            # > Grid (while active) + its TV / identity / mean pulls, and the optional raw-render term (--lambda_raw).
            loss, _ = appearance_photometric_loss(
                image, gt_image, iteration, bilagrid=bilagrid, view_idx=viewpoint_cam.uid, compensated=compensated,
                # Exposure compensation stays close to GW's rule (L1 on the compensated image, SSIM on the raw render).
                decoupled=args.appearance in ("bilagrid+decoupled", "exposure"), lambda_raw=args.lambda_raw,
                lambda_dssim=opt.lambda_dssim, confidence_map=confidence_map, conf_beta=args.conf_beta,
                ignore_mask=load_ignore_mask(args.ignore_masks, viewpoint_cam.image_name, image.shape[1], image.shape[2]),
            )
        else:
            if args.decoupled_appearance or args.exposure_compensation:
                Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.uid)
            else:
                Ll1 = l1_loss(image, gt_image)
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), padding="valid")
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Colour and normal variance along rays (CoMe)
        if use_variance_losses and (iteration >= args.var_from_iter) and not is_generated_view:
            color_var_loss, normal_var_loss = variance_losses(viewpoint_cam, gaussians, pipe, gt_image, bg_color=bg)
            loss = loss + args.lambda_color_var * color_var_loss + args.lambda_normal_var * normal_var_loss

        # Hollowing losses (0 when off)
        # ---B9: pull the rendered alpha towards 1 where the ray already hits something---
        # Weighted by (alpha > alpha_floor) so the term completes thin, partly-transparent surfaces
        # without seeding geometry in empty sky, which has no background model on the 360 scenes.
        if opt.lambda_alpha > 0:
            _alpha = render_pkg["alpha"]
            _w = (_alpha.detach() > opt.alpha_floor).float()
            loss = loss + opt.lambda_alpha * (_w * (1.0 - _alpha)).sum() / _w.sum().clamp_min(1.0)
        loss = loss + hollow.loss(iteration)

        # Depth-Normal Consistency Regularization
        if reg_kick_on:
            if args.mask_depth_normal:
                reg_depth_ratio = 0.6
                
                # Blending that respects the median depthmap validity mask
                depth_blend = torch.where(
                    render_pkg["median_depth"] > 0,  # (1, H, W)
                    (1. - reg_depth_ratio) * render_pkg["expected_depth"] + reg_depth_ratio * render_pkg["median_depth"],  # (1, H, W)
                    render_pkg["median_depth"],  # (1, H, W)
                )
                
                depth_normal, valid_points = depth_to_normal_with_mask(viewpoint_cam, depth_blend)  # (3, H, W), (H, W)
                normal_error_map = 1 - torch.linalg.vecdot(render_pkg["normal"], depth_normal, dim=0)  # (H, W)
                depth_normal_loss = torch.where(valid_points.squeeze(), normal_error_map, torch.zeros_like(normal_error_map)).mean()
                depth_normal_loss = args.lambda_depth_normal * depth_normal_loss
                
            else:
                rendered_depth_to_normals: torch.Tensor = depth_to_normal(
                    viewpoint_cam, 
                    render_pkg["expected_depth"],  # 1, H, W
                    render_pkg["median_depth"],  # 1, H, W
                )  # 3, H, W or 2, 3, H, W
                rendered_normals: torch.Tensor = render_pkg["normal"]  # 3, H, W
            
                if rendered_depth_to_normals.ndim == 4:
                    # If shape is 2, 3, H, W
                    reg_depth_ratio = 0.6
                    normal_error_map = 1. - (rendered_normals[None] * rendered_depth_to_normals).sum(dim=1)  # 2, H, W
                    depth_normal_loss = args.lambda_depth_normal * (
                        (1. - reg_depth_ratio) * normal_error_map[0]  # (H, W)
                        + reg_depth_ratio * normal_error_map[1]  # (H, W)
                    )  # (H, W)
                else:
                    # If shape is 3, H, W
                    depth_normal_loss = args.lambda_depth_normal * (1 - (rendered_normals * rendered_depth_to_normals).sum(dim=0))  # (H, W)
            
            # Weight by image gradient as in PGSR
            if args.weight_by_img_grad:
                image_weight = get_img_grad_weight(img=gt_image)  # (H, W)
                image_weight = (1.0 - image_weight).clamp(min=0.0, max=1.0)  # (H, W)
                image_weight = image_weight ** 2  # (H, W)
                depth_normal_loss = depth_normal_loss * image_weight  # (H, W)
            
            depth_normal_loss = depth_normal_loss.mean()
            
            loss = loss + depth_normal_loss
            
        # Min scale regularization (from PGSR)
        if args.use_scale_loss:
            if visibility_filter.sum() > 0:
                min_scaling_loss = torch.sort(
                    gaussians.get_scaling_with_3D_filter[visibility_filter],  # (N_visible_gaussians, 3)
                    dim=-1
                ).values[..., 0]  # (N_visible_gaussians,)
                min_scaling_loss = args.scale_loss_weight * min_scaling_loss.mean()
                loss = loss + min_scaling_loss
            
        # Depth Order Regularization
        # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
        # > In the paper, MILo does not rely on any learned prior. 
        if depth_order_kick_on:
            if depth_order_config["depth_ratio"] < 1.:
                depth_for_depth_order = (
                    (1. - depth_order_config["depth_ratio"]) * render_pkg["expected_depth"]
                    + depth_order_config["depth_ratio"] * render_pkg["median_depth"]
                )
            else:
                depth_for_depth_order = render_pkg["median_depth"]
                
            depth_prior_loss, _, do_supervision_depth, lambda_depth_order = compute_depth_order_regularization(
                iteration=iteration,
                rendered_depth=depth_for_depth_order,
                depth_priors=depth_priors,
                viewpoint_idx=viewpoint_idx,
                gaussians=gaussians,
                config=depth_order_config,
            )
                
            loss = loss + depth_prior_loss
            depth_order_kick_on = lambda_depth_order > 0

        # Monocular prior (normals / depth from MoGe-2)
        if mono_kick_on:
            mono_loss, mono_terms = mono_prior.loss(viewpoint_cam, render_pkg, iteration)
            loss = loss + mono_loss
            if iteration % 1000 == 0 and mono_terms:
                print(f"[mono] it {iteration} weights {mono_prior.weights(iteration)} terms {mono_terms}")
            
        # Multiview Regularization
        if multiview_kick_on:
            
            if multiview_render is None:
                multiview_render_pkg = render_pkg
            else:
                multiview_render_pkg = multiview_render(
                    viewpoint_cam, gaussians, pipe, bg,
                    require_coord=False, 
                    require_depth=True,
                )
            
            multiview_render_pkg = compute_multiview_regularization(
                iteration=iteration,
                scene=scene,
                render_pkg=multiview_render_pkg,
                viewpoint_cam=viewpoint_cam,
                viewpoint_idx=viewpoint_idx,
                gaussians=gaussians,
                render_func=render,
                pipe=pipe,
                background=bg,
                multiview_config=multiview_config,
                multiview_state=multiview_state,
                kernel_size=0.0,
                rasterizer=args.rasterizer,
            )
            multiview_loss = multiview_render_pkg["multiview_loss"] * args.multiview_factor
            loss = loss + multiview_loss
        
        # Normal Field Regularization
        if normal_field_kick_on:
            if args.detach_gaussian_rendering:
                detached_render_pkg = {
                    "render": render_pkg["render"].detach(),
                    "median_depth": render_pkg["median_depth"].detach(),
                    "expected_depth": render_pkg["expected_depth"].detach(),
                    "normal": render_pkg["normal"].detach(),
                }
            
            normal_field_render_pkg = compute_normal_field_regularization(
                iteration=iteration,
                render_pkg=detached_render_pkg if args.detach_gaussian_rendering else render_pkg,
                viewpoint_cam=viewpoint_cam,
                viewpoint_idx=viewpoint_idx,
                gaussians=gaussians,
                scene=scene,
                pipe=pipe,
                background=bg,
                kernel_size=0.0,
                config=normal_field_config,
                normal_field_state=normal_field_state,
                render_func=partial(render, require_coord=False, require_depth=True),
                args=args,
            )
            normal_field_loss = normal_field_render_pkg["normal_field_loss"]
            loss = loss + normal_field_loss
            
            normal_field_alignment_loss = normal_field_render_pkg["normal_field_alignment_loss"]
            front_pivots_visibility_loss = normal_field_render_pkg["front_pivots_visibility_loss"]
            back_pivots_occlusion_loss = normal_field_render_pkg["back_pivots_occlusion_loss"]
            gaussian_flattening_loss = normal_field_render_pkg["gaussian_flattening_loss"]
            sdf_and_normal_field_consistency_loss = normal_field_render_pkg["sdf_and_normal_field_consistency_loss"]
            
        # Mesh-In-the-Loop Regularization
        if milo_kick_on:
            milo_pkg = compute_mesh_in_the_loop_regularization(
                iteration=iteration,
                train_cameras=all_viewpoints,
                viewpoint_cam=viewpoint_cam,
                viewpoint_idx=viewpoint_idx,
                render_pkg=render_pkg,
                gaussians=gaussians,
                pipe=pipe,
                background=bg,
                kernel_size=0.0,
                milo_config=milo_config,
                milo_state=milo_state,
                args=args,
            )
            milo_loss = milo_pkg["milo_loss"]
            loss = loss + milo_loss
            
            mesh_depth_loss = milo_pkg["mesh_depth_loss"]
            mesh_normal_loss = milo_pkg["mesh_normal_loss"]
            occupied_centers_loss = milo_pkg["occupied_centers_loss"]
        
        # ---Backward pass---
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # ---Final hollowing prune---
            # > Runs before the logging below, which saves the last iteration.
            hollow_has_changed = False
            if iteration == opt.iterations:
                hollow_has_changed = hollow.after_step(iteration)
                if hollow_has_changed and not use_mip_filter:
                    gaussians.reset_3D_filter()

            # ---Logging---
            (
                postfix_dict,
                ema_loss_for_log, 
                ema_depth_normal_loss_for_log, 
                ema_normal_field_alignment_loss_for_log,
                ema_front_pivots_visibility_loss_for_log,
                ema_back_pivots_occlusion_loss_for_log,
                ema_gaussian_flattening_loss_for_log,
                ema_sdf_and_normal_field_consistency_loss_for_log,
                ema_depth_order_loss_for_log,
                ema_multiview_loss_for_log,
                ema_mesh_depth_loss_for_log,
                ema_mesh_normal_loss_for_log,
                ema_occupied_centers_loss_for_log,
            ) = log_normal_field_training_progress(
                args, iteration, log_interval, progress_bar, run,
                scene, gaussians, pipe, opt, bg,
                viewpoint_idx, viewpoint_cam, render_pkg, 
                normal_field_render_pkg if normal_field_kick_on else None, 
                milo_pkg if milo_kick_on else None,
                do_supervision_depth if depth_order_kick_on else None,
                reg_kick_on, normal_field_kick_on, depth_order_kick_on, multiview_kick_on, milo_kick_on,
                loss, depth_normal_loss if reg_kick_on else None, 
                normal_field_alignment_loss if normal_field_kick_on else None,
                front_pivots_visibility_loss if normal_field_kick_on else None,
                back_pivots_occlusion_loss if normal_field_kick_on else None,
                gaussian_flattening_loss if normal_field_kick_on else None,
                sdf_and_normal_field_consistency_loss if normal_field_kick_on else None,
                depth_prior_loss if depth_order_kick_on else None,
                multiview_loss if multiview_kick_on else None,
                mesh_depth_loss if milo_kick_on else None,
                mesh_normal_loss if milo_kick_on else None,
                occupied_centers_loss if milo_kick_on else None,
                normal_field_config if normal_field_kick_on else None, 
                milo_config if milo_kick_on else None,
                postfix_dict, ema_loss_for_log, ema_depth_normal_loss_for_log, 
                ema_normal_field_alignment_loss_for_log,
                ema_front_pivots_visibility_loss_for_log,
                ema_back_pivots_occlusion_loss_for_log,
                ema_gaussian_flattening_loss_for_log,
                ema_sdf_and_normal_field_consistency_loss_for_log,
                ema_depth_order_loss_for_log, 
                ema_multiview_loss_for_log,
                ema_mesh_depth_loss_for_log,
                ema_mesh_normal_loss_for_log,
                ema_occupied_centers_loss_for_log,
                testing_iterations, saving_iterations, render,
            )
            # > The grid next to every saved PLY (not only checkpoints); a checkpoint at the same iteration overwrites
            # > it with the post-step grid below.
            if bilagrid is not None and iteration in saving_iterations:
                torch.save(bilagrid.state_dict(), os.path.join(scene.model_path, f"bilagrid{iteration}.pth"))
            
            if iteration % 100 == 0:
                if dataset.use_unbounded_opacity:
                    _opacity = gaussians.get_opacity_with_3D_filter.detach()
                    _contribution = gaussians.get_contribution(viewpoint_cam).detach()
                    print(f"Min contribution: {_contribution.min()}, Max contribution: {_contribution.max()}")
                    print(f"Min opacity: {_opacity.min()}, Max opacity: {_opacity.max()}")

            # ---Densification---
            gaussians_have_changed = hollow_has_changed
            if iteration < opt.densify_until_iter and not hollow_has_changed:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats_radegs(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if not args.use_max_size_threshold:
                        size_threshold = None
                    gaussians.densify_and_prune_radegs(
                        opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold,
                        use_abs_grad=args.use_abs_grad_for_densification,
                        viewpoint_cameras=scene.getTrainCameras().copy(),
                        budget=current_budget(iteration, opt.densify_from_iter, opt.densify_until_iter,
                                              opt.budget_cap, opt.budget_warmup_offset),
                    )
                    gaussians_have_changed = True
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras().copy()
                        )
                    else:
                        gaussians.reset_3D_filter()
                        
                if opt.opacity_reset and (
                    iteration % opt.opacity_reset_interval == 0
                    or (dataset.white_background and iteration == opt.densify_from_iter)
                ):
                    gaussians.reset_opacity()

                # ---gs-atoms B6: gated opacity decay, the replacement for the reset---
                # decay_opacity_ does logit(sigmoid(raw) * factor) in place on gated rows only. Safe under
                # the 3D Mip filter, whose coefficient depends only on scaling and filter_3D, but WRONG
                # under --use_unbounded_opacity, where the opacity activation is exp rather than sigmoid.
                if opt.opa_decay < 1.0 and atom_gate is not None:
                    if gaussians.use_unbounded_opacity:
                        raise ValueError("--opa_decay assumes a sigmoid opacity activation and cannot be "
                                         "combined with --use_unbounded_opacity.")
                    atoms_decay_opacity_(gaussians._opacity.data, atom_gate, opt.opa_decay)

            # ---gs-atoms B7: the per-epoch contribution prune---
            # Placed after densification so the row count is settled. Rows whose blend mass stayed below
            # contrib_min for a whole epoch die; newborns are exempt for their first window.
            if epoch_prune is not None and atom_contrib is not None:
                epoch_prune.update(atom_contrib, atom_gate)
                if epoch_prune.due() and iteration <= epoch_prune_until and not gaussians_have_changed:
                    dead = epoch_prune.rule()
                    if dead is not None and bool(dead.any()):
                        gaussians.prune_points(dead)
                        gaussians_have_changed = True
                        if use_mip_filter:
                            gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())
                        else:
                            gaussians.reset_3D_filter()
                    epoch_prune.start(gaussians.get_xyz.shape[0], gaussians.get_xyz.device)
                    
            # ---Non-maximal pruning---
            if iteration in args.non_maximal_pruning_iterations:
                print(f"[INFO] Pruning non-maximal Gaussians at iteration {iteration+1}.")
                print(f"        > Number of Gaussians before pruning: {gaussians._xyz.shape[0]}.")
                prune_non_maximal_gaussians(
                    gaussians=gaussians,
                    cameras=scene.getTrainCameras().copy(),
                    pipe=pipe,
                    background=bg,
                )
                gaussians_have_changed = True
                print(f"        > Number of Gaussians after pruning: {gaussians._xyz.shape[0]}.")
                if use_mip_filter:
                    gaussians.compute_3D_filter(
                        cameras=scene.getTrainCameras().copy()
                    )
                
            # ---Normal Field Densification---
            if args.use_normal_field:
                cond_1 = (
                    normal_field_kick_on 
                    and normal_field_config["use_densification"]
                )
                cond_2 = (
                    (iteration+1 >= normal_field_config["start_iter_densification"])
                    and (iteration+1 <= normal_field_config["end_iter_densification"])
                )
                cond_3 = (
                    (iteration+1 - normal_field_config["start_iter_densification"]) % normal_field_config["densify_every_n_iterations"] == 0
                )
                if cond_1 and cond_2 and cond_3:
                    print(f"[INFO] Densifying normal field at iteration {iteration+1}.")
                    print(f"        > Using normalization method: {normal_field_config['densification_normalization_method']}.")
                    print(f"        > Using normal computed from: {normal_field_config['densification_normal_to_use']}.")
                    print(f"        > Using normal errors quantile: {normal_field_config['densification_normal_errors_quantile']}.")
                    print(f"        > Maintaining constant volume: {normal_field_config['maintain_constant_volume']}.")
                    print(f"        > Number of Gaussians before densification: {gaussians._xyz.shape[0]}.")
                    densify_normal_field(
                        iteration=iteration, 
                        gaussians=gaussians, 
                        cameras=scene.getTrainCameras().copy(),
                        scene=scene, 
                        pipe=pipe, 
                        background=bg, 
                        kernel_size=0.0, 
                        config=normal_field_config,
                        normal_field_state=normal_field_state, 
                        render_func=render, 
                        args=args,
                        maintain_constant_volume=normal_field_config["maintain_constant_volume"],
                    )
                    gaussians_have_changed = True
                    print(f"        > Number of Gaussians after densification: {gaussians._xyz.shape[0]}.")
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras().copy()
                        )
                        
                    if normal_field_config["reset_normals_after_densification"]:
                        print(f"[INFO] Resetting normal features after densification.")
                        gaussians.reset_normal_features(
                            reset_directions=normal_field_config["reset_normal_directions"],
                            reset_signs=normal_field_config["reset_normal_signs"],
                        )
                        
            # ---Normal field pruning---
            if args.use_normal_field:
                cond_1 = (
                    normal_field_kick_on 
                    and normal_field_config["use_pruning"]
                )
                cond_2 = (
                    (iteration+1 >= normal_field_config["start_iter_pruning"])
                    and (iteration+1 <= normal_field_config["end_iter_pruning"])
                )
                cond_3 = (
                    (iteration+1 - normal_field_config["start_iter_pruning"]) % normal_field_config["prune_every_n_iterations"] == 0
                )
                if cond_1 and cond_2 and cond_3:
                    print(f"[INFO] Pruning non-maximal Gaussians at iteration {iteration+1}.")
                    print(f"        > Number of Gaussians before pruning: {gaussians._xyz.shape[0]}.")
                    prune_non_maximal_gaussians(
                        gaussians=gaussians,
                        cameras=scene.getTrainCameras().copy(),
                        pipe=pipe,
                        background=bg,
                    )
                    gaussians_have_changed = True
                    print(f"        > Number of Gaussians after pruning: {gaussians._xyz.shape[0]}.")
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras().copy()
                        )

            # ---Hollowing: opacity decay, unseen pruning, see-through carving---
            # > HollowSchedule recomputes the 3D filter itself after a prune.
            if iteration < opt.iterations and hollow.after_step(iteration):
                gaussians_have_changed = True
                if not use_mip_filter:
                    gaussians.reset_3D_filter()

            # ---Reset Normal field state if Gaussians have changed---
            if normal_field_kick_on and gaussians_have_changed:
                normal_field_state = reset_normal_field_state_at_next_iteration(normal_field_state)
            
            # ---Reset MILO state if Gaussians have changed---
            if milo_kick_on and gaussians_have_changed:
                milo_state = reset_milo_state_at_next_iteration(milo_state)
            
            # ---Update 3D Mip Filter---
            if use_mip_filter and (iteration > opt.densify_until_iter) and (
                (iteration == args.warn_until_iter)
                or (iteration % args.update_mip_filter_every == 0)
            ):
                if iteration < opt.iterations - args.update_mip_filter_every:
                    gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())
                else:
                    print(f"[INFO] Skipping 3D Mip Filter update at iteration {iteration}")

            # ---Optimizer step---
            if iteration < opt.iterations:
                if gaussians.use_appearance_network or gaussians.use_exposure_compensation:
                    gaussians.optimizer.step()
                else:
                    visible = radii>0
                    # > The free-space prior also reaches Gaussians outside the view.
                    if hollow.touched is not None:
                        visible = visible | hollow.touched
                    if atom_gate is not None:
                        # The contrib gate is the row set gs-atoms updates; keep the hollow union or the
                        # free-space prior can no longer move the rows it samples (B5).
                        visible = visible | atom_gate
                    if opt.optimizer_type == "view_adam":
                        gaussians.optimizer.step(visible, row_signal=atom_cover)
                    elif opt.optimizer_type == "row_adam":
                        gaussians.optimizer.step(visible)
                    else:
                        gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none = True)
                if bilagrid_on:
                    bilagrid.step()
                    bilagrid.zero_grad()

            # ---Save checkpoint---
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if bilagrid is not None:
                    torch.save(bilagrid.state_dict(), scene.model_path + "/bilagrid" + str(iteration) + ".pth")

        if iteration % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print('Num of Gaussians: %d'%(gaussians._xyz.shape[0]))

    # ---Gaussian statistics (fog, size, time)---
    with torch.no_grad():
        opacity = gaussians.get_opacity_with_3D_filter.squeeze(-1)
        p = opacity.clamp(1e-6, 1.0 - 1e-6)
        stats = {
            "n_gaussians": int(opacity.numel()),
            "opacity_below_0.05": float((opacity < 0.05).float().mean()),
            "opacity_below_0.1": float((opacity < 0.1).float().mean()),
            "opacity_median": float(opacity.median()),
            "opacity_entropy": float((-(p * p.log() + (1 - p) * (1 - p).log())).mean()),
            "train_seconds": time.time() - time_start,
            "lpips_only_images": len(lpips_names),
            "appearance": args.appearance,
            "lambda_raw": args.lambda_raw,
            "bilagrid": None if bilagrid is None else {**robust.bilagrid_config(args), **bilagrid.drift_stats()},
            "confidence": bool(args.confidence),
            "hollow": bool(hollow.enabled),
        }
    with open(os.path.join(scene.model_path, "gs_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Stats: {stats}")

    if WANDB_FOUND:
        run.finish()
    
    return 


def prepare_output_and_logger(dataset, args):    
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))

    # Create WandB run       
    global WANDB_FOUND
    WANDB_FOUND = (
        WANDB_FOUND
        and (args.wandb_project is not None)
        and (args.wandb_entity is not None)
    )
    if WANDB_FOUND:
        run = wandb.init(
            name=args.wandb_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
        )
    else:
        run=None
        print("[INFO] WandB not found, skipping logging.")
    return run


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # ----- Usual arguments -----
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=-1)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # ----- Rasterization technique -----
    parser.add_argument("--rasterizer", type=str, default="radegs", choices=["radegs", "ours"])
    
    # ----- Normal Field -----
    parser.add_argument("--no_normal_field", action="store_true")
    parser.add_argument("--normal_field_config", type=str, default="default_regular_densification")
    # Gaussians management
    parser.add_argument("--dense_gaussians", action="store_true")
    parser.add_argument("--detach_gaussian_rendering", action="store_true")

    # ----- Densification and Simplification -----
    # > Inspired by Mini-Splatting2.
    # > Used for pruning, densification and Gaussian pivots selection.
    parser.add_argument("--N_max_gaussians", type=int, default=None,
        help="Cap Gaussian count during Normal Field Densification. If the next densification would exceed this, only the highest-error Gaussians are added up to the cap. None = no limit.")
    parser.add_argument("--warn_until_iter", type=int, default=3000)
    parser.add_argument("--non_maximal_pruning_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--use_max_size_threshold", action=BooleanOptionalAction, default=False)
    
    # ----- Depth-Normal consistency Regularization -----
    # > Inspired by 2DGS, GOF, RaDe-GS...
    parser.add_argument("--regularization_from_iter", type=int, default = 7_000)
    parser.add_argument("--lambda_depth_normal", type=float, default = 0.05)
    parser.add_argument("--mask_depth_normal", action="store_true")
    
    # ----- Multiview Regularization -----
    parser.add_argument("--multiview", action=BooleanOptionalAction, default=True)
    parser.add_argument("--multiview_config", type=str, default="fast")
    parser.add_argument("--multiview_rasterizer", type=str, default=None)
    parser.add_argument("--multiview_factor", type=float, default=1.0)
    
    # ----- Mesh-In-the-Loop Regularization -----
    parser.add_argument("--milo", action="store_true")
    parser.add_argument("--milo_config", type=str, default="default_regular_densification")
    
    # ----- Depth Order Regularization (Learned Prior) -----
    # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
    # > In the paper, MILo does not rely on any learned prior.
    parser.add_argument("--depth_order", action="store_true")
    parser.add_argument("--depth_order_config", type=str, default="default")
    # ----- Monocular prior (MoGe-2 depth + normals, mono_prior_prepare.py) for weakly conditioned scenes -----
    parser.add_argument("--mono_prior_dir", type=str, default=None)
    parser.add_argument("--mono_prior_config", type=str, default="normal")
    parser.add_argument("--mono_prior_gate", type=str, default=None, choices=["auto", "on", "off"],
                        help="overrides the config's gate")

    # ----- 3D Mip Filter -----
    # > Inspired by Mip-Splatting.
    parser.add_argument("--disable_mip_filter", action="store_true", default=False)
    parser.add_argument("--update_mip_filter_every", type=int, default=100)

    # ----- Appearance Network for Exposure-aware loss -----
    # > Inspired by GOF.
    parser.add_argument("--decoupled_appearance", action="store_true")
    # > Inspired by PGSR. Default None: on unless --appearance or --decoupled_appearance picks another mode.
    parser.add_argument("--exposure_compensation", action=BooleanOptionalAction, default=None)

    # ----- Robustness to non-clean data (robust/) -----
    # > --appearance {none,exposure,network,bilagrid,bilagrid+decoupled}; unset = GW's legacy flags above.
    robust.add_args(parser)
    parser.set_defaults(appearance=None)

    # ----- Hollowing (hollow/) -----
    HollowSchedule.add_args(parser)

    # ----- BEA-GS labels -----
    parser.add_argument("--labels", type=str, default=None,
        help="labels.pt (int32, one per Gaussian) registered as a per-Gaussian extra; needs a matching --start_checkpoint.")

    # ----- ArtiFixer3D distillation -----
    parser.add_argument("--lpips_images", type=str, default=None,
        help="File listing image names supervised by lambda_lpips * LPIPS only (ArtiFixer3D generated frames).")
    parser.add_argument("--lambda_lpips", type=float, default=0.1)

    # ----- PGSR losses -----
    parser.add_argument("--use_scale_loss", action="store_true")
    parser.add_argument("--scale_loss_weight", type=float, default=100.0)
    parser.add_argument("--weight_by_img_grad", action="store_true")

    # ----- Logging -----
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    if not -1 in args.test_iterations:
        args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    args.use_normal_field = not args.no_normal_field
    resolve_appearance(args)
    robust.check_args(args)
    print(f"[INFO] Appearance: {args.appearance}.")
    
    if args.port == -1:
        args.port = np.random.randint(5000, 9000)
        print(f"Using random port: {args.port}")
        
    # Load multiview regularization config
    if args.multiview:
        # Get multiview config file
        multiview_config_file = os.path.join(BASE_DIR, "configs", "multiview", f"{args.multiview_config}.yaml")
        with open(multiview_config_file, "r") as f:
            multiview_config = yaml.safe_load(f)
        print(f"[INFO] Using multiview regularization with config: {args.multiview_config}")
    else:
        multiview_config = None
        
    # Load mesh-in-the-loop regularization config
    if args.milo:
        # Get mesh regularization config file
        milo_config_file = os.path.join(BASE_DIR, "configs", "mesh_in_the_loop", f"{args.milo_config}.yaml")
        with open(milo_config_file, "r") as f:
            milo_config = yaml.safe_load(f)
        print(f"[INFO] Using mesh-in-the-loop regularization with config: {args.milo_config}")
    else:
        milo_config = None
    
    # Load depth order regularization config (not used in MILo paper)
    if args.depth_order:
        # Get depth order config file
        depth_order_config_file = os.path.join(BASE_DIR, "configs", "depth_order", f"{args.depth_order_config}.yaml")
        with open(depth_order_config_file, "r") as f:
            depth_order_config = yaml.safe_load(f)
        print(f"[INFO] Using depth order regularization with config: {args.depth_order_config}")
    else:
        depth_order_config = None
        
    # Load mesh-in-the-loop regularization config
    if args.use_normal_field:
        # Get mesh regularization config file
        normal_field_config_file = os.path.join(BASE_DIR, "configs", "normal_field", f"{args.normal_field_config}.yaml")
        with open(normal_field_config_file, "r") as f:
            normal_field_config = yaml.safe_load(f)
        print(f"[INFO] Using normal field with config: {args.normal_field_config}")
    else:
        normal_field_config = None
    
    # Message for detach_gaussian_rendering
    if args.detach_gaussian_rendering:
        print(f"[INFO] Detaching Gaussian rendering for mesh regularization.")
    
    # Import rendering function
    print(f"[INFO] Using {args.rasterizer} as rasterizer.")
    if args.rasterizer == "radegs":
        from gaussian_renderer.radegs import render_radegs as render
        from gaussian_renderer.radegs import integrate_radegs as integrate
        args.use_abs_grad_for_densification = True
    elif args.rasterizer == "ours":
        from gaussian_renderer.ours import render_ours as render
        from gaussian_renderer.ours import integrate_ours as integrate
        args.use_abs_grad_for_densification = True
        args.mask_depth_normal = True
        print(f"[INFO] Using Ours rasterizer. Setting mask_depth_normal to True.")
    else:
        raise ValueError(f"Invalid rasterizer: {args.rasterizer}")
    
    if args.multiview_rasterizer == "ours":
        print(f"[INFO] Using Ours rasterizer for multiview regularization.")
        from gaussian_renderer.ours import render_ours as multiview_render
    elif args.multiview_rasterizer is None:
        print(f"[INFO] Using default rasterizer for multiview regularization.")
        multiview_render = None
    else:
        raise ValueError(f"Invalid multiview rasterizer: {args.multiview_rasterizer}")
        
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    torch.cuda.synchronize()
    time_start=time.time()
    
    training(
        lp.extract(args), op.extract(args), pp.extract(args), 
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args,
        depth_order_config,
        normal_field_config,
        multiview_config,
        milo_config,
        args.log_interval,
    )

    torch.cuda.synchronize()
    time_end=time.time()
    time_total=time_end-time_start
    print('Training time: %fs'%(time_total))

    time_txt_path=os.path.join(args.model_path, r'time.txt')
    with open(time_txt_path, 'w') as f:  
        f.write(str(time_total)) 

    # All done
    print("\nTraining complete.")
