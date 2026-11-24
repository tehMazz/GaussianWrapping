"""Seed2GS (arXiv 2608.11928, reimplemented): training-free, camera-free text-prompted object extraction.

    python gaussian_wrapping/seed2gs.py -m <model> --iteration N --prompt "<text>" [--prompt "<text 2>" ...]
        [--ref_view <image_name|index>] [--out <dir> | --out_root <dir>] [--track_side 1008] [--mask_split all]

Per prompt:
1. render the reference view (a training camera, default the first) with render_ours;
2. SAM3 text prompt on that render (seg/sam3_runner.py, a separate SAM3 Python env);
3. seed lift: Gaussian centres inside the mask at the rendered median depth, largest 3D connected component;
4. virtual cameras: a flat orbit and a spiral (--n_views each) about the seed centroid, starting at the reference camera,
   rendered once as jpg at the reference resolution, short side capped to --track_side;
5. SAM3 tracker propagates the reference mask along each orbit (no repeated detection);
6. votes (alpha*T per Gaussian, 0 = background, 1 = object) over the virtual views, FlashSplat closed-form rule;
7. writes <out>/labels.pt (int32 [P], 1 = object), <out>/masks/<image_name>.npy ((2, H, W) float16, the BEA-GS mask
   format of beags/masks.py at the camera resolution: label coverage >= 0.5, see seg/masks.py) on the training
   cameras, and on the test cameras too with --mask_split all (default: like the provided SAM2 mask dirs, so
   beags_eval.py --masks can match GT ids and seed2gs_eval.py can score test views; beags_prepare.py reads only
   the training cameras), and <out>/seed2gs.json (scores, seed, tracking, per-step and SAM3 per-op seconds).
   A prompt SAM3 does not find (or whose seed is empty) writes <out>/seed2gs_failed.json instead and removes the
   previous labels.pt / seed2gs.json / seed2gs_eval.json, so done markers (seed2gs.json, labels.pt) and
   seed2gs_eval.py never take a stale result for current; the run exits non-zero after the other prompts.
Default --out: <out_root>/<prompt slug>, --out_root output/seed2gs/<scene dir name>; --out only with one prompt.
beags_prepare.py --masks <out>/masks runs unchanged on the result.

Speed (fern smoke: 175 s, of which ~110 s SAM3 import + load and 42 s the first SAM3 call; tracking 48 frames ~5 s):
- the SAM3 worker is spawned before this script imports torch, so its imports and checkpoint read overlap with ours;
- sam3_runner.py prefetches the checkpoint and skips random init (identical weights); Triton kernels compiled by the
  first call are cached in TRITON_CACHE_DIR, so later runs skip that JIT;
- several --prompt share one SAM3 worker, one model load and one scene load (LERF figurines: 7 prompts, one job);
- --track_side: SAM3 resizes every frame to a 1008 x 1008 square (aspect ignored), so a frame whose SHORT side is
  above 1008 only costs rendering and JPEG IO; it is scaled down until the short side is 1008 (both axes still >=
  SAM3's input, so tracking sees the same detail). The virtual cameras are built at that size (same FoV) and their
  votes are rendered at it too, so masks need no upsampling. A no-op when the short side is <= 1008 (LLFF -r 8,
  Mip-NeRF 360 images_4, LERF); above it only the orbit votes count fewer pixels per Gaussian (the FlashSplat rule
  uses the vote ratio, so labels barely move).
- Kept as is: one SAM3 session per orbit. Concatenating both orbits in one video saves one init_state (~0.5 s) but
  feeds flat-orbit frames into the spiral's memory bank, so results would change.
"""
import json
import os
import sys
import time
from argparse import ArgumentParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'submodules'))

_SAM = None
if __name__ == "__main__" and not {"-h", "--help"} & set(sys.argv):
    from seg.sam3 import worker_from_argv  # stdlib only: SAM3 starts loading before we import torch
    _SAM = worker_from_argv()

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from beags.labels import label_coverage
from gaussian_renderer.beags_api import render_depth_alpha, render_labels
from gaussian_renderer.ours import render_ours
from scene import Scene, GaussianModel
from scene.cameras import Camera
from seg.lift import knn_spacing, largest_component, seed_from_mask
from seg.masks import capped_size, prompt_slug as slug, save_beags_mask
from seg.orbits import flat_orbit, spiral, world_up
from seg.sam3 import add_worker_args, worker_from_argv
from seg.votes import flashsplat_labels
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal


FAILED_JSON = "seed2gs_failed.json"  # a failed prompt's record (never seed2gs.json: that is the done marker)


class PromptFailed(Exception):
    pass


def pick_reference(cameras, ref_view):
    if ref_view is None:
        return cameras[0]
    if ref_view.isdigit():
        return cameras[int(ref_view)]
    name = os.path.splitext(ref_view)[0]
    for cam in cameras:
        if cam.image_name == ref_view or os.path.splitext(cam.image_name)[0] == name:
            return cam
    raise ValueError(f"no training camera named {ref_view}")


def cam_to_world(cam):
    """OpenCV camera-to-world (4, 4) of a GW camera."""
    return np.linalg.inv(cam.world_view_transform.T.double().cpu().numpy())


def virtual_camera(ref, c2w, name, uid, size=None):
    """GW Camera at OpenCV c2w with the reference FoV, at `size` (H, W) (default the reference's); dummy CPU image."""
    w2c = np.linalg.inv(c2w)
    H, W = size or (int(ref.image_height), int(ref.image_width))
    image = torch.zeros(3, H, W)
    return Camera(colmap_id=-1, R=w2c[:3, :3].T, T=w2c[:3, 3], FoVx=ref.FoVx, FoVy=ref.FoVy, image=image,
                  gt_alpha_mask=None, image_name=name, uid=uid, data_device="cpu")


def save_render(image, path):
    Image.fromarray((image.clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy()).save(path, quality=95)


def out_dirs(prompts, out, out_root):
    if out:
        if len(prompts) > 1:
            raise SystemExit("[seed2gs] --out takes one --prompt; use --out_root with several")
        return [os.path.abspath(out)]
    dirs = [os.path.abspath(os.path.join(out_root, slug(p))) for p in prompts]
    if len(set(dirs)) < len(dirs):
        raise SystemExit(f"[seed2gs] two prompts share a directory name: {prompts}")
    return dirs


def track_orbits(args, sam, gaussians, pipe, bg, cameras, ref, ref_mask_path, anchor, frames_dir, out, info, timings):
    """Steps 4-5 for one prompt -> [(virtual camera, orbit mask path)] of the tracked views."""
    t = time.time()
    up = world_up(np.stack([cam_to_world(c) for c in cameras]))
    ref_c2w = cam_to_world(ref)
    size = capped_size(ref.image_height, ref.image_width, args.track_side)
    orbits = {"flat": flat_orbit(ref_c2w, anchor, up, args.n_views),
              "spiral": spiral(ref_c2w, anchor, up, args.n_views, args.spiral_deg)}
    cams, n = {}, 0
    for name, c2ws in orbits.items():
        odir = os.path.join(frames_dir, name)
        os.makedirs(odir, exist_ok=True)
        cams[name] = [virtual_camera(ref, c2w, f"{name}_{i:03d}", 100000 + n + i, size) for i, c2w in enumerate(c2ws)]
        n += len(c2ws)
        for i, cam in enumerate(cams[name]):
            save_render(render_ours(cam, gaussians, pipe, bg)["render"], os.path.join(odir, f"{i}.jpg"))
    torch.cuda.synchronize()
    timings["render_orbits"] = time.time() - t

    t = time.time()
    views, tracked = [], {}
    for name in orbits:
        mdir = os.path.join(out, "orbit_masks", name)
        tracked[name] = sam.track(os.path.join(frames_dir, name), ref_mask_path, mdir)
        print(f"[seed2gs] {name}: object tracked on {tracked[name]['nonempty']}/{tracked[name]['frames']} views")
        first = 1 if views else 0  # every orbit starts at the reference camera: vote with it once
        views += [(cam, os.path.join(mdir, f"{i}.npy")) for i, cam in enumerate(cams[name])
                  if i >= first and os.path.exists(os.path.join(mdir, f"{i}.npy"))]
    timings["track"] = time.time() - t
    info.update(up=up.tolist(), track_size=list(size), tracked=tracked)
    return views


def segment(args, prompt, out, sam, gaussians, pipe, bg, cameras, ref, ref_png):
    """Steps 2-5 for one prompt (SAM3 side). -> (info, timings, views)."""
    timings, info = {}, dict(prompt=prompt, out=out)
    frames_dir = os.path.join(out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    H, W = int(ref.image_height), int(ref.image_width)

    # 2. SAM3 text prompt
    t = time.time()
    ref_mask_path = os.path.join(out, "reference_mask.npy")
    det = sam.detect(ref_png, prompt, ref_mask_path)
    timings["detect"] = time.time() - t
    info.update(sam3_score=det["score"], sam3_scores=det["scores"], sam3_detect_seconds=det.get("seconds"))
    if not det["found"]:
        raise PromptFailed(f"SAM3 found no '{prompt}' in {ref.image_name}; try another --ref_view")
    ref_mask = torch.from_numpy(np.load(ref_mask_path)).cuda()
    print(f"[seed2gs] '{prompt}' on {ref.image_name}: SAM3 score {det['score']:.3f}, {int(ref_mask.sum())} px")

    # 3. seed lift
    t = time.time()
    depth, _ = render_depth_alpha(ref, gaussians, pipe)
    xyz = gaussians.get_xyz
    intrinsics = (fov2focal(ref.FoVx, W), fov2focal(ref.FoVy, H), (W - 1) / 2, (H - 1) / 2)
    seed = seed_from_mask(xyz, ref.world_view_transform, intrinsics, ref_mask, depth, args.depth_tol)
    seed &= gaussians.get_opacity_with_3D_filter[:, 0] >= args.min_opacity
    seed_idx = torch.nonzero(seed).squeeze(1).cpu().numpy()
    info.update(reference_mask_px=int(ref_mask.sum()), seed_raw=len(seed_idx))
    if len(seed_idx) < 10:
        raise PromptFailed(f"only {len(seed_idx)} seed Gaussians: the mask misses the rendered surface")
    points = xyz[seed].cpu().numpy().astype(np.float64)
    voxel = knn_spacing(points, args.knn) * args.voxel_mult
    keep = largest_component(points, voxel)
    anchor = points[keep].mean(0)
    print(f"[seed2gs] seed {int(keep.sum())}/{len(seed_idx)} Gaussians (voxel {voxel:.4g}), centre {anchor.round(3).tolist()}")
    timings["lift"] = time.time() - t
    info.update(seed=int(keep.sum()), voxel=voxel, anchor=anchor.tolist())

    # 4-5. virtual orbits, rendered and tracked
    views = track_orbits(args, sam, gaussians, pipe, bg, cameras, ref, ref_mask_path, anchor, frames_dir, out,
                         info, timings)
    return info, timings, views


@torch.no_grad()
def label_and_mask(args, out, views, gaussians, pipe, bg, mask_cameras, info, timings):
    """Steps 6-7 for one prompt."""
    # 6. votes over the virtual views, FlashSplat rule
    t = time.time()
    n = gaussians.get_xyz.shape[0]
    votes = torch.zeros((n, 2), device="cuda")
    dummy = torch.zeros(n, dtype=torch.int32, device="cuda")  # labels only switch voting on
    for cam, mask_path in views:
        pixel_labels = torch.from_numpy(np.load(mask_path)).cuda().int()
        votes += render_labels(cam, gaussians, pipe, bg, gaussian_labels=dummy, pixel_labels=pixel_labels,
                               num_labels=2)["votes"]
    labels = flashsplat_labels(votes, args.gamma)
    torch.save(labels.cpu(), os.path.join(out, "labels.pt"))
    n_obj = int(labels.sum())
    print(f"[seed2gs] {n_obj}/{n} Gaussians labelled object ({int((votes.sum(1) > 0).sum())} voted)")
    timings["votes"] = time.time() - t

    # 7. masks on the training (and test) cameras
    t = time.time()
    masks_dir = os.path.join(out, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    coverage_px = {}
    for split, cams in mask_cameras.items():
        coverage_px[split] = {}
        for cam in cams:
            obj = (label_coverage(cam, gaussians, pipe, labels, 1) >= 0.5).cpu().numpy()
            save_beags_mask(masks_dir, cam.image_name, obj)
            coverage_px[split][cam.image_name] = int(obj.sum())
    timings["masks"] = time.time() - t
    info.update(n_gaussians=n, n_object=n_obj, n_voting_views=len(views),
                views_with_object={s: sum(v > 0 for v in c.values()) for s, c in coverage_px.items()},
                n_cameras={s: len(c) for s, c in coverage_px.items()})


def seed2gs(args, dataset, pipe, sam=None):
    t0 = time.time()
    prompts = args.prompt
    outs = out_dirs(prompts, args.out, args.out_root or os.path.join(
        ROOT_DIR, "output", "seed2gs", os.path.basename(os.path.normpath(dataset.source_path))))
    for p, o in zip(prompts, outs):
        print(f"[seed2gs] '{p}' -> {o}")
    sam = sam or worker_from_argv()  # usually spawned at import time, so it loads while the scene loads

    results, failed = [], []
    try:
        with torch.no_grad():
            gaussians = GaussianModel(dataset.sh_degree, use_mip_filter=not args.disable_mip_filter,
                                      use_unbounded_opacity=dataset.use_unbounded_opacity)
            scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
            cameras = scene.getTrainCameras()
            mask_cameras = {"train": cameras}
            if args.mask_split == "all":
                mask_cameras["test"] = scene.getTestCameras()
            bg = torch.tensor([1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0], device="cuda")
            ref = pick_reference(cameras, args.ref_view)
            load_s = time.time() - t0

            # 1. reference render (shared by every prompt)
            os.makedirs(outs[0], exist_ok=True)
            ref_png = os.path.join(outs[0], "reference.png")
            save_render(render_ours(ref, gaussians, pipe, bg)["render"], ref_png)

            t = time.time()
            ready = sam.wait_ready()
            sam3_wait = time.time() - t
            print(f"[seed2gs] SAM3 ready {ready['since_start']:.1f}s after spawn (waited {sam3_wait:.1f}s here): "
                  f"{json.dumps({k: round(v, 1) for k, v in ready['seconds'].items()})}")

            for i, (prompt, out) in enumerate(zip(prompts, outs)):
                try:
                    info, timings, views = segment(args, prompt, out, sam, gaussians, pipe, bg, cameras, ref, ref_png)
                except PromptFailed as e:
                    print(f"[seed2gs] '{prompt}': {e}")
                    failed.append(prompt)
                    os.makedirs(out, exist_ok=True)
                    for f in ("labels.pt", "seed2gs.json", "seed2gs_eval.json"):  # a stale result must not look current
                        if os.path.exists(os.path.join(out, f)):
                            os.remove(os.path.join(out, f))
                    with open(os.path.join(out, FAILED_JSON), "w") as f:
                        json.dump(dict(prompt=prompt, failed=str(e), model_path=dataset.model_path,
                                       iteration=scene.loaded_iter, reference=ref.image_name), f, indent=2)
                    continue
                if i == 0:
                    timings.update(load=load_s, sam3_wait=sam3_wait)
                results.append((out, info, timings, views))
    finally:
        sam.close()

    with torch.no_grad():
        for out, info, timings, views in results:
            label_and_mask(args, out, views, gaussians, pipe, bg, mask_cameras, info, timings)
            timings["total"] = sum(timings.values())  # this prompt's share; the first prompt carries load + SAM3 wait
            info.update(model_path=dataset.model_path, iteration=scene.loaded_iter, reference=ref.image_name,
                        n_views=args.n_views, spiral_deg=args.spiral_deg, track_side=args.track_side,
                        gamma=args.gamma, depth_tol=args.depth_tol, mask_split=args.mask_split,
                        sam3_ready=ready, seconds=timings, run_seconds=time.time() - t0, n_prompts=len(prompts))
            with open(os.path.join(out, "seed2gs.json"), "w") as f:
                json.dump(info, f, indent=2)
            if os.path.exists(os.path.join(out, FAILED_JSON)):
                os.remove(os.path.join(out, FAILED_JSON))
            print(f"[seed2gs] '{info['prompt']}' done: {out}")
    print(f"[seed2gs] {len(results)}/{len(prompts)} prompts in {time.time() - t0:.1f}s")
    if failed:
        raise SystemExit(f"[seed2gs] no object for: {failed}")
    return outs


if __name__ == "__main__":
    parser = ArgumentParser(description="Seed2GS: text-prompted object labels and masks on a trained GW model")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--prompt", required=True, action="append", type=str,
                        help="SAM3 text prompt; repeat for several objects (one SAM3 load, one output dir each)")
    parser.add_argument("--ref_view", default=None, type=str, help="training camera image name or index (default: first)")
    parser.add_argument("--out", default=None, type=str, help="output dir (one prompt only)")
    parser.add_argument("--out_root", default=None, type=str,
                        help="per-prompt dirs <out_root>/<prompt slug>; default output/seed2gs/<scene dir name>")
    parser.add_argument("--n_views", default=24, type=int, help="views per orbit (flat and spiral)")
    parser.add_argument("--spiral_deg", default=30.0, type=float, help="elevation the spiral gains over its turn")
    parser.add_argument("--track_side", default=1008, type=int,
                        help="orbit frames (and their votes) scaled down until their SHORT side is this many px "
                             "(SAM3's square input side); 0 = reference size")
    parser.add_argument("--mask_split", default="all", choices=["train", "all"],
                        help="cameras that get masks/<image_name>.npy: training only, or training + test")
    parser.add_argument("--depth_tol", default=0.05, type=float, help="seed: |z - median depth| <= tol * depth")
    parser.add_argument("--min_opacity", default=0.1, type=float, help="seed: minimum filtered opacity")
    parser.add_argument("--knn", default=8, type=int, help="voxel = median k-NN spacing of the seed x --voxel_mult")
    parser.add_argument("--voxel_mult", default=1.0, type=float)
    parser.add_argument("--disable_mip_filter", action="store_true", help="the model was trained with --disable_mip_filter")
    parser.add_argument("--gamma", default=0.0, type=float, help="FlashSplat background bias in [-1, 1]")
    add_worker_args(parser)
    parser.add_argument("--quiet", action="store_true")
    try:
        args = get_combined_args(parser)
        safe_state(args.quiet)
        seed2gs(args, model.extract(args), pipeline.extract(args), _SAM)
    except BaseException:  # bad flags, no object, crash: do not leave the SAM3 worker loading
        if _SAM is not None:
            _SAM.kill()
        raise
