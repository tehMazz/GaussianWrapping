"""Background-evidence ledger: which Gaussians were never constrained by a photo of real background.

    python gaussian_wrapping/beags/artifact_ledger.py -m <model> [--iteration N] --remove <k> --masks <dir>
        --out <dir> [--vis] [--source_path <COLMAP>] [--images <folder>]

A Gaussian that only ever accumulated render weight UNDER the object's 2D mask was never supervised by a photo
showing what is actually behind the object. After the object is deleted it has nothing holding it to the truth:
that is the smear (HANDOFF 7.2). The existing hollow ledger (hollow/ledger.py) sums alpha*T over ALL pixels, so
it cannot tell these apart from well-supervised Gaussians; this splits the same sum by pixel label.

On the FULL scene (object present, so the training-time occlusion T is reproduced), for every training view:
    bg_i  += sum of alpha_i T_i over pixels OUTSIDE the object mask
    obj_i += sum of alpha_i T_i over pixels INSIDE  the object mask
and the per-Gaussian score is the share with no background evidence behind it:
    s_i = obj_i / (bg_i + obj_i)        1 = only ever seen under the object, 0 = always had background evidence
Rendering s of the scene WITHOUT label k, normalized by alpha, gives a continuous per-view artifact score in
[0, 1]. No depth, no post-removal geometry, no SAM: the score lives on Gaussians, so the per-view maps are
multi-view consistent by construction.
Writes into --out (never inside the model dir):
    <image stem>.png  uint8 L, 255 * A, white = unsupported = artifact
    ledger.pt         bg, obj, score per Gaussian + meta, reused while the source PLY and masks match
    info.json         per-view mean A and the per-Gaussian score distribution
    vis/              with --vis: photo | scene after removal | A heatmap, for up to 8 views
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

ENCODING = "png8: 255 * A, A = rendered score / alpha without the removed label; high = no background evidence"
HIGH = 0.5


def encode(a):
    return np.round(np.clip(np.nan_to_num(np.asarray(a, np.float32), nan=0.0), 0, 1) * 255).astype(np.uint8)


def decode(png):
    return np.asarray(png, dtype=np.float32) / 255.0


def heatmap(a):
    """(H, W) in [0, 1] -> (H, W, 3) uint8, black -> red -> yellow -> white."""
    a = np.clip(a, 0, 1)[..., None]
    return np.round(np.concatenate([np.clip(3 * a, 0, 1), np.clip(3 * a - 1, 0, 1),
                                    np.clip(3 * a - 2, 0, 1)], -1) * 255).astype(np.uint8)


def evidence_score(bg, obj):
    """(P,) in [0, 1]: share of a Gaussian's render weight that came from under the object mask; 0 when unseen."""
    import torch
    total = bg + obj
    return torch.where(total > 0, obj / total.clamp_min(1e-12), torch.zeros_like(total))


def build_ledger(gaussians, cameras, pipe, remove):
    """(bg, obj) per Gaussian over the training views, split by the object mask (full scene, object present)."""
    import torch
    from gaussian_renderer.beags_api import render_labels
    n = gaussians.get_xyz.shape[0]
    bg = torch.zeros(n, device="cuda")
    obj = torch.zeros(n, device="cuda")
    bgcol = torch.zeros(3, device="cuda")
    gaussian_labels = torch.zeros(n, dtype=torch.int32, device="cuda")  # votes do not depend on these
    used = 0
    with torch.no_grad():
        for cam in cameras:
            pixel_labels = cam.masks.labels
            if int((pixel_labels >= 0).sum()) == 0:  # no mask for this view: it cannot say where the object is
                continue
            num_labels = int(pixel_labels.max().item()) + 1
            if num_labels <= remove:
                continue
            out = render_labels(cam, gaussians, pipe, bgcol, gaussian_labels=gaussian_labels,
                                pixel_labels=pixel_labels, num_labels=num_labels)
            votes = out["votes"]
            obj += votes[:, remove]
            bg += votes.sum(1) - votes[:, remove]
            used += 1
    return bg, obj, used


def render_view(cam, gaussians, pipe, score, keep, render_fn):
    """(A, rgb): alpha-weighted mean score at each pixel of the scene without the removed label, and its render."""
    import torch
    feat = torch.stack([score, torch.zeros_like(score), torch.zeros_like(score)], dim=1)
    bg = torch.zeros(3, device=score.device)
    pkg = render_fn(cam, gaussians, pipe, bg, colors_precomp=feat, mask=keep, require_depth=False)
    alpha = pkg["alpha"][0]
    rgb = render_fn(cam, gaussians, pipe, bg, mask=keep, require_depth=False)["render"]
    return pkg["render"][0] / alpha.clamp_min(1e-3), rgb


def write_maps(cams, gaussians, pipe, score, keep, out_dir, render_fn, vis_names=()):
    import torch
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    with torch.no_grad():
        for cam in cams:
            a, rgb = render_view(cam, gaussians, pipe, score, keep, render_fn)
            png = encode(a.float().cpu().numpy())
            Image.fromarray(png).save(os.path.join(out_dir, f"{cam.image_name}.png"))
            v = decode(png)
            stats[cam.image_name] = {"mean": round(float(v.mean()), 4),
                                     "high_fraction": round(float((v > HIGH).mean()), 4)}
            if cam.image_name in vis_names:
                os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)
                to8 = lambda t: np.round(t.clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                row = [to8(cam.original_image[:3]), to8(rgb), heatmap(v)]
                Image.fromarray(np.concatenate(row, 1)).save(os.path.join(out_dir, "vis", f"{cam.image_name}.jpg"))
    return stats


def add_args(parser):
    parser.add_argument("--out", required=True)
    parser.add_argument("--remove", type=int, required=True, help="label k of the removed object")
    parser.add_argument("--masks", required=True, help="per-view object masks <image stem>.npy (beags/masks.py)")
    parser.add_argument("--vis", action="store_true")


def main():
    import torch
    from beags.common import load_model, parse_args
    from beags.masks import attach_masks
    from beags.unseen_map import keep_mask
    from gaussian_renderer.ours import render_ours

    dataset, pipe, _, args = parse_args("Background-evidence ledger (artifact score without geometry)", add_args)
    out_dir = os.path.abspath(args.out)
    model = os.path.normpath(os.path.abspath(dataset.model_path))
    assert not out_dir.startswith(model + os.sep), f"{out_dir} is inside the model dir"
    scene, gaussians = load_model(dataset, args.iteration, with_labels=True)
    ply = os.path.join(model, "point_cloud", f"iteration_{scene.loaded_iter}", "point_cloud.ply")
    labels = gaussians.labels.reshape(-1)
    keep = keep_mask(labels, args.remove)
    train = list(scene.getTrainCameras())
    cams = train + list(scene.getTestCameras())
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, "ledger.pt")
    expect = dict(source=ply, iteration=scene.loaded_iter, remove=args.remove,
                  masks=os.path.abspath(args.masks), num_gaussians=int(labels.numel()))
    led = torch.load(path, map_location="cuda") if os.path.exists(path) else None
    if led is not None and (any(led.get(k) != v for k, v in expect.items())
                            or os.path.getmtime(path) < os.path.getmtime(ply)):
        led = None
    if led is None:
        attach_masks(train, args.masks)
        print(f"[INFO] ledger over {len(train)} training views (full scene, object present)")
        bg, obj, used = build_ledger(gaussians, train, pipe, args.remove)
        led = {**expect, "bg": bg, "obj": obj, "views": used}
        torch.save({**led, "bg": bg.cpu(), "obj": obj.cpu()}, path)
    else:
        bg, obj = led["bg"].cuda(), led["obj"].cuda()
        print(f"[INFO] reusing {path}")
    score = evidence_score(bg, obj)

    kept = score[keep]
    q = [float(torch.quantile(kept.float(), p)) for p in (0.5, 0.9, 0.99)]
    print(f"[INFO] kept Gaussians: score median {q[0]:.3f}, p90 {q[1]:.3f}, p99 {q[2]:.3f}; "
          f"{float((kept > 0.9).float().mean()) * 100:.2f}% above 0.9 (no background evidence at all)")

    vis_names = {c.image_name for c in cams[:: max(1, len(cams) // 8)][:8]} if args.vis else set()
    print(f"[INFO] artifact maps of {len(cams)} cameras without label {args.remove} -> {out_dir}")
    stats = write_maps(cams, gaussians, pipe, score, keep, out_dir, render_ours, vis_names)
    info = {"source": ply, "iteration": scene.loaded_iter, "remove": args.remove,
            "masks": os.path.abspath(args.masks), "source_path": dataset.source_path, "images": dataset.images,
            "ledger_views": led["views"], "cameras": len(stats), "encoding": ENCODING, "high_threshold": HIGH,
            "score_quantiles": {"p50": q[0], "p90": q[1], "p99": q[2]},
            "mean": float(np.mean([s["mean"] for s in stats.values()])),
            "high_fraction_mean": float(np.mean([s["high_fraction"] for s in stats.values()])), "views": stats}
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"[INFO] {len(stats)} maps: mean A {info['mean']:.3f}, "
          f"{100 * info['high_fraction_mean']:.2f}% of pixels above {HIGH}")


if __name__ == "__main__":
    main()
