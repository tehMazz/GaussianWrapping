"""Seed2GS against provided masks (plan: LERF figurines mIoU). CPU only: reads mask files and labels.pt, no model.

    python gaussian_wrapping/seed2gs_eval.py --seed <seed2gs out> [<seed2gs out 2> ...] --gt_masks <dir>
        [--labels <provided-mask labels.pt>] [--gt_channel j ...] [--provided_label k ...] [--out <json>]

2D, on every view with a GT mask (<gt_masks>/<image_name>.npy, (C, H, W), the beags_eval.py --gt_masks format) and a
seed2gs mask (<seed>/masks/<image_name>.npy, written for test views too by seed2gs.py --mask_split all): the GT is
resized to the seed mask (= camera) resolution as beags/masks.load_npy_mask does, then
  IoU, BIoU (beags/metrics.py) of the object vs GT channel j; bg_IoU of the complements; mIoU2 = mean(IoU, bg_IoU);
  acc = pixel accuracy. GT channel per prompt: --gt_channel (one per --seed), else Hungarian matching of prompts to
  GT objects on IoU summed over the GT views.
3D, with --labels (beags_prepare.py labels.pt of the SAME model iteration, -1 = never hit, excluded): per-Gaussian
  agreement of the seed2gs labels.pt (1 = object) with provided label k (--provided_label, else the label of max
  IoU): IoU, precision, recall, accuracy, and the IoU against every provided label.
Failed prompts (seed2gs.py found no object: <seed>/seed2gs_failed.json, or no labels.pt) are scored as an empty
prediction on every GT view (IoU 0 where the object is in the GT), any stale masks/ ignored, and get no Gaussian
agreement; they are listed with "failed".
Writes <seed>/seed2gs_eval.json per prompt; with several --seed also a summary (--out, default
<parent of the first seed>/seed2gs_eval.json): mIoU / mBIoU = mean object IoU / BIoU over all matched prompts, failed
ones included (LERF-Mask style), mIoU_found / mBIoU_found over the prompts seed2gs.py found.
"""
import json
import os
import sys
from argparse import ArgumentParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from beags.masks import load_npy_mask
from beags.metrics import boundary_iou, iou
from seg.masks import object_channel


def view_metrics(pred, gt):
    """(H, W) bool prediction vs GT -> IoU, BIoU, bg_IoU, mIoU2, acc."""
    obj, bg = float(iou(pred, gt)), float(iou(~pred, ~gt))
    return dict(IoU=obj, BIoU=float(boundary_iou(gt, pred)), bg_IoU=bg, mIoU2=(obj + bg) / 2,
                acc=float((pred == gt).mean()))


def gt_views(gt_dir):
    return sorted(os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.endswith(".npy"))


def seed_failure(seed_dir):
    """Why seed2gs.py has no current result in seed_dir, or None."""
    for name in ("seed2gs_failed.json", "seed2gs.json"):  # seed2gs.json: older runs wrote the failure there
        path = os.path.join(seed_dir, name)
        if os.path.exists(path):
            try:
                failed = json.load(open(path)).get("failed")
            except ValueError:
                failed = f"unreadable {name}"
            if failed:
                return str(failed)
    if not os.path.exists(os.path.join(seed_dir, "labels.pt")):
        return "no labels.pt (seed2gs.py did not finish)"
    return None


def load_pairs(seed_dir, gt_dir, failed=False):
    """{view: (pred (H, W) bool, gt (C, H, W) bool)} on the GT views that have a seed2gs mask; missing views.
    failed: every GT view, empty prediction at the GT resolution (masks/ is not read)."""
    pairs, missing = {}, []
    for view in gt_views(gt_dir):
        if failed:
            gt_path = os.path.join(gt_dir, f"{view}.npy")
            pred = np.zeros(np.load(gt_path, mmap_mode="r").shape[-2:], bool)
            pairs[view] = (pred, load_npy_mask(gt_path, *pred.shape).numpy() > 0.5)
            continue
        path = os.path.join(seed_dir, "masks", f"{view}.npy")
        if not os.path.exists(path):
            missing.append(view)
            continue
        pred = object_channel(np.load(path))
        gt = load_npy_mask(os.path.join(gt_dir, f"{view}.npy"), *pred.shape).numpy() > 0.5
        pairs[view] = (pred, gt)
    return pairs, missing


def match_channels(score):
    """(prompts, C) IoU sums, column 0 = background -> per prompt GT channel (>= 1) or None, Hungarian."""
    out = [None] * score.shape[0]
    if score.shape[1] > 1:
        rows, cols = linear_sum_assignment(-score[:, 1:])
        for r, c in zip(rows, cols):
            out[int(r)] = int(c) + 1
    return out


def label_agreement(seed, provided, label=None):
    """Per-Gaussian agreement of seed labels (1 = object) with provided labels (-1 never hit: excluded)."""
    seed, provided = np.asarray(seed).reshape(-1), np.asarray(provided).reshape(-1)
    if seed.shape != provided.shape:
        raise ValueError(f"labels of different models: seed2gs {seed.shape[0]} vs provided {provided.shape[0]} Gaussians")
    valid = provided >= 0
    obj, prov = (seed == 1)[valid], provided[valid]
    ious = {int(k): float(iou(obj, prov == k)) for k in np.unique(prov) if k > 0}
    if label is None:
        label = max(ious, key=ious.get) if ious else 1
    ref = prov == label
    tp = int((obj & ref).sum())
    return dict(provided_label=int(label), IoU=float(iou(obj, ref)),
                precision=tp / max(int(obj.sum()), 1), recall=tp / max(int(ref.sum()), 1),
                accuracy=float((obj == ref).mean()) if len(ref) else 1.0, n_seed_object=int((seed == 1).sum()),
                n_provided_object=int(ref.sum()), n_never_hit=int((~valid).sum()), IoU_per_label=ious)


def evaluate(seed_dirs, gt_dir, labels=None, gt_channels=None, provided_labels=None):
    """-> per-prompt result dicts (also written to <seed>/seed2gs_eval.json) and the summary dict."""
    failures = [seed_failure(d) for d in seed_dirs]
    data = [load_pairs(d, gt_dir, failed=f is not None) for d, f in zip(seed_dirs, failures)]
    num_gt = max((gt.shape[0] for pairs, _ in data for _, gt in pairs.values()), default=0)
    if gt_channels is None:
        score = np.zeros((len(seed_dirs), max(num_gt, 1)))
        for i, (pairs, _) in enumerate(data):
            for pred, gt in pairs.values():
                for j in range(1, gt.shape[0]):
                    score[i, j] += iou(pred, gt[j])
        gt_channels = match_channels(score)
    provided = None
    if labels:
        provided = torch.load(labels, map_location="cpu")
        provided = (provided.numpy() if torch.is_tensor(provided) else np.asarray(provided)).astype(np.int64)

    results = []
    for i, (seed_dir, (pairs, missing)) in enumerate(zip(seed_dirs, data)):
        info = {}
        for name in ("seed2gs.json", "seed2gs_failed.json"):
            if not info.get("prompt") and os.path.exists(os.path.join(seed_dir, name)):
                try:
                    info = json.load(open(os.path.join(seed_dir, name)))
                except ValueError:
                    pass
        j, failed = gt_channels[i], failures[i]
        res = dict(prompt=info.get("prompt"), seed_dir=os.path.abspath(seed_dir), gt_masks=os.path.abspath(gt_dir),
                   gt_channel=j, n_views=len(pairs), missing_views=missing)
        if failed:
            res["failed"] = failed
            print(f"[seed2gs_eval] {seed_dir}: failed ({failed}): scored as an empty prediction")
        elif not pairs:
            print(f"[seed2gs_eval] {seed_dir}: no seed2gs mask on any GT view ({missing}); rerun seed2gs.py with "
                  f"--mask_split all")
        if j is not None and pairs:
            rows = [dict(view=v, **view_metrics(pred, gt[j] if j < gt.shape[0] else np.zeros_like(pred)))
                    for v, (pred, gt) in pairs.items()]
            res.update({k: float(np.mean([r[k] for r in rows])) for k in ("IoU", "BIoU", "bg_IoU", "mIoU2", "acc")})
            res["per_view"] = rows
        if provided is not None and failed:
            res["gaussians"] = dict(labels=os.path.abspath(labels), error=f"seed2gs failed: {failed}")
        elif provided is not None:
            seed_labels = torch.load(os.path.join(seed_dir, "labels.pt"), map_location="cpu").numpy()
            k = None if provided_labels is None else provided_labels[i]
            try:
                res["gaussians"] = dict(labels=os.path.abspath(labels), **label_agreement(seed_labels, provided, k))
            except ValueError as e:
                print(f"[seed2gs_eval] {seed_dir}: {e}")
                res["gaussians"] = dict(labels=os.path.abspath(labels), error=str(e))
        os.makedirs(seed_dir, exist_ok=True)  # a prompt that never ran still gets its record
        with open(os.path.join(seed_dir, "seed2gs_eval.json"), "w") as f:
            json.dump(res, f, indent=2)
        short = {k: round(v, 4) for k, v in res.items() if k in ("IoU", "BIoU", "mIoU2")}
        if "gaussians" in res and "IoU" in res["gaussians"]:
            short["gaussian_IoU"] = round(res["gaussians"]["IoU"], 4)
        print(f"[seed2gs_eval] '{res['prompt']}' GT channel {j}, {len(pairs)} views: {short}")
        results.append(res)

    scored = [r for r in results if "IoU" in r]
    found = [r for r in scored if "failed" not in r]

    def mean(rows, k):
        return float(np.mean([r[k] for r in rows])) if rows else None
    summary = dict(gt_masks=os.path.abspath(gt_dir), n_prompts=len(results), n_scored=len(scored),
                   n_failed=sum("failed" in r for r in results),
                   mIoU=mean(scored, "IoU"), mBIoU=mean(scored, "BIoU"),
                   mIoU_found=mean(found, "IoU"), mBIoU_found=mean(found, "BIoU"),
                   prompts=[{k: r.get(k) for k in ("prompt", "seed_dir", "gt_channel", "n_views", "IoU", "BIoU", "failed")}
                            for r in results])
    gauss = [r["gaussians"]["IoU"] for r in results if "IoU" in r.get("gaussians", {})]
    if gauss:
        summary["gaussian_mIoU"] = float(np.mean(gauss))
    return results, summary


def main():
    parser = ArgumentParser(description="Seed2GS masks / labels vs GT masks and provided-mask labels")
    parser.add_argument("--seed", required=True, nargs="+", help="seed2gs.py output dir(s) (masks/, labels.pt)")
    parser.add_argument("--gt_masks", required=True, help="GT (C, H, W) <image_name>.npy (beags_eval.py --gt_masks)")
    parser.add_argument("--labels", default=None, help="provided-mask labels.pt of the same model iteration")
    parser.add_argument("--gt_channel", type=int, nargs="+", default=None, help="GT channel per --seed (default: matched)")
    parser.add_argument("--provided_label", type=int, nargs="+", default=None,
                        help="provided label per --seed (default: max-IoU label)")
    parser.add_argument("--out", default=None, help="summary json with several --seed (default <parent>/seed2gs_eval.json)")
    args = parser.parse_args()
    for name in ("gt_channel", "provided_label"):
        if getattr(args, name) is not None and len(getattr(args, name)) != len(args.seed):
            parser.error(f"--{name} needs one value per --seed")
    if args.labels and not os.path.exists(args.labels):
        print(f"[seed2gs_eval] no provided labels at {args.labels}: 2D metrics only")
        args.labels = None
    results, summary = evaluate(args.seed, args.gt_masks, args.labels, args.gt_channel, args.provided_label)
    if len(args.seed) > 1:
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.seed[0])), "seed2gs_eval.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[seed2gs_eval] summary -> {out}")
    print(f"[seed2gs_eval] mIoU {summary['mIoU']}, mBIoU {summary['mBIoU']} ({summary['n_failed']} failed prompts "
          f"count as empty; found only: mIoU {summary['mIoU_found']})"
          + (f", Gaussian mIoU {summary['gaussian_mIoU']:.4f}" if "gaussian_mIoU" in summary else ""))


if __name__ == "__main__":
    main()
