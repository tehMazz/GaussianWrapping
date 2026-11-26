"""Finetune probe: snapshots of one label during beags_finetune.py, to see when (and why) it erodes.

Enabled by `beags_finetune.py --snap_every N --snap_dir <dir>`. At iteration 0 (before the first step), every N
steps and at the last step it writes, without touching the optimisation:
    <dir>/stats.jsonl              one JSON line per snapshot: label count, opacity, test IoU, loss terms, and the
                                   per-term push on the label's opacity (see `term_push`)
    <dir>/label_<k>/<it>_<view>.png   label k rendered alone (white bg)
    <dir>/full/<it>_<view>.png        full model
Views: --snap_views cameras spread evenly over test + train (same rule as beags_render_labels.py).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from beags.masks import attach_gt_masks
from beags.metrics import match_gt_to_labels, segmentation_metrics
from gaussian_renderer.beags_api import render_labels


def add_args(parser):
    parser.add_argument("--snap_every", type=int, default=0, help="probe snapshot every N finetune steps; 0 = off")
    parser.add_argument("--snap_dir", default="", help="probe output dir (flat: output/<method>/<scene>/<mode>)")
    parser.add_argument("--snap_views", type=int, default=5)
    parser.add_argument("--snap_label", type=int, default=1, help="label whose erosion is tracked")
    parser.add_argument("--snap_gt_masks", default="", help="GT masks dir: test IoU at every snapshot")
    parser.add_argument("--snap_sam_masks", default="", help="SAM2 masks dir, to match GT ids to labels")
    parser.add_argument("--snap_scale", type=float, default=0.5, help="saved render scale")
    parser.add_argument("--snap_dense_until", type=int, default=0, help="also snapshot every --snap_dense_every "
                        "steps up to this step (the early steps, where the erosion starts)")
    parser.add_argument("--snap_dense_every", type=int, default=25)


class FinetuneProbe:
    def __init__(self, args, scene):
        self.dir, self.every, self.label, self.scale = args.snap_dir, args.snap_every, args.snap_label, args.snap_scale
        self.dense_until, self.dense_every = args.snap_dense_until, max(1, args.snap_dense_every)
        for sub in ("full", f"label_{self.label}"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        open(os.path.join(self.dir, "stats.jsonl"), "w").close()
        tests = list(scene.getTestCameras())
        self.gt_cams, self.gt_to_label = [], {}
        if args.snap_gt_masks:
            attach_gt_masks(tests, args.snap_gt_masks)
            self.gt_cams = [c for c in tests if c.gt is not None]
            if self.gt_cams:
                self.gt_to_label = (match_gt_to_labels(self.gt_cams, args.snap_sam_masks) if args.snap_sam_masks
                                    else {j: j for j in range(1, max(c.gt.shape[0] for c in self.gt_cams))})
        cams = tests + list(scene.getTrainCameras())
        idx = np.unique(np.linspace(0, len(cams) - 1, min(args.snap_views, len(cams))).round().astype(int))
        self.views = [cams[i] for i in idx]
        print(f"[probe] label {self.label}, {len(self.views)} views, {len(self.gt_cams)} GT views, -> {self.dir}")

    def due(self, it, last):
        return self.every > 0 and (it % self.every == 0 or it == last or
                                   (it <= self.dense_until and it % self.dense_every == 0))

    @torch.no_grad()
    def snapshot(self, it, gaussians, pipe, terms=None, push=None):
        sel = gaussians.labels == self.label
        op = gaussians.get_opacity[:, 0]
        row = dict(iteration=int(it), n_total=int(sel.numel()), n_label=int(sel.sum()))
        if bool(sel.any()):
            o = op[sel]
            row.update(opacity_mean=float(o.mean()), opacity_median=float(o.median()),
                       frac_below_0p5=float((o < 0.5).float().mean()))
        if self.gt_cams:
            seg = segmentation_metrics(self.gt_cams, gaussians, pipe, gaussians.labels, self.gt_to_label)
            row.update({k: seg[k] for k in ("IoU", "BIoU", "EIoU") if seg.get(k) is not None})
        if terms:
            row["loss"] = {k: float(v) for k, v in terms.items()}
        if push:
            row["push"] = push
        with open(os.path.join(self.dir, "stats.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[probe] {json.dumps(row)}")
        white = torch.ones(3, device="cuda")
        for cam in self.views:
            for sub, mask in (("full", None), (f"label_{self.label}", sel)):
                img = render_labels(cam, gaussians, pipe, white, mask=mask)["render"].clamp(0, 1)
                if self.scale != 1.0:
                    img = F.interpolate(img[None], scale_factor=self.scale, mode="area")[0]
                torchvision.utils.save_image(img, os.path.join(self.dir, sub, f"{it:05d}_{cam.image_name}.png"))


def term_push(terms, gaussians, label):
    """Per weighted loss term: the descent step it asks of the label's opacity logits, -mean(dL/dlogit) over the
    label's Gaussians (negative = pushes opacity DOWN), plus the share of those Gaussians it pushes down.
    `terms` maps name -> weighted scalar still attached to the graph; call before loss.backward()."""
    sel = gaussians.labels == label
    out = {}
    if not bool(sel.any()):
        return out
    for name, value in terms.items():
        if not (torch.is_tensor(value) and value.requires_grad):
            continue
        try:
            (g,) = torch.autograd.grad(value, gaussians._opacity, retain_graph=True, allow_unused=True)
        except RuntimeError as e:  # a custom backward that cannot run twice: report, do not stop the finetune
            out[name] = {"error": str(e)[:200]}
            continue
        if g is None:
            out[name] = {"mean": 0.0, "down_frac": 0.0}
            continue
        g = g[sel, 0]
        out[name] = {"mean": float(-g.mean()), "down_frac": float((g > 0).float().mean())}
    return out
