"""Figures for slurm/gw_ftprobe.sh (CPU only, login node is fine).

    python gaussian_wrapping/beags_ftprobe_figure.py output/ftprobe/<scene> [--label 1] [--cell 220]

Reads <dir>/snaps_<arm>/stats.jsonl + renders and writes into <dir>/figures/:
    timeline_<arm>.jpg      rows = snapshot iterations, columns = views, label alone; row header = count, opacity, IoU
    compare_final.jpg       one row per arm at the last snapshot (+ the iteration-0 row once), label alone
    compare_full_final.jpg  same with the full model
    curves.png              IoU, label count, mean opacity vs finetune step, one line per arm
    push.png                per-term push on the label's opacity (from term_push) vs step, one panel per arm
    summary.json            first / last row of every arm and the step where IoU first drops below half its start
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ARM_ORDER = ["base", "b10", "occ0", "sag0", "piv0", "cons_base", "cons_b10", "dense_base"]
ARM_NAME = {"base": "as run (break)", "b10": "B10 (alpha_obj 2.5)", "occ0": "occ off", "sag0": "sag off",
            "piv0": "piv off", "dense_base": "as run, dense", "cons_base": "consensus fix",
            "cons_b10": "consensus fix + B10"}


def load(snap_dir):
    with open(os.path.join(snap_dir, "stats.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def font(size):
    for path in ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def views_of(snap_dir, sub, it):
    return sorted(glob.glob(os.path.join(snap_dir, sub, f"{it:05d}_*.png")))


CROP = {}  # view name -> (l, t, r, b), from the object's footprint at the first snapshot


def view_name(path):
    return os.path.basename(path).split("_", 1)[1]


def set_crops(paths, pad=0.25):
    import numpy as np
    for p in paths:
        a = np.asarray(Image.open(p).convert("L"))
        ys, xs = np.nonzero(a < 245)
        if len(xs) == 0:
            continue
        l, r, t, b = xs.min(), xs.max(), ys.min(), ys.max()
        side = max(r - l, b - t) * (1 + 2 * pad) / 2
        cx, cy = (l + r) / 2, (t + b) / 2
        CROP[view_name(p)] = (int(max(0, cx - side)), int(max(0, cy - side)),
                              int(min(a.shape[1], cx + side)), int(min(a.shape[0], cy + side)))


def cell(path, w):
    im = Image.open(path).convert("RGB")
    if view_name(path) in CROP:
        im = im.crop(CROP[view_name(path)])
    return im.resize((w, max(1, round(im.height * w / im.width))), Image.LANCZOS)


def header(row):
    parts = [f"it {row['iteration']}", f"n={row.get('n_label', 0)}"]
    if "opacity_mean" in row:
        parts.append(f"op={row['opacity_mean']:.3f}")
    if "IoU" in row:
        parts.append(f"IoU={row['IoU']:.3f}")
    return "  ".join(parts)


def grid(rows, w, path, title):
    """rows: [(text, [image paths])]"""
    rows = [(t, ps) for t, ps in rows if ps]
    cells = [[cell(p, w) for p in ps] for _, ps in rows]
    h = max(c.height for r in cells for c in r)
    head, pad, top = 260, 4, 34
    ncol = max(len(r) for r in cells)
    canvas = Image.new("RGB", (head + ncol * (w + pad), top + len(rows) * (h + pad)), "white")
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), title, fill="black", font=font(18))
    f = font(13)
    for i, ((text, _), r) in enumerate(zip(rows, cells)):
        y = top + i * (h + pad)
        for j, line in enumerate(text.split("  ")):
            d.text((6, y + 6 + 17 * j), line, fill="black", font=f)
        for j, im in enumerate(r):
            canvas.paste(im, (head + j * (w + pad), y))
    canvas.save(path, quality=90)
    print(f"[figure] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--label", type=int, default=1)
    ap.add_argument("--cell", type=int, default=220)
    ap.add_argument("--no_crop", action="store_true", help="full frames instead of a crop around the object")
    ap.add_argument("--arms", nargs="*", default=None, help="only these arms (default: every snaps_<arm>)")
    a = ap.parse_args()
    sub = f"label_{a.label}"
    out = os.path.join(a.dir, "figures")
    os.makedirs(out, exist_ok=True)
    found = [os.path.basename(p)[len("snaps_"):] for p in glob.glob(os.path.join(a.dir, "snaps_*"))]
    arms = [x for x in ARM_ORDER if x in found] + sorted(x for x in found if x not in ARM_ORDER)
    if a.arms:
        arms = [x for x in a.arms if x in found]
    out = os.path.join(a.dir, "figures", "_".join(a.arms)) if a.arms else out
    os.makedirs(out, exist_ok=True)
    stats = {arm: load(os.path.join(a.dir, f"snaps_{arm}")) for arm in arms}
    stats = {k: v for k, v in stats.items() if v}
    if stats and not a.no_crop:
        first = next(iter(stats))
        set_crops(views_of(os.path.join(a.dir, f"snaps_{first}"), sub, stats[first][0]["iteration"]))
    summary = {}
    for arm, rows in stats.items():
        snap = os.path.join(a.dir, f"snaps_{arm}")
        grid([(header(r), views_of(snap, sub, r["iteration"])) for r in rows], a.cell,
             os.path.join(out, f"timeline_{arm}.jpg"), f"{ARM_NAME.get(arm, arm)}: label {a.label} alone")
        start = rows[0].get("IoU")
        half = next((r["iteration"] for r in rows if start and r.get("IoU", start) < start / 2), None)
        summary[arm] = {"first": rows[0], "last": rows[-1], "iou_half_at": half}
    if stats:
        first_arm = next(iter(stats))
        r0 = stats[first_arm][0]
        for name, s in (("compare_final.jpg", sub), ("compare_full_final.jpg", "full")):
            rows = [(f"start  {header(r0)}", views_of(os.path.join(a.dir, f"snaps_{first_arm}"), s, r0["iteration"]))]
            rows += [(f"{ARM_NAME.get(arm, arm)}  {header(st[-1])}",
                      views_of(os.path.join(a.dir, f"snaps_{arm}"), s, st[-1]["iteration"])) for arm, st in stats.items()]
            grid(rows, a.cell, os.path.join(out, name), f"end of finetune, {'label %d alone' % a.label if s == sub else 'full model'}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for arm, rows in stats.items():
            its = [r["iteration"] for r in rows]
            for ax, key in zip(axes, ("IoU", "n_label", "opacity_mean")):
                ax.plot(its, [r.get(key, float("nan")) for r in rows], marker="o", ms=3, label=ARM_NAME.get(arm, arm))
        for ax, t in zip(axes, ("test IoU (label render)", f"label-{a.label} Gaussians", f"mean opacity, label {a.label}")):
            ax.set_title(t)
            ax.set_xlabel("finetune step")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "curves.png"), dpi=110)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(stats), figsize=(4.2 * len(stats), 3.6), squeeze=False, sharey=True)
        for ax, (arm, rows) in zip(axes[0], stats.items()):
            rows = [r for r in rows if "push" in r]
            terms = sorted({t for r in rows for t, v in r["push"].items() if "mean" in v})
            for t in terms:
                ax.plot([r["iteration"] for r in rows], [r["push"].get(t, {}).get("mean", float("nan")) for r in rows],
                        marker="o", ms=3, label=t)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.set_title(ARM_NAME.get(arm, arm))
            ax.set_xlabel("finetune step")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        axes[0][0].set_ylabel("push on object opacity (<0 = down)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "push.png"), dpi=110)
        plt.close(fig)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for arm, s in summary.items():
        print(f"[figure] {arm:6s} {header(s['first'])}  ->  {header(s['last'])}   IoU halved at: {s['iou_half_at']}")


if __name__ == "__main__":
    main()
