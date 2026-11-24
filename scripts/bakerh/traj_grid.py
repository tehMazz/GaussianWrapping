#!/usr/bin/env python3
"""Image grid of ArtiFixer novel-view frames: one column per --col NAME=DIR, one row per frame (N evenly spaced frames of
the first column, matched by file name in the others, else by index). A single column is tiled --wrap frames per row.

  python scripts/bakerh/traj_grid.py --col "GW render=<dir>" --col "ArtiFixer=<dir>" --n 8 --out grid.jpg
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw

EXT = (".png", ".jpg", ".jpeg")


def frames(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(EXT)) if d and os.path.isdir(d) else []


def tile(path, w):
    im = Image.open(path).convert("RGB")
    return im.resize((w, max(1, round(im.height * w / im.width))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--col", action="append", required=True, help="NAME=DIR, repeatable; the first one picks the frames")
    p.add_argument("--n", type=int, default=8, help="frames")
    p.add_argument("--wrap", type=int, default=4, help="frames per row when there is a single column")
    p.add_argument("--width", type=int, default=360, help="tile width in pixels")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cols = [c.split("=", 1) for c in a.col]
    names = {n: frames(d) for n, d in cols}
    first = names[cols[0][0]]
    if not first:
        sys.exit(f"no frames in {cols[0][1]}")
    idx = sorted({round(i * (len(first) - 1) / max(1, a.n - 1)) for i in range(min(a.n, len(first)))})
    rows = []
    for i in idx:
        row = []
        for n, d in cols:
            f = first[i] if first[i] in names[n] else (names[n][i] if i < len(names[n]) else None)
            row.append(tile(os.path.join(d, f), a.width) if f else None)
        rows.append(row)
    if len(cols) == 1:  # contact sheet
        flat = [r[0] for r in rows]
        rows = [flat[k:k + a.wrap] for k in range(0, len(flat), a.wrap)]
        header = [f"{cols[0][0]} ({len(first)} frames; showing {len(idx)})"]
    else:
        header = [f"{n} ({len(names[n])})" for n, _ in cols]
    th = max((t.height for r in rows for t in r if t), default=a.width)
    ncol = max(len(r) for r in rows)
    pad = 22
    grid = Image.new("RGB", (ncol * a.width, pad + len(rows) * th), "white")
    draw = ImageDraw.Draw(grid)
    for c, h in enumerate(header):
        draw.text((c * a.width + 4, 4), h, fill="black")
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            if t is None:
                draw.text((c * a.width + 4, pad + r * th + th // 2), "missing", fill="red")
            else:
                grid.paste(t, (c * a.width, pad + r * th))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    grid.save(a.out, quality=90)
    print(f"{a.out}: {len(rows)} rows x {ncol} columns")


if __name__ == "__main__":
    main()
