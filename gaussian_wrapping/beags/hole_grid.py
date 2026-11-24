"""Side-by-side ArtiFixer removal holes (artifixer_gs.removal split outputs) for a few frames, CPU only.

    python gaussian_wrapping/beags/hole_grid.py --split alpha=<removal_split_alpha> --split gavis=<removal_split_gavis> \
        [--unseen_dir <out>/unseen] [--n 6] [--width 320] --out grid.jpg

One row per frame (evenly spaced): photo | removal render | the render with each split's hole in red | unseen map
(beags/unseen_map.py heatmap). The header row names the columns and gives each split's hole fraction on that frame.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load(path, size, mode="RGB"):
    with Image.open(path) as im:
        return np.asarray(im.convert(mode).resize(size, Image.BILINEAR if mode != "L" else Image.NEAREST))


def frames(split_dir):
    """[(photo path, stem)] of a split: the photos of its scene/ copy (same order as renders/ and hole/)."""
    entry = next(iter(json.loads((split_dir / "scene" / "split.json").read_text())["test"].values()))
    scene = json.loads(Path(entry["transforms_path"]).read_text())["frames"]
    return [(Path(entry["image_root"]) / f["file_path"], Path(f["file_path"]).stem) for f in scene]


def heatmap(u):
    u = u[..., None]
    rgb = np.concatenate([np.clip(3 * u, 0, 1), np.clip(3 * u - 1, 0, 1), np.clip(3 * u - 2, 0, 1)], axis=-1)
    return np.round(rgb * 255).astype(np.uint8)


def label(text, width, height=22):
    im = Image.new("RGB", (width, height), (255, 255, 255))
    ImageDraw.Draw(im).text((4, 4), text, fill=(0, 0, 0))
    return np.asarray(im)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", action="append", required=True, help="name=<removal_split dir>, repeatable")
    parser.add_argument("--unseen_dir", type=Path, default=None)
    parser.add_argument("--n", type=int, default=6, help="rows (evenly spaced frames)")
    parser.add_argument("--width", type=int, default=320, help="tile width in pixels")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    splits = [(s.split("=", 1)[0], Path(s.split("=", 1)[1]).expanduser().resolve()) for s in args.split]
    first = splits[0][1]
    photos = frames(first)
    n = len(photos)
    for name, d in splits:
        assert len(list((d / "hole").glob("*.png"))) == n, f"{name}: {d} has another frame count than {first}"
    with Image.open(first / "renders" / "00000.png") as im:
        w0, h0 = im.size
    size = (args.width, round(args.width * h0 / w0))
    rows = []
    for k in np.linspace(0, n - 1, min(args.n, n)).round().astype(int):
        photo, stem = photos[k]
        render = load(first / "renders" / f"{k:05d}.png", size)
        tiles, names = [load(photo, size), render], [stem, "removal render"]
        for name, d in splits:
            hole = load(d / "hole" / f"{k:05d}.png", size, "L") > 127
            tiles.append(np.where(hole[..., None], (0.4 * render + [153, 0, 0]).astype(np.uint8), render))
            names.append(f"{name} {100 * hole.mean():.1f}%")
        if args.unseen_dir is not None:
            u = args.unseen_dir / f"{stem}.png"
            tiles.append(heatmap(load(u, size, "L") / 255.0) if u.is_file() else np.zeros_like(render))
            names.append("unseen U")
        rows.append(np.concatenate([np.concatenate([label(t, size[0]), im], axis=0) for t, im in zip(names, tiles)], axis=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out, quality=90)
    print(f"wrote {args.out}: {len(rows)} frames x {len(rows[0][0]) // size[0]} columns")


if __name__ == "__main__":
    main()
