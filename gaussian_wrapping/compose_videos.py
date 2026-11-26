"""Side-by-side videos from render_path_gw.py / render_path_pgsr.py / render_path_mesh.py frame folders. CPU only.

    python gaussian_wrapping/compose_videos.py --out output/compare/<scene> \
        --method "GW today=output/gw_today/<scene>/video" --method "PGSR=output/pgsr/<scene>/video" [--fps 30]

Each method folder holds rgb/, depth/ (float16 .npy), normal/ and mesh/ with frames %05d. Writes, in --out:
    <slug>.mp4      one row per method: RGB | depth | normal | mesh (columns missing for a method are left grey)
    compare.mp4     all methods stacked, one row each (the GW-vs-PGSR video)
    still_<k>.jpg   the compare frame at 0%, 25%, 50%, 75% of the path
Depth uses ONE turbo range per scene (2-98th percentile of the first method's depth over the path, near = red,
no depth = black), so colours mean the same distance in every frame and for every method.
Encoder: a static ffmpeg with libx264 (H.264 mp4, plays in browsers); --ffmpeg to override.
"""
import argparse
import glob
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FFMPEG = "/leonardo_scratch/fast/IscrC_EditGS/envs/aurafusion_official/bin/ffmpeg"
COLS = ["rgb", "depth", "normal", "mesh"]


def font(size):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def frame_ids(d):
    ids = set()
    for sub in COLS:
        for p in glob.glob(os.path.join(d, sub, "*")):
            m = re.match(r"(\d{5})\.", os.path.basename(p))
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def depth_range(d, ids):
    vals = []
    for i in ids[::max(1, len(ids) // 30)]:
        p = os.path.join(d, "depth", f"{i:05d}.npy")
        if os.path.exists(p):
            x = np.load(p).astype(np.float32)
            vals.append(x[x > 0][::37])
    v = np.concatenate(vals) if vals else np.array([0.0, 1.0])
    return float(np.percentile(v, 2)), float(np.percentile(v, 98))


def panel(d, sub, i, w, h, rng):
    p = os.path.join(d, sub, f"{i:05d}." + ("npy" if sub == "depth" else "png"))
    if not os.path.exists(p):
        return Image.new("RGB", (w, h), (90, 90, 90))
    if sub == "depth":
        x = np.load(p).astype(np.float32)
        t = np.clip((x - rng[0]) / max(rng[1] - rng[0], 1e-6), 0, 1)
        rgb = (cm.turbo(1 - t)[..., :3] * 255).astype(np.uint8)
        rgb[x <= 0] = 0
        im = Image.fromarray(rgb)
    else:
        im = Image.open(p).convert("RGB")
    return im.resize((w, h), Image.LANCZOS)


def label(im, text, f):
    d = ImageDraw.Draw(im)
    tw = d.textlength(text, font=f)
    d.rectangle([4, 4, 12 + tw, 10 + f.size], fill=(0, 0, 0))
    d.text((8, 6), text, fill=(255, 255, 255), font=f)


def encode(frames_iter, size, path, fps, ffmpeg):
    w, h = size
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r",
           str(fps), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for im in frames_iter:
        proc.stdin.write(np.asarray(im, dtype=np.uint8).tobytes())
        n += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    print(f"[video] {path}: {n} frames, {w}x{h}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", action="append", required=True, help='"Name=frames_dir", first one sets the depth range')
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--panel_width", type=int, default=640)
    ap.add_argument("--ffmpeg", default=FFMPEG)
    a = ap.parse_args()
    methods = [m.split("=", 1) for m in a.method]
    methods = [(n, d) for n, d in methods if os.path.isdir(d)]
    if not methods:
        raise SystemExit("no method folder exists")
    ids = sorted(set.intersection(*[set(frame_ids(d)) for _, d in methods]))
    rng = depth_range(methods[0][1], ids)
    first = next(p for p in glob.glob(os.path.join(methods[0][1], "rgb", "*.png")))
    W0, H0 = Image.open(first).size
    w = a.panel_width
    h = round(H0 * w / W0)
    f = font(max(12, w // 32))
    os.makedirs(a.out, exist_ok=True)

    def row(name, d, i):
        canvas = Image.new("RGB", (w * len(COLS), h))
        for c, sub in enumerate(COLS):
            im = panel(d, sub, i, w, h, rng)
            label(im, f"{name} - {sub}" if c == 0 else sub, f)
            canvas.paste(im, (c * w, 0))
        return canvas

    for name, d in methods:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        encode((row(name, d, i) for i in ids), (w * len(COLS), h), os.path.join(a.out, f"{slug}.mp4"), a.fps, a.ffmpeg)

    def stacked(i):
        canvas = Image.new("RGB", (w * len(COLS), h * len(methods)))
        for r, (name, d) in enumerate(methods):
            canvas.paste(row(name, d, i), (0, r * h))
        return canvas

    if len(methods) > 1:
        encode((stacked(i) for i in ids), (w * len(COLS), h * len(methods)), os.path.join(a.out, "compare.mp4"),
               a.fps, a.ffmpeg)
    for k, q in enumerate((0.0, 0.25, 0.5, 0.75)):
        stacked(ids[int(q * (len(ids) - 1))]).save(os.path.join(a.out, f"still_{k}.jpg"), quality=90)
    print(f"[video] depth range {rng[0]:.3f}..{rng[1]:.3f}, {len(ids)} frames, methods {[n for n, _ in methods]}")


if __name__ == "__main__":
    main()
