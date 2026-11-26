"""Column titles for image grids: a white header strip with one title centred over each equal-width column.

    from utils.grid_titles import add_column_titles
    add_column_titles("grid.png", ["photo", "mesh", "overlay"])          # rewrites the file in place

CLI, for grids saved before titles existed (skips files that already have a header, see HEADER_TAG):
    python gaussian_wrapping/utils/grid_titles.py --titles "photo|mesh|overlay" <img> [<img> ...]
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

HEADER_TAG = "column_titles"          # PNG text chunk / JPEG comment marking a grid that already has its header


def _font(size):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def has_titles(path):
    im = Image.open(path)
    return HEADER_TAG in getattr(im, "text", {}) or HEADER_TAG.encode() in (im.info.get("comment") or b"")


def titled(im, titles):
    """Return a copy of PIL image `im` with a header row: titles[k] centred over column k (equal widths)."""
    im = im.convert("RGB")
    w, h = im.size
    cw = w / len(titles)
    size = max(12, min(int(cw / 14), 40))
    f = _font(size)
    while size > 10 and max(ImageDraw.Draw(im).textlength(t, font=f) for t in titles) > cw - 8:
        size -= 1
        f = _font(size)
    hh = int(size * 1.7)
    out = Image.new("RGB", (w, h + hh), (255, 255, 255))
    out.paste(im, (0, hh))
    d = ImageDraw.Draw(out)
    for k, t in enumerate(titles):
        tw = d.textlength(t, font=f)
        d.text((k * cw + (cw - tw) / 2, (hh - size) / 2 - 1), t, fill=(0, 0, 0), font=f)
        if k:
            d.line([(int(k * cw), 0), (int(k * cw), hh)], fill=(200, 200, 200))
    return out


def add_column_titles(path, titles):
    """Rewrite image file `path` with a column-title header (marked, so a second call is a no-op)."""
    if has_titles(path):
        return
    out = titled(Image.open(path), titles)
    if path.lower().endswith(".png"):
        meta = PngImagePlugin.PngInfo()
        meta.add_text(HEADER_TAG, "|".join(titles))
        out.save(path, pnginfo=meta)
    else:
        out.save(path, quality=92, comment=HEADER_TAG.encode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", required=True, help='"a|b|c"')
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    for p in a.images:
        add_column_titles(p, a.titles.split("|"))
        print(f"[titles] {p}")
