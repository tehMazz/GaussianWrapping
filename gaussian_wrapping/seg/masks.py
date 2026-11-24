"""Seed2GS mask IO in the BEA-GS mask format (beags/masks.py): <dir>/<image_name>.npy, (2, H, W) float16,
channel 0 = background, channel 1 = the prompted object, at the camera's (training) resolution."""
import os
import re

import numpy as np


def prompt_slug(text):
    """Prompt -> output directory name; slurm/common.sh `slug` applies the same rule."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "object"


def beags_mask(obj):
    """(H, W) bool object mask -> (2, H, W) float16 [background, object]."""
    obj = np.asarray(obj, dtype=bool)
    return np.stack([~obj, obj]).astype(np.float16)


def save_beags_mask(mask_dir, image_name, obj):
    np.save(os.path.join(mask_dir, f"{image_name}.npy"), beags_mask(obj))


def object_channel(mask, channel=1):
    """(H, W) bool from a (C, H, W) mask file array (channel > 0.5) or an (H, W) bool / 0-1 array."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[channel]
    return mask.astype(np.float32) > 0.5


def capped_size(height, width, side):
    """(h, w) scaled down (aspect kept, rounded) until min(h, w) == side; unchanged when side <= 0 or min(h, w) <= side.
    The SHORT side is capped: SAM3 resizes every frame to a side x side square (sam3/model/io_utils.py), so a frame
    whose short side is still >= side gives SAM3 the same detail on both axes as the full-size frame."""
    if side <= 0 or min(height, width) <= side:
        return int(height), int(width)
    s = side / min(height, width)
    return max(side, int(round(height * s))), max(side, int(round(width * s)))
