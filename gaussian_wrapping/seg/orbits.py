"""Virtual cameras around a seed: a flat orbit and a spiral, both starting at the reference camera.

orbit_pose is copied from ArtiFixer-GS (artifixer_gs/trajectory.py, SPDX Apache-2.0, NVIDIA). It rotates the
whole camera-to-world rigidly, so it works on OpenCV c2w as well as OpenGL.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def orbit_pose(c2w, anchor, yaw, pitch, up, max_elev=None):
    """``c2w`` rigidly rotated about ``anchor``: by ``yaw`` about the vertical ``up``, then by ``pitch`` about
    the horizontal axis across the anchor->camera direction (positive raises the camera; the elevation stays
    within +-``max_elev``). The anchor keeps its image position, and a level camera stays level."""
    offset = c2w[:3, 3] - anchor
    rot = Rotation.from_rotvec(yaw * up).as_matrix()
    offset1 = rot @ offset
    across = np.cross(offset1, up)
    if np.linalg.norm(across) > 1e-9 * np.linalg.norm(offset1) and pitch != 0:
        elev = np.arcsin(np.clip(offset1 @ up / np.linalg.norm(offset1), -1, 1))
        if max_elev is not None:
            pitch = float(np.clip(elev + pitch, -max_elev, max_elev) - elev)
        rot = Rotation.from_rotvec(pitch * across / np.linalg.norm(across)).as_matrix() @ rot
    out = np.eye(4)
    out[:3, :3] = rot @ c2w[:3, :3]
    out[:3, 3] = anchor + rot @ offset
    return out


def world_up(c2ws):
    """Unit world up from OpenCV camera-to-worlds (N, 4, 4): minus the mean camera y axis."""
    up = -np.asarray(c2ws)[:, :3, 1].mean(0)
    return up / np.linalg.norm(up)


def elevation(c2w, anchor, up):
    offset = c2w[:3, 3] - anchor
    return float(np.arcsin(np.clip(offset @ up / np.linalg.norm(offset), -1, 1)))


def flat_orbit(ref_c2w, anchor, up, n=24):
    """n cameras, one turn about the vertical through anchor at the reference elevation and distance."""
    return [orbit_pose(ref_c2w, anchor, 2 * np.pi * i / n, 0.0, up) for i in range(n)]


def spiral(ref_c2w, anchor, up, n=24, climb_deg=30.0, max_elev_deg=80.0):
    """n cameras, one turn while the elevation rises linearly by climb_deg (capped at max_elev_deg)."""
    climb, max_elev = np.radians(climb_deg), np.radians(max_elev_deg)
    return [orbit_pose(ref_c2w, anchor, 2 * np.pi * i / n, climb * i / max(n - 1, 1), up, max_elev)
            for i in range(n)]
