"""
Multi-label Primal Adaptive Meshing: per-label occupancy on one Delaunay, tet labels and labelled faces.

Labels: 0 = background, 1..L-1 = objects, -1 = empty (tets) / ignored (Gaussians).
Face convention: every face is stored once, with its normal pointing from label_a into label_b.
Hole of a removed label k (plan §5): the label-k boundary plus the k<->0 contact patch dilated by h
(contact_patch, removal_hole); export_meshes writes every PLY + hole_<k>.json, save_tets / load_tets keep the
labelled Delaunay for CPU re-runs.
"""
import json
import os
from typing import List, Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

# Face opposite to vertex j of a tet; scipy's neighbors[:, j] is the tet across that face.
OPPOSITE_FACE = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


'''
Gaussian subsets and support gating
'''

class GaussianSubset:
    """Read-only view of the Gaussians selected by `mask`, exposing what integrate_ours reads."""

    @torch.no_grad()
    def __init__(self, gaussians, mask: torch.Tensor):
        self._gaussians = gaussians
        self._mask = mask
        self.active_sh_degree = gaussians.active_sh_degree
        self.get_xyz = gaussians.get_xyz[mask]
        self.get_opacity_with_3D_filter = gaussians.get_opacity_with_3D_filter[mask]
        self.get_scaling_with_3D_filter = gaussians.get_scaling_with_3D_filter[mask]
        self.get_rotation = gaussians.get_rotation[mask]

    def get_covariance(self, scaling_modifier=1):
        return self._gaussians.get_covariance(scaling_modifier)[self._mask]

    def __len__(self):
        return self.get_xyz.shape[0]


def knn_spacing(xyz: np.ndarray, k: int = 8) -> float:
    """Median over centres of the mean distance to their k nearest neighbours."""
    if xyz.shape[0] < 2:
        return 0.0
    k = min(k, xyz.shape[0] - 1)
    dist, _ = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    return float(np.median(dist[:, 1:].mean(axis=1)))


@torch.no_grad()
def support_gates(points: torch.Tensor, xyz: np.ndarray, labels: np.ndarray, num_labels: int,
                  radius_mult: float = 2.5, strict: bool = False, chunk_size: int = 4_000_000):
    """
    (L, N) bool on points.device: label k may be occupied at point x iff x is within
    r_k = radius_mult * knn_spacing(label-k centres) of a label-k centre, or (unless strict)
    label k owns the centre nearest to x (keeps the inside of thick objects, far from any centre).
    Gaussians with label -1 are ignored. Returns gates and the radii r_k.
    """
    valid = labels >= 0
    xyz, labels = xyz[valid], labels[valid]
    trees, radii = [], []
    for k in range(num_labels):
        xyz_k = xyz[labels == k]
        trees.append(cKDTree(xyz_k) if xyz_k.shape[0] > 0 else None)
        radii.append(radius_mult * knn_spacing(xyz_k))
    nearest_tree = None if strict else cKDTree(xyz)

    gates = torch.zeros((num_labels, points.shape[0]), dtype=torch.bool, device=points.device)
    for start in range(0, points.shape[0], chunk_size):
        chunk = points[start:start + chunk_size].detach().cpu().numpy().astype(np.float64)
        nearest_label = None
        if nearest_tree is not None:
            nearest_label = labels[nearest_tree.query(chunk, k=1, workers=-1)[1]]
        for k, tree in enumerate(trees):
            if tree is None:
                continue
            dist, _ = tree.query(chunk, k=1, distance_upper_bound=radii[k], workers=-1)
            gate = np.isfinite(dist)
            if nearest_label is not None:
                gate |= nearest_label == k
            gates[k, start:start + chunk.shape[0]] = torch.from_numpy(gate).to(points.device)
    return gates, radii


'''
Per-label occupancy (min over views of integrate_ours on the label-k Gaussians only)
'''

def project_to_grid(view, points: torch.Tensor):
    """grid_sample coordinates (N, 2) in [-1, 1] (align_corners=True) and camera depth (N,)."""
    R = torch.from_numpy(view.R).float().to(points.device)
    T = torch.from_numpy(view.T).float().to(points.device)
    points_cam = points @ R + T
    z = points_cam[:, 2]
    pts2d = points_cam[:, :2] / z.clamp_min(1e-8)[:, None]
    pts2d = torch.addcmul(
        pts2d.new_tensor(
            [
                (view.Cx * 2.0 + 1.0) / view.image_width - 1.0,
                (view.Cy * 2.0 + 1.0) / view.image_height - 1.0,
            ]
        ),
        pts2d.new_tensor([view.Fx * 2.0 / view.image_width, view.Fy * 2.0 / view.image_height]),
        pts2d,
    )
    return pts2d, z


def sample_image(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """(C, H, W) image sampled at (N, 2) grid coordinates -> (C, N)."""
    return torch.nn.functional.grid_sample(image[None], grid[None, None], align_corners=True)[0, :, 0]


@torch.no_grad()
def label_occupancy(
    points: torch.Tensor,
    gates: torch.Tensor,
    subsets: List[Optional[GaussianSubset]],
    views,
    pipeline,
    kernel_size: float,
    integrate_func,
    gt_mask_func,
    label_mask_func=None,
    depth_func=None,
    mask_bg_threshold: float = 0.01,
    chunk_size: int = 1_000_000,
):
    """
    (L, N) occupancy of `points` for each label, as compute_global_occupancy but with the label-k
    Gaussians only and 0 outside gates[k] or outside every frustum.

    Mask carving per view (label_mask_func(view) -> (C, H, W) soft masks or None): a point where the
    label-k mask (channel k) is < mask_bg_threshold is voted empty for k, unless the pixel is covered
    by another object (channel j >= 1, j != k, >= 0.5: the point may be occluded by it) or the point lies
    behind the rendered surface (depth_func(view) -> (H, W), 0 = no surface); both abstain.
    """
    num_labels, n_points = gates.shape
    occupancy = torch.ones((num_labels, n_points), dtype=torch.float32, device=points.device)
    seen = torch.zeros(n_points, dtype=torch.bool, device=points.device)
    for view in views:
        soft = label_mask_func(view) if label_mask_func is not None else None
        depth = None
        if soft is not None:
            soft = soft.to(points.device)
            depth = depth_func(view)[None]
            depth = torch.where(depth > 0, depth, torch.full_like(depth, float("inf")))
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            point_chunk = points[start:end]
            mask_alpha = gt_mask_func(view, point_chunk)
            if soft is not None:
                grid, z = project_to_grid(view, point_chunk)
                soft_at = sample_image(soft, grid)  # (C, n)
                in_front = z < sample_image(depth, grid)[0]
                covered = soft_at >= 0.5
                covered[0] = False  # only objects abstain; behind the visible background, the depth test does
            for k, subset in enumerate(subsets):
                sel = gates[k, start:end]
                if subset is None or not sel.any():
                    continue
                ret = integrate_func(point_chunk[sel], view, subset, pipeline, kernel_size)
                in_frustum = ret["inside"]
                seen_chunk = seen[start:end]
                seen_chunk[sel] = seen_chunk[sel] | in_frustum

                empty = mask_alpha[sel] < mask_bg_threshold
                abstain = torch.zeros_like(empty)
                if soft is not None and k < soft.shape[0]:
                    not_k = soft_at[k, sel] < mask_bg_threshold
                    other = torch.cat([covered[:k], covered[k + 1:]]).any(dim=0)[sel]
                    carve = not_k & ~other & in_front[sel]
                    empty = empty | carve
                    abstain = not_k & ~carve

                occ_chunk = occupancy[k, start:end]
                current = occ_chunk[sel]
                occ_chunk[sel] = torch.where(
                    in_frustum & empty,
                    torch.zeros_like(current),
                    torch.where(in_frustum & ~abstain, torch.min(ret["alpha_integrated"], current), current),
                )
    occupancy[:, ~seen] = 0.0
    occupancy[~gates] = 0.0
    return occupancy


'''
Tet labels and labelled faces
'''

def tet_labels_from_fields(fields: np.ndarray, iso: float) -> np.ndarray:
    """(L, T) per-label occupancy of each tet -> (T,) argmax label, -1 (empty) where the max < iso."""
    return np.where(fields.max(axis=0) >= iso, fields.argmax(axis=0), -1).astype(np.int64)


def half_faces(points: np.ndarray, simplices: np.ndarray, neighbors: np.ndarray):
    """
    The 4 faces of every tet, (4T, 3), oriented with the normal pointing out of the tet,
    and (4T, 2) [tet, tet across the face] (-1 = outside the convex hull).
    """
    tri = simplices[:, OPPOSITE_FACE].reshape(-1, 3)
    opposite = points[simplices.reshape(-1)]
    p = points[tri]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    inward = (normal * (opposite - p.mean(axis=1))).sum(-1) > 0
    tri[inward] = tri[inward][:, ::-1]
    pairs = np.column_stack((np.repeat(np.arange(simplices.shape[0]), 4), neighbors.reshape(-1)))
    return tri, pairs


def label_interfaces(tri: np.ndarray, pairs: np.ndarray, tet_labels: np.ndarray):
    """
    Faces between tets of different labels (outside the hull counts as empty, -1), each once,
    from the side with the larger label: normal from label_a into label_b, label_a > label_b.
    Returns faces (F, 3), label_a (F,), label_b (F,), tets (F, 2) [tet_a, tet_b] (tet_b = -1 outside).
    """
    lab = np.append(tet_labels, -1)  # pairs == -1 -> outside
    label_a, label_b = lab[pairs[:, 0]], lab[pairs[:, 1]]
    keep = label_a > label_b
    return tri[keep], label_a[keep], label_b[keep], pairs[keep]


def region_boundary(faces: np.ndarray, label_a: np.ndarray, label_b: np.ndarray, k: int):
    """Closed boundary of the label-k region, normals pointing out of it: faces, label_a (= k), label_b (other side)."""
    sel = (label_a == k) | (label_b == k)
    faces, label_a, label_b = faces[sel], label_a[sel], label_b[sel]
    flip = label_b == k
    faces = np.where(flip[:, None], faces[:, ::-1], faces)
    other = np.where(flip, label_a, label_b)
    return faces, np.full_like(other, k), other


def removal_closure(tri: np.ndarray, pairs: np.ndarray, tet_labels: np.ndarray, k: int, background: int = 0):
    """
    Background surface left behind by removing label k (k tets relabelled empty): the faces between
    `background` tets and former label-k tets, normals pointing into the emptied region.
    Returns faces, label_a (= background), label_b (= -1).
    """
    removed = np.where(tet_labels == k, -1, tet_labels)
    faces, label_a, label_b, tets = label_interfaces(tri, pairs, removed)
    was_k = np.append(tet_labels, -1)[tets[:, 1]] == k
    sel = (label_a == background) & was_k
    return faces[sel], label_a[sel], label_b[sel]


'''
Holes left by removing a label
'''

def triangle_samples(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(F, 7, 3) samples of each face: its 3 vertices, 3 edge midpoints and centroid."""
    p = points[faces]
    return np.concatenate([p, (p + np.roll(p, -1, axis=1)) / 2, p.mean(axis=1, keepdims=True)], axis=1)


def faces_within(points: np.ndarray, faces: np.ndarray, targets: np.ndarray, h: float,
                 chunk_size: int = 1_000_000) -> np.ndarray:
    """
    (F,) bool: some triangle_samples of the face lies within h of a target point. With targets = the samples
    of a surface, the face-to-surface distance is under-estimated by less than half the longest edge involved.
    """
    within = np.zeros(faces.shape[0], dtype=bool)
    if faces.shape[0] == 0 or targets.shape[0] == 0 or h < 0:
        return within
    p = points[faces]
    lo, hi = targets.min(axis=0) - h, targets.max(axis=0) + h
    idx = np.flatnonzero(np.all((p.max(axis=1) >= lo) & (p.min(axis=1) <= hi), axis=1))  # face box meets targets' box + h
    if idx.size == 0:
        return within
    tree = cKDTree(targets)
    for start in range(0, idx.size, chunk_size):
        sel = idx[start:start + chunk_size]
        samples = triangle_samples(points, faces[sel]).reshape(-1, 3)
        dist, _ = tree.query(samples, k=1, distance_upper_bound=h * (1 + 1e-9) + 1e-12, workers=-1)
        within[sel] = (dist.reshape(-1, 7) <= h).any(axis=1)
    return within


def contact_patch(points: np.ndarray, tri: np.ndarray, pairs: np.ndarray, tet_labels: np.ndarray, k: int, h: float,
                  background: int = 0):
    """
    The k<->background contact patch dilated by h: with the k tets relabelled empty, the faces between
    `background` tets and empty tets (convex-hull faces excluded) that bounded a k tet (removal_closure) or lie
    within h of the label-k boundary (faces_within, Euclidean). h <= 0: removal_closure only. Euclidean >= the
    geodesic dilation along the background surface, and it still finds the supporting surface when the k tets
    and the background tets are separated by a thin layer of empty tets (kitchen: 73 contact faces).
    Normals point into the empty side. Returns faces, label_a (= background), label_b (= -1),
    tets (F, 2) [background tet, tet across] (tet_labels[tets[:, 1]] == k: the face touched label k).
    """
    removed = np.where(tet_labels == k, -1, tet_labels)
    faces, label_a, label_b, tets = label_interfaces(tri, pairs, removed)
    inside = tets[:, 1] >= 0
    touched = np.zeros(faces.shape[0], dtype=bool)
    touched[inside] = tet_labels[tets[inside, 1]] == k
    candidate = (label_a == background) & (label_b == -1) & inside
    keep = candidate & touched
    if h > 0:
        lab = np.append(tet_labels, -1)
        boundary = tri[(lab[pairs[:, 0]] == k) & (lab[pairs[:, 1]] != k)]  # label-k half-faces facing another label
        targets = triangle_samples(points, boundary).reshape(-1, 3)
        idx = np.flatnonzero(candidate & ~touched)
        keep[idx] = faces_within(points, faces[idx], targets, h)
    return faces[keep], label_a[keep], label_b[keep], tets[keep]


def face_area(points: np.ndarray, faces: np.ndarray) -> float:
    p = points[faces]
    return float(0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=-1).sum())


def removal_hole(points: np.ndarray, tri: np.ndarray, pairs: np.ndarray, tet_labels: np.ndarray, k: int, h: float,
                 background: int = 0, interfaces=None):
    """
    Hole after removing label k (plan §5): the closed boundary of the label-k tets (region_boundary,
    label_a = k) plus the dilated contact patch (contact_patch, label_a = background, label_b = -1) without the
    patch faces on the k boundary (already there, with the opposite orientation). h <= 0: the label-k boundary only.
    interfaces: label_interfaces(tri, pairs, tet_labels) if already computed.
    Returns hole (faces, label_a, label_b), closure = the whole patch (faces, label_a, label_b), stats (dict).
    """
    faces, label_a, label_b, _ = interfaces if interfaces is not None else label_interfaces(tri, pairs, tet_labels)
    obj = region_boundary(faces, label_a, label_b, k)
    patch_faces, patch_a, patch_b, patch_tets = contact_patch(points, tri, pairs, tet_labels, k, h, background)
    touched = tet_labels[patch_tets[:, 1]] == k
    add = ~touched
    hole = tuple(np.concatenate([o, p[add]]) for o, p in zip(obj, (patch_faces, patch_a, patch_b)))
    stats = {
        "hole_faces": int(hole[0].shape[0]),
        "object_faces": int(obj[0].shape[0]),
        "object_contact_faces": int((obj[2] == background).sum()),
        "patch_faces": int(add.sum()),
        "closure_faces": int(patch_faces.shape[0]),
        "closure_contact_faces": int(touched.sum()),
        "object_area": face_area(points, obj[0]),
        "patch_area": face_area(points, patch_faces[add]),
        "closure_area": face_area(points, patch_faces),
    }
    return hole, (patch_faces, patch_a, patch_b), stats


def export_meshes(out_dir: str, points: np.ndarray, simplices: np.ndarray, neighbors: np.ndarray,
                  tet_labels: np.ndarray, num_labels: int, remove=(), h: float = 0.0, h_info: Optional[dict] = None):
    """
    Writes mesh_labels.ply, label_<k>.ply (every label) and, for each k in `remove`, hole_<k>.ply (removal_hole),
    hole_<k>_closure.ply (the dilated contact patch) and hole_<k>.json (h, face counts, areas).
    Returns {k: stats}.
    """
    tri, pairs = half_faces(points, simplices, neighbors)
    interfaces = label_interfaces(tri, pairs, tet_labels)
    faces, label_a, label_b, _ = interfaces
    path = os.path.join(out_dir, "mesh_labels.ply")
    write_labelled_ply(path, points, faces, label_a, label_b)
    print(f"[INFO] Saved {faces.shape[0]} labelled faces to {path}")

    for k in range(num_labels):
        region = region_boundary(faces, label_a, label_b, k)
        path = os.path.join(out_dir, f"label_{k}.ply")
        write_labelled_ply(path, points, *region)
        print(f"[INFO] Saved label {k} region ({region[0].shape[0]} faces) to {path}")

    tet_counts = np.bincount(tet_labels + 1, minlength=num_labels + 1)
    all_stats = {}
    for k in remove:
        assert 0 < k < num_labels, f"--remove {k}: object labels are 1..{num_labels - 1}"
        hole, closure, stats = removal_hole(points, tri, pairs, tet_labels, k, h, interfaces=interfaces)
        path = os.path.join(out_dir, f"hole_{k}.ply")
        write_labelled_ply(path, points, *hole)
        print(f"[INFO] Saved hole of label {k} ({stats['object_faces']} label-{k} faces, "
              f"{stats['object_contact_faces']} in contact with the background, + {stats['patch_faces']} contact "
              f"patch faces within h = {h:.4g}) to {path}")
        path = os.path.join(out_dir, f"hole_{k}_closure.ply")
        write_labelled_ply(path, points, *closure)
        print(f"[INFO] Saved background closure of label {k} ({stats['closure_faces']} faces, "
              f"{stats['closure_contact_faces']} touching label {k}) to {path}")
        info = {"label": int(k), "contact_h": float(h), **(h_info or {}),
                "tets": {"empty": int(tet_counts[0]), "label": int(tet_counts[k + 1]), "total": int(tet_labels.size)},
                **stats}
        with open(os.path.join(out_dir, f"hole_{k}.json"), "w") as f:
            json.dump(info, f, indent=1)
        all_stats[k] = info
    return all_stats


def save_tets(path: str, points: np.ndarray, simplices: np.ndarray, neighbors: np.ndarray, tet_labels: np.ndarray,
              num_labels: int, voxel_edge: float = float("nan"), voxel_points: str = ""):
    """Labelled Delaunay for CPU re-runs (multilabel_mesh_extraction.py --from_tets); voxel_points: source of voxel_edge."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, points=points.astype(np.float64), simplices=simplices.astype(np.int32),
             neighbors=neighbors.astype(np.int32), tet_labels=tet_labels.astype(np.int32),
             num_labels=np.int64(num_labels), voxel_edge=np.float64(voxel_edge), voxel_points=np.str_(voxel_points or ""))


def load_tets(path: str) -> dict:
    """save_tets -> dict(points, simplices, neighbors, tet_labels (int64), num_labels, voxel_edge (nan = unknown),
    voxel_points)."""
    with np.load(path) as data:
        return {"points": data["points"], "simplices": data["simplices"].astype(np.int64),
                "neighbors": data["neighbors"].astype(np.int64), "tet_labels": data["tet_labels"].astype(np.int64),
                "num_labels": int(data["num_labels"]), "voxel_edge": float(data["voxel_edge"]),
                "voxel_points": str(data["voxel_points"]) if "voxel_points" in data.files else ""}


'''
Export
'''

def compact_mesh(points: np.ndarray, faces: np.ndarray):
    """Keeps only the referenced vertices: (V', 3), (F, 3)."""
    if faces.shape[0] == 0:
        return np.zeros((0, 3), dtype=points.dtype), faces.reshape(0, 3)
    used, inverse = np.unique(faces, return_inverse=True)
    return points[used], inverse.reshape(-1, 3)


def write_labelled_ply(path: str, points: np.ndarray, faces: np.ndarray, label_a: np.ndarray, label_b: np.ndarray):
    """Triangle mesh PLY with int32 face properties 'label_a', 'label_b' (normal from a into b)."""
    verts, faces = compact_mesh(points, faces)
    vertex = np.empty(verts.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex["x"], vertex["y"], vertex["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    face = np.empty(faces.shape[0], dtype=[("vertex_indices", "i4", (3,)), ("label_a", "i4"), ("label_b", "i4")])
    face["vertex_indices"] = faces
    face["label_a"], face["label_b"] = label_a, label_b
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")]).write(path)
