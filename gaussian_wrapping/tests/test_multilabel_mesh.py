"""Multi-label PAM (utils/multilabel_meshing.py): tet labels, labelled faces, holes (+ contact patch dilated by h on
a synthetic table + box Delaunay), gating, mask carving, tets.npz round trip and the --from_tets CLI.

    python -m pytest -q gaussian_wrapping/tests/test_multilabel_mesh.py   (CPU; the *_cuda test skips without a GPU)
"""
import math
import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "submodules"))

from utils.multilabel_meshing import (  # noqa: E402
    contact_patch, export_meshes, faces_within, half_faces, label_interfaces, label_occupancy, load_tets,
    region_boundary, removal_closure, removal_hole, save_tets, support_gates, tet_labels_from_fields,
    triangle_samples, write_labelled_ply,
)

# Three tets around the axis (0,0,-1)-(0,0,1), each sharing one face with each of the other two:
# tet i = (axis bottom, axis top, ring i, ring i+1).
POINTS = np.array([[0, 0, -1], [0, 0, 1], [1, 0, 0], [-0.5, 0.8, 0], [-0.5, -0.8, 0]], dtype=np.float64)
SIMPLICES = np.array([[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]])
# neighbors[t, j] = tet across the face opposite vertex j (scipy convention), -1 = outside
NEIGHBORS = np.array([[-1, -1, 1, 2], [-1, -1, 2, 0], [-1, -1, 0, 1]])


def tet_fields(vertex_fields):
    """(L, V) vertex fields -> (L, T) mean over the tet vertices."""
    return vertex_fields[:, SIMPLICES].mean(axis=-1)


def face_normals(points, faces):
    p = points[faces]
    return np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])


def points_out_of(points, faces, simplices, tets):
    """True where each face normal points away from the centroid of tets[i]."""
    centre = points[simplices[tets]].mean(axis=1)
    return (face_normals(points, faces) * (points[faces].mean(axis=1) - centre)).sum(-1) > 0


def directed_edges(faces):
    return Counter((int(f[i]), int(f[(i + 1) % 3])) for f in faces for i in range(3))


def is_closed_oriented(faces):
    """Every directed edge appears once and its reverse once (closed, consistently oriented)."""
    edges = directed_edges(faces)
    return all(n == 1 and edges.get((b, a), 0) == 1 for (a, b), n in edges.items())


def canonical(face):
    i = int(np.argmin(face))
    return tuple(int(v) for v in np.roll(face, -i))


def test_tet_labels_argmax_and_iso():
    # label 0 high on the ring, label 1 high on the axis
    vertex_fields = np.array([[0.0, 0.0, 1.0, 1.0, 0.0],
                              [1.0, 1.0, 0.0, 0.0, 0.0]])
    fields = tet_fields(vertex_fields)  # tet 0: (0.5, 0.5), tet 1: (0.25, 0.5), tet 2: (0.25, 0.5)
    assert tet_labels_from_fields(fields, 0.5).tolist() == [0, 1, 1]  # ties -> lowest label
    assert tet_labels_from_fields(fields, 0.6).tolist() == [-1, -1, -1]
    assert tet_labels_from_fields(np.array([[0.2, 0.7, 0.4]]), 0.5).tolist() == [-1, 0, -1]


def test_half_faces_orientation_and_pairs():
    tri, pairs = half_faces(POINTS, SIMPLICES, NEIGHBORS)
    assert tri.shape == (12, 3) and pairs.shape == (12, 2)
    assert np.all(points_out_of(POINTS, tri, SIMPLICES, pairs[:, 0]))
    for t in range(3):  # each tet's own 4 faces close it
        assert is_closed_oriented(tri[pairs[:, 0] == t])
    for (t, n), f in zip(pairs, tri):  # shared faces have the same vertices in both tets
        if n >= 0:
            twin = tri[(pairs[:, 0] == n) & (pairs[:, 1] == t)]
            assert len(twin) == 1 and sorted(twin[0]) == sorted(f)


def test_label_interfaces_once_per_change():
    tri, pairs = half_faces(POINTS, SIMPLICES, NEIGHBORS)
    faces, la, lb, tets = label_interfaces(tri, pairs, np.array([2, 0, -1]))
    got = Counter(zip(la.tolist(), lb.tolist()))
    # tet0 (2): 2 hull faces, face vs tet1 (0), face vs tet2 (empty); tet1 (0): 2 hull faces, face vs tet2 (empty)
    assert got == Counter({(2, -1): 3, (2, 0): 1, (0, -1): 3})
    assert np.all(la > lb)
    assert np.all(points_out_of(POINTS, faces, SIMPLICES, tets[:, 0]))  # normal from label_a into label_b
    assert len({tuple(sorted(f)) for f in faces}) == len(faces)

    faces, la, lb, _ = label_interfaces(tri, pairs, np.array([1, 1, 1]))
    assert len(faces) == 6 and np.all(lb == -1) and is_closed_oriented(faces)  # convex hull only
    assert len(label_interfaces(tri, pairs, np.array([-1, -1, -1]))[0]) == 0


def test_hole_and_closure():
    tri, pairs = half_faces(POINTS, SIMPLICES, NEIGHBORS)
    tet_labels = np.array([1, 0, 0])
    faces, la, lb, _ = label_interfaces(tri, pairs, tet_labels)

    hole, ha, hb = region_boundary(faces, la, lb, 1)
    assert len(hole) == 4 and np.all(ha == 1) and is_closed_oriented(hole)
    assert sorted(hb.tolist()) == [-1, -1, 0, 0]  # 2 hull faces + contact with tets 1 and 2
    assert np.all(points_out_of(POINTS, hole, SIMPLICES, np.zeros(len(hole), dtype=int)))

    # object 2 vs object 1: boundary of the lower label is flipped to point out of it
    faces2, la2, lb2, _ = label_interfaces(tri, pairs, np.array([1, 2, 0]))
    hole2, ha2, hb2 = region_boundary(faces2, la2, lb2, 1)
    assert len(hole2) == 4 and np.all(ha2 == 1) and is_closed_oriented(hole2)
    assert sorted(hb2.tolist()) == [-1, -1, 0, 2]

    closure, ca, cb = removal_closure(tri, pairs, tet_labels, 1)
    assert len(closure) == 2 and np.all(ca == 0) and np.all(cb == -1)
    contact = {tuple(sorted(f)) for f, b in zip(hole, hb) if b == 0}
    assert {tuple(sorted(f)) for f in closure} == contact
    assert not np.any(points_out_of(POINTS, closure, SIMPLICES, np.zeros(2, dtype=int)))  # into the emptied tet
    # background after removal + closure = boundary of the background region in the relabelled mesh
    removed = np.where(tet_labels == 1, -1, tet_labels)
    f_r, la_r, lb_r, _ = label_interfaces(tri, pairs, removed)
    assert is_closed_oriented(f_r) and len(f_r) == 6 and np.all(la_r == 0)
    assert contact <= {tuple(sorted(f)) for f in f_r}


def test_matches_gw_get_surface():
    """Single label: interior faces equal MeshFromDelaunay.get_surface (which omits hull faces), same orientation."""
    pytest.importorskip("scene", exc_type=ImportError)  # first, as the GW scripts do (utils.geometry_utils <-> scene cycle)
    pam = pytest.importorskip("utils.primal_adaptive_meshing_utils", exc_type=ImportError)
    rng = np.random.default_rng(0)
    pts = rng.random((60, 3))
    mesh = pam.MeshFromDelaunay(pts, add_corners=False)
    occupied = np.linalg.norm(mesh.barycenters - 0.5, axis=1) < 0.3
    mesh.tet_colors = occupied
    _, gw_faces, used = mesh.get_surface(return_indices=True)
    gw = {canonical(used[f]) for f in gw_faces}
    tri, pairs = half_faces(mesh.points, mesh.simplices, mesh.neighbors)
    faces, la, lb, tets = label_interfaces(tri, pairs, np.where(occupied, 0, -1))
    ours = {canonical(f) for f in faces[tets[:, 1] >= 0]}
    assert len(ours) > 0 and ours == gw


def test_support_gates():
    # chains of 30 centres: mean 8-NN distance 2.5 * spacing, gate radius 2.5 * 2.5 * spacing
    xyz = np.concatenate([np.stack([np.arange(30) * 0.1, np.zeros(30), np.zeros(30)], 1),       # label 0, r ~ 0.625
                          np.stack([np.arange(30) * 0.01, np.ones(30), np.zeros(30)], 1),       # label 1, r ~ 0.0625
                          [[5.0, 5.0, 5.0]]])                                                    # ignored
    labels = np.array([0] * 30 + [1] * 30 + [-1])
    query = torch.tensor([[1.45, 0.05, 0.0],    # near label 0
                          [0.05, 1.02, 0.0],    # near label 1
                          [0.15, 0.8, 0.0],     # outside both radii, nearest centre label 1
                          [5.0, 5.0, 5.0]],     # on the ignored Gaussian, far from both labels
                         dtype=torch.float32)
    gates, radii = support_gates(query, xyz, labels, 3, radius_mult=2.5, strict=True)
    assert gates.shape == (3, 4) and not gates[2].any() and radii[2] == 0.0
    assert gates[0].tolist() == [True, False, False, False]
    assert gates[1].tolist() == [False, True, False, False]
    loose, _ = support_gates(query, xyz, labels, 3, radius_mult=2.5, strict=False)
    assert loose[1, 2] and not loose[0, 2]
    assert (loose | ~gates).all()  # the nearest-label rule only adds points


def make_view(width=8, height=6):
    return SimpleNamespace(R=np.eye(3, dtype=np.float32), T=np.zeros(3, dtype=np.float32), image_name="v",
                           image_width=width, image_height=height, Fx=4.0, Fy=4.0,
                           Cx=(width - 1) / 2, Cy=(height - 1) / 2)


def test_label_occupancy_mask_carving():
    """Label 1 carved where its mask is 0 and nothing else covers the pixel; abstains under another
    object or behind the rendered surface. Label 0 (background mask 1) keeps min(alpha)."""
    points = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 2.0]])
    subsets = [SimpleNamespace(alpha=0.3), SimpleNamespace(alpha=0.8)]
    integrate = lambda p, view, subset, pipe, ks: {"alpha_integrated": torch.full((p.shape[0],), subset.alpha),
                                                   "inside": torch.ones(p.shape[0], dtype=torch.bool)}
    ones = lambda view, p: torch.ones(p.shape[0])
    gates = torch.ones((2, 2), dtype=torch.bool)
    view = make_view()

    def run(soft, depth, gate=gates):
        return label_occupancy(points, gate, subsets, [view], None, 0.0, integrate, ones,
                               label_mask_func=lambda v: soft, depth_func=lambda v: depth)

    hw = (view.image_height, view.image_width)
    background = torch.stack([torch.ones(hw), torch.zeros(hw), torch.zeros(hw)])
    other = torch.stack([torch.zeros(hw), torch.zeros(hw), torch.ones(hw)])
    far, none = torch.full(hw, 10.0), torch.zeros(hw)

    occ = run(background, far)
    assert torch.allclose(occ[0], torch.tensor([0.3, 0.3])) and torch.all(occ[1] == 0)
    assert torch.all(run(background, none)[1] == 0)                   # no surface: in front of it
    assert torch.allclose(run(other, far)[1], torch.ones(2))          # covered by object 2: abstain
    assert torch.allclose(run(background, torch.full(hw, 1.5))[1], torch.tensor([0.0, 1.0]))  # 2nd point behind
    occ = run(None, None)                                             # no mask file: min accumulation
    assert torch.allclose(occ, torch.tensor([[0.3, 0.3], [0.8, 0.8]]))
    gate = torch.tensor([[True, False], [False, True]])
    assert torch.allclose(run(None, None, gate), torch.tensor([[0.3, 0.0], [0.0, 0.8]]))


def test_write_labelled_ply(tmp_path):
    plyfile = pytest.importorskip("plyfile")
    tri, pairs = half_faces(POINTS, SIMPLICES, NEIGHBORS)
    faces, la, lb, _ = label_interfaces(tri, pairs, np.array([1, -1, -1]))
    path = str(tmp_path / "mesh" / "m.ply")
    write_labelled_ply(path, POINTS, faces, la, lb)
    ply = plyfile.PlyData.read(path)
    assert ply["vertex"].count == 4 and ply["face"].count == 4  # unused vertex dropped
    assert np.all(ply["face"]["label_a"] == 1) and np.all(ply["face"]["label_b"] == -1)
    out_faces = np.stack(ply["face"]["vertex_indices"])
    verts = np.stack([ply["vertex"][c] for c in "xyz"], 1)
    assert is_closed_oriented(out_faces)
    assert np.allclose(face_normals(verts, out_faces), face_normals(POINTS, faces))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_label_occupancy_integrate_cuda():
    """Real integrate_ours on two blobs: inside blob 1 only O_1 is high; in front of both, both are 0."""
    pytest.importorskip("diff_gaussian_rasterization_ours")
    from gaussian_renderer.ours import integrate_ours
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2
    from utils.multilabel_meshing import GaussianSubset

    width, height, fov = 64, 48, math.radians(60.0)
    fovy = 2 * math.atan(math.tan(fov / 2) * height / width)
    w2v = torch.tensor(getWorld2View2(np.eye(3), np.zeros(3)), dtype=torch.float32).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fovy).transpose(0, 1).cuda()
    view = SimpleNamespace(FoVx=fov, FoVy=fovy, image_width=width, image_height=height, world_view_transform=w2v,
                           full_proj_transform=(w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0),
                           camera_center=w2v.inverse()[3, :3])
    g = torch.Generator(device="cuda").manual_seed(0)
    n = 200
    centres = torch.tensor([[0.8, 0.0, 3.0], [0.0, 0.0, 3.0]], device="cuda")
    labels = torch.arange(2 * n, device="cuda") // n
    xyz = centres[labels] + 0.05 * torch.randn(2 * n, 3, generator=g, device="cuda")
    model = SimpleNamespace(get_xyz=xyz, active_sh_degree=0,
                            get_opacity_with_3D_filter=torch.full((2 * n, 1), 0.9, device="cuda"),
                            get_scaling_with_3D_filter=torch.full((2 * n, 3), 0.08, device="cuda"),
                            get_rotation=torch.tensor([[1.0, 0, 0, 0]], device="cuda").repeat(2 * n, 1))
    subsets = [GaussianSubset(model, labels == k) for k in range(2)]
    assert len(subsets[0]) == n and torch.equal(subsets[1].get_xyz, xyz[n:])
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False)
    points = torch.tensor([[0.0, 0.0, 3.1], [0.0, 0.0, 1.0]], device="cuda")
    ones = lambda view, p: torch.ones(p.shape[0], device=p.device)
    occ = label_occupancy(points, torch.ones((2, 2), dtype=torch.bool, device="cuda"), subsets, [view], pipe, 0.0,
                          integrate_ours, ones)
    assert occ[1, 0] > 0.5 and occ[0, 0] < 0.1
    assert occ[:, 1].abs().max() < 0.1


# Synthetic scene for the contact patch: jittered grid Delaunay on [0, 1]^3, table (label 0) below z = 0.3,
# box (label 1) over the table centre from z = 0.3 + gap to 0.65, the rest empty.
TABLE_TOP, BOX_HALF, BOX_TOP = 0.3, 0.15, 0.65


def table_box(gap, n=13, seed=0):
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    g = np.linspace(0, 1, n)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    pts = pts + rng.uniform(-0.15, 0.15, pts.shape) / (n - 1)
    tri = Delaunay(pts)
    bary = tri.points[tri.simplices].mean(axis=1)
    box = (np.abs(bary[:, :2] - 0.5).max(axis=1) < BOX_HALF) & (bary[:, 2] > TABLE_TOP + gap) & (bary[:, 2] < BOX_TOP)
    labels = np.where(bary[:, 2] < TABLE_TOP, 0, np.where(box, 1, -1)).astype(np.int64)
    return tri.points, tri.simplices.astype(np.int64), tri.neighbors.astype(np.int64), labels


def face_set(faces):
    return {tuple(sorted(int(v) for v in f)) for f in faces}


def k_boundary_vertices(points, tri, pairs, labels, k=1):
    lab = np.append(labels, -1)
    return points[np.unique(tri[(lab[pairs[:, 0]] == k) & (lab[pairs[:, 1]] != k)])]


def test_triangle_samples_and_faces_within():
    pts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    samples = triangle_samples(pts, np.array([[0, 1, 2]]))
    assert samples.shape == (1, 7, 3) and np.allclose(samples[0, 3], [0.5, 0, 0]) and np.allclose(samples[0, 6], [1 / 3, 1 / 3, 0])
    faces = np.array([[0, 1, 2]])
    target = np.array([[0.5, 0.0, 0.3]])  # 0.3 above the midpoint of edge 0-1
    assert faces_within(pts, faces, target, 0.31)[0] and not faces_within(pts, faces, target, 0.29)[0]
    assert not faces_within(pts, faces, np.zeros((0, 3)), 1.0)[0]
    assert faces_within(pts, faces, np.array([[0.0, 0, 0]]), 0.0)[0]  # shared vertex at h = 0


@pytest.mark.parametrize("gap", [0.0, 0.1])
def test_contact_patch_h0_is_closure(gap):
    points, simplices, neighbors, labels = table_box(gap)
    tri, pairs = half_faces(points, simplices, neighbors)
    faces, la, lb, tets = contact_patch(points, tri, pairs, labels, 1, 0.0)
    closure = removal_closure(tri, pairs, labels, 1)
    assert face_set(faces) == face_set(closure[0]) and np.all(la == 0) and np.all(lb == -1)
    assert np.all(labels[tets[:, 1]] == 1)
    if gap == 0.0:
        assert len(faces) > 0  # the box sits on the table
    else:
        assert len(faces) == 0  # a layer of empty tets between them: no contact faces at all


def test_contact_patch_dilation():
    points, simplices, neighbors, labels = table_box(0.1)
    tri, pairs = half_faces(points, simplices, neighbors)
    removed = np.where(labels == 1, -1, labels)
    rf, rla, rlb, rtets = label_interfaces(tri, pairs, removed)
    surface = (rla == 0) & (rlb == -1) & (rtets[:, 1] >= 0)  # every candidate: table top (no hull faces)

    patches = {h: contact_patch(points, tri, pairs, labels, 1, h) for h in (0.1, 0.2, 5.0)}
    sets = {h: face_set(p[0]) for h, p in patches.items()}
    assert 0 < len(sets[0.1]) < len(sets[0.2]) < len(sets[5.0])
    assert sets[0.1] <= sets[0.2] <= sets[5.0] == face_set(rf[surface])  # monotone; large h = the whole table top

    k_verts = k_boundary_vertices(points, tri, pairs, labels)
    from scipy.spatial import cKDTree
    tree = cKDTree(k_verts)
    h = 0.1
    faces, la, lb, tets = patches[h]
    assert np.all(la == 0) and np.all(lb == -1) and np.all(labels[tets[:, 0]] == 0) and np.all(labels[tets[:, 1]] == -1)
    edge = np.linalg.norm(points[faces] - points[np.roll(faces, 1, axis=1)], axis=-1).max()
    assert tree.query(points[faces].reshape(-1, 3))[0].reshape(-1, 3).min(axis=1).max() <= h + 2 * edge
    centre = points[faces].mean(axis=1)
    assert np.all(np.abs(centre[:, :2] - 0.5).max(axis=1) < BOX_HALF + h + 2 * edge)  # under / around the box
    assert np.all(np.abs(centre[:, 2] - TABLE_TOP) < 2 * edge)                         # on the table top
    rest = face_set(rf[surface]) - sets[h]
    rest = np.array(sorted(rest))
    assert np.all(tree.query(points[rest].mean(axis=1))[0] > h)  # left-out faces: centroid farther than h
    # normals out of the table tet, into the emptied side (up, on average)
    assert np.all(points_out_of(points, faces, simplices, tets[:, 0]))
    assert face_normals(points, faces)[:, 2].mean() > 0


@pytest.mark.parametrize("gap", [0.0, 0.1])
def test_removal_hole(gap):
    points, simplices, neighbors, labels = table_box(gap)
    tri, pairs = half_faces(points, simplices, neighbors)
    faces, la, lb, _ = label_interfaces(tri, pairs, labels)
    obj = region_boundary(faces, la, lb, 1)
    edges = directed_edges(obj[0])  # closed and oriented; tets meeting along an edge make it non-manifold there
    assert all(n == edges.get((b, a), 0) for (a, b), n in edges.items())

    hole0, closure0, stats0 = removal_hole(points, tri, pairs, labels, 1, 0.0)
    for a, b in zip(hole0, obj):  # h = 0: the old outputs
        assert np.array_equal(a, b)
    assert face_set(closure0[0]) == face_set(removal_closure(tri, pairs, labels, 1)[0]) and stats0["patch_faces"] == 0

    h = 0.15
    hole, closure, stats = removal_hole(points, tri, pairs, labels, 1, h)
    patch = face_set(closure[0])
    assert len(face_set(hole[0])) == len(hole[0])  # no triangle twice (contact faces only once, from the box)
    assert face_set(hole[0]) == face_set(obj[0]) | patch
    assert np.array_equal(hole[0][:len(obj[0])], obj[0]) and np.all(hole[1][:len(obj[0])] == 1)
    assert np.all(hole[1][len(obj[0]):] == 0) and np.all(hole[2][len(obj[0]):] == -1)
    assert stats["hole_faces"] == len(hole[0]) == stats["object_faces"] + stats["patch_faces"]
    assert stats["closure_faces"] == len(closure[0]) == stats["patch_faces"] + stats["closure_contact_faces"]
    assert stats["closure_contact_faces"] == len(closure0[0]) and stats["object_contact_faces"] == int((obj[2] == 0).sum())
    assert stats["patch_area"] > 0 and stats["closure_area"] >= stats["patch_area"]


def test_export_meshes_and_tets_roundtrip(tmp_path):
    plyfile = pytest.importorskip("plyfile")
    import json
    points, simplices, neighbors, labels = table_box(0.1)
    path = str(tmp_path / "mesh" / "tets.npz")
    save_tets(path, points, simplices, neighbors, labels, 2, voxel_edge=0.02, voxel_points="pts.npz")
    tets = load_tets(path)
    assert tets["voxel_points"] == "pts.npz"
    assert np.array_equal(tets["tet_labels"], labels) and np.array_equal(tets["neighbors"], neighbors)
    assert tets["num_labels"] == 2 and tets["voxel_edge"] == 0.02 and np.allclose(tets["points"], points)

    out = str(tmp_path / "out")
    stats = export_meshes(out, tets["points"], tets["simplices"], tets["neighbors"], tets["tet_labels"], 2,
                          remove=[1], h=0.15, h_info={"h_source": "test"})
    for name in ("mesh_labels", "label_0", "label_1", "hole_1", "hole_1_closure"):
        assert os.path.isfile(os.path.join(out, f"{name}.ply"))
    info = json.load(open(os.path.join(out, "hole_1.json")))
    assert info == json.loads(json.dumps(stats[1]))
    assert info["contact_h"] == 0.15 and info["h_source"] == "test" and info["label"] == 1
    assert info["tets"]["label"] == int((labels == 1).sum()) and info["tets"]["total"] == len(labels)
    count = lambda name: plyfile.PlyData.read(os.path.join(out, f"{name}.ply"))["face"].count
    assert count("hole_1") == info["hole_faces"] and count("hole_1_closure") == info["closure_faces"]
    assert count("label_1") == info["object_faces"] and info["patch_faces"] > 0


def test_from_tets_cli(tmp_path):
    """multilabel_mesh_extraction.py --from_tets on CPU: auto h = voxel edge x --contact_vicinity (h = 0: test_removal_hole)."""
    pytest.importorskip("plyfile")
    pytest.importorskip("scene", exc_type=ImportError)
    pytest.importorskip("gaussian_renderer.ours", exc_type=ImportError)
    import json
    import subprocess
    points, simplices, neighbors, labels = table_box(0.1)
    path = str(tmp_path / "tets.npz")
    save_tets(path, points, simplices, neighbors, labels, 2, voxel_edge=0.02)
    script = os.path.join(BASE_DIR, "multilabel_mesh_extraction.py")
    run = lambda *extra: subprocess.run([sys.executable, script, "--from_tets", path, "--remove", "1", *extra],
                                        capture_output=True, text=True, timeout=600)
    res = run()  # one run: the script imports take ~40 s on the login node
    assert res.returncode == 0, res.stderr[-2000:]
    info = json.load(open(tmp_path / "hole_1.json"))
    assert info["contact_h"] == pytest.approx(0.02 * 4.30) and info["contact_vicinity"] == 4.30
    assert info["h_source"] == "voxel_edge x contact_vicinity" and info["voxel_edge"] == 0.02
    assert info["patch_faces"] > 0 and info["closure_contact_faces"] == 0  # gap: all of the patch comes from h
    for name in ("mesh_labels", "label_0", "label_1", "hole_1", "hole_1_closure"):
        assert (tmp_path / f"{name}.ply").is_file()
