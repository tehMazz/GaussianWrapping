"""Unit tests for beags/mesh_field.py (mesh occupancy for L_occ / L_piv). CPU. Run:
    pytest -q gaussian_wrapping/tests/test_mesh_field.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial import Delaunay

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "submodules"))

import beags.occupancy as occupancy
from beags.mesh_field import MeshField, rasterize_tets
from beags.occupancy import LabeledVoxels, occupancy_loss
from utils.multilabel_meshing import save_tets

A = 0.5      # label 1: cube [-A, A]^3
B = 1.0      # label 0: the rest of the hull [-B, B]^3
CELL = 0.02


def _cube_distance(p):
    return torch.relu(p.abs() - A).norm(dim=1)


@pytest.fixture(scope="module")
def mesh_dir(tmp_path_factory):
    """Delaunay of a lattice on [-B, B]^3 (step A / 2): label 1 = tets with centroid in the inner cube."""
    d = str(tmp_path_factory.mktemp("mesh"))
    g = np.arange(-B, B + 1e-9, A / 2)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    pts = pts + np.random.default_rng(0).normal(scale=1e-6, size=pts.shape)  # break cospherical ties
    tri = Delaunay(pts)
    centroid = pts[tri.simplices].mean(axis=1)
    labels = (np.abs(centroid) < A).all(axis=1).astype(np.int64)
    save_tets(os.path.join(d, "tets.npz"), pts, tri.simplices, tri.neighbors, labels, 2)
    return d


@pytest.fixture(scope="module")
def field(mesh_dir):
    return MeshField.build(mesh_dir, [0, 1], cell=CELL, band=2 * CELL, max_ring=3, device="cpu")


def test_rasterize_single_tet():
    tet = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    inside = rasterize_tets(tet, torch.zeros(3), 0.25, (5, 5, 5), budget=7)   # budget < bbox: chunked
    ijk = torch.stack(torch.meshgrid(*[torch.arange(5)] * 3, indexing="ij"), -1)
    assert torch.equal(inside, ijk.sum(-1) <= 4)


def test_cube_distance_matches_analytic(field):
    torch.manual_seed(0)
    p = (torch.rand(5000, 3) * 2 - 1) * 0.6           # inside the label-1 grid (A + margin = 0.66)
    d = field.lookup(p, torch.ones(5000, dtype=torch.long))
    assert torch.allclose(d, _cube_distance(p), atol=2 * CELL)   # inside nodes: up to sqrt(3) cells off at corners


def test_contains_band_and_distance(field):
    lab = torch.ones(4, dtype=torch.long)
    p = torch.tensor([[0.0, 0, 0], [A + 1.5 * CELL, 0, 0], [A + 3 * CELL, 0, 0], [A + 7.9 * CELL, 0, 0]])
    assert field.contains(p, lab).tolist() == [True, True, False, False]
    d = field.distance(p, lab, max_ring=3)
    assert d[0] == 0 and abs(d[3] - 7.9 * CELL) < 0.5 * CELL
    assert field.distance(torch.tensor([[0.9, 0, 0]]), lab[:1], max_ring=1) == 2 * 2 * CELL   # clamped


def test_background_region(field):
    lab = torch.zeros(3, dtype=torch.long)
    p = torch.tensor([[0.8, 0.8, 0.8], [0.0, 0, 0], [10.0, 10, 10]])
    d = field.lookup(p, lab)
    assert d[0] == 0 and abs(d[1] - A) < CELL          # inner cube is label 1, not background
    assert d[2] == 0                                   # far background, off the grid: no constraint


def test_outside_grid_object_and_unknown_label(field):
    far = torch.tensor([[10.0, 10, 10]])
    assert field.lookup(far, torch.tensor([1])).item() == float("inf")
    assert field.lookup(far, torch.tensor([2])).item() == float("inf")    # no grid for label 2
    assert not field.contains(far, torch.tensor([1])).item()


def test_cache_reused(mesh_dir, field, capsys):
    MeshField.build(mesh_dir, [1], cell=CELL, band=2 * CELL, device="cpu")
    assert "cached" in capsys.readouterr().out
    assert os.path.isfile(os.path.join(mesh_dir, "dist_1.npz"))


def test_missing_tets_or_label(mesh_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        MeshField.build(str(tmp_path), [1], device="cpu")
    with pytest.raises(ValueError):
        MeshField.build(mesh_dir, [3], cell=CELL, device="cpu")


def test_labeled_voxels_distance_matches_rings():
    from beags.pivots import label_distance
    vox = LabeledVoxels.build(torch.tensor([[0.5, 0.5, 0.5]]), torch.tensor([1]), 1.0)
    points = torch.tensor([[0.5, 0.5, 0.5], [2.5, 0.5, 0.5], [3.5, 0.5, 0.5], [9.5, 0.5, 0.5]])
    labels = torch.ones(4, dtype=torch.long)
    assert label_distance(vox, points, labels).tolist() == [0.0, 2.0, 3.0, 4.0]


def test_per_label_cell_and_band(mesh_dir, tmp_path):
    os.symlink(os.path.join(mesh_dir, "tets.npz"), tmp_path / "tets.npz")
    f = MeshField.build(str(tmp_path), [0, 1], max_nodes=100_000, device="cpu", cache=False)
    assert f.grids[0].band == pytest.approx(2 * f.grids[1].band, rel=0.01)    # hull edge 2B, object edge 2A
    assert f.grids[1].band == pytest.approx(2 * (2 * A) / 100_000 ** (1 / 3), rel=0.01)
    h = f.tolerance(torch.tensor([0, 1, 2]))
    assert h[0] == f.grids[0].band and h[1] == f.grids[1].band and h[2] == f.grids[0].band


def _gaussians(center, scale, opacity=0.8):
    return SimpleNamespace(
        get_xyz=torch.tensor([center], dtype=torch.float32, requires_grad=True),
        get_rotation=torch.tensor([[1.0, 0, 0, 0]]),
        get_scaling_with_3D_filter=torch.full((1, 3), scale),
        get_opacity_with_3D_filter=torch.full((1, 1), opacity, requires_grad=True))


def test_occupancy_loss_with_mesh(field, monkeypatch):
    monkeypatch.setattr(occupancy, "build_rotation", lambda q: torch.eye(3)[None].expand(q.shape[0], 3, 3))
    lab = torch.tensor([1])
    inside = occupancy_loss(_gaussians([0.0, 0, 0], 0.05), lab, field, samples_per_gaussian=200)
    outside = occupancy_loss(_gaussians([0.62, 0, 0], 0.03), lab, field, samples_per_gaussian=200)
    assert inside.item() == 0 and outside.item() > 0


def test_pivot_loss_with_mesh(field, monkeypatch):
    import beags.pivots as pivots
    points = torch.tensor([[0.0, 0, 0], [A + 8 * CELL, 0, 0]])                 # inside; beyond the clamp
    monkeypatch.setattr(pivots, "front_pivots", lambda g, idx: points[idx])
    g = SimpleNamespace(get_xyz=points, get_opacity_with_3D_filter=torch.full((2, 1), 0.5))
    loss = pivots.pivot_loss(g, torch.ones(2, dtype=torch.long), field, max_ring=1, max_gaussians=None)
    assert loss.item() == pytest.approx(0.5 * (0 + 1) / 2, abs=0.05)           # max_ring 1: d <= 2 h -> relu(d - h) / h = 1
