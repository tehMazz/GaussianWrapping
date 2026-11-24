"""
Multi-label watertight meshing with Primal Adaptive Meshing (PAM).

    python gaussian_wrapping/multilabel_mesh_extraction.py -m <model> --iteration N [--labels <labels.pt>] \
        [--label_masks <dir>] [--remove k ...] [--contact_h h] [--input_mesh <pivot mesh>] [--out_dir <dir>] [--save_tets]
    python gaussian_wrapping/multilabel_mesh_extraction.py --from_tets <out_dir>/tets.npz --remove k [--contact_h h]
        (CPU: step 3 only, on the labelled Delaunay saved by --save_tets; writes next to tets.npz unless --out_dir)

1. PAM candidate points on the global occupancy field (as primal_adaptive_meshing_extraction.py) and one Delaunay.
2. Per-label occupancy O_k (integrate_ours on the label-k Gaussians only, min over views), gated by the distance
   to the label-k centres; tet label = argmax_k mean O_k over the tet samples, or empty (-1) below the iso value.
3. Faces wherever the tet label changes (tet vs empty / outside the hull included):
    <out_dir>/mesh_labels.ply        face properties label_a, label_b (normal from label_a into label_b)
    <out_dir>/label_<k>.ply          closed boundary of the label-k tets, every label (for viewing)
    <out_dir>/hole_<k>.ply           hole after removing k (plan §5): closed boundary of the label-k tets (label_a = k;
                                     label_b = 0: contact with the background) + the contact patch faces not on it
                                     (label_a = 0, label_b = -1)
    <out_dir>/hole_<k>_closure.ply   contact patch: background surface (k tets relabelled empty) that bounded a k tet
                                     or lies within h of the label-k boundary; normals into the emptied side
    <out_dir>/hole_<k>.json          h and its source, tet / face counts, areas
    <out_dir>/tets.npz               with --save_tets: Delaunay + tet labels + voxel edge (for --from_tets)
h (--contact_h, default < 0 = auto) = the BEA-GS voxel edge: beags/occupancy.voxel_size of the labelled object
points x --contact_vicinity (4.30, as beags_finetune --voxel_vicinity). Points: --contact_points, else
<model>/beags/labeled_points.npz, else <model>/../train/beags/labeled_points.npz (flat layout), else the object
Gaussian centres. h = 0: hole = label-k boundary, closure = faces that touched label k (the old outputs).
hole_<k>.ply carries the patch because ArtiFixer-GS hole_mesh.py (--hole_mode mesh) marks a pixel when the mesh
is within (1 + depth_tol) x the removed scene's depth, which keeps the patch lying on the supporting surface:
the region around and under the object (contact shadow, mislabelled residue) is repainted too.
No largest-cluster post-processing.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'submodules'))

import glob
import math
from argparse import ArgumentParser

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from utils.camera_utils import get_cameras_spatial_extent
from utils.general_utils import safe_state
from utils.primal_adaptive_meshing_utils import MeshFromDelaunay, GaussianVectorField
from utils.multilabel_meshing import (
    GaussianSubset,
    support_gates,
    label_occupancy,
    tet_labels_from_fields,
    export_meshes,
    save_tets,
    load_tets,
)
from primal_adaptive_meshing_extraction import (
    compute_global_occupancy,
    sample_gt_mask_value,
    get_candidate_points,
)
from gaussian_renderer.ours import integrate_ours
from gaussian_renderer.beags_api import render_depth_alpha
from beags.masks import load_npy_mask
from beags.occupancy import voxel_size

# Script-specific options read from the command line only (never from <model>/cfg_args)
SCRIPT_ONLY_ARGS = ["labels", "label_masks", "input_mesh", "out_dir", "remove",
                    "contact_h", "contact_vicinity", "contact_points", "save_tets", "from_tets"]


def load_model(args):
    gaussians = GaussianModel(
        args.sh_degree,
        learn_occupancy=True,
        n_gaussian_features=4,
        use_unbounded_opacity=args.use_unbounded_opacity,
        use_appearance_network=False,
        use_mip_filter=not args.disable_mip_filter,
    )
    print(f"[INFO] Loading Gaussian Model at iteration {args.iteration}")
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

    if args.labels is not None:
        print(f"[INFO] Loading labels from {args.labels}")
        labels = torch.load(args.labels, map_location="cuda").int()
    else:
        labels = gaussians.labels
    if labels is None:
        raise FileNotFoundError(f"No labels.pt in {args.model_path}/point_cloud/iteration_{scene.loaded_iter}; pass --labels")
    if labels.shape[0] != gaussians.get_xyz.shape[0]:
        raise ValueError(f"{labels.shape[0]} labels for {gaussians.get_xyz.shape[0]} Gaussians")
    return scene, gaussians, labels.cuda()


def find_input_mesh(model_path):
    """Newest pivot-based mesh in the model dir (mesh_<sdf_mode>_<n>pivots*.ply), post-processed ones excluded."""
    candidates = [p for p in glob.glob(os.path.join(model_path, "mesh_*pivots*.ply")) if not p.endswith("_post.ply")]
    if not candidates:
        raise FileNotFoundError(f"No pivot-based mesh in {model_path}; run pivot_based_mesh_extraction.py or pass --input_mesh")
    return max(candidates, key=os.path.getmtime)


def find_contact_points(args):
    """labeled_points.npz for the voxel edge: --contact_points, <model>/beags/, <model>/../train/beags/, or None."""
    if args.contact_points is not None:
        return args.contact_points
    model = os.path.normpath(args.model_path)
    for path in (os.path.join(model, "beags", "labeled_points.npz"),
                 os.path.join(os.path.dirname(model), "train", "beags", "labeled_points.npz")):
        if os.path.isfile(path):
            return path
    return None


def voxel_edge(args, gaussians, labels):
    """BEA-GS voxel edge before the vicinity factor (beags_finetune.load_voxels) and where its points came from."""
    path = find_contact_points(args)
    if path is not None:
        with np.load(path) as data:
            xyz = torch.from_numpy(data["xyz"]).float()
        source = path
    else:
        xyz = gaussians.get_xyz[labels >= 1].detach().cpu()
        source = "object Gaussian centres"
    if xyz.shape[0] < 2:
        print(f"[WARN] {xyz.shape[0]} points for the voxel edge ({source}); pass --contact_h")
        return float("nan"), source
    edge = voxel_size(xyz)
    print(f"[INFO] BEA-GS voxel edge {edge:.5g} from {xyz.shape[0]} points ({source})")
    return edge, source


def contact_radius(args, edge, source):
    """h of the contact patch: --contact_h if >= 0, else voxel edge x --contact_vicinity. Returns h, info for the JSON."""
    info = {"voxel_edge": None if math.isnan(edge) else edge, "contact_vicinity": args.contact_vicinity,
            "voxel_points": source}
    if args.contact_h >= 0:
        return args.contact_h, {**info, "h_source": "--contact_h"}
    if math.isnan(edge):
        raise ValueError("No voxel edge for the automatic contact h; pass --contact_h")
    return edge * args.contact_vicinity, {**info, "h_source": "voxel_edge x contact_vicinity"}


def main_from_tets(args):
    """Step 3 on CPU from a --save_tets file."""
    tets = load_tets(args.from_tets)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.from_tets))
    print(f"[INFO] {args.from_tets}: {tets['points'].shape[0]} points, {tets['simplices'].shape[0]} tets, "
          f"{tets['num_labels']} labels")
    h, h_info = 0.0, {}
    if args.remove:
        h, h_info = contact_radius(args, tets["voxel_edge"], tets["voxel_points"] or f"{args.from_tets} (voxel_edge)")
    export_meshes(out_dir, tets["points"], tets["simplices"], tets["neighbors"], tets["tet_labels"],
                  tets["num_labels"], args.remove, h, h_info)


@torch.no_grad()
def main(args):
    safe_state(args.quiet)

    scene, gaussians, labels = load_model(args)
    views = scene.getTrainCameras()
    print(f"[INFO] Loaded {len(views)} views, {gaussians.get_xyz.shape[0]} Gaussians.")
    pipeline = PipelineParams(ArgumentParser()).extract(args)
    kernel_size = args.kernel_size if hasattr(args, 'kernel_size') else 0.0

    num_labels = int(labels.max().item()) + 1
    counts = torch.bincount(labels[labels >= 0].long(), minlength=num_labels).tolist()
    print(f"[INFO] {num_labels} labels, Gaussians per label: {counts}, unlabelled: {int((labels < 0).sum())}")

    occupancy_threshold = 0.5 - args.iso_surface_value
    assert 0 <= occupancy_threshold <= 1, "Occupancy threshold must be between 0 and 1"
    print(f"[INFO] Iso surface shift of {args.iso_surface_value} corresponds to using occupancy threshold: {occupancy_threshold}")
    args.iso_surface_value = occupancy_threshold  # as in primal_adaptive_meshing_extraction.py

    # 1. PAM candidate points on the global field (all Gaussians) and one Delaunay
    global_occupancy_func = lambda query_points: compute_global_occupancy(
        query_points, views, gaussians, pipeline, kernel_size, integrate_ours,
        mask_bg_threshold=args.mask_bg_threshold, chunk_size=args.chunk_size, device="cuda",
    )
    vector_field = GaussianVectorField(gaussians, nearest_neighbors_type="kdtree", k_neighbors=args.n_neighbors_vector_field)
    grad_occupancy_func = lambda query_points: vector_field.get_vector_field_quantities(
        query_points=query_points,
        gaussians=gaussians,
        k_neighbors=args.n_neighbors_vector_field,
        compute_curl=False,
        compute_normal_field=True,
    )["normal_field"]
    scene_pckg = {"scene_radius": get_cameras_spatial_extent(cameras=views)['radius'].item(), "cameras": views,
                  "gaussians_xyz": gaussians.get_xyz.detach()}
    if args.input_mesh is None:
        args.input_mesh = find_input_mesh(args.model_path)
    points = get_candidate_points(args, global_occupancy_func, grad_occupancy_function=grad_occupancy_func, scene_pckg=scene_pckg)

    print(f"[INFO] Initializing Delaunay Triangulation on {points.shape[0]} points...")
    delaunay = MeshFromDelaunay(points, add_corners=False)
    n_tets = delaunay.simplices.shape[0]
    if args.p_per_tet == 1:
        query_points = delaunay.barycenters
    else:
        query_points = delaunay.sample_random_tet_points(args.p_per_tet).reshape(-1, 3)  # (p_per_tet * T, 3)
    query_points = torch.tensor(query_points, dtype=torch.float32, device="cuda")

    # 2. Per-label occupancy, gated by the label-k support
    labels_np = labels.cpu().numpy()
    gates, radii = support_gates(
        query_points, gaussians.get_xyz.cpu().numpy(), labels_np, num_labels,
        radius_mult=args.gate_radius_mult, strict=args.strict_gate,
    )
    for k in range(num_labels):
        print(f"[INFO] Label {k}: gate radius {radii[k]:.4g}, {int(gates[k].sum())}/{query_points.shape[0]} query points")
    subsets = [GaussianSubset(gaussians, labels == k) if counts[k] > 0 else None for k in range(num_labels)]

    label_mask_func, depth_func = None, None
    if args.label_masks is not None:
        print(f"[INFO] Mask carving with {args.label_masks} (pixels covered by other labels abstain)")
        label_mask_func = lambda view: load_npy_mask(
            os.path.join(args.label_masks, f"{view.image_name}.npy"), view.image_height, view.image_width)
        depth_func = lambda view: render_depth_alpha(view, gaussians, pipeline)[0]

    print(f"[INFO] Computing {num_labels} label occupancies on {query_points.shape[0]} points...")
    occupancy = label_occupancy(
        query_points, gates, subsets, views, pipeline, kernel_size, integrate_ours, sample_gt_mask_value,
        label_mask_func=label_mask_func, depth_func=depth_func,
        mask_bg_threshold=args.mask_bg_threshold, chunk_size=args.chunk_size,
    )
    del gates, query_points
    fields = occupancy.view(num_labels, -1, n_tets).mean(dim=1).cpu().numpy()  # (L, T)
    del occupancy

    tet_labels = tet_labels_from_fields(fields, args.iso_surface_value)
    tet_counts = np.bincount(tet_labels + 1, minlength=num_labels + 1)
    print(f"[INFO] Tets: {tet_counts[0]} empty, per label: {tet_counts[1:].tolist()}")

    # 3. Labelled faces, holes (label-k boundary + contact patch dilated by h)
    out_dir = args.out_dir or os.path.join(args.model_path, "mesh")
    edge, source = float("nan"), None
    if args.remove or args.save_tets:
        edge, source = voxel_edge(args, gaussians, labels)
    if args.save_tets:
        path = os.path.join(out_dir, "tets.npz")
        save_tets(path, delaunay.points, delaunay.simplices, delaunay.neighbors, tet_labels, num_labels, edge, source)
        print(f"[INFO] Saved the labelled Delaunay to {path} (CPU re-runs: --from_tets {path})")
    h, h_info = 0.0, {}
    if args.remove:
        h, h_info = contact_radius(args, edge, source)
        print(f"[INFO] Contact patch radius h = {h:.5g} ({h_info['h_source']})")
    export_meshes(out_dir, delaunay.points, delaunay.simplices, delaunay.neighbors, tet_labels, num_labels,
                  args.remove, h, h_info)


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-label Primal Adaptive Meshing")
    model = ModelParams(parser, sentinel=True)
    PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_mip_filter", action="store_true")

    # Multi-label
    parser.add_argument("--labels", type=str, default=None, help="labels.pt (int32, -1 = ignored); default point_cloud/iteration_<it>/labels.pt")
    parser.add_argument("--label_masks", type=str, default=None,
                        help="dir with <image_name>.npy (C, H, W) masks (beags/masks.py format) for per-label mask carving")
    parser.add_argument("--remove", type=int, nargs="+", default=[], help="object labels to write hole_<k>.ply and hole_<k>_closure.ply for")
    parser.add_argument("--contact_h", type=float, default=-1.0,
                        help="dilation h (scene units) of the k<->0 contact patch; < 0: BEA-GS voxel edge x --contact_vicinity; "
                             "0: no dilation (hole = label-k faces, closure = faces that touched label k)")
    parser.add_argument("--contact_vicinity", type=float, default=4.30,
                        help="voxel edge multiplier for the automatic h (beags_finetune --voxel_vicinity)")
    parser.add_argument("--contact_points", type=str, default=None,
                        help="labeled_points.npz for the voxel edge; default <model>/beags/, then <model>/../train/beags/, "
                             "else the object Gaussian centres")
    parser.add_argument("--save_tets", action="store_true", help="also write <out_dir>/tets.npz (Delaunay + tet labels) for --from_tets")
    parser.add_argument("--from_tets", type=str, default=None,
                        help="tets.npz from --save_tets: rewrite the meshes / holes on CPU (no model, no GPU)")
    parser.add_argument("--gate_radius_mult", type=float, default=2.5,
                        help="label k can occupy points within this multiple of the median kNN spacing of its centres")
    parser.add_argument("--strict_gate", action="store_true",
                        help="gate by radius only (default also keeps points whose nearest centre has label k)")
    parser.add_argument("--chunk_size", type=int, default=1_000_000, help="query points per integrate call")
    parser.add_argument("--out_dir", type=str, default=None, help="default <model>/mesh")

    # PAM (as primal_adaptive_meshing_extraction.py)
    parser.add_argument("--input_mesh", type=str, default=None, help="pivot-based mesh; default newest <model>/mesh_*pivots*.ply")
    parser.add_argument("--max_points", type=int, default=1000000, help="Max points to process")
    parser.add_argument("--p_per_tet", type=int, default=10, help="Points per tet for occupancy check")
    parser.add_argument("--n_steps", default=10, type=int, help="Number of refinement steps")
    parser.add_argument("--vacancy_threshold", default=0.1, type=float, help="Vacancy threshold for filtering")
    parser.add_argument("--bounding_box_method", default="gaussians", type=str,
                        choices=["ground_truth", "scene", "gaussians", "blender"])
    parser.add_argument("--bounding_box_scaling", default=1.0, type=float)
    parser.add_argument("--bounding_box_file", default=None, type=str)
    parser.add_argument("--n_neighbors_vector_field", type=int, default=32, help="Number of neighbors for grad occupancy computation")
    parser.add_argument("--mesh_sampling_method", default="proportional_to_camera", type=str, choices=["surface_even", "proportional_to_camera"])
    parser.add_argument("--plot_vacancy_histogram", action="store_true", help="Plot histogram of vacancy values")
    parser.add_argument("--no_force_use_all_points", action="store_true")
    parser.add_argument("--oversampling_factor", type=int, default=2)
    parser.add_argument("--iso_surface_value", type=float, default=0.0, help="Isosurface value for mesh extraction")
    parser.add_argument("--mask_bg_threshold", type=float, default=0.01,
                        help="Mask alpha below this value is treated as definitely-background (occupancy forced to 0)")

    cmdline = parser.parse_args()
    if cmdline.from_tets is not None:
        main_from_tets(cmdline)
        sys.exit(0)

    args = get_combined_args(parser)
    for key in SCRIPT_ONLY_ARGS:
        setattr(args, key, getattr(cmdline, key))

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    main(args)
