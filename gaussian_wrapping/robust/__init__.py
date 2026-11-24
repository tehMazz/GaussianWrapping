"""Robustness to non-clean data: per-view bilateral grids, CoMe confidence, decoupled appearance, variance losses."""

APPEARANCE_MODES = ["none", "exposure", "network", "bilagrid", "bilagrid+decoupled"]


def add_args(parser):
    g = parser.add_argument_group("robust")
    g.add_argument("--appearance", choices=APPEARANCE_MODES, default="none",
                   help="exposure / network = GW's exposure compensation / appearance network; "
                        "bilagrid = per-view bilateral grid; +decoupled = CoMe SSIM-decoupled loss")
    # Bilateral-grid drift (HANDOFF 7.1): the losses see the graded render, eval/export the raw one. All opt-in.
    g.add_argument("--bilagrid_tv_weight", type=float, default=10.0, help="TV weight on the grids (gsplat: 10)")
    g.add_argument("--bilagrid_identity_weight", type=float, default=0.0,
                   help="weight of mean((G_v - I)^2), pulls every per-view grid toward identity; 0 = off")
    g.add_argument("--bilagrid_mean_weight", type=float, default=0.0,
                   help="weight of mean((G_v - mean G)^2), pulls every per-view grid toward the mean grid; 0 = off")
    g.add_argument("--bilagrid_until", type=int, default=-1,
                   help="last iteration using the grid; later its use the raw render and the grid is frozen. "
                        "-1 = grid until the end")
    g.add_argument("--lambda_raw", type=float, default=0.0,
                   help="weight of an extra photometric loss on the raw (uncompensated) render whenever an "
                        "appearance-compensated render drives the loss (bilagrid, or exposure in the robust path)")
    g.add_argument("--confidence", action="store_true", help="per-Gaussian confidence (CoMe)")
    g.add_argument("--conf_beta", type=float, default=0.075)
    g.add_argument("--conf_from_iter", type=int, default=500)
    g.add_argument("--lambda_color_var", type=float, default=0.0, help="CoMe uses 0.5")
    g.add_argument("--lambda_normal_var", type=float, default=0.0, help="CoMe uses 0.005")
    g.add_argument("--var_from_iter", type=int, default=15000)
    g.add_argument("--ignore_masks", type=str, default="",
                   help="dir of per-image <image_name>.png/.npy, nonzero = pixel ignored by the photometric loss")
    return g


def check_args(args):
    """Raise on knobs that would be silently ignored: --bilagrid_* without a bilagrid appearance, --lambda_raw without
    a compensated render in train.py's robust path (bilagrid, or exposure with --confidence / --ignore_masks)."""
    grid = args.appearance.startswith("bilagrid")
    set_ = [f for f, off in (("bilagrid_identity_weight", 0.0), ("bilagrid_mean_weight", 0.0), ("bilagrid_until", -1),
                             ("bilagrid_tv_weight", 10.0)) if getattr(args, f, off) != off]
    if set_ and not grid:
        raise ValueError(f"--{', --'.join(set_)} need --appearance bilagrid[+decoupled], got {args.appearance}")
    robust_exposure = args.appearance == "exposure" and (args.confidence or bool(args.ignore_masks))
    if args.lambda_raw > 0 and not (grid or robust_exposure):
        raise ValueError("--lambda_raw needs --appearance bilagrid[+decoupled], or exposure with --confidence / "
                         f"--ignore_masks; got {args.appearance}")


def bilagrid_config(args):
    """The bilagrid / raw-loss knobs, as recorded in gs_stats.json."""
    return {"tv_weight": args.bilagrid_tv_weight, "identity_weight": args.bilagrid_identity_weight,
            "mean_weight": args.bilagrid_mean_weight, "until": args.bilagrid_until, "lambda_raw": args.lambda_raw}
