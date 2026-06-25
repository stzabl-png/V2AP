#!/usr/bin/env python3
"""Visualize eval_pool batch candidates under an evaluation result directory.

Scans ``{result_dir}/candidates/**/**/*_grasp.hdf5`` (pool layout from
``--generate-candidate-each-trial``) and writes overlays to
``{result_dir}/vis_candidates/``.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from model.pdm.dataset import DEFAULT_ROTATED_MESH_DIR  # noqa: E402
from model.pdm.visualize import visualize  # noqa: E402


def discover_pool_hdf5(result_dir: Path, obj_ids: list[str] | None) -> list[str]:
    cand_root = result_dir / "candidates"
    if not cand_root.is_dir():
        raise FileNotFoundError(f"candidates directory not found: {cand_root}")
    paths = sorted(cand_root.glob("**/*_grasp.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no *_grasp.hdf5 under {cand_root}")
    if obj_ids:
        want = set(obj_ids)
        paths = [p for p in paths if p.parent.name in want or p.stem.split("_")[0] in want]
    return [str(p.resolve()) for p in paths]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize eval_pool PDM candidate HDF5 files in-place under output/evaluation/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--result-dir",
        required=True,
        help="eval_pool --result-dir (contains candidates/{obj_id}/*.hdf5)",
    )
    p.add_argument("--obj", nargs="*", default=None, help="Optional subset of obj_id")
    p.add_argument(
        "--outdir",
        default=None,
        help="Default: {result-dir}/vis_candidates",
    )
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--mesh-root", default=DEFAULT_ROTATED_MESH_DIR)
    p.add_argument("--bg-points", type=int, default=8000)
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--overview-cols", type=int, default=5)
    p.add_argument("--overview-only", action="store_true", help="Only write overview.png")
    p.add_argument("--no-overview", action="store_true")
    return p


def main() -> None:
    cli = build_parser().parse_args()
    result_dir = Path(cli.result_dir).expanduser().resolve()
    outdir = Path(cli.outdir).expanduser().resolve() if cli.outdir else result_dir / "vis_candidates"
    files = discover_pool_hdf5(result_dir, cli.obj)
    vis_args = Namespace(
        hdf5=files,
        candidates_dir=str(result_dir / "candidates"),
        condition_h5=None,
        use_condition_cache=False,
        outdir=str(outdir),
        obj=None,
        all=False,
        random=0,
        top=int(cli.top),
        bg_points=int(cli.bg_points),
        mesh_root=str(cli.mesh_root),
        width_scale=1.0,
        dpi=int(cli.dpi),
        elev=22.0,
        azim=132.0,
        seed=42,
        overview=not cli.no_overview,
        overview_only=bool(cli.overview_only),
        overview_name="overview.png",
        overview_cols=int(cli.overview_cols),
    )
    print(f"Visualizing {len(files)} candidate file(s) -> {outdir}")
    visualize(vis_args)


if __name__ == "__main__":
    main()
