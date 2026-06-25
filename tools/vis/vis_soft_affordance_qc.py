#!/usr/bin/env python3
"""
Visualize soft affordance GT for all objects in affordance_*_soft.h5.

Writes per-object PNGs and one (or more) grid figures under qc/.

Usage::

    python3 tools/vis_soft_affordance_qc.py \\
        --dataset-dir output/affordance_no_rot_executed

    # custom layout
    python3 tools/vis_soft_affordance_qc.py --cols 11 --dpi 120 --max-per-page 44
    python3 tools/vis_soft_affordance_qc.py --view 2d   # faster flat grid
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from math import ceil

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MERGED_DIR = os.path.join(PROJ, "output", "grasp_collect_no_rot", "merged")
sys.path.insert(0, PROJ)


def decode_obj_ids(raw) -> list[str]:
    return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]


def is_trusted_grasp(g: h5py.Group) -> bool:
    if "gripper_tips_loc" not in g:
        return False
    if bool(g.attrs.get("gripper_tips_trusted", False)):
        return True
    if str(g.attrs.get("gripper_tips_source", "")) == "legacy_post_lift":
        return False
    return str(g.attrs.get("gripper_tips_snapshot", "at_close")) == "at_close"


def count_trusted_grasps(merged_path: str) -> int:
    with h5py.File(merged_path, "r") as f:
        if "successful_grasps" not in f:
            return 0
        grp = f["successful_grasps"]
        return sum(1 for key in grp.keys() if is_trusted_grasp(grp[key]))


def load_pose_counts(
    obj_ids: list[str],
    *,
    qc_summary: str | None,
    merged_dir: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if qc_summary and os.path.isfile(qc_summary):
        with open(qc_summary, newline="") as f:
            for row in csv.DictReader(f):
                oid = row.get("obj_id", "").strip()
                if not oid:
                    continue
                try:
                    counts[oid] = int(row.get("n_grasps_trusted", row.get("n_trusted", 0)))
                except ValueError:
                    pass
    for oid in obj_ids:
        if oid in counts:
            continue
        mp = os.path.join(merged_dir, f"{oid}_robot_gt_merged.hdf5")
        counts[oid] = count_trusted_grasps(mp) if os.path.isfile(mp) else 0
    return counts


def load_soft_h5(path: str) -> list[dict]:
    rows: list[dict] = []
    with h5py.File(path, "r") as f:
        if "data/soft_labels" not in f:
            raise KeyError(f"no data/soft_labels in {path}")
        pts = f["data/points"][:]
        soft = f["data/soft_labels"][:]
        obj_ids = decode_obj_ids(f["data/obj_ids"][:])
        cats = None
        if "data/categories" in f:
            cats = decode_obj_ids(f["data/categories"][:])
        for i, oid in enumerate(obj_ids):
            rows.append({
                "obj_id": oid,
                "dataset": cats[i] if cats is not None else "",
                "points": pts[i].astype(np.float32),
                "soft": soft[i].astype(np.float32),
            })
    return rows


def _scatter_soft_2d(ax, pts: np.ndarray, soft: np.ndarray, *, title: str) -> None:
    """XY projection — readable in small grid cells."""
    order = np.argsort(soft)
    ax.scatter(
        pts[order, 0], pts[order, 1],
        c=plt.cm.jet(soft[order]), s=1.2, alpha=0.9, linewidths=0,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.set_title(title, fontsize=6, color="#ddd", pad=2)


def save_object_png(
    row: dict,
    n_trusted: int,
    out_path: str,
    *,
    dpi: int,
    view: str,
) -> None:
    pts, soft = row["points"], row["soft"]
    oid, ds = row["obj_id"], row.get("dataset", "")
    title = f"{oid}  trusted={n_trusted}  max={soft.max():.2f}"
    if ds:
        title = f"{oid} ({ds})  trusted={n_trusted}  max={soft.max():.2f}"

    if view == "3d":
        fig = plt.figure(figsize=(5, 4.5), facecolor="#1a1a2e")
        ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")
        order = np.argsort(soft)
        ax.scatter(
            pts[order, 0], pts[order, 1], pts[order, 2],
            c=plt.cm.jet(soft[order]), s=2, alpha=0.85, linewidths=0,
        )
        lo, hi = pts.min(0), pts.max(0)
        c = (lo + hi) / 2
        r = float((hi - lo).max()) * 0.55 + 1e-6
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.view_init(elev=22, azim=-58)
        ax.set_title(title, fontsize=9, color="#ddd")
        ax.tick_params(colors="#666", labelsize=6)
    else:
        fig, ax = plt.subplots(figsize=(4, 4), facecolor="#1a1a2e")
        _scatter_soft_2d(ax, pts, soft, title=title)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


def save_grid_page(
    page_rows: list[tuple[dict, int]],
    out_path: str,
    *,
    cols: int,
    dpi: int,
    view: str,
    page_label: str = "",
) -> None:
    n = len(page_rows)
    rows_n = ceil(n / cols)
    fig_w = max(6, cols * 1.35)
    fig_h = max(4, rows_n * 1.25)
    fig, axes = plt.subplots(
        rows_n, cols,
        figsize=(fig_w, fig_h),
        facecolor="#1a1a2e",
        squeeze=False,
    )
    cmap = plt.cm.jet
    for idx in range(rows_n * cols):
        ax = axes[idx // cols, idx % cols]
        ax.set_facecolor("#1a1a2e")
        if idx >= n:
            ax.axis("off")
            continue
        row, n_trusted = page_rows[idx]
        pts, soft = row["points"], row["soft"]
        oid = row["obj_id"]
        if view == "3d":
            ax.remove()
            ax = fig.add_subplot(rows_n, cols, idx + 1, projection="3d", facecolor="#1a1a2e")
            order = np.argsort(soft)
            ax.scatter(
                pts[order, 0], pts[order, 1], pts[order, 2],
                c=cmap(soft[order]), s=0.8, alpha=0.85, linewidths=0,
            )
            lo, hi = pts.min(0), pts.max(0)
            c = (lo + hi) / 2
            r = float((hi - lo).max()) * 0.55 + 1e-6
            ax.set_xlim(c[0] - r, c[0] + r)
            ax.set_ylim(c[1] - r, c[1] + r)
            ax.set_zlim(c[2] - r, c[2] + r)
            ax.view_init(elev=22, azim=-58)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            _scatter_soft_2d(
                ax, pts, soft,
                title=f"{oid}\nN={n_trusted}  max={soft.max():.2f}",
            )
        if view == "3d":
            ax.set_title(
                f"{oid}\nN={n_trusted}  max={soft.max():.2f}",
                fontsize=5.5, color="#ddd", pad=1,
            )

    supt = "Soft affordance GT (jet)"
    if page_label:
        supt += f"  —  {page_label}"
    fig.suptitle(supt, fontsize=10, color="#ccc", y=1.002)
    plt.tight_layout(pad=0.35, h_pad=0.6, w_pad=0.4)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="QC grid of soft affordance heatmaps")
    p.add_argument(
        "--dataset-dir",
        default=os.path.join(PROJ, "output", "affordance_no_rot_executed"),
    )
    p.add_argument(
        "--h5",
        default=None,
        help="default: {dataset-dir}/affordance_all_soft.h5",
    )
    p.add_argument(
        "--qc-dir",
        default=None,
        help="default: {dataset-dir}/qc",
    )
    p.add_argument(
        "--merged-dir",
        default=DEFAULT_MERGED_DIR,
        help="fallback trusted pose counts if qc/summary.csv missing",
    )
    p.add_argument("--cols", type=int, default=10, help="grid columns")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument(
        "--max-per-page",
        type=int,
        default=0,
        help="split grid into pages (0 = all objects on one figure)",
    )
    p.add_argument(
        "--view",
        choices=("2d", "3d"),
        default="3d",
        help="3d=scatter3d (default); 2d=XY projection (faster grid)",
    )
    p.add_argument("--no-individual", action="store_true", help="skip per-object PNGs")
    p.add_argument(
        "--sort",
        choices=("obj_id", "n_trusted_desc", "n_trusted_asc"),
        default="n_trusted_desc",
    )
    args = p.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    h5_path = args.h5 or os.path.join(dataset_dir, "affordance_all_soft.h5")
    qc_dir = args.qc_dir or os.path.join(dataset_dir, "qc")
    vis_dir = os.path.join(qc_dir, "vis_soft")
    summary_csv = os.path.join(qc_dir, "summary.csv")

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(h5_path)

    items = load_soft_h5(h5_path)
    pose_counts = load_pose_counts(
        [r["obj_id"] for r in items],
        qc_summary=summary_csv,
        merged_dir=os.path.abspath(args.merged_dir),
    )

    if args.sort == "obj_id":
        items.sort(key=lambda r: r["obj_id"])
    elif args.sort == "n_trusted_desc":
        items.sort(key=lambda r: (-pose_counts[r["obj_id"]], r["obj_id"]))
    else:
        items.sort(key=lambda r: (pose_counts[r["obj_id"]], r["obj_id"]))

    paired = [(r, pose_counts[r["obj_id"]]) for r in items]

    print(f"HDF5: {h5_path}  ({len(items)} objects)")
    print(f"Pose counts: {summary_csv if os.path.isfile(summary_csv) else args.merged_dir}")
    print(f"Output: {qc_dir}")

    if not args.no_individual:
        os.makedirs(vis_dir, exist_ok=True)
        for row, n_tr in paired:
            out = os.path.join(vis_dir, f"{row['obj_id']}.png")
            save_object_png(row, n_tr, out, dpi=args.dpi, view=args.view)
        print(f"  Per-object: {vis_dir}/ ({len(paired)} PNG)")

    max_per = args.max_per_page if args.max_per_page > 0 else len(paired)
    n_pages = ceil(len(paired) / max_per)
    grid_paths: list[str] = []
    for pi in range(n_pages):
        chunk = paired[pi * max_per : (pi + 1) * max_per]
        if n_pages == 1:
            name = "all_soft_heatmap_grid.png"
            label = f"{len(chunk)} objects"
        else:
            name = f"all_soft_heatmap_grid_p{pi + 1:02d}.png"
            label = f"page {pi + 1}/{n_pages} ({len(chunk)} objects)"
        out_grid = os.path.join(qc_dir, name)
        save_grid_page(
            chunk, out_grid,
            cols=args.cols, dpi=args.dpi, view=args.view, page_label=label,
        )
        grid_paths.append(out_grid)
        print(f"  Grid: {out_grid}")

    # colorbar legend (separate small figure)
    fig, ax = plt.subplots(figsize=(4, 0.45), facecolor="#1a1a2e")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label("soft label", color="#ccc", fontsize=8)
    cbar.ax.tick_params(colors="#aaa", labelsize=7)
    legend_path = os.path.join(qc_dir, "soft_heatmap_colorbar.png")
    fig.savefig(legend_path, dpi=100, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Colorbar: {legend_path}")


if __name__ == "__main__":
    main()
