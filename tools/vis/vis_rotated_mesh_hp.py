#!/usr/bin/env python3
"""
vis_rotated_mesh_hp.py — rotated SAM3D mesh + Human Prior 叠加可视化
====================================================================
默认与 random_grasp_sampler / convert_obj_usd 一致:
  - mesh: SAM3DMesh/rotated_mesh/{dataset|ycb}/{obj}/mesh.ply × scale.json
  - HP:   train_fp_rotated/{dataset|dexycb}/{obj}.hdf5（OakInk HP 再 × scale）

用法:
    python3 tools/vis_rotated_mesh_hp.py --obj A01001 --dataset oakink
    python3 tools/vis_rotated_mesh_hp.py --dataset oakink
    python3 tools/vis_rotated_mesh_hp.py --dataset ycb
    python3 tools/vis_rotated_mesh_hp.py --all

    # 回退 obj_meshes + training_fp:
    python3 tools/vis_rotated_mesh_hp.py --obj A01001 --legacy-assets

输出:
    output/rotated_mesh_hp_vis/{dataset}/{obj_id}.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "tools"))
import random_grasp_sampler as rgs

OUT_DIR = os.path.join(PROJ, "output", "rotated_mesh_hp_vis")
DEFAULT_DATASETS = ("oakink", "ycb")
N_SURFACE = 6000
N_HP_MAX = 3000
VIEW_ELEV, VIEW_AZIM = 20, 130
HP_LABEL_MIN = 0.05
MESH_POINT_SIZE = 4.0
HP_POINT_SIZE = 6.0
BG = "#1a1a2e"


def mesh_hp_nn_stats(mesh: trimesh.Trimesh, hp_pc: np.ndarray) -> dict:
    """HP 点到 mesh 表面的最近邻（米）。"""
    from scipy.spatial import cKDTree
    pts = np.asarray(hp_pc, dtype=np.float64)
    if len(pts) == 0:
        return {"nn_median_cm": float("nan"), "nn_p90_cm": float("nan")}
    tree = cKDTree(mesh.vertices)
    d, _ = tree.query(pts, k=1)
    d_cm = d * 100.0
    return {
        "nn_median_cm": float(np.median(d_cm)),
        "nn_p90_cm": float(np.percentile(d_cm, 90)),
        "nn_max_cm": float(d.max()),
    }


def subsample_hp(hp_pc: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(hp_pc, dtype=np.float64)
    if len(pts) <= max_points:
        return pts
    idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
    return pts[idx]


def draw_human_prior(
    ax,
    hp_pc: np.ndarray | None,
    hp_labels: np.ndarray | None,
    *,
    max_points: int,
    mesh: trimesh.Trimesh | None = None,
    color_by_surface_dist: bool = False,
    label_min: float = HP_LABEL_MIN,
    hp_point_size: float = HP_POINT_SIZE,
) -> np.ndarray | None:
    """mesh + HP；HP 按 human_prior 用 hot colormap 着色。"""
    if hp_pc is None or hp_labels is None:
        return None
    labels = np.asarray(hp_labels, dtype=np.float32).reshape(-1)
    mask = labels > label_min
    if not np.any(mask):
        return None
    pts = np.asarray(hp_pc[mask], dtype=np.float64)
    vals = labels[mask]
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, vals = pts[idx], vals[idx]

    if color_by_surface_dist and mesh is not None:
        from scipy.spatial import cKDTree
        d_m = cKDTree(mesh.vertices).query(pts, k=1)[0] * 100.0
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=d_m, cmap="viridis_r", vmin=0.0, vmax=max(2.0, float(np.percentile(d_m, 95))),
            s=hp_point_size, marker="o", alpha=0.9, linewidths=0, zorder=4, depthshade=False,
        )
    else:
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=vals, cmap="hot", vmin=0.0, vmax=1.0,
            s=hp_point_size, marker="o", alpha=0.88, linewidths=0,
            zorder=4, depthshade=False,
        )
    return pts


def set_equal_aspect(ax, pts: np.ndarray, pad: float = 1.12):
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    c = (mn + mx) / 2
    r = (mx - mn).max() / 2 * pad
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def load_mesh_and_hp(
    obj_id: str,
    dataset: str | None,
    *,
    use_legacy_assets: bool,
):
    mesh_path, scale_factor, store_ds, apply_scale = rgs.find_obj_mesh(
        obj_id, dataset=dataset, use_legacy_assets=use_legacy_assets,
    )
    if mesh_path is None:
        return None, None, None, None, None, "mesh not found"

    ds = store_ds or rgs.infer_obj_dataset(obj_id, dataset)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if apply_scale and abs(scale_factor - 1.0) > 1e-8:
        mesh.vertices = mesh.vertices * float(scale_factor)

    hp_pc, hp_labels, hp_path = rgs.load_human_prior(
        obj_id, dataset=ds, use_rotated_hp=not use_legacy_assets,
    )
    if hp_pc is not None and rgs.apply_metric_scale_to_hp_on_load(obj_id, ds):
        hp_pc = rgs.scale_hp_to_metric(hp_pc, scale_factor)

    return mesh, hp_pc, hp_labels, hp_path, scale_factor, ds


def render_one(
    obj_id: str,
    dataset: str | None,
    out_path: str,
    *,
    use_legacy_assets: bool = False,
    show_hp: bool = True,
    n_surface: int = N_SURFACE,
    n_hp_max: int = N_HP_MAX,
    elev: float = VIEW_ELEV,
    azim: float = VIEW_AZIM,
    show_axes: bool = True,
    color_hp_by_dist: bool = False,
    mesh_alpha: float = 0.55,
    mesh_point_size: float = MESH_POINT_SIZE,
    hp_point_size: float = HP_POINT_SIZE,
) -> tuple[bool, str]:
    mesh, hp_pc, hp_labels, hp_path, sf, ds = load_mesh_and_hp(
        obj_id, dataset, use_legacy_assets=use_legacy_assets,
    )
    if mesh is None:
        return False, "mesh not found"

    mesh_pts, _ = trimesh.sample.sample_surface(mesh, n_surface)
    all_pts = [mesh_pts]
    combined = np.vstack(all_pts)

    fig = plt.figure(figsize=(8, 7), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    ax.scatter(
        mesh_pts[:, 0], mesh_pts[:, 1], mesh_pts[:, 2],
        c="#5ab4d4", s=mesh_point_size, marker="o", alpha=mesh_alpha,
        linewidths=0, zorder=3, depthshade=False,
    )
    hp_drawn = None
    if show_hp:
        hp_drawn = draw_human_prior(
            ax, hp_pc, hp_labels, max_points=n_hp_max,
            mesh=mesh, color_by_surface_dist=color_hp_by_dist,
            hp_point_size=hp_point_size,
        )
        if hp_drawn is not None:
            combined = np.vstack([combined, hp_drawn])

    if show_axes:
        origin = mesh.centroid
        axis_len = float(mesh.bounding_box.extents.max()) * 0.42
        for direction, color in [([1, 0, 0], "#e74c3c"), ([0, 1, 0], "#2ecc71"), ([0, 0, 1], "#3498db")]:
            ax.quiver(
                origin[0], origin[1], origin[2],
                direction[0], direction[1], direction[2],
                length=axis_len, color=color, arrow_length_ratio=0.2, linewidth=1.8,
            )

    ext_cm = mesh.bounding_box.extents * 100
    mode = "legacy obj_meshes + training_fp" if use_legacy_assets else "rotated_mesh + train_fp_rotated"
    hp_note = "no HP" if hp_pc is None else os.path.relpath(hp_path or "?", PROJ)
    align_note = ""
    if hp_pc is not None:
        chk = rgs.verify_mesh_hp_scale(
            mesh, hp_pc, apply_scale=False, scale_factor=sf, obj_id=obj_id,
        )
        surf = mesh_hp_nn_stats(mesh, hp_pc)
        align_note = (
            f"  bbox-NN≈{chk['nn_cm']:.1f}cm ({'OK' if chk['ok'] else 'MISMATCH'})"
            f"  |  surface-NN med={surf['nn_median_cm']:.2f}cm p90={surf['nn_p90_cm']:.2f}cm"
        )
    ax.view_init(elev=elev, azim=azim)
    set_equal_aspect(ax, combined)
    ax.set_xlabel("X", color="#aaa", fontsize=8)
    ax.set_ylabel("Y", color="#aaa", fontsize=8)
    ax.set_zlabel("Z", color="#aaa", fontsize=8)
    ax.tick_params(colors="#666", labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333")

    title = (
        f"{obj_id}  ({ds})  —  {mode}\n"
        f"mesh×scale={sf:.4f}  bbox {ext_cm[0]:.1f}×{ext_cm[1]:.1f}×{ext_cm[2]:.1f} cm\n"
        f"HP: {hp_note}{align_note}\n"
        f"cyan·=mesh  hot=HP (human_prior)"
        + ("  viridis=HP dist to mesh (cm)" if color_hp_by_dist else "")
    )
    fig.suptitle(title, color="#ddd", fontsize=9, y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return True, out_path


def iter_jobs(
    datasets: list[str],
    obj_filter: str | None,
    *,
    use_legacy_assets: bool,
) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    for ds in datasets:
        list_ds = "dexycb" if ds == "ycb" else ds
        for obj_id in rgs.list_dataset_objs(list_ds, use_legacy_assets=use_legacy_assets):
            if obj_filter and obj_id != obj_filter:
                continue
            jobs.append((obj_id, list_ds))
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Visualize rotated SAM3D mesh with Human Prior overlay (PNG)",
    )
    parser.add_argument("--obj", help="单个物体 ID")
    parser.add_argument(
        "--dataset", choices=[*DEFAULT_DATASETS, "all"],
        help="oakink / ycb / all（批量）",
    )
    parser.add_argument("--all", action="store_true", help="oakink + ycb 全部 ready 物体")
    parser.add_argument("--outdir", default=OUT_DIR, help="输出根目录")
    parser.add_argument("--legacy-assets", action="store_true", help="obj_meshes + training_fp")
    parser.add_argument("--no-hp", action="store_true", help="只画 mesh")
    parser.add_argument("--no-axes", action="store_true", help="不画 XYZ 轴")
    parser.add_argument("--n-surface", type=int, default=N_SURFACE)
    parser.add_argument("--n-hp", type=int, default=N_HP_MAX, help="HP 点云 subsample 上限")
    parser.add_argument("--elev", type=float, default=VIEW_ELEV)
    parser.add_argument("--azim", type=float, default=VIEW_AZIM)
    parser.add_argument(
        "--color-hp-dist", action="store_true",
        help="HP 按到 mesh 表面距离(cm)上色，用于查 frame/scale 是否对齐",
    )
    parser.add_argument("--mesh-alpha", type=float, default=0.55, help="mesh 点透明度")
    parser.add_argument("--mesh-size", type=float, default=MESH_POINT_SIZE, help="mesh scatter 点大小")
    parser.add_argument("--hp-size", type=float, default=HP_POINT_SIZE, help="HP scatter 点大小")
    args = parser.parse_args()

    if args.obj:
        ds = args.dataset if args.dataset and args.dataset != "all" else None
        sub = "legacy" if args.legacy_assets else "rotated"
        out_ds = ds or rgs.infer_obj_dataset(args.obj, None)
        if out_ds == "dexycb":
            out_ds = "ycb"
        out_path = os.path.join(args.outdir, sub, out_ds, f"{args.obj}.png")
        ok, msg = render_one(
            args.obj, ds, out_path,
            use_legacy_assets=args.legacy_assets,
            show_hp=not args.no_hp,
            n_surface=args.n_surface,
            n_hp_max=args.n_hp,
            elev=args.elev,
            azim=args.azim,
            show_axes=not args.no_axes,
            color_hp_by_dist=args.color_hp_dist,
            mesh_alpha=args.mesh_alpha,
            mesh_point_size=args.mesh_size,
            hp_point_size=args.hp_size,
        )
        if ok:
            print(f"✅ {msg}")
        else:
            print(f"❌ {args.obj}: {msg}")
            sys.exit(1)
        return

    if args.all:
        datasets = list(DEFAULT_DATASETS)
    elif args.dataset and args.dataset != "all":
        datasets = [args.dataset]
    else:
        parser.print_help()
        print("\n示例: --obj A01001 --dataset oakink | --dataset oakink | --all")
        sys.exit(1)

    sub = "legacy" if args.legacy_assets else "rotated"
    jobs = iter_jobs(datasets, None, use_legacy_assets=args.legacy_assets)
    print(f"输出: {args.outdir}/{sub}/  共 {len(jobs)} 个物体")
    ok = fail = 0
    for i, (obj_id, list_ds) in enumerate(jobs):
        out_ds = "ycb" if list_ds == "dexycb" else list_ds
        out_path = os.path.join(args.outdir, sub, out_ds, f"{obj_id}.png")
        success, msg = render_one(
            obj_id, list_ds, out_path,
            use_legacy_assets=args.legacy_assets,
            show_hp=not args.no_hp,
            n_surface=args.n_surface,
            n_hp_max=args.n_hp,
            elev=args.elev,
            azim=args.azim,
            show_axes=not args.no_axes,
            color_hp_by_dist=args.color_hp_dist,
            mesh_alpha=args.mesh_alpha,
            mesh_point_size=args.mesh_size,
            hp_point_size=args.hp_size,
        )
        if success:
            ok += 1
            print(f"  [{i+1}/{len(jobs)}] ✅ {out_ds}/{obj_id}")
        else:
            fail += 1
            print(f"  [{i+1}/{len(jobs)}] ❌ {out_ds}/{obj_id}: {msg}")
    print(f"\n完成: ok={ok}  fail={fail}  → {args.outdir}/{sub}/")


if __name__ == "__main__":
    main()
