#!/usr/bin/env python3
"""
vis_train_fp_rotated_hp.py — train_fp_rotated Human Prior 可视化（白底、无坐标轴）
================================================================================

从 data_hub/ProcessedData/train_fp_rotated/{dataset}/{obj}.hdf5 读取 HP，
整 mesh 密集采样 + KNN 插值后二值着色：human_prior > 0.1 为红色，其余为灰色。

用法:
    python3 tools/vis_train_fp_rotated_hp.py --obj A01001 --dataset oakink
    python3 tools/vis_train_fp_rotated_hp.py --obj A01001 --sparse-hp
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "tools"))
import random_grasp_sampler as rgs

DEFAULT_DATA_ROOT = os.path.join(PROJ, "data_hub", "ProcessedData", "train_fp_rotated")
DEFAULT_OUT_DIR = os.path.join(PROJ, "output", "vis_train_fp_rotated_hp")
DEFAULT_DATASETS = ("oakink", "dexycb")
BG = "white"

VIEW_ELEV, VIEW_AZIM = 20, 130
MULTI_VIEWS = [
    ("Front", 20, -60),
    ("Right", 20, 30),
    ("Top", 80, -60),
    ("Back", 20, 120),
]
HP_THRESH_DEFAULT = 0.1
COLOR_HIGH = "#e74c3c"
COLOR_LOW = "#d8d8d8"
N_DENSE_DEFAULT = 25000
DENSE_POINT_SIZE = 1.2
SPARSE_POINT_SIZE = 6.0


def load_manifest(data_root: str) -> list[tuple[str, str]]:
    """Return (subdir, obj_id) pairs from manifest.json if present."""
    manifest_path = os.path.join(data_root, "manifest.json")
    if not os.path.isfile(manifest_path):
        return []
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    jobs: list[tuple[str, str]] = []
    for entry in data.get("files", []):
        if entry.get("status") != "ok":
            continue
        jobs.append((entry["subdir"], entry["obj_id"]))
    return jobs


def list_objects(data_root: str, dataset: str | None) -> list[tuple[str, str]]:
    manifest_jobs = load_manifest(data_root)
    if manifest_jobs:
        if dataset is None:
            return manifest_jobs
        return [(ds, oid) for ds, oid in manifest_jobs if ds == dataset]

    jobs: list[tuple[str, str]] = []
    datasets = [dataset] if dataset else sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    for ds in datasets:
        ds_dir = os.path.join(data_root, ds)
        for name in sorted(os.listdir(ds_dir)):
            if name.endswith(".hdf5"):
                jobs.append((ds, name[:-5]))
    return jobs


def load_hp_hdf5(hdf5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with h5py.File(hdf5_path, "r") as f:
        pc = f["point_cloud"][()].astype(np.float32)
        hp = f["human_prior"][()].astype(np.float32).reshape(-1)
        fc = (
            f["force_center"][()].astype(np.float32)
            if "force_center" in f
            else None
        )
    return pc, hp, fc



def load_mesh_for_obj(
    obj_id: str,
    dataset: str,
    *,
    apply_oakink_scale: bool,
) -> tuple["trimesh.Trimesh | None", float]:
    import trimesh

    list_ds = "dexycb" if dataset == "ycb" else dataset
    mesh_path, scale_factor, _, apply_scale = rgs.find_obj_mesh(
        obj_id, dataset=list_ds, use_legacy_assets=False,
    )
    if mesh_path is None:
        return None, 1.0
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    sf = float(scale_factor)
    if apply_oakink_scale and apply_scale and abs(sf - 1.0) > 1e-8:
        mesh.vertices = mesh.vertices * sf
    return mesh, sf


def scale_hp_to_metric_frame(
    obj_id: str,
    dataset: str,
    pc: np.ndarray,
    fc: np.ndarray | None,
    scale_factor: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """OakInk HP / force_center 与 rotated mesh 同 metric 尺度。"""
    if not rgs.apply_metric_scale_to_hp_on_load(obj_id, dataset):
        return pc, fc
    pc = rgs.scale_hp_to_metric(pc, scale_factor)
    if fc is not None:
        fc = rgs.scale_hp_to_metric(fc.reshape(1, -1), scale_factor).reshape(-1)
    return pc, fc


def interpolate_hp_on_mesh(
    mesh,
    hp_pc: np.ndarray,
    hp: np.ndarray,
    *,
    n_dense: int = N_DENSE_DEFAULT,
    k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """rotated_mesh 表面密集采样，KNN 插值 human_prior。"""
    import trimesh

    vis_pc, _ = trimesh.sample.sample_surface(mesh, n_dense)
    vis_pc = vis_pc.astype(np.float32)
    hp_pc = np.asarray(hp_pc, dtype=np.float32)
    hp = np.asarray(hp, dtype=np.float32).reshape(-1)

    _, idx = cKDTree(hp_pc).query(vis_pc, k=k)
    if k == 1:
        hp_dense = hp[idx]
    else:
        dists = np.linalg.norm(vis_pc[:, None, :] - hp_pc[idx], axis=2)
        weights = 1.0 / (dists + 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        hp_dense = (weights * hp[idx]).sum(axis=1)
    return vis_pc, hp_dense.astype(np.float32)


def prepare_vis_points(
    obj_id: str,
    dataset: str,
    pc: np.ndarray,
    hp: np.ndarray,
    fc: np.ndarray | None,
    *,
    dense_mesh: bool,
    apply_oakink_scale: bool,
    n_dense: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    mesh, scale_factor = load_mesh_for_obj(
        obj_id, dataset, apply_oakink_scale=apply_oakink_scale,
    )
    pc_metric, fc_metric = scale_hp_to_metric_frame(
        obj_id, dataset, pc, fc, scale_factor,
    )

    if dense_mesh:
        if mesh is None:
            return pc_metric, hp, fc_metric, "sparse (mesh not found)"
        vis_pc, vis_hp = interpolate_hp_on_mesh(
            mesh, pc_metric, hp, n_dense=n_dense,
        )
        return vis_pc, vis_hp, fc_metric, f"dense mesh {len(vis_pc):,} pts"

    return pc_metric, hp, fc_metric, f"sparse HP {len(pc_metric)} pts"


def set_equal_aspect(ax, pts: np.ndarray, pad: float = 1.08) -> None:
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    c = (mn + mx) / 2
    r = (mx - mn).max() / 2 * pad
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def style_clean_3d(ax) -> None:
    """白底、无坐标轴/网格/pane。"""
    ax.set_axis_off()
    ax.set_facecolor(BG)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(BG)
        axis.pane.set_alpha(0.0)
        axis.line.set_color(BG)
        axis.set_ticklabels([])
        axis.set_ticks([])



def draw_hp_binary(
    ax,
    pc: np.ndarray,
    hp: np.ndarray,
    *,
    hp_thresh: float,
    point_size: float,
    show_force_center: bool,
    fc: np.ndarray | None,
) -> np.ndarray:
    labels = np.asarray(hp, dtype=np.float32).reshape(-1)
    pts = np.asarray(pc, dtype=np.float64)
    high = labels > hp_thresh
    low = ~high

    if np.any(low):
        ax.scatter(
            pts[low, 0], pts[low, 1], pts[low, 2],
            c=COLOR_LOW, s=point_size, alpha=0.75, linewidths=0,
            depthshade=False, zorder=2,
        )
    if np.any(high):
        ax.scatter(
            pts[high, 0], pts[high, 1], pts[high, 2],
            c=COLOR_HIGH, s=point_size, alpha=0.92, linewidths=0,
            depthshade=False, zorder=3,
        )

    if show_force_center and fc is not None:
        ax.scatter(
            fc[0], fc[1], fc[2],
            c="#2ecc71", s=80, marker="*",
            edgecolors="#1a7a3a", linewidths=0.6, zorder=5,
        )

    return pts


def render_one(
    obj_id: str,
    dataset: str,
    out_path: str,
    *,
    data_root: str = DEFAULT_DATA_ROOT,
    dense_mesh: bool = True,
    apply_oakink_scale: bool = True,
    n_dense: int = N_DENSE_DEFAULT,
    multi_view: bool = False,
    title: bool,
    hp_thresh: float = HP_THRESH_DEFAULT,
    point_size: float | None = None,
    elev: float = VIEW_ELEV,
    azim: float = VIEW_AZIM,
    dpi: int = 150,
    show_force_center: bool = False,
) -> tuple[bool, str]:
    hdf5_path = os.path.join(data_root, dataset, f"{obj_id}.hdf5")
    if not os.path.isfile(hdf5_path):
        return False, f"HDF5 not found: {hdf5_path}"

    pc, hp, fc = load_hp_hdf5(hdf5_path)
    vis_pc, vis_hp, vis_fc, mode_note = prepare_vis_points(
        obj_id, dataset, pc, hp, fc,
        dense_mesh=dense_mesh,
        apply_oakink_scale=apply_oakink_scale,
        n_dense=n_dense,
    )
    if point_size is None:
        point_size = DENSE_POINT_SIZE if dense_mesh else SPARSE_POINT_SIZE

    all_pts = vis_pc.astype(np.float64)
    views = MULTI_VIEWS if multi_view else [("", elev, azim)]

    n_cols = len(views)
    fig = plt.figure(figsize=(5.0 * n_cols, 5.0), facecolor=BG)

    for i, (view_name, view_elev, view_azim) in enumerate(views):
        ax = fig.add_subplot(1, n_cols, i + 1, projection="3d", facecolor=BG)
        draw_hp_binary(
            ax, vis_pc, vis_hp,
            hp_thresh=hp_thresh,
            point_size=point_size,
            show_force_center=show_force_center,
            fc=vis_fc,
        )
        ax.view_init(elev=view_elev, azim=view_azim)
        set_equal_aspect(ax, all_pts)
        style_clean_3d(ax)
        if multi_view and view_name:
            ax.set_title(view_name, fontsize=10, color="#333333", pad=2)

    if title:
        cov = float((vis_hp > hp_thresh).mean()) * 100
        fig.suptitle(
            f"{obj_id} ({dataset})  {mode_note}  red(>{hp_thresh})={cov:.1f}%",
            fontsize=11, color="#333333", y=0.98,
        )

    plt.tight_layout(pad=0.2)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=BG, pad_inches=0.05)
    plt.close(fig)
    return True, out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize train_fp_rotated Human Prior (red/gray binary, white bg)",
    )
    parser.add_argument("--obj", help="单个物体 ID")
    parser.add_argument(
        "--dataset", choices=[*DEFAULT_DATASETS, "all"],
        help="oakink / dexycb / all",
    )
    parser.add_argument("--all", action="store_true", help="处理 manifest 中全部物体")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--outdir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--sparse-hp", action="store_true",
        help="只画原始 4096 HP 稀疏点（默认整 mesh 密集采样）",
    )
    parser.add_argument("--n-dense", type=int, default=N_DENSE_DEFAULT, help="mesh 采样点数")
    parser.add_argument(
        "--no-oakink-scale", action="store_true",
        help="OakInk mesh/HP 不乘 scale.json（默认会乘以对齐）",
    )
    parser.add_argument("--multi-view", action="store_true", help="四视角拼图")
    parser.add_argument("--title", action="store_true", help="显示标题统计")
    parser.add_argument(
        "--hp-thresh", type=float, default=HP_THRESH_DEFAULT,
        help="human_prior 红色阈值（默认 0.1）",
    )
    parser.add_argument("--point-size", type=float, default=None, help="scatter 点大小（默认 dense=1.2 sparse=6）")
    parser.add_argument("--elev", type=float, default=VIEW_ELEV)
    parser.add_argument("--azim", type=float, default=VIEW_AZIM)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show-fc", action="store_true", help="显示 force_center")
    args = parser.parse_args()

    render_kwargs = dict(
        data_root=args.data_root,
        dense_mesh=not args.sparse_hp,
        apply_oakink_scale=not args.no_oakink_scale,
        n_dense=args.n_dense,
        multi_view=args.multi_view,
        title=args.title,
        hp_thresh=args.hp_thresh,
        point_size=args.point_size,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
        show_force_center=args.show_fc,
    )

    if args.obj:
        ds = args.dataset if args.dataset and args.dataset != "all" else None
        if ds is None:
            for candidate in DEFAULT_DATASETS:
                if os.path.isfile(os.path.join(args.data_root, candidate, f"{args.obj}.hdf5")):
                    ds = candidate
                    break
        if ds is None:
            print(f"❌ 无法推断 dataset，请指定 --dataset")
            sys.exit(1)
        out_path = os.path.join(args.outdir, ds, f"{args.obj}.png")
        ok, msg = render_one(args.obj, ds, out_path, **render_kwargs)
        print(f"{'✅' if ok else '❌'} {msg}")
        sys.exit(0 if ok else 1)

    if args.all:
        datasets = list(DEFAULT_DATASETS)
    elif args.dataset and args.dataset != "all":
        datasets = [args.dataset]
    else:
        parser.print_help()
        print("\n示例: --obj A01001 --dataset oakink | --dataset oakink | --all")
        sys.exit(1)

    jobs: list[tuple[str, str]] = []
    for ds in datasets:
        jobs.extend(list_objects(args.data_root, ds))

    print(f"输出: {args.outdir}/  共 {len(jobs)} 个物体")
    ok = fail = 0
    for i, (ds, obj_id) in enumerate(jobs):
        out_path = os.path.join(args.outdir, ds, f"{obj_id}.png")
        success, msg = render_one(obj_id, ds, out_path, **render_kwargs)
        if success:
            ok += 1
            print(f"  [{i + 1}/{len(jobs)}] ✅ {ds}/{obj_id}")
        else:
            fail += 1
            print(f"  [{i + 1}/{len(jobs)}] ❌ {ds}/{obj_id}: {msg}")
    print(f"\n完成: ok={ok}  fail={fail}  → {args.outdir}/")


if __name__ == "__main__":
    main()
