#!/usr/bin/env python3
"""
vis_merged_grasps.py — 从 merged robot_gt 可视化成功抓取（候选 vs 执行）

每个物体输出两张 PNG（默认前 5 条成功，按 score 降序）:
  - {obj}_candidate_top{N}.png   候选 grasp_point + 夹爪几何（规划 pose）
  - {obj}_executed_top{N}.png    executed_panda_hand@at_close 腕部 + 指尖

仅可视化 gripper_tips_trusted=True 的成功条目（含真 gripper_tips_loc@close）。

用法:
    python3 tools/vis_merged_grasps.py --obj A01001
    python3 tools/vis_merged_grasps.py --all \\
        --merged-dir output/grasp_collect_no_rot/merged \\
        --outdir output/grasp_collect_no_rot/vis_merged
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "tools"))

from mesh_utils import infer_dataset  # noqa: E402
import vis_grasp_candidates as vgc  # noqa: E402

TCP_OFFSET = vgc.TCP_OFFSET
DEFAULT_MERGED_DIR = os.path.join(PROJ, "output", "grasp_collect_no_rot", "merged")
DEFAULT_OUT_DIR = os.path.join(PROJ, "output", "grasp_collect_no_rot", "vis_merged")
FACE_COLOR = "#1a1a2e"


def _read_executed(g: h5py.Group, label: str) -> dict | None:
    sub = f"executed_panda_hand_{label}"
    if sub not in g:
        return None
    sg = g[sub]
    if "position" not in sg:
        return None
    return {
        "position": np.asarray(sg["position"][:], dtype=np.float64),
        "rotation": np.asarray(sg["rotation"][:], dtype=np.float64),
        "approach_dir": np.asarray(sg["approach_dir"][:], dtype=np.float64),
        "finger_dir": np.asarray(sg["finger_dir"][:], dtype=np.float64),
    }


def _round_from_source(path: str) -> str:
    m = re.search(r"round_(\d+)", path or "")
    return m.group(1) if m else "?"


def load_merged_successes(merged_path: str, top_n: int) -> list[dict]:
    rows: list[dict] = []
    with h5py.File(merged_path, "r") as f:
        if "successful_grasps" not in f:
            return rows
        sg = f["successful_grasps"]
        for key in sg.keys():
            g = sg[key]
            row = {
                "key": key,
                "name": str(g.attrs.get("name", key)),
                "score": float(g.attrs.get("score", 0.0)),
                "gripper_width": float(g.attrs.get("gripper_width", 0.04)),
                "grasp_point": np.asarray(g["grasp_point"][:], dtype=np.float64),
                "rotation": np.asarray(g["rotation"][:], dtype=np.float64),
                "approach_dir": np.asarray(g["approach_dir"][:], dtype=np.float64),
                "finger_dir": np.asarray(g["finger_dir"][:], dtype=np.float64),
                "source_file": str(g.attrs.get("source_file", "")),
                "round": _round_from_source(str(g.attrs.get("source_file", ""))),
                "executed_at_close": _read_executed(g, "at_close"),
                "tips_source": str(g.attrs.get("gripper_tips_source", "none")),
            }
            if "gripper_tips_loc" not in g:
                continue
            trusted = bool(g.attrs.get("gripper_tips_trusted", False))
            if not trusted:
                src = str(g.attrs.get("gripper_tips_source", ""))
                if src == "legacy_post_lift":
                    continue
                trusted = str(g.attrs.get("gripper_tips_snapshot", "at_close")) == "at_close"
            if not trusted:
                continue
            row["gripper_tips_loc"] = np.asarray(g["gripper_tips_loc"][:], dtype=np.float64)
            row["finger_width_actual"] = float(
                g.attrs.get("finger_width_actual", row["gripper_width"])
            )
            row["tips_trusted"] = True
            rows.append(row)
    rows.sort(key=lambda r: (-r["score"], r["name"]))
    return rows[:top_n]


def _draw_mesh_surface(ax, mesh: trimesh.Trimesh) -> None:
    pts, _ = trimesh.sample.sample_surface(mesh, 6000)
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c="#5ab4d4", s=2.0, alpha=0.5, linewidths=0, zorder=2, depthshade=False,
    )


def _set_axes_equal(mesh: trimesh.Trimesh, ax) -> None:
    b = mesh.bounds
    cx, cy, cz = (b[0] + b[1]) / 2
    r = max(b[1] - b[0]) * 0.9
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(b[0][2] - r * 0.15, b[0][2] + 2.1 * r)
    ax.set_xlabel("X", color="#aaa", fontsize=7)
    ax.set_ylabel("Y", color="#aaa", fontsize=7)
    ax.set_zlabel("Z", color="#aaa", fontsize=7)
    ax.tick_params(colors="#555", labelsize=6)
    ax.view_init(elev=22, azim=132)


def _style_ax(ax, title: str, title_color: str = "#ccc") -> None:
    ax.set_facecolor(FACE_COLOR)
    ax.set_title(title, color=title_color, fontsize=7.5, pad=4)


def _draw_candidate_ax(ax, mesh: trimesh.Trimesh, g: dict, panel_idx: int) -> None:
    _draw_mesh_surface(ax, mesh)
    pos = g["grasp_point"]
    approach = g["approach_dir"] / (np.linalg.norm(g["approach_dir"]) + 1e-9)
    finger = g["finger_dir"] / (np.linalg.norm(g["finger_dir"]) + 1e-9)
    hw = g["gripper_width"] / 2.0
    fl = pos - hw * finger
    fr = pos + hw * finger
    wrist = pos - approach * TCP_OFFSET

    ax.scatter(*pos, c="white", s=55, zorder=7, edgecolors="#2ca02c", linewidths=1.0)
    for fp in (fl, fr):
        ax.scatter(*fp, c="#17becf", s=40, marker="s", zorder=6, edgecolors="white", linewidths=0.3)
    ax.plot([fl[0], fr[0]], [fl[1], fr[1]], [fl[2], fr[2]], color="#17becf", linewidth=1.8, alpha=0.9)
    ax.scatter(*wrist, c="#ffdd57", s=70, marker="D", zorder=6, edgecolors="#666", linewidths=0.3)
    ax.plot(
        [wrist[0], pos[0]], [wrist[1], pos[1]], [wrist[2], pos[2]],
        color="#ffdd57", linestyle="--", linewidth=1.2, alpha=0.75,
    )
    ax.quiver(*pos, *approach, length=0.022, color="#2ca02c", arrow_length_ratio=0.35, linewidth=1.8)

    title = (
        f"[{panel_idx}] {g['name']}  s={g['score']:.1f}  r{g['round']}\n"
        f"candidate  w={g['gripper_width']*100:.1f}cm"
    )
    _style_ax(ax, title, "#8fd19e")
    _set_axes_equal(mesh, ax)


def _draw_executed_ax(ax, mesh: trimesh.Trimesh, g: dict, panel_idx: int) -> None:
    _draw_mesh_surface(ax, mesh)
    ex = g.get("executed_at_close")
    if ex is None:
        _style_ax(ax, f"[{panel_idx}] {g['name']}  (no executed@close)", "#f08080")
        _set_axes_equal(mesh, ax)
        return

    wrist = ex["position"]
    approach = ex["approach_dir"] / (np.linalg.norm(ex["approach_dir"]) + 1e-9)
    finger = ex["finger_dir"] / (np.linalg.norm(ex["finger_dir"]) + 1e-9)

    ax.scatter(*wrist, c="#ffdd57", s=80, marker="D", zorder=7, edgecolors="white", linewidths=0.4)
    ax.quiver(*wrist, *approach, length=0.025, color="#98df8a", arrow_length_ratio=0.35, linewidth=2.0)

    tips = g["gripper_tips_loc"]
    t0, t1 = tips[0], tips[1]
    ax.scatter(*t0, c="#17becf", s=50, marker="s", zorder=8, edgecolors="white", linewidths=0.4)
    ax.scatter(*t1, c="#17becf", s=50, marker="s", zorder=8, edgecolors="white", linewidths=0.4)
    ax.plot([t0[0], t1[0]], [t0[1], t1[1]], [t0[2], t1[2]], color="#17becf", linewidth=2.0)
    mid = (t0 + t1) / 2.0
    ax.plot(
        [wrist[0], mid[0]], [wrist[1], mid[1]], [wrist[2], mid[2]],
        color="#aaa", linestyle=":", linewidth=1.0,
    )
    tips_label = f"tips@close w={g['finger_width_actual']*100:.1f}cm"

    # 候选 grasp 点（淡）便于对比执行偏差
    gp = g["grasp_point"]
    ax.scatter(*gp, c="none", edgecolors="#ffffff", s=40, marker="o", linewidths=0.8, alpha=0.5)

    title = (
        f"[{panel_idx}] {g['name']}  s={g['score']:.1f}  r{g['round']}\n"
        f"executed@close  {tips_label}"
    )
    _style_ax(ax, title, "#8fd19e")
    _set_axes_equal(mesh, ax)


def _save_panel(
    mesh: trimesh.Trimesh,
    grasps: list[dict],
    mode: str,
    out_path: str,
    obj_id: str,
    *,
    dpi: int = 130,
) -> None:
    n = len(grasps)
    fig = plt.figure(figsize=(3.4 * n, 3.6), facecolor=FACE_COLOR)
    for i, g in enumerate(grasps):
        ax = fig.add_subplot(1, n, i + 1, projection="3d", facecolor=FACE_COLOR)
        if mode == "candidate":
            _draw_candidate_ax(ax, mesh, g, i + 1)
        else:
            _draw_executed_ax(ax, mesh, g, i + 1)
    fig.suptitle(
        f"{obj_id} — {mode} (top {n} successes)",
        color="#ddd",
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=FACE_COLOR)
    plt.close(fig)


def vis_one_object(
    obj_id: str,
    merged_path: str,
    outdir: str,
    *,
    top_n: int = 5,
    show_hp: bool = False,
    dpi: int = 130,
) -> bool:
    grasps = load_merged_successes(merged_path, top_n)
    if not grasps:
        print(f"  ⬛ {obj_id}: no trusted successes in merged")
        return False

    dataset = infer_dataset(obj_id)
    mesh, _sf, _apply, note = vgc.load_mesh_rotated_pipeline(obj_id, dataset)
    print(f"  {obj_id}: {len(grasps)} grasps  mesh ({note})")

    obj_out = os.path.join(outdir, obj_id)
    os.makedirs(obj_out, exist_ok=True)
    n = len(grasps)
    cand_path = os.path.join(obj_out, f"{obj_id}_candidate_top{n}.png")
    exec_path = os.path.join(obj_out, f"{obj_id}_executed_top{n}.png")
    _save_panel(mesh, grasps, "candidate", cand_path, obj_id, dpi=dpi)
    _save_panel(mesh, grasps, "executed", exec_path, obj_id, dpi=dpi)
    print(f"    → {os.path.basename(cand_path)}")
    print(f"    → {os.path.basename(exec_path)}")
    return True


def count_trusted_successes(merged_path: str) -> int:
    return len(load_merged_successes(merged_path, top_n=10_000))


def iter_merged_objects(merged_dir: str) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    for path in sorted(glob.glob(os.path.join(merged_dir, "*_merged.hdf5"))):
        obj_id = os.path.basename(path).replace("_robot_gt_merged.hdf5", "")
        try:
            n = count_trusted_successes(path)
        except OSError:
            continue
        if n > 0:
            jobs.append((obj_id, path))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize merged successful grasps")
    parser.add_argument("--obj", help="单个物体 ID")
    parser.add_argument(
        "--all", action="store_true",
        help="批量处理 merged-dir 下所有有成功的物体",
    )
    parser.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR)
    parser.add_argument("--outdir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--top", type=int, default=5, help="每物体最多画几条成功 (默认 5)")
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--show-hp", action="store_true", help="(预留) 默认不画 HP 以加快 batch")
    args = parser.parse_args()

    if not args.obj and not args.all:
        parser.error("需要 --obj ID 或 --all")

    merged_dir = os.path.abspath(args.merged_dir)
    outdir = os.path.abspath(args.outdir)

    if args.obj:
        merged_path = os.path.join(merged_dir, f"{args.obj}_robot_gt_merged.hdf5")
        if not os.path.isfile(merged_path):
            sys.exit(f"❌ merged 不存在: {merged_path}")
        ok = vis_one_object(
            args.obj, merged_path, outdir, top_n=args.top, show_hp=args.show_hp, dpi=args.dpi,
        )
        if not ok:
            sys.exit(1)
        print(f"\n✅ {os.path.join(outdir, args.obj)}")
        return

    jobs = iter_merged_objects(merged_dir)
    print(f"Batch: {len(jobs)} objects with trusted successes under {merged_dir}")
    ok_n = 0
    for obj_id, path in jobs:
        if vis_one_object(obj_id, path, outdir, top_n=args.top, show_hp=args.show_hp, dpi=args.dpi):
            ok_n += 1
    print(f"\n✅ Done: {ok_n}/{len(jobs)} objects → {outdir}")


if __name__ == "__main__":
    main()
