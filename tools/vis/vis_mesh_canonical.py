#!/usr/bin/env python3
"""
vis_mesh_canonical.py — 仅可视化 canonical mesh + 物体坐标轴 (无抓取)
=====================================================================
用于检查 estimate_obj_rotation / rotation.json 后的摆放朝向。
使用 mesh_utils.load_mesh_canonical，与 USD/Sim 一致。

坐标轴: 红=X  绿=Y  蓝=Z（从 mesh 质心出发）

用法:
    python3 tools/vis_mesh_canonical.py --obj A01001
    python3 tools/vis_mesh_canonical.py --obj A01001 --dataset oakink
    python3 tools/vis_mesh_canonical.py --dataset oakink
    python3 tools/vis_mesh_canonical.py --all
    python3 tools/vis_mesh_canonical.py --obj A01001 --no-rotation   # 仅 scale，不应用 rotation.json

输出:
    output/mesh_dir_vis/{dataset}/{obj_id}.png
    output/mesh_dir_vis/raw/{dataset}/{obj_id}.png   (--no-rotation)
"""
import os
import sys
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "tools"))
from mesh_utils import (
    PROC_MESH_DIR,
    DATASETS,
    load_mesh_canonical,
    load_mesh_raw,
    get_canonical_euler,
    find_ply,
)

OUT_DIR = os.path.join(PROJ, "output", "mesh_dir_vis")
N_SURFACE = 6000
VIEW_ELEV, VIEW_AZIM = 20, 130


def list_objs(dataset: str):
    ds_dir = os.path.join(PROC_MESH_DIR, dataset)
    if not os.path.isdir(ds_dir):
        return []
    return sorted(
        o
        for o in os.listdir(ds_dir)
        if os.path.isfile(os.path.join(ds_dir, o, "mesh.ply"))
        and "_" not in o.split("_")[0]
    )


def render_one(
    obj_id: str,
    dataset: str,
    out_path: str,
    n_surface: int = N_SURFACE,
    apply_rotation: bool = True,
):
    """绘制并保存单张 PNG。返回 True 成功。"""
    euler = get_canonical_euler(obj_id, dataset)
    if apply_rotation:
        mesh = load_mesh_canonical(obj_id, dataset, verbose=False)
        mode_label = "canonical"
    else:
        mesh = load_mesh_raw(obj_id, dataset, apply_scale=True)
        mode_label = "raw (rotation not applied)"
    pts, _ = trimesh.sample.sample_surface(mesh, n_surface)

    fig = plt.figure(figsize=(7, 7), facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c="#5ab4d4", s=2.5, alpha=0.6, linewidths=0, depthshade=False,
    )

    origin = mesh.centroid
    axis_len = float(mesh.bounding_box.extents.max()) * 0.45
    for direction, color, label in [
        ([1, 0, 0], "r", "+X"),
        ([0, 1, 0], "g", "+Y"),
        ([0, 0, 1], "b", "+Z"),
    ]:
        ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            length=axis_len, color=color, arrow_length_ratio=0.25, linewidth=2.0,
        )

    ext_cm = mesh.bounding_box.extents * 100
    method_path = os.path.join(PROC_MESH_DIR, dataset, obj_id, "rotation.json")
    method = "?"
    if os.path.isfile(method_path):
        import json
        method = json.load(open(method_path)).get("method", "?")

    ax.set_xlabel("X", color="#ccc")
    ax.set_ylabel("Y", color="#ccc")
    ax.set_zlabel("Z", color="#ccc")
    ax.tick_params(colors="#888")
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    _set_equal_aspect(ax, pts)

    title = (
        f"{obj_id} ({dataset})  "
        f"{'rot=' + str([round(e, 1) for e in euler]) + '°  method=' + method if apply_rotation else 'rotation.json NOT applied'}"
        f"\n"
        f"bbox {ext_cm[0]:.1f}x{ext_cm[1]:.1f}x{ext_cm[2]:.1f} cm  "
        f"R=X G=Y B=Z ({mode_label})"
    )
    fig.suptitle(title, color="#ddd", fontsize=9, y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return True


def _set_equal_aspect(ax, pts):
    """3D 等比例显示。"""
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    c = (mn + mx) / 2
    r = (mx - mn).max() / 2 * 1.15
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def run_batch(
    datasets,
    obj_filter=None,
    out_root=OUT_DIR,
    n_surface=N_SURFACE,
    apply_rotation=True,
):
    ok, skip, fail = 0, 0, 0
    for ds in datasets:
        for obj_id in list_objs(ds):
            if obj_filter and obj_id != obj_filter:
                continue
            sub = ds if apply_rotation else os.path.join("raw", ds)
            out_path = os.path.join(out_root, sub, f"{obj_id}.png")
            if find_ply(obj_id, ds)[0] is None:
                skip += 1
                continue
            try:
                render_one(
                    obj_id, ds, out_path,
                    n_surface=n_surface,
                    apply_rotation=apply_rotation,
                )
                print(f"  ✅ {ds}/{obj_id} → {out_path}")
                ok += 1
            except Exception as e:
                print(f"  ❌ {ds}/{obj_id}: {e}")
                fail += 1
    return ok, skip, fail


def _out_path(out_root, dataset, obj_id, apply_rotation):
    sub = dataset if apply_rotation else os.path.join("raw", dataset)
    return os.path.join(out_root, sub, f"{obj_id}.png")


def main():
    parser = argparse.ArgumentParser(description="Canonical mesh + XYZ 轴 → PNG")
    parser.add_argument("--obj", help="单个物体 ID")
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default=None)
    parser.add_argument("--all", action="store_true", help="所有 DATASETS 下全部物体")
    parser.add_argument("--outdir", default=OUT_DIR, help="输出根目录")
    parser.add_argument("--n-surface", type=int, default=N_SURFACE, help="表面采样点数")
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help="不应用 rotation.json（仅 scale），输出到 outdir/raw/{dataset}/",
    )
    args = parser.parse_args()

    out_root = os.path.abspath(args.outdir)
    apply_rotation = not args.no_rotation

    if args.obj:
        ds = args.dataset
        if ds is None or ds == "all":
            ply, ds_found = find_ply(args.obj, None)
            if ply is None:
                print(f"❌ 未找到 mesh.ply: {args.obj}")
                sys.exit(1)
            ds = ds_found
        out_path = _out_path(out_root, ds, args.obj, apply_rotation)
        render_one(
            args.obj, ds, out_path,
            n_surface=args.n_surface,
            apply_rotation=apply_rotation,
        )
        print(f"✅ {out_path}")
        return

    if args.all:
        datasets = DATASETS
    elif args.dataset and args.dataset != "all":
        datasets = [args.dataset]
    else:
        parser.print_help()
        print("\n示例: --obj A01001 | --dataset oakink | --all")
        sys.exit(1)

    print(f"输出目录: {out_root}" + ("  [raw, no rotation]" if args.no_rotation else ""))
    for ds in datasets:
        n = len(list_objs(ds))
        print(f"\n=== {ds} ({n} 物体) ===")
    ok, skip, fail = run_batch(
        datasets,
        out_root=out_root,
        n_surface=args.n_surface,
        apply_rotation=apply_rotation,
    )
    print(f"\n完成: ok={ok}  skip={skip}  fail={fail}  → {out_root}")


if __name__ == "__main__":
    main()
