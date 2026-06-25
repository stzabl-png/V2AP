#!/usr/bin/env python3
"""
vis_ego_contact.py — 接触图可视化

从 human_prior HDF5 生成彩色 3D 点云图（matplotlib 无需 display），
输出 PNG 可直接用于论文 / 汇报。

用法:
    conda run -n hawor python tools/vis_ego_contact.py --obj assemble_tile
    conda run -n hawor python tools/vis_ego_contact.py --obj assemble_tile --mesh
"""

import os, sys, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")   # 无 display 模式
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401

PROJ   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HP_DIR = os.path.join(PROJ, "data_hub", "human_prior")
OUT    = os.path.join(PROJ, "output", "affordance_ego")
os.makedirs(OUT, exist_ok=True)


def load_hp(obj_name):
    path = os.path.join(HP_DIR, f"{obj_name}.hdf5")
    if not os.path.exists(path):
        print(f"❌ HDF5 not found: {path}"); return None, None, None
    with h5py.File(path, "r") as f:
        pc  = f["point_cloud"][()].astype(np.float32)
        nrm = f["normals"][()].astype(np.float32)
        hp  = f["human_prior"][()].astype(np.float32)
        attrs = dict(f.attrs)
    print(f"✅ Loaded {path}")
    print(f"   points={len(pc)}  coverage(>0.1)={float((hp>0.1).mean()):.1%}  max_hp={hp.max():.3f}")
    print(f"   source={attrs.get('source','')}  n_valid_mano={attrs.get('n_valid_mano','?')}")
    return pc, nrm, hp


def render_pointcloud(ax, points, values, title,
                      cmap_name='jet', vmin=0, vmax=1,
                      elev=25, azim=135, binary=False):
    """与 vis_3panel.py 完全一致的渲染函数."""
    cmap = plt.get_cmap(cmap_name)

    if binary:
        colors = np.zeros((len(values), 4))
        colors[values < 0.5]  = [0.75, 0.75, 0.85, 1.0]   # 浅灰蓝 = 无接触
        colors[values >= 0.5] = [0.90, 0.15, 0.15, 1.0]   # 红     = 有接触
    else:
        colors = cmap(np.clip(values, vmin, vmax))

    order = np.argsort(values)   # 低值先画，高值覆盖
    ax.scatter(points[order, 0], points[order, 1], points[order, 2],
               c=colors[order], s=1.5, alpha=0.9, edgecolors='none')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.view_init(elev=elev, azim=azim)

    extents  = points.max(axis=0) - points.min(axis=0)
    max_ext  = extents.max() * 0.6
    center   = (points.max(axis=0) + points.min(axis=0)) / 2
    ax.set_xlim(center[0] - max_ext, center[0] + max_ext)
    ax.set_ylim(center[1] - max_ext, center[1] + max_ext)
    ax.set_zlim(center[2] - max_ext, center[2] + max_ext)
    ax.set_axis_off()


MESH_BASE_EGO = os.path.join(PROJ, "data_hub", "ProcessedData",
                              "obj_meshes", "egocentric")


def make_vis(obj_name, pc, hp, out_path):
    """
    与 vis_3panel.py 风格一致:
      - 白底, jet colormap
      - 左: Human Prior（二值: 灰=无接触, 红=有接触）
      - 右: Human Prior（连续 jet, 显示概率梯度）
    使用 mesh 密集重采样 + KNN 插值（与第三人称 20000-point 做法相同）
    """
    from scipy.spatial import cKDTree

    # ── 尝试从 mesh 密集采样 ────────────────────────────────────────────
    mesh_path = os.path.join(MESH_BASE_EGO, obj_name, "mesh.ply")
    if os.path.exists(mesh_path):
        import trimesh, json
        mesh = trimesh.load(mesh_path, force="mesh")
        # 应用 scale（如果有）
        scale_json = os.path.join(os.path.dirname(mesh_path), "scale.json")
        if os.path.exists(scale_json):
            sf = float(json.load(open(scale_json)).get("scale_factor", 1.0))
            mesh.vertices = mesh.vertices * sf
        N_VIS = 20000
        vis_pc, _ = trimesh.sample.sample_surface(mesh, N_VIS)
        vis_pc = vis_pc.astype(np.float32)
        tree = cKDTree(pc)
        _, idx = tree.query(vis_pc, k=3)
        dists   = np.linalg.norm(vis_pc[:, None, :] - pc[idx], axis=2)
        weights = 1.0 / (dists + 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        hp_dense = (weights * hp[idx]).sum(axis=1)
        print(f"   mesh dense sampling: {N_VIS} pts (KNN from {len(pc)})")
    else:
        vis_pc   = pc
        hp_dense = hp
        print("   (no mesh found, using 4096 pts directly)")

    coverage  = float((hp_dense > 0.1).mean())
    n_contact = int((hp_dense >= 0.5).sum())

    # ── 画图：白底, 与 vis_3panel 一致 ─────────────────────────────────
    fig = plt.figure(figsize=(12, 6), facecolor='white')
    fig.suptitle(
        f"{obj_name}  [EgoDex]   coverage={coverage:.1%}   "
        f"contact pts={n_contact}/{len(vis_pc)}   max_hp={hp_dense.max():.3f}",
        fontsize=14, fontweight='bold', y=0.98
    )

    ax1 = fig.add_subplot(121, projection='3d')
    render_pointcloud(ax1, vis_pc, hp_dense,
                      'Human Prior (binary)', binary=True)

    ax2 = fig.add_subplot(122, projection='3d')
    render_pointcloud(ax2, vis_pc, hp_dense,
                      'Human Prior (continuous)', cmap_name='jet')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   → saved: {out_path}")


def make_histogram(obj_name, hp, out_path):
    """Distribution of contact probability."""
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#1a1a2e")
    ax.set_facecolor("#0d0d1a")
    ax.hist(hp, bins=50, color="#e040fb", edgecolor="none", alpha=0.85)
    ax.axvline(0.1, color="#ffcc00", linestyle="--", lw=1.5, label="threshold 0.1")
    ax.set_xlabel("Contact Probability", color="white", fontsize=11)
    ax.set_ylabel("# Points", color="white", fontsize=11)
    ax.set_title(f"Contact Distribution — {obj_name}", color="white",
                 fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"   → saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="接触图可视化 (与 vis_3panel.py 风格一致)")
    parser.add_argument("--obj", default="assemble_tile",
                        help="human_prior/{obj}.hdf5")
    args = parser.parse_args()

    pc, nrm, hp = load_hp(args.obj)
    if pc is None:
        return

    out_path = os.path.join(OUT, f"{args.obj}_contact_3d.png")
    print("\nRendering...")
    make_vis(args.obj, pc, hp, out_path)
    print(f"\n✅ Done → {out_path}")


if __name__ == "__main__":
    main()
