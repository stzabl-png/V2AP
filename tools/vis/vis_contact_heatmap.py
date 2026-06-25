#!/usr/bin/env python3
"""
vis_contact_heatmap.py — 统一接触热力图可视化工具
==================================================

整合了以下脚本:
  • analysis/vis_contacts.py
  • analysis/vis_human_contacts.py
  • data/vis_merged_heatmap.py
  • tools/vis_ego_contact.py
  • data/vis_contact_regions.py

支持数据集:
  --dataset dexycb   : DexYCB training_fp HDF5
  --dataset ho3d_v3  : HO3D  training_fp HDF5
  --dataset egodex   : EgoDex 第一人称 human_prior HDF5
  --dataset oakink   : OakInk v1
  --dataset grab     : GRAB

渲染模式 (--mode):
  static      : matplotlib PNG，无需 display（默认）
  interactive : Open3D 交互窗口

用法:
    # 第三人称 training_fp
    python tools/vis_contact_heatmap.py --dataset dexycb --obj 003_cracker_box
    python tools/vis_contact_heatmap.py --dataset ho3d_v3 --obj 004_sugar_box
    python tools/vis_contact_heatmap.py --dataset dexycb --batch             # 批量全部

    # 第一人称 EgoDex
    python tools/vis_contact_heatmap.py --dataset egodex --obj assemble_tile
    python tools/vis_contact_heatmap.py --dataset egodex --batch

    # OakInk / GRAB
    python tools/vis_contact_heatmap.py --dataset oakink --obj A16013
    python tools/vis_contact_heatmap.py --dataset grab   --obj mug

    # 对比模式: GT vs HaWoR 预测（仅 oakink/grab）
    python tools/vis_contact_heatmap.py --dataset oakink --obj A16013 --compare

    # 交互模式 (Open3D)
    python tools/vis_contact_heatmap.py --dataset dexycb --obj 003_cracker_box --mode interactive
"""

import os, sys, glob, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError:
    cKDTree = None

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import config

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAINING_FP_DIR  = os.path.join(PROJ, "data_hub", "ProcessedData", "training_fp")
EGO_HP_DIR       = os.path.join(PROJ, "data_hub", "ProcessedData", "human_prior_fp")
YCB_MESH_DIR     = os.path.join(PROJ, "data_hub", "ProcessedData", "obj_meshes", "ycb")
EGO_MESH_DIR     = os.path.join(PROJ, "data_hub", "ProcessedData", "obj_meshes", "egocentric")
EGODEX_SCALED_MESH_DIR = os.path.join(PROJ, "data_hub", "ProcessedData", "Scaled_Mesh", "Egodex")
OAKINK_MESH_DIR  = config.MESH_V1_DIR
GRAB_MESH_DIR    = os.path.join(config.DATA_HUB, "meshes", "grab")
CONTACTS_DIR     = config.CONTACTS_DIR
OUT_BASE         = os.path.join(PROJ, "output", "vis_contact_heatmap")

EGODEX_TRAINING_FP_DIRS = [
    os.path.join(TRAINING_FP_DIR, "egodex"),
    os.path.join(TRAINING_FP_DIR, "EgoDex"),
    os.path.join(PROJ, "data_hub", "human_prior"),
]

EGODEX_MESH_DIRS = [
    EGO_MESH_DIR,
    EGODEX_SCALED_MESH_DIR,
    os.path.join(PROJ, "data_hub", "meshes", "SAM3DMesh", "meshes", "egodex"),
    os.path.join(PROJ, "data_hub", "meshes", "SAM3DMesh", "rotated_mesh", "egodex"),
]

GRAB_MESH_MAP = {
    "mug": "coffeemug", "phone": "phone", "fryingpan": "fryingpan",
    "gamecontroller": "gamecontroller", "stanfordbunny": "stanfordbunny",
    "piggybank": "piggybank", "doorknob": "doorknob",
    "waterbottle": "waterbottle", "lightbulb": "lightbulb",
    "alarmclock": "alarmclock", "eyeglasses": "eyeglasses",
}

N_DENSE = 20000   # dense sampling for mesh-based interpolation


# ══════════════════════════════════════════════════════════════════════════════
# Data Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_training_fp(dataset, obj_name):
    """Load from training_fp/{dataset}/{obj_name}.hdf5 (DexYCB / HO3D)."""
    # First try training_fp subdir, then legacy human_prior_fp
    for base in [TRAINING_FP_DIR, EGO_HP_DIR]:
        for ds_sub in [dataset, ""]:
            path = os.path.join(base, ds_sub, f"{obj_name}.hdf5")
            if not os.path.exists(path):
                path = os.path.join(base, f"{obj_name}.hdf5")
            if os.path.exists(path):
                with h5py.File(path, "r") as f:
                    pc  = f["point_cloud"][()].astype(np.float32)
                    nrm = f["normals"][()].astype(np.float32)
                    hp  = f["human_prior"][()].astype(np.float32)
                    fc  = f["force_center"][()].astype(np.float32) if "force_center" in f else None
                print(f"✅ Loaded {path}")
                return pc, nrm, hp, fc
    return None, None, None, None


def load_egodex_hp(obj_name):
    """Load EgoDex HDF5 from HF training_fp/EgoDex or legacy local folders."""
    for base in EGODEX_TRAINING_FP_DIRS:
        path = os.path.join(base, f"{obj_name}.hdf5")
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                pc  = f["point_cloud"][()].astype(np.float32)
                nrm = f["normals"][()].astype(np.float32)
                hp  = f["human_prior"][()].astype(np.float32)
                fc  = f["force_center"][()].astype(np.float32) if "force_center" in f else None
            print(f"✅ Loaded {path}")
            return pc, nrm, hp, fc
    return None, None, None, None


def load_oakink_grab_contacts(obj_id, source, max_frames=200):
    """Load raw finger-contact points from OakInk v1 / GRAB HDF5 files."""
    if source == "grab":
        pattern = os.path.join(os.path.dirname(CONTACTS_DIR),
                               "contacts_grab", f"{obj_id}/*/*.hdf5")
    else:
        pattern = os.path.join(CONTACTS_DIR, f"**/{obj_id}_*/*.hdf5")

    files = sorted(glob.glob(pattern, recursive=True))[:max_frames]
    pts = []
    for fpath in files:
        try:
            with h5py.File(fpath, "r") as h:
                for key in ("finger_contact_pts", "finger_contact_points"):
                    if key in h:
                        pts.append(h[key][:])
                        break
        except Exception:
            continue
    if pts:
        return np.vstack(pts)
    return None


def find_mesh(obj_id, dataset):
    """Find mesh file for any dataset."""
    if dataset in ("dexycb", "ho3d_v3"):
        # YCB meshes: data_hub/ProcessedData/obj_meshes/ycb/{obj_id}/mesh.ply
        for ext in ("mesh.ply", "mesh.obj"):
            p = os.path.join(YCB_MESH_DIR, obj_id, ext)
            if os.path.exists(p):
                return p
        # fallback: flat ycb dir
        for ext in (".ply", ".obj"):
            p = os.path.join(YCB_MESH_DIR, f"{obj_id}{ext}")
            if os.path.exists(p):
                return p
        return None
    elif dataset == "egodex":
        for root in EGODEX_MESH_DIRS:
            for candidate in (
                os.path.join(root, obj_id, "mesh.ply"),
                os.path.join(root, obj_id, "mesh.obj"),
                os.path.join(root, obj_id, f"{obj_id}.ply"),
                os.path.join(root, obj_id, f"{obj_id}.obj"),
                os.path.join(root, f"{obj_id}.ply"),
                os.path.join(root, f"{obj_id}.obj"),
            ):
                if os.path.exists(candidate):
                    return candidate
        return None
    elif dataset == "oakink":
        for ext in (".obj", ".ply"):
            p = os.path.join(OAKINK_MESH_DIR, f"{obj_id}{ext}")
            if os.path.exists(p):
                return p
        return None
    elif dataset == "grab":
        for name in [obj_id, GRAB_MESH_MAP.get(obj_id, obj_id)]:
            for ext in (".ply", ".obj"):
                p = os.path.join(GRAB_MESH_DIR, f"{name}{ext}")
                if os.path.exists(p):
                    return p
        # fuzzy
        if os.path.exists(GRAB_MESH_DIR):
            for f in os.listdir(GRAB_MESH_DIR):
                if obj_id.replace("_", "").lower() in f.replace("_", "").lower():
                    return os.path.join(GRAB_MESH_DIR, f)
    return None


def dense_from_mesh(mesh_path, pc, hp, n_dense=N_DENSE):
    """Sample dense points from mesh and KNN-interpolate hp values."""
    if cKDTree is None:
        raise ImportError("scipy is required for mesh dense interpolation")
    vis_pc = sample_mesh_surface(mesh_path, n_dense)
    _, idx  = cKDTree(pc).query(vis_pc, k=3)
    dists   = np.linalg.norm(vis_pc[:, None, :] - pc[idx], axis=2)
    weights = 1.0 / (dists + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    hp_dense = (weights * hp[idx]).sum(axis=1)
    return vis_pc, hp_dense


def sample_mesh_surface(mesh_path, n_dense=N_DENSE):
    """Sample points from a mesh after applying colocated scale.json, if present."""
    import json
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    scale_json = os.path.join(os.path.dirname(mesh_path), "scale.json")
    if os.path.exists(scale_json):
        sf = float(json.load(open(scale_json)).get("scale_factor", 1.0))
        mesh.vertices = mesh.vertices * sf
    vis_pc, _ = trimesh.sample.sample_surface(mesh, n_dense)
    return vis_pc.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Rendering Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _set_ax_equal(ax, pts, scale=0.6):
    extents = pts.max(0) - pts.min(0)
    r = extents.max() * scale
    c = (pts.max(0) + pts.min(0)) / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_xlabel("X (m)", fontsize=8, labelpad=2)
    ax.set_ylabel("Y (m)", fontsize=8, labelpad=2)
    ax.set_zlabel("Z (m)", fontsize=8, labelpad=2)
    ax.tick_params(axis="both", which="major", labelsize=7, pad=0)
    ax.grid(True, linewidth=0.45, alpha=0.45)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor((0.96, 0.96, 0.96, 0.22))
        axis.pane.set_edgecolor((0.82, 0.82, 0.82, 0.75))


def draw_xyz_frame(ax, pts):
    """Draw a colored XYZ frame from the true object-frame origin."""
    mn, mx = pts.min(0), pts.max(0)
    span = mx - mn
    axis_len = float(span.max()) * 0.35
    origin = np.zeros(3, dtype=np.float32)
    axes = [
        ((1, 0, 0), "#d62728", "X"),
        ((0, 1, 0), "#2ca02c", "Y"),
        ((0, 0, 1), "#1f77b4", "Z"),
    ]
    ax.scatter(origin[0], origin[1], origin[2],
               color="black", s=36, marker="o", depthshade=False, zorder=10)
    ax.text(origin[0], origin[1], origin[2], " O",
            color="black", fontsize=9, fontweight="bold", zorder=11)
    for direction, color, label in axes:
        d = np.asarray(direction, dtype=np.float32)
        ax.quiver(
            origin[0], origin[1], origin[2],
            d[0], d[1], d[2],
            length=axis_len,
            color=color,
            linewidth=2.0,
            arrow_length_ratio=0.18,
            normalize=False,
        )
        end = origin + d * axis_len * 1.08
        ax.text(end[0], end[1], end[2], label, color=color,
                fontsize=10, fontweight="bold")


def render_pc(ax, pts, vals, title="", cmap="jet", vmin=0, vmax=None,
              binary=False, binary_thr=None, elev=25, azim=135, fc=None):
    """Render a point cloud with colormap on a 3D axis.

    binary_thr: threshold for binary coloring (default: 50% of val max)
    vmax:       jet colormap upper bound (default: actual val max)
    """
    order = np.argsort(vals)
    p, v = pts[order], vals[order]

    val_max = float(v.max()) if v.max() > 1e-8 else 1.0
    if vmax is None:
        vmax = val_max          # ← 修复3: 按实际最大值归一化
    if binary_thr is None:
        binary_thr = val_max * 0.5  # ← 修复1: 阈值 = 50% of max

    if binary:
        colors = np.zeros((len(v), 4))
        colors[v < binary_thr]  = [0.75, 0.75, 0.85, 1.0]  # 浅蓝=无接触
        colors[v >= binary_thr] = [0.90, 0.15, 0.15, 1.0]  # 红=有接触
    else:
        colors = plt.get_cmap(cmap)(np.clip((v - vmin) / (vmax - vmin + 1e-8), 0, 1))

    ax.scatter(p[:, 0], p[:, 1], p[:, 2],
               c=colors, s=3, alpha=0.9, edgecolors="none")  # s=3 更清晰
    if fc is not None:
        ax.scatter(*fc, c="lime", s=120, marker="*",
                   edgecolors="darkgreen", linewidth=1.2, zorder=10,
                   label="force center")
        ax.legend(fontsize=7)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.view_init(elev=elev, azim=azim)
    _set_ax_equal(ax, p)


def render_mesh_only(ax, pts, title="", elev=25, azim=135, show_xyz_frame=False):
    """Render mesh surface samples without HP coloring."""
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               color=(0.72, 0.74, 0.78, 1.0), s=2.5, alpha=0.92, edgecolors="none")
    if show_xyz_frame:
        draw_xyz_frame(ax, pts)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.view_init(elev=elev, azim=azim)
    _set_ax_equal(ax, pts)


def render_overlay(ax, mesh_pts, hp_pts, hp_vals, title="", elev=25, azim=135):
    """Render gray mesh surface with continuous HP points overlaid."""
    hp_max = float(hp_vals.max()) if hp_vals.max() > 1e-8 else 1.0
    order = np.argsort(hp_vals)
    hp_pts = hp_pts[order]
    hp_vals = hp_vals[order]
    colors = plt.get_cmap("jet")(np.clip(hp_vals / (hp_max + 1e-8), 0, 1))

    ax.scatter(mesh_pts[:, 0], mesh_pts[:, 1], mesh_pts[:, 2],
               color=(0.72, 0.74, 0.78, 0.22), s=2.0, alpha=0.22, edgecolors="none")
    ax.scatter(hp_pts[:, 0], hp_pts[:, 1], hp_pts[:, 2],
               c=colors, s=3.2, alpha=0.95, edgecolors="none")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.view_init(elev=elev, azim=azim)
    _set_ax_equal(ax, np.vstack([mesh_pts, hp_pts]))


def make_stats_title(obj_name, hp, dataset):
    cov = float((hp > 0.1).mean()) * 100
    cov_01 = float((hp > 0.01).mean()) * 100
    return (f"{obj_name}  [{dataset}]   "
            f"cov(>0.1)={cov:.0f}%  cov(>0.01)={cov_01:.0f}%  "
            f"max={hp.max():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Static (matplotlib PNG) rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_static(obj_name, dataset, pc, hp, fc, out_path,
                  mesh_path=None, contact_pts=None, compare=False, show_binary=False):
    """Generate a static PNG.

    Default: single jet heatmap panel.
    show_binary=True: add binary panel on the left.
    compare=True:     add GT contact panel on the right.
    """
    mesh_pts = None
    mesh_note = "mesh missing; showing HDF5 point_cloud"
    if mesh_path and os.path.exists(mesh_path):
        try:
            mesh_pts = sample_mesh_surface(mesh_path, N_DENSE)
            mesh_note = f"mesh {N_DENSE:,}pts"
        except ImportError as e:
            print(f"  ⚠️  {e}; using raw {len(pc)} HP pts")
        except ModuleNotFoundError as e:
            print(f"  ⚠️  {e}; using raw {len(pc)} HP pts")

    if mesh_pts is not None:
        if cKDTree is None:
            vis_pc, hp_dense = pc, hp
            dense_label = f"raw {len(pc)}pts — scipy missing"
            print(f"  ⚠️  scipy is required for mesh HP interpolation; using raw {len(pc)} HP pts")
        else:
            vis_pc = mesh_pts
            _, idx = cKDTree(pc).query(vis_pc, k=3)
            dists = np.linalg.norm(vis_pc[:, None, :] - pc[idx], axis=2)
            weights = 1.0 / (dists + 1e-8)
            weights /= weights.sum(axis=1, keepdims=True)
            hp_dense = (weights * hp[idx]).sum(axis=1)
            dense_label = f"mesh {N_DENSE:,}pts"
    else:
        mesh_pts = pc
        vis_pc, hp_dense = pc, hp
        dense_label = f"raw {len(pc)}pts — mesh not found"
        print(f"  ⚠️  Mesh not found for {obj_name}, using raw {len(pc)} pts")

    hp_max = float(hp_dense.max()) if hp_dense.max() > 1e-8 else 1.0

    # With --binary, produce the requested 4-panel diagnostic:
    # 1 mesh, 2 HP binary, 3 HP continuous, 4 mesh + continuous HP overlay.
    n_panels = 4 if show_binary else 1
    if not show_binary and compare and contact_pts is not None:
        n_panels += 1

    fig = plt.figure(figsize=(7 * n_panels, 7), facecolor="white")
    fig.suptitle(make_stats_title(obj_name, hp_dense, dataset),
                 fontsize=12, fontweight="bold", y=0.98)

    col = 1
    if show_binary:
        ax = fig.add_subplot(1, n_panels, col, projection="3d"); col += 1
        render_mesh_only(ax, mesh_pts, f"1. Mesh ({mesh_note})", show_xyz_frame=True)

        ax = fig.add_subplot(1, n_panels, col, projection="3d"); col += 1
        render_pc(ax, vis_pc, hp_dense,
                  f"2. HP binary (thr={hp_max*0.5:.3f})",
                  binary=True, fc=fc)

    ax = fig.add_subplot(1, n_panels, col, projection="3d"); col += 1
    render_pc(ax, vis_pc, hp_dense,
              f"{'3. ' if show_binary else ''}HP continuous ({dense_label})",
              cmap="jet", vmax=hp_max, fc=fc)

    if show_binary:
        ax = fig.add_subplot(1, n_panels, col, projection="3d"); col += 1
        render_overlay(ax, mesh_pts, vis_pc, hp_dense,
                       f"4. Mesh overlay HP continuous")

    if (not show_binary) and compare and contact_pts is not None:
        if cKDTree is None:
            print("  ⚠️  scipy is required for compare mode; skipping GT contact panel")
            contact_pts = None
        else:
            dists, _ = cKDTree(contact_pts).query(vis_pc)
            gt_on_mesh = (dists < 0.012).astype(np.float32)
            ax = fig.add_subplot(1, n_panels, col, projection="3d")
            render_pc(ax, vis_pc, gt_on_mesh, "GT Contact (raw finger pts)", binary=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  → saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Interactive (Open3D) rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_interactive(obj_name, dataset, pc, hp, fc, mesh_path=None):
    """Open3D interactive viewer: mesh colored by human_prior heatmap."""
    import open3d as o3d

    # Prefer mesh
    if mesh_path and os.path.exists(mesh_path):
        try:
            vis_pc, hp_dense = dense_from_mesh(mesh_path, pc, hp)
        except ImportError as e:
            print(f"  ⚠️  {e}; using raw {len(pc)} HP pts")
            vis_pc, hp_dense = pc, hp
    else:
        vis_pc, hp_dense = pc, hp

    colors_rgba = plt.get_cmap("RdYlBu_r")(np.clip(hp_dense, 0, hp_dense.max() + 1e-6))[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(vis_pc)
    pcd.colors  = o3d.utility.Vector3dVector(colors_rgba)
    geoms = [pcd]

    if fc is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        sphere.translate(fc)
        sphere.paint_uniform_color([0.0, 1.0, 0.3])
        geoms.append(sphere)

    cov = float((hp_dense > 0.1).mean()) * 100
    print(f"\n🖱️  左键=旋转  滚轮=缩放  Shift+左键=平移  Q=退出")
    print(f"  coverage(>0.1)={cov:.1f}%  max={hp_dense.max():.3f}")
    print(f"  🔴 红=高接触  🔵 蓝=低接触  ⭐=force_center\n")

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Contact Heatmap — {obj_name} [{dataset}]  cov={cov:.0f}%",
        width=1200, height=800,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Per-dataset dispatch
# ══════════════════════════════════════════════════════════════════════════════

def process_obj(dataset, obj_name, mode, compare, out_dir, show_binary=False):
    """Main dispatch: load data and render."""
    pc = hp = fc = None

    if dataset in ("dexycb", "ho3d_v3"):
        pc, nrm, hp, fc = load_training_fp(dataset, obj_name)
    elif dataset == "egodex":
        pc, nrm, hp, fc = load_egodex_hp(obj_name)
    elif dataset in ("oakink", "grab"):
        contact_pts = load_oakink_grab_contacts(obj_name, dataset)
        if contact_pts is None:
            print(f"❌ No contact data for {obj_name} [{dataset}]")
            return False
        # Build pseudo-hp from contact density on mesh points
        mesh_path = find_mesh(obj_name, dataset)
        if mesh_path is None:
            print(f"❌ Mesh not found for {obj_name}")
            return False
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        pc, _ = trimesh.sample.sample_surface(mesh, 4096)
        pc = pc.astype(np.float32)
        if cKDTree is None:
            print("❌ scipy is required for oakink/grab contact density")
            return False
        dists, _ = cKDTree(contact_pts).query(pc)
        hp = np.exp(-dists / 0.01).astype(np.float32)
        hp = hp / (hp.max() + 1e-8)
        fc = None
    else:
        print(f"❌ Unknown dataset: {dataset}")
        return False

    if pc is None:
        print(f"❌ No data loaded for {obj_name} [{dataset}]")
        return False

    print(f"  pc={pc.shape}  hp: cov(>0.1)={float((hp>0.1).mean()):.1%}  "
          f"cov(>0.01)={float((hp>0.01).mean()):.1%}  max={hp.max():.3f}")

    mesh_path = find_mesh(obj_name, dataset)

    contact_pts = None
    if compare and dataset in ("oakink", "grab"):
        contact_pts = load_oakink_grab_contacts(obj_name, dataset)

    if mode == "interactive":
        render_interactive(obj_name, dataset, pc, hp, fc, mesh_path)
    else:
        out_path = os.path.join(out_dir, dataset, f"{obj_name}.png")
        render_static(obj_name, dataset, pc, hp, fc, out_path,
                      mesh_path=mesh_path,
                      contact_pts=contact_pts, compare=compare,
                      show_binary=show_binary)
    return True


def list_objects(dataset):
    """List available objects for the given dataset."""
    if dataset in ("dexycb", "ho3d_v3"):
        folder = os.path.join(TRAINING_FP_DIR, dataset)
        if not os.path.isdir(folder):
            folder = EGO_HP_DIR
        return [os.path.splitext(f)[0]
                for f in os.listdir(folder) if f.endswith(".hdf5")]
    elif dataset == "egodex":
        for folder in EGODEX_TRAINING_FP_DIRS:
            if os.path.isdir(folder):
                return [os.path.splitext(f)[0]
                        for f in os.listdir(folder) if f.endswith(".hdf5")]
        return []
    elif dataset == "oakink":
        return [os.path.splitext(f)[0]
                for f in os.listdir(OAKINK_MESH_DIR)
                if f.endswith((".obj", ".ply"))]
    elif dataset == "grab":
        return list(GRAB_MESH_MAP.keys())
    return []


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="统一接触热力图可视化 (vis_contact_heatmap.py)")
    p.add_argument("--dataset", required=True,
                   choices=["dexycb", "ho3d_v3", "egodex", "oakink", "grab"],
                   help="数据集名称")
    p.add_argument("--obj",  default=None,
                   help="物体名（不含 .hdf5）")
    p.add_argument("--batch", action="store_true",
                   help="批量处理该数据集全部物体")
    p.add_argument("--mode", default="static",
                   choices=["static", "interactive"],
                   help="渲染模式 (static=PNG | interactive=Open3D)")
    p.add_argument("--compare", action="store_true",
                   help="同时对比 GT finger contacts（仅 oakink/grab）")
    p.add_argument("--binary", action="store_true",
                   help="同时显示 binary 面板（默认关闭）")
    p.add_argument("--out", default=OUT_BASE,
                   help=f"输出目录 (default: {OUT_BASE})")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.batch:
        objs = list_objects(args.dataset)
        if not objs:
            print(f"❌ 找不到 {args.dataset} 的物体列表")
            return
        print(f"批量处理 {len(objs)} 个物体 [{args.dataset}]")
        ok = 0
        for obj in sorted(objs):
            print(f"\n  [{obj}]")
            if process_obj(args.dataset, obj, args.mode, args.compare, args.out,
                           show_binary=args.binary):
                ok += 1
        print(f"\n✅ {ok}/{len(objs)} 完成 → {args.out}/{args.dataset}/")
    elif args.obj:
        process_obj(args.dataset, args.obj, args.mode, args.compare, args.out,
                    show_binary=args.binary)
    else:
        objs = list_objects(args.dataset)
        print(f"可用物体 [{args.dataset}]: {objs}")
        print("请指定 --obj <名称> 或 --batch")


if __name__ == "__main__":
    main()
