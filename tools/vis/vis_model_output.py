#!/usr/bin/env python3
"""
vis_model_output.py — 统一模型预测可视化工具
=============================================

整合了以下脚本:
  • analysis/vis_model_prediction.py
  • analysis/vis_m5_result.py
  • tools/vis_m5_predict.py

用法:
    # 单物体预测（自动找最新 checkpoint）
    python tools/vis_model_output.py --obj 003_cracker_box
    python tools/vis_model_output.py --obj A16013

    # 指定 checkpoint
    python tools/vis_model_output.py --obj 004_sugar_box --ckpt output/checkpoints/best.pth

    # 对比 GT（左=GT Human Prior, 中=Robot GT, 右=模型预测）
    python tools/vis_model_output.py --obj 003_cracker_box --compare

    # 批量生成所有物体
    python tools/vis_model_output.py --batch --out output/vis_model

    # 交互模式 (Open3D，需要 display)
    python tools/vis_model_output.py --obj A16013 --mode interactive

    # 指定数据集（默认 dexycb）
    python tools/vis_model_output.py --obj 003_cracker_box --dataset ho3d_v3
"""

import os, sys, glob, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "model"))
import config

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAINING_FP_DIR = os.path.join(PROJ, "data_hub", "ProcessedData", "training_fp")
MESH_V1_DIR     = config.MESH_V1_DIR
GRAB_MESH_DIR   = os.path.join(config.DATA_HUB, "meshes", "grab")
OUT_BASE        = os.path.join(PROJ, "output", "vis_model_output")

CKPT_SEARCH = [
    os.path.join(PROJ, "output", "checkpoints_m5", "best_m5_model.pth"),
    os.path.join(PROJ, "output", "checkpoints",    "best_model.pth"),
    os.path.join(PROJ, "output", "checkpoints_v1v2", "best_model.pth"),
]
N_DENSE = 20000
N_POINTS = 4096


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def find_ckpt(user_ckpt=None):
    if user_ckpt and os.path.exists(user_ckpt):
        return user_ckpt
    for c in CKPT_SEARCH:
        if os.path.exists(c):
            return c
    return None


def load_training_data(dataset, obj_name):
    """Load HDF5 from training_fp/{dataset}/{obj_name}.hdf5."""
    for base_ds in [dataset, ""]:
        path = os.path.join(TRAINING_FP_DIR, base_ds, f"{obj_name}.hdf5")
        if not os.path.exists(path):
            path = os.path.join(TRAINING_FP_DIR, f"{obj_name}.hdf5")
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                return {
                    "path": path,
                    "pc":    f["point_cloud"][()].astype(np.float32),
                    "nrm":   f["normals"][()].astype(np.float32),
                    "hp":    f["human_prior"][()].astype(np.float32),
                    "rgt":   f["robot_gt"][()].astype(np.float32) if "robot_gt" in f else None,
                    "fc":    f["force_center"][()].astype(np.float32) if "force_center" in f else None,
                }
    return None


def find_mesh(obj_name):
    """Find mesh file — search v1, GRAB, YCB."""
    for d, exts in [
        (MESH_V1_DIR, (".obj", ".ply")),
        (GRAB_MESH_DIR, (".ply", ".obj")),
    ]:
        for ext in exts:
            p = os.path.join(d, f"{obj_name}{ext}")
            if os.path.exists(p):
                return p
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Model inference
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path, device):
    import torch
    from pointnet2 import PointNet2Seg

    model = PointNet2Seg(num_classes=1, in_channel=7).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val   = ckpt.get("val_loss", float("nan"))
    print(f"✅ Model: epoch={epoch}  val_loss={val:.4f}  [{os.path.basename(ckpt_path)}]")
    return model


def predict(model, data, device):
    import torch
    pc, nrm, hp = data["pc"], data["nrm"], data["hp"]
    feats = np.concatenate([pc, nrm, hp[:, None]], axis=-1)
    pts_t = torch.from_numpy(pc).unsqueeze(0).to(device)
    ft_t  = torch.from_numpy(feats).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(pts_t, ft_t).squeeze(0).cpu().numpy()
    return pred.ravel()


# ══════════════════════════════════════════════════════════════════════════════
# Rendering
# ══════════════════════════════════════════════════════════════════════════════

def _set_ax(ax, pts, scale=0.6):
    r = (pts.max(0) - pts.min(0)).max() * scale
    c = (pts.max(0) + pts.min(0)) / 2
    ax.set_xlim(c[0]-r, c[0]+r)
    ax.set_ylim(c[1]-r, c[1]+r)
    ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_axis_off()


def render_pc_panel(ax, pts, vals, title, cmap="jet", binary=False,
                    elev=25, azim=135, fc=None):
    order = np.argsort(vals)
    p, v  = pts[order], vals[order]
    if binary:
        colors = np.where(v[:, None] >= 0.5,
                          [[0.9, 0.15, 0.15, 1.0]], [[0.75, 0.75, 0.85, 1.0]])
    else:
        colors = plt.get_cmap(cmap)(np.clip(v, 0, 1))
    ax.scatter(p[:,0], p[:,1], p[:,2], c=colors, s=1.5, alpha=0.9, edgecolors="none")
    if fc is not None:
        ax.scatter(*fc, c="lime", s=100, marker="*",
                   edgecolors="darkgreen", linewidth=1, zorder=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.view_init(elev=elev, azim=azim)
    _set_ax(ax, p)


def dense_interpolate(mesh_path, pc, *val_arrays):
    """Dense sample mesh, KNN-interpolate multiple value arrays."""
    import trimesh
    mesh = trimesh.load(mesh_path, force="mesh")
    vis_pc, _ = trimesh.sample.sample_surface(mesh, N_DENSE)
    vis_pc = vis_pc.astype(np.float32)
    _, idx  = cKDTree(pc).query(vis_pc, k=3)
    dists   = np.linalg.norm(vis_pc[:,None,:] - pc[idx], axis=2)
    w = 1.0 / (dists + 1e-8); w /= w.sum(1, keepdims=True)
    results = [vis_pc]
    for arr in val_arrays:
        results.append((w * arr[idx]).sum(1))
    return results


def render_static(obj_name, dataset, data, pred, out_path, compare=True):
    """Generate 2-panel (hp + pred) or 3-panel (hp + rgt + pred) PNG."""
    pc  = data["pc"]
    hp  = data["hp"]
    rgt = data.get("rgt")
    fc  = data.get("fc")
    has_rgt = compare and rgt is not None and rgt.max() > 0.01

    mesh_path = find_mesh(obj_name)
    if mesh_path:
        all_vals = [hp, pred] + ([rgt] if has_rgt else [])
        res = dense_interpolate(mesh_path, pc, *all_vals)
        vis_pc, hp_d, pred_d = res[0], res[1], res[2]
        rgt_d = res[3] if has_rgt else None
        src = f"mesh {N_DENSE:,}pts"
    else:
        vis_pc, hp_d, pred_d = pc, hp, pred
        rgt_d = rgt if has_rgt else None
        src = f"{len(pc)} pts"

    n_panels = 3 if has_rgt else 2
    fig = plt.figure(figsize=(7*n_panels, 7), facecolor="white")
    fig.suptitle(
        f"{obj_name}  [{dataset}]   "
        f"pred_max={pred_d.max():.3f}  pred_cov={float((pred_d>0.3).mean()):.1%}  ({src})",
        fontsize=12, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(1, n_panels, 1, projection="3d")
    render_pc_panel(ax1, vis_pc, hp_d, "Human Prior (binary)", binary=True, fc=fc)

    if has_rgt and rgt_d is not None:
        ax2 = fig.add_subplot(1, n_panels, 2, projection="3d")
        render_pc_panel(ax2, vis_pc, rgt_d, "Robot GT", fc=fc)
        ax3 = fig.add_subplot(1, n_panels, 3, projection="3d")
        render_pc_panel(ax3, vis_pc, pred_d,
                        f"Model Prediction\nmax={pred_d.max():.3f}", fc=fc)
    else:
        ax2 = fig.add_subplot(1, n_panels, 2, projection="3d")
        render_pc_panel(ax2, vis_pc, pred_d,
                        f"Model Prediction\nmax={pred_d.max():.3f}", fc=fc)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  → saved: {out_path}")


def render_interactive(obj_name, data, pred):
    """Open3D interactive: side-by-side mesh with hp heatmap (left) and pred heatmap (right)."""
    import open3d as o3d
    pc, hp, fc = data["pc"], data["hp"], data.get("fc")

    def pc_to_pcd(pts, vals, cmap_name="RdYlBu_r", offset=None):
        cols = plt.get_cmap(cmap_name)(np.clip(vals, 0, 1))[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts + (offset or 0))
        pcd.colors = o3d.utility.Vector3dVector(cols)
        return pcd

    span = (pc.max(0) - pc.min(0))[0] * 1.5
    geoms = [
        pc_to_pcd(pc, hp, "RdYlBu_r"),
        pc_to_pcd(pc, pred, "jet", offset=np.array([span, 0, 0])),
    ]
    if fc is not None:
        for offset in [np.zeros(3), np.array([span, 0, 0])]:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
            s.translate(fc + offset); s.paint_uniform_color([0,1,0.3])
            geoms.append(s)

    print(f"\n🖱️  左=Human Prior  右=Model Prediction")
    print(f"  pred_max={pred.max():.3f}  pred_cov={float((pred>0.3).mean()):.1%}")
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Model Output — {obj_name}  | Left=HP  Right=Pred",
        width=1400, height=800,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def process_obj(obj_name, dataset, model, device, mode, compare, out_dir):
    data = load_training_data(dataset, obj_name)
    if data is None:
        print(f"  ⚠️ 无训练数据: {obj_name} [{dataset}]")
        return False

    pred = predict(model, data, device)

    print(f"  hp_cov={float((data['hp']>0.1).mean()):.1%}  "
          f"pred_max={pred.max():.3f}  pred_cov={float((pred>0.3).mean()):.1%}")

    if mode == "interactive":
        render_interactive(obj_name, data, pred)
    else:
        out_path = os.path.join(out_dir, dataset, f"{obj_name}.png")
        render_static(obj_name, dataset, data, pred, out_path, compare=compare)
    return True


def main():
    p = argparse.ArgumentParser(description="统一模型预测可视化 (vis_model_output.py)")
    p.add_argument("--obj",     default=None, help="物体名（不含 .hdf5）")
    p.add_argument("--dataset", default="dexycb",
                   choices=["dexycb", "ho3d_v3", "egodex"],
                   help="数据集")
    p.add_argument("--ckpt",    default=None,  help="模型 checkpoint 路径")
    p.add_argument("--batch",   action="store_true", help="批量处理全部物体")
    p.add_argument("--compare", action="store_true",
                   help="3列显示: HP | Robot GT | Prediction")
    p.add_argument("--mode",    default="static",
                   choices=["static", "interactive"],
                   help="渲染模式")
    p.add_argument("--out",     default=OUT_BASE, help="输出目录")
    args = p.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = find_ckpt(args.ckpt)
    if ckpt is None:
        print("❌ 找不到模型 checkpoint，请用 --ckpt 指定路径")
        return
    model = load_model(ckpt, device)

    os.makedirs(args.out, exist_ok=True)

    if args.batch:
        folder = os.path.join(TRAINING_FP_DIR, args.dataset)
        if not os.path.isdir(folder):
            folder = TRAINING_FP_DIR
        objs = sorted(os.path.splitext(f)[0]
                      for f in os.listdir(folder) if f.endswith(".hdf5"))
        print(f"批量处理 {len(objs)} 个物体 [{args.dataset}]")
        ok = 0
        for obj in objs:
            print(f"\n  [{obj}]")
            if process_obj(obj, args.dataset, model, device,
                           args.mode, args.compare, args.out):
                ok += 1
        print(f"\n✅ {ok}/{len(objs)} 完成 → {args.out}/{args.dataset}/")
    elif args.obj:
        process_obj(args.obj, args.dataset, model, device,
                    args.mode, args.compare, args.out)
    else:
        print("请指定 --obj <名称> 或 --batch")


if __name__ == "__main__":
    main()
