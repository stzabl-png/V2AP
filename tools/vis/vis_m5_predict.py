#!/usr/bin/env python3
"""
用 M5 模型对物体做 affordance 预测, 可视化.

用法:
    # Run from project root
    # 交互模式 (Open3D)
    python3 tools/vis_m5_predict.py --obj A01001

    # 保存 PNG (无需 display)
    python3 tools/vis_m5_predict.py --obj A01001 --save output/vis_m5/A01001.png
    python3 tools/vis_m5_predict.py --obj A16013  --save output/vis_m5/A16013.png
"""
import os, sys, argparse
import numpy as np
import trimesh
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))
from pointnet2 import PointNet2Seg

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'v1')
PROC_MESH_DIR = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
HP_DIR = os.path.join(PROJ, 'data_hub', 'human_prior')
HP_INFER_DIR = os.path.join(PROJ, 'data_hub', 'human_prior_infer')
CKPT_DEFAULT = os.path.join(PROJ, 'output', 'checkpoints_m5', 'best_m5_model.pth')
N_POINTS = 4096


def find_mesh(obj_id):
    """先找 .obj，再找 ProcessedData .ply。"""
    p = os.path.join(MESH_DIR, f'{obj_id}.obj')
    if os.path.exists(p): return p
    for ds in ['oakink', 'dexycb', 'egodex']:
        p = os.path.join(PROC_MESH_DIR, ds, obj_id, 'mesh.ply')
        if os.path.exists(p): return p
    return None


def load_human_prior(obj_id):
    # 先找 infer 第，再找旧版
    for path in [
        os.path.join(HP_INFER_DIR, f'oakink_{obj_id}.hdf5'),
        os.path.join(HP_DIR, f'{obj_id}.hdf5'),
        os.path.join(HP_DIR, f'grab_{obj_id}.hdf5'),
    ]:
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                return f['point_cloud'][()].astype(np.float32), \
                       f['normals'][()].astype(np.float32), \
                       f['human_prior'][()].astype(np.float32)
    return None, None, None


def predict(obj_id):
    import open3d as o3d
    from scipy.spatial import cKDTree

    mesh_path = find_mesh(obj_id)
    if not mesh_path:
        print(f"❌ mesh 不存在: {obj_id}"); return

    mesh = trimesh.load(mesh_path, force='mesh')

    # 采样 4096 点用于模型推理
    hp_pc, hp_nrm, hp_labels = load_human_prior(obj_id)
    if hp_pc is not None:
        pc, normals, hp = hp_pc, hp_nrm, hp_labels
        if len(pc) != N_POINTS:
            idx = np.random.choice(len(pc), N_POINTS, replace=len(pc) < N_POINTS)
            pc, normals, hp = pc[idx], normals[idx], hp[idx]
        print(f"📂 使用 Human Prior 点云")
    else:
        pc, face_idx = trimesh.sample.sample_surface(mesh, N_POINTS)
        pc = pc.astype(np.float32)
        normals = mesh.face_normals[face_idx].astype(np.float32)
        hp = np.zeros(N_POINTS, dtype=np.float32)
        print(f"📂 从 mesh 采样点云")

    # 加载模型 (v5 multi-task, 6ch: xyz+normals)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet2Seg(num_classes=2, in_channel=6, predict_force_center=True).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    epoch = ckpt.get('epoch', '?')
    f1    = ckpt.get('val_f1', ckpt.get('val_loss', '?'))
    print(f"✅ 模型加载: epoch={epoch}, val_f1={f1}")

    # 推理 (4096 点)
    pts_t = torch.from_numpy(pc).unsqueeze(0).to(device)
    features = np.concatenate([pc, normals], axis=-1)  # 6ch: xyz+normals
    feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_pred, fc_pred = model(pts_t, feat_t)   # v5 returns tuple
        # seg_pred: (1, N, 2) → softmax → class-1 prob
        probs = torch.softmax(seg_pred[0], dim=-1)[:, 1].cpu().numpy()  # (N,)
    pred_sparse = probs

    # 密集采样 30K 点用于可视化
    N_DENSE = 30000
    dense_pc, dense_fidx = trimesh.sample.sample_surface(mesh, N_DENSE)
    dense_pc = dense_pc.astype(np.float32)

    # 用 KNN 把 4096 点的预测插值到 30K 点
    tree = cKDTree(pc)
    dists, idx = tree.query(dense_pc, k=3)
    weights = 1.0 / (dists + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    pred = np.sum(pred_sparse[idx] * weights, axis=1)

    print(f"\n📊 预测统计 ({N_DENSE} 点):")
    print(f"   min={pred.min():.3f}  max={pred.max():.3f}  mean={pred.mean():.3f}")
    print(f"   > 0.3: {(pred > 0.3).sum()} 点")
    print(f"   > 0.5: {(pred > 0.5).sum()} 点")

    if args.save:
        # ── matplotlib 多视角 PNG（无需 display）──────────────────────
        _save_matplotlib(obj_id, dense_pc, pred, args.save)
    else:
        # ── Open3D 交互窗口 ───────────────────────────────────────────
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dense_pc)
        cmap = mcm.get_cmap('jet')
        colors = cmap(pred)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"M5 — {obj_id} (max={pred.max():.2f})", width=1200, height=800)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = 4.0
        opt.background_color = np.array([1, 1, 1])
        vis.run()
        vis.destroy_window()


def _save_matplotlib(obj_id, pts, vals, out_path):
    """Render 4-view matplotlib scatter → PNG."""
    from scipy.spatial import cKDTree
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    cmap = mcm.get_cmap('jet')
    order = np.argsort(vals)          # low → high, high rendered on top
    p, v = pts[order], vals[order]
    colors = cmap(np.clip(v, 0, 1))   # RGBA

    # 4 viewing angles
    views = [
        ("Front",    20, -60),
        ("Right",    20,  30),
        ("Top-down", 80, -60),
        ("Back",     20, 120),
    ]

    fig = plt.figure(figsize=(20, 5), facecolor='white')
    cov_03 = float((v > 0.3).mean()) * 100
    cov_05 = float((v > 0.5).mean()) * 100
    fig.suptitle(
        f"{obj_id}  —  M5 Affordance Prediction\n"
        f"max={v.max():.3f}  mean={v.mean():.3f}  "
        f"cov(>0.3)={cov_03:.1f}%  cov(>0.5)={cov_05:.1f}%",
        fontsize=13, fontweight='bold', y=1.01
    )

    for i, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                   c=colors, s=1.2, alpha=0.85, edgecolors='none')
        ax.view_init(elev=elev, azim=azim)

        # equal axes
        extents = p.max(0) - p.min(0)
        r = extents.max() * 0.55
        c = (p.max(0) + p.min(0)) / 2
        ax.set_xlim(c[0]-r, c[0]+r)
        ax.set_ylim(c[1]-r, c[1]+r)
        ax.set_zlim(c[2]-r, c[2]+r)
        ax.set_axis_off()
        ax.set_facecolor('white')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=4)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.65])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Contact Probability', fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n✅ Saved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj',  required=True, help='物体 ID')
    parser.add_argument('--ckpt', default=CKPT_DEFAULT, help='checkpoint 路径')
    parser.add_argument('--save', default=None, help='保存图片路径 (不开窗口，自动用 matplotlib)')
    args = parser.parse_args()
    predict(args.obj)
