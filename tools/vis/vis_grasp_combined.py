#!/usr/bin/env python3
"""
vis_grasp_combined.py — Open3D 交互可视化

① Affordance 热力图（PointNet++ 预测，蓝→红连续梯度）
② force_center 受力中心球（橙色）
③ GraspDiffusion 预测的夹爪（N 个，每个不同颜色）
   - 左右指尖球（青色）
   - 夹爪横线（finger_dir 方向）
   - 手指纵向杆（approach_dir 方向，4cm 深）
   - 夹爪背板
   - approach 方向箭头
   - 手腕球（黄色）

用法:
  python3 tools/vis_grasp_combined.py --obj S10024
  python3 tools/vis_grasp_combined.py --obj S10024 --n_samples 5

鼠标操作:
  左键拖拽 旋转 / 右键拖拽 平移 / 滚轮 缩放 / Q 退出
"""

import os, sys, json, argparse, colorsys
import numpy as np
import torch
from scipy.spatial import cKDTree

try:
    import open3d as o3d
except ImportError:
    o3d = None

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from model.pointnet2 import PointNet2Seg
from model.grasp_diffusion import GraspDiffusion, rotation_from_6d

# ── 默认路径 ──────────────────────────────────────────────────
PN2_CKPT  = os.path.join(PROJ, 'output', 'checkpoints', 'v2_sigma04', 'best_model.pth')
DIFF_CKPT = os.path.join(PROJ, 'output', 'checkpoints', 'v2_grasp_diff', 'best_model.pth')
ROT_STATS = os.path.join(PROJ, 'output', 'checkpoints', 'v2_grasp_diff', 'rot_stats.pt')
PC_DIR    = os.path.join(PROJ, 'data_hub', 'training_m5')
TCP_OFFSET = 0.105   # Franka 手腕→指尖 (m)
GRIPPER_W  = 0.08    # 可视化夹爪宽度 (m)


# ─────────────────────────────────────────────────────────────
# PointNet++ 推理
# ─────────────────────────────────────────────────────────────

def run_pn2(obj_id, device, pn2_ckpt):
    import h5py, matplotlib.cm as mcm

    pc_path = os.path.join(PC_DIR, f'{obj_id}.hdf5')
    if not os.path.exists(pc_path):
        raise FileNotFoundError(f'找不到点云: {pc_path}')

    with h5py.File(pc_path, 'r') as h:
        pc_all  = h['point_cloud'][:].astype(np.float32)
        nrm_all = h['normals'][:].astype(np.float32)

    # 采 4096 点送模型
    n = pc_all.shape[0]
    idx4096   = np.random.choice(n, 4096, replace=(n < 4096))
    pc_model  = pc_all[idx4096]
    nrm_model = nrm_all[idx4096]

    cloud   = np.concatenate([pc_model, nrm_model], axis=1)  # (4096,6)
    cloud_t = torch.from_numpy(cloud).unsqueeze(0).to(device)
    xyz_t   = cloud_t[:, :, :3]

    pn2 = PointNet2Seg(num_classes=2, in_channel=6,
                       predict_force_center=True).to(device)
    ckpt = torch.load(pn2_ckpt, map_location=device, weights_only=False)
    pn2.load_state_dict(ckpt['model_state_dict'])
    pn2.eval()

    with torch.no_grad():
        seg_logits, fc_pred = pn2(xyz_t, cloud_t)
        global_feat         = pn2.extract_global_feat(xyz_t, cloud_t)
        probs = torch.softmax(seg_logits[0], dim=-1)[:, 1].cpu().numpy()  # (4096,)

    force_center = fc_pred[0].cpu().numpy()   # (3,)

    # ── 显示点云：直接用 training_m5 全量点云，保证坐标系一致 ──
    # 注意：绝对不能用 mesh 重采样（mesh 有 rotation.json 可能已旋转）
    N_DISP     = 30000
    disp_idx   = np.random.choice(n, N_DISP, replace=True)
    disp_pts   = pc_all[disp_idx]                          # (30K, 3)
    # KNN 插值 affordance 到所有显示点
    tree       = cKDTree(pc_model)
    _, knn_idx = tree.query(disp_pts, k=3)
    affordance = probs[knn_idx].mean(axis=1)               # (30K,)

    cmap   = mcm.get_cmap('jet')
    colors = cmap(np.clip(affordance, 0, 1))[:, :3]        # (30K, 3)

    print(f'  Affordance max={probs.max():.3f} '
          f'mean={probs.mean():.3f} >0.5: {(probs>0.5).sum()}pts')
    print(f'  force_center = {force_center.round(4)} m')
    print(f'  Object bbox:  X={float(pc_all[:,0].max()-pc_all[:,0].min()):.3f}  '
          f'Y={float(pc_all[:,1].max()-pc_all[:,1].min()):.3f}  '
          f'Z={float(pc_all[:,2].max()-pc_all[:,2].min()):.3f} m')

    return disp_pts, colors, force_center, global_feat, fc_pred, pc_all


# ─────────────────────────────────────────────────────────────
# GraspDiffusion 推理
# ─────────────────────────────────────────────────────────────

def run_diffusion(global_feat, fc_pred, device, diff_ckpt, rot_stats,
                  n_samples=5, ddim_steps=50):
    diff = GraspDiffusion(T=1000).to(device)
    diff.load(diff_ckpt, device=device)
    diff.eval()

    stats    = torch.load(rot_stats, map_location=device, weights_only=False)
    rot_mean = stats['rot_mean'].to(device)
    rot_std  = stats['rot_std'].to(device)

    with torch.no_grad():
        rot_norm = diff.sample(global_feat, fc_pred,
                               n_samples=n_samples, ddim_steps=ddim_steps)
    rot_6d = rot_norm * rot_std.unsqueeze(0) + rot_mean.unsqueeze(0)

    rot_mat      = rotation_from_6d(rot_6d)           # (N,3,3)
    approach_dir = rot_mat[:, :, 2].cpu().numpy()     # Z列
    finger_dir   = rot_mat[:, :, 0].cpu().numpy()     # X列
    fc_np        = fc_pred[0].cpu().numpy()

    grasps = []
    for i in range(n_samples):
        grasps.append({
            'force_center': fc_np,
            'approach_dir': approach_dir[i],
            'finger_dir':   finger_dir[i],
        })
    return grasps


# ─────────────────────────────────────────────────────────────
# Open3D 几何体工具
# ─────────────────────────────────────────────────────────────

def _rot_to_direction(direction):
    """返回将 Z 轴旋转到 direction 的旋转矩阵"""
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction) + 1e-8
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, direction)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(z, direction))
    if sin_a > 1e-6:
        axis /= sin_a
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            axis * np.arctan2(sin_a, cos_a))
    elif cos_a < 0:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([1.0, 0.0, 0.0]) * np.pi)
    else:
        R = np.eye(3)
    return R


def sphere(center, radius=0.005, color=(1, 0.5, 0)):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(np.array(center, dtype=float))
    s.paint_uniform_color(list(color))
    s.compute_vertex_normals()
    return s


def cylinder(p1, p2, radius=0.002, color=(0.1, 0.8, 0.9)):
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return None
    mid = (p1 + p2) / 2.0
    c = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length,
                                                   resolution=12)
    c.rotate(_rot_to_direction(direction), center=[0, 0, 0])
    c.translate(mid)
    c.paint_uniform_color(list(color))
    c.compute_vertex_normals()
    return c


def arrow(start, direction, length=0.03, radius=0.002, color=(0.9, 0.2, 0.2)):
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction) + 1e-8
    a = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius, cone_radius=radius * 2.5,
        cylinder_height=length * 0.7, cone_height=length * 0.3,
        resolution=12)
    a.rotate(_rot_to_direction(direction), center=[0, 0, 0])
    a.translate(np.array(start, dtype=float))
    a.paint_uniform_color(list(color))
    a.compute_vertex_normals()
    return a


# ─────────────────────────────────────────────────────────────
# 单个夹爪的几何体组
# ─────────────────────────────────────────────────────────────

def build_gripper(grasp, idx, n_total, width=GRIPPER_W,
                  finger_depth=0.04, tcp_vis_len=0.06):
    """
    夹爪几何体。所有尺寸（width/finger_depth/tcp_vis_len）
    由外部根据物体尺寸自动缩放后传入，不在这里硬编码。
    """
    geoms = []
    fc   = np.array(grasp['force_center'], dtype=float)
    app  = np.array(grasp['approach_dir'], dtype=float)
    app /= np.linalg.norm(app) + 1e-8
    fing = np.array(grasp['finger_dir'], dtype=float)
    fing /= np.linalg.norm(fing) + 1e-8

    # HSV 颜色（每个 grasp 不同色相）
    h, s, v = idx / max(n_total, 1), 0.85, 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    col = (r, g, b)

    hw           = width / 2.0

    tip_l      = fc - hw * fing
    tip_r      = fc + hw * fing
    tip_l_back = tip_l - app * finger_depth
    tip_r_back = tip_r - app * finger_depth
    back_mid   = (tip_l_back + tip_r_back) / 2.0
    wrist      = fc - app * TCP_OFFSET

    # 半径随夹爪宽度比例缩放
    tip_r_ball = width * 0.04     # 指尖球半径
    rod_r      = width * 0.025    # 杆半径
    wrist_r    = width * 0.055    # 手腕球半径
    fc_r       = width * 0.035    # force_center 小球

    # 指尖球（青色）
    geoms.append(sphere(tip_l, radius=tip_r_ball, color=(0.1, 0.85, 0.9)))
    geoms.append(sphere(tip_r, radius=tip_r_ball, color=(0.1, 0.85, 0.9)))

    # 指尖横线（夹爪宽度）
    c = cylinder(tip_l, tip_r, radius=rod_r, color=(0.1, 0.85, 0.9))
    if c: geoms.append(c)

    # 左右手指纵向杆
    c = cylinder(tip_l, tip_l_back, radius=rod_r, color=col)
    if c: geoms.append(c)
    c = cylinder(tip_r, tip_r_back, radius=rod_r, color=col)
    if c: geoms.append(c)

    # 背板横线（连接两指根）
    c = cylinder(tip_l_back, tip_r_back, radius=rod_r, color=col)
    if c: geoms.append(c)

    # approach 方向箭头（长度 = tcp_vis_len）
    a = arrow(fc, app, length=tcp_vis_len,
              radius=rod_r * 0.6, color=col)
    if a: geoms.append(a)

    # 手腕→背板中心连杆
    c = cylinder(wrist, back_mid, radius=rod_r * 0.7, color=(0.85, 0.85, 0.2))
    if c: geoms.append(c)

    # 手腕球（黄色）
    geoms.append(sphere(wrist, radius=wrist_r, color=(1.0, 0.9, 0.1)))

    # force_center 小球（橙色，每个 grasp 单独标注）
    geoms.append(sphere(fc, radius=fc_r, color=(1.0, 0.45, 0.0)))

    return geoms


def compute_surface_width(pc, force_center, finger_dir, pct=88):
    """
    沿 finger_dir 方向探测物体表面，返回夹爪显示宽度。
    将全部点云投影到 finger_dir 轴，用两侧 pct 百分位数找到表面。
    返回: 宽度 (m)  按 5% margin 略宽于表面以确保夹爪将封住物体。
    """
    fing = np.array(finger_dir, dtype=float)
    fing /= np.linalg.norm(fing) + 1e-8
    offsets = pc - np.array(force_center, dtype=float)
    projs   = offsets @ fing
    pos = projs[projs > 0]
    neg = projs[projs < 0]
    if len(pos) < 10 or len(neg) < 10:
        return None   # fallback
    right = np.percentile(pos, pct)
    left  = abs(np.percentile(neg, 100 - pct))
    return (right + left) * 1.05   # 5% 外延以封住物体


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────

def main(args):
    if o3d is None:
        print('❌ 请安装 open3d: pip install open3d'); return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obj_id = args.obj

    # 自动选 checkpoint
    diff_ckpt = args.diff_ckpt or DIFF_CKPT
    if not os.path.exists(diff_ckpt):
        diff_ckpt = os.path.join(PROJ, 'output', 'checkpoints',
                                 'v1_grasp_diff', 'best_model.pth')
        print(f'  ⚠ v2 checkpoint 不存在，退到 v1: {diff_ckpt}')

    rot_stats = args.rot_stats or ROT_STATS
    if not os.path.exists(rot_stats):
        rot_stats = os.path.join(PROJ, 'output', 'checkpoints',
                                 'v1_grasp_diff', 'pose_stats.pt')

    # ── PN2 推理 ────────────────────────────────────────────
    print(f'\n[{obj_id}] PointNet++ 推理 ...')
    disp_pts, colors, force_center, global_feat, fc_pred, pc_all = \
        run_pn2(obj_id, device, args.pn2_ckpt)

    # ── Diffusion 采样 ──────────────────────────────
    print(f'[{obj_id}] GraspDiffusion 采样 '
          f'({args.n_samples} poses, DDIM {args.ddim_steps} steps) ...')
    grasps = run_diffusion(global_feat, fc_pred, device,
                           diff_ckpt, rot_stats,
                           n_samples=args.n_samples,
                           ddim_steps=args.ddim_steps)

    # 全局参考尺寸（当 per-grasp 表面探测失败时的退化方案）
    obj_ext    = pc_all.max(0) - pc_all.min(0)
    fallback_w = float(np.sort(obj_ext)[-2]) * 0.80  # 第二大轴的 80%

    # ── 构建 Open3D 场景 ─────────────────────────────────────
    all_geoms = []

    # 热力图点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(disp_pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    all_geoms.append(pcd)

    # force_center 大橙球（大小 = 第二大轴宽度的 2%）
    fc_ball_r = float(np.sort(obj_ext)[-2]) * 0.02
    all_geoms.append(sphere(force_center,
                            radius=fc_ball_r,
                            color=(1.0, 0.35, 0.0)))

    # 各夹爪：per-grasp 根据 finger_dir 探测表面宽度
    for i, g in enumerate(grasps):
        fdir = np.array(g['finger_dir'], dtype=float)
        fdir /= np.linalg.norm(fdir) + 1e-8

        # 沿 finger_dir 探测物体表面，得到正确显示宽度
        surf_w = compute_surface_width(pc_all, force_center, fdir)
        if surf_w is None or surf_w < GRIPPER_W:
            surf_w = fallback_w

        # 手指深度和第二轴方向的物体宽度（approach 方向的宽度）
        adir   = np.array(g['approach_dir'], dtype=float)
        adir  /= np.linalg.norm(adir) + 1e-8
        depth_w = compute_surface_width(pc_all, force_center, adir)
        if depth_w is None:
            depth_w = surf_w * 0.6
        fdepth   = min(depth_w * 0.5, surf_w * 0.6)   # 手指深度
        tcp_vis  = surf_w * 0.8                        # approach 算头长度

        scale = surf_w / GRIPPER_W
        gripper = build_gripper(g, i, args.n_samples,
                                width=surf_w,
                                finger_depth=fdepth,
                                tcp_vis_len=tcp_vis)
        all_geoms.extend(gripper)
        app = g['approach_dir']
        fin = g['finger_dir']
        print(f'  [{i:02d}] approach=({app[0]:+.2f},{app[1]:+.2f},{app[2]:+.2f})'
              f'  finger=({fin[0]:+.2f},{fin[1]:+.2f},{fin[2]:+.2f})'
              f'  disp_w={surf_w:.3f}m ({scale:.1f}x)')

    # ── Open3D 窗口 ──────────────────────────────────────────
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f'Affordance Heatmap + GraspDiffusion v2  [{obj_id}]',
        width=1400, height=900)

    for g in all_geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size       = 3.0
    opt.background_color = np.array([0.06, 0.06, 0.10])

    print('\n🖱  左键旋转  右键平移  滚轮缩放  Q退出')
    print('   🔴→🔵 热力图: 红=高affordance')
    print('   🟠 大橙球: force_center (TCP指尖中心)')
    print('   🩵 青球+横线: 指尖 + 夹爪宽度')
    print('   彩色纵杆+背板: 夹爪结构')
    print('   彩色箭头: approach方向')
    print('   🟡 黄球: 手腕EE')
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj',           required=True)
    parser.add_argument('--pn2_ckpt',      default=PN2_CKPT)
    parser.add_argument('--diff_ckpt',     default=None)
    parser.add_argument('--rot_stats',     default=None)
    parser.add_argument('--n_samples',     type=int,   default=5)
    parser.add_argument('--ddim_steps',    type=int,   default=50)
    parser.add_argument('--gripper_width', type=float, default=GRIPPER_W)
    args = parser.parse_args()
    main(args)
