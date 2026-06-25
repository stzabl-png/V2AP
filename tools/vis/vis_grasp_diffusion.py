#!/usr/bin/env python3
"""
vis_grasp_diffusion.py — Grasp Diffusion 预测位姿可视化

复用 vis_grasp_candidates.py 的 draw_grasp() 函数，
每个预测 pose 生成一张完整的夹爪图（与 GT 对比）。

用法（项目根目录）:
  # 单物体
  python3 tools/vis_grasp_diffusion.py --obj S10024

  # 批量保存
  python3 tools/vis_grasp_diffusion.py --obj S10024 \
      --save_dir output/vis_diff --n_samples 10

输出:
  output/vis_diff/{obj_id}/pred_00.png  ~ pred_09.png  (预测)
  output/vis_diff/{obj_id}/gt_00.png    ~ gt_N.png     (GT 对比)
  output/vis_diff/{obj_id}/overview.png (总览)
"""
import os, sys, json, argparse
import numpy as np
import h5py
import trimesh
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from PIL import Image

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from model.pointnet2 import PointNet2Seg
from model.grasp_diffusion import GraspDiffusion, decode_pose

# ── 默认路径 ──────────────────────────────────────────────────
PN2_CKPT   = os.path.join(PROJ, 'output', 'checkpoints', 'v2_sigma04', 'best_model.pth')
DIFF_CKPT  = os.path.join(PROJ, 'output', 'checkpoints', 'v1_grasp_diff', 'best_model.pth')
POSE_STATS = os.path.join(PROJ, 'output', 'checkpoints', 'v1_grasp_diff', 'pose_stats.pt')
PC_DIR     = os.path.join(PROJ, 'data_hub', 'training_m5')
GT_DIRS    = [
    os.path.join(PROJ, 'output', 'robot_gt_merged_oakink'),
    os.path.join(PROJ, 'output', 'robot_gt_merged_dexycb'),
]
OBJ_MESHES_DIR = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
DATASETS = ['oakink', 'ycb', 'arctic', 'dexycb', 'egocentric', 'ho3d_v3']
TCP_OFFSET = 0.105  # Franka finger-tip 到手腕距离

# ─────────────────────────────────────────────────────────────
# 数据加载（复用 vis_grasp_candidates 的逻辑）
# ─────────────────────────────────────────────────────────────

def find_obj_mesh(obj_id):
    for ds in DATASETS:
        mp = os.path.join(OBJ_MESHES_DIR, ds, obj_id, 'mesh.ply')
        sp = os.path.join(OBJ_MESHES_DIR, ds, obj_id, 'scale.json')
        if not os.path.exists(mp):
            continue
        sf = 1.0
        if os.path.exists(sp):
            with open(sp) as f:
                sf = float(json.load(f)['scale_factor'])
        return mp, sf
    return None, 1.0


def load_mesh(obj_id):
    mp, sf = find_obj_mesh(obj_id)
    if mp is None:
        return None
    mesh = trimesh.load(mp, force='mesh', process=False)
    if sf != 1.0:
        mesh.apply_scale(sf)
    rot_p = os.path.join(os.path.dirname(mp), 'rotation.json')
    if os.path.exists(rot_p):
        with open(rot_p) as f:
            data = json.load(f)
        if 'rotation_matrix' in data:
            R = np.array(data['rotation_matrix'], dtype=np.float64)
        elif 'euler_xyz_deg' in data:
            from scipy.spatial.transform import Rotation as ScipyR
            R = ScipyR.from_euler('xyz', data['euler_xyz_deg'], degrees=True).as_matrix()
        else:
            R = np.eye(3)
        T = np.eye(4); T[:3, :3] = R
        mesh.apply_transform(T)

    return mesh


def load_point_cloud(obj_id, n_points=4096):
    pc_path = os.path.join(PC_DIR, f'{obj_id}.hdf5')
    with h5py.File(pc_path, 'r') as h:
        pc  = h['point_cloud'][:].astype(np.float32)
        nrm = h['normals'][:].astype(np.float32)
    n = pc.shape[0]
    idx = np.random.choice(n, n_points, replace=(n < n_points))
    return pc[idx], nrm[idx]


def load_gt_poses(obj_id, max_gt=8):
    """返回最多 max_gt 个成功 GT pose 的 dict 列表（与 draw_grasp 兼容格式）"""
    poses = []
    for gt_dir in GT_DIRS:
        p = os.path.join(gt_dir, f'{obj_id}_robot_gt.hdf5')
        if not os.path.exists(p):
            continue
        with h5py.File(p, 'r') as h:
            for key in list(h.get('successful_grasps', {}).keys())[:max_gt]:
                g = h[f'successful_grasps/{key}']
                poses.append({
                    'pos':      g['grasp_point'][:],
                    'approach': g['approach_dir'][:],
                    'finger':   g['finger_dir'][:],
                    'width':    float(g.attrs.get('gripper_width', 0.04)),
                    'name':     key,
                    'score':    float(g.attrs.get('score', 0.0)),
                    'idx':      len(poses),
                })
        if len(poses) >= max_gt:
            break
    return poses


# ─────────────────────────────────────────────────────────────
# 推理：采样预测 pose
# ─────────────────────────────────────────────────────────────

def run_diffusion(obj_id, n_samples=10, ddim_steps=50, device='cuda'):
    pc, nrm = load_point_cloud(obj_id)
    cloud   = np.concatenate([pc, nrm], axis=1)
    cloud_t = torch.from_numpy(cloud).unsqueeze(0).to(device)

    pn2 = PointNet2Seg(num_classes=2, in_channel=6,
                       predict_force_center=True).to(device)
    ckpt = torch.load(PN2_CKPT, map_location=device, weights_only=False)
    pn2.load_state_dict(ckpt['model_state_dict'])
    pn2.eval()

    diff = GraspDiffusion(T=1000).to(device)
    diff.load(DIFF_CKPT, device=device)
    diff.eval()

    stats     = torch.load(POSE_STATS, map_location=device, weights_only=False)
    pose_mean = stats['mean'].to(device)
    pose_std  = stats['std'].to(device)

    with torch.no_grad():
        feat       = pn2.extract_global_feat(cloud_t[:, :, :3], cloud_t)
        poses_norm = diff.sample(feat, n_samples=n_samples, ddim_steps=ddim_steps)

    poses = poses_norm * pose_std.unsqueeze(0) + pose_mean.unsqueeze(0)
    gp, rot, width = decode_pose(poses)

    # 把 rotation 矩阵分解成 approach_dir 和 finger_dir
    # approach_dir = rot[:, :, 2] (Z列), finger_dir = rot[:, :, 0] (X列)
    approach_dirs = rot[:, :, 2].cpu().numpy()
    finger_dirs   = rot[:, :, 0].cpu().numpy()
    gp_np         = gp.cpu().numpy()
    width_np      = width.cpu().numpy()

    pred_poses = []
    for i in range(n_samples):
        pred_poses.append({
            'pos':      gp_np[i],
            'approach': approach_dirs[i],
            'finger':   finger_dirs[i],
            'width':    float(width_np[i]),
            'name':     f'pred_{i:02d}',
            'score':    0.0,
            'idx':      i,
        })
    return pred_poses


# ─────────────────────────────────────────────────────────────
# 复用 vis_grasp_candidates 的 draw_grasp 函数
# ─────────────────────────────────────────────────────────────

def draw_grasp(ax, mesh, cand, color='#f85149', label_prefix='Pred'):
    """直接从 vis_grasp_candidates.py 移植，支持自定义颜色"""
    pts, _ = trimesh.sample.sample_surface(mesh, 6000)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2],
               c='#5ab4d4', s=2, alpha=0.45,
               linewidths=0, zorder=2, depthshade=False)

    pos      = cand['pos']
    approach = cand['approach']
    finger   = cand['finger']
    width    = cand['width']
    hw       = width / 2.0

    # ① 受力中心 (CoM)
    com = mesh.center_mass
    ax.scatter(*com, c='#ff7f0e', s=100, marker='*', zorder=6,
               edgecolors='white', linewidths=0.4)

    # ② 抓取点
    ax.scatter(*pos, c='white', s=80, zorder=7,
               edgecolors=color, linewidths=1.5)

    # ③ 两个指尖 + 夹爪宽度横线
    fl = pos - hw * finger
    fr = pos + hw * finger
    ax.scatter(*fl, c='#17becf', s=55, marker='s', zorder=6,
               edgecolors='white', linewidths=0.4)
    ax.scatter(*fr, c='#17becf', s=55, marker='s', zorder=6,
               edgecolors='white', linewidths=0.4)
    ax.plot([fl[0], fr[0]], [fl[1], fr[1]], [fl[2], fr[2]],
            color='#17becf', linewidth=2.5, alpha=0.9, zorder=5)

    # ④ 腕部
    wrist = pos - approach * 0.05
    ax.scatter(*wrist, c='#ffdd57', s=90, marker='D', zorder=6,
               edgecolors='#888', linewidths=0.4)
    ax.plot([wrist[0], pos[0]], [wrist[1], pos[1]], [wrist[2], pos[2]],
            color='#ffdd57', linewidth=1.5, linestyle='--', alpha=0.7)

    # ⑤ 接近方向箭头
    arr_len = 0.025
    ax.quiver(*pos, *approach, length=arr_len, color=color,
              arrow_length_ratio=0.4, linewidth=2.0)

    return com, pos, fl, fr, wrist


def make_single_fig(mesh, cand, title_prefix='Pred', color='#f85149',
                    bg='#1a1a2e'):
    fig = plt.figure(figsize=(7, 7), facecolor=bg)
    ax  = fig.add_subplot(111, projection='3d', facecolor=bg)

    com, pos, fl, fr, wrist = draw_grasp(ax, mesh, cand, color=color)

    # 坐标轴范围
    b  = mesh.bounds
    cx, cy, cz = (b[0] + b[1]) / 2
    r  = max(b[1] - b[0]) * 0.9
    ax.set_xlim(cx-r, cx+r)
    ax.set_ylim(cy-r, cy+r)
    ax.set_zlim(b[0][2] - r*0.2, b[0][2] + 2*r)
    ax.set_xlabel('X', color='#aaa', fontsize=9)
    ax.set_ylabel('Y', color='#aaa', fontsize=9)
    ax.set_zlabel('Z', color='#aaa', fontsize=9)
    ax.tick_params(colors='#555', labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    com_dist = np.linalg.norm(pos - com) * 100
    title = (
        f"{title_prefix}  {cand['name']}\n"
        f"approach=({cand['approach'][0]:+.2f},{cand['approach'][1]:+.2f},{cand['approach'][2]:+.2f})"
        f"  w={cand['width']*100:.1f}cm  CoM偏差={com_dist:.1f}cm"
    )
    ax.set_title(title, color='white', fontsize=8.5, pad=8)

    legend_items = [
        plt.Line2D([0],[0], marker='*',  color='w', markerfacecolor='#ff7f0e',
                   markersize=10, label='受力中心 CoM'),
        plt.Line2D([0],[0], marker='o',  color='w', markerfacecolor='white',
                   markersize=8,  label='抓取点 (grasp_point)'),
        plt.Line2D([0],[0], marker='s',  color='w', markerfacecolor='#17becf',
                   markersize=7,  label=f'夹爪指尖  width={cand["width"]*100:.1f}cm'),
        plt.Line2D([0],[0], marker='D',  color='w', markerfacecolor='#ffdd57',
                   markersize=7,  label='腕部 EE'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=7,
              facecolor='#2a2a3e', edgecolor='#555', labelcolor='#ccc')
    ax.view_init(elev=20, azim=130)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj',        required=True)
    parser.add_argument('--n_samples',  type=int, default=10)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--max_gt',     type=int, default=6,
                        help='最多显示几张 GT 对比图')
    parser.add_argument('--save_dir',   default='output/vis_diff')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obj_id = args.obj

    print(f'[{obj_id}] 加载 mesh ...')
    mesh = load_mesh(obj_id)
    if mesh is None:
        print(f'❌ 找不到 {obj_id} 的 mesh'); return

    outdir = os.path.join(PROJ, args.save_dir, obj_id)
    os.makedirs(outdir, exist_ok=True)
    print(f'  输出目录: {outdir}')

    # ── 预测 pose ──────────────────────────────────────────
    print(f'  采样 {args.n_samples} 个 pose (DDIM {args.ddim_steps}步) ...')
    pred_poses = run_diffusion(obj_id, n_samples=args.n_samples,
                               ddim_steps=args.ddim_steps, device=device)

    print('  生成预测可视化 ...')
    pred_paths = []
    for cand in pred_poses:
        fig  = make_single_fig(mesh, cand, title_prefix='[PRED]', color='#f85149')
        path = os.path.join(outdir, f"{cand['name']}.png")
        fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
        pred_paths.append(path)
        print(f'    ✅ {cand["name"]}  w={cand["width"]*100:.1f}cm')

    # ── GT pose ────────────────────────────────────────────
    gt_poses = load_gt_poses(obj_id, max_gt=args.max_gt)
    print(f'  生成 GT 可视化 ({len(gt_poses)} 个) ...')
    gt_paths = []
    for cand in gt_poses:
        fig  = make_single_fig(mesh, cand, title_prefix='[GT]', color='#58a6ff')
        path = os.path.join(outdir, f"gt_{cand['name']}.png")
        fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
        gt_paths.append(path)
        print(f'    ✅ GT {cand["name"]}  w={cand["width"]*100:.1f}cm')

    # ── 总览图：PRED 上排 / GT 下排 ───────────────────────
    all_paths = pred_paths + gt_paths
    cols = 5
    rows = (len(all_paths) + cols - 1) // cols
    tiles = [Image.open(p) for p in all_paths[:cols*rows]]
    w, h  = tiles[0].size
    ov    = Image.new('RGB', (w * cols, h * rows), (26, 26, 46))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        ov.paste(t, (c * w, r * h))
    ov_path = os.path.join(outdir, 'overview.png')
    ov.save(ov_path)

    print(f'\n📊 总览图 → {ov_path}')
    print(f'✅ 完成: {outdir}')


if __name__ == '__main__':
    main()
