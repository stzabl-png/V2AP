#!/usr/bin/env python3
"""
detect_table_plane.py
=====================
从 DepthPro 深度图中检测桌面平面，估计相机到世界坐标系的旋转矩阵。

原理:
  1. 加载若干序列的 depths.npz + K.txt
  2. 取中间帧，反投影为 3D 点云
  3. RANSAC 平面拟合 → 最大平面法线 n_cam (相机坐标系中)
  4. 该法线即世界 Z 轴方向 → R_cam_to_world 使 n_cam → [0,0,1]
  5. 保存到 depth/{type}/{dataset}/R_cam_to_world.json

输出:
  data_hub/ProcessedData/depth/ThirdPerson/oakink/R_cam_to_world.json
  data_hub/ProcessedData/depth/ThirdPerson/dexycb/R_cam_to_world.json

用法:
  python data/detect_table_plane.py --dataset oakink
  python data/detect_table_plane.py --dataset dexycb
  python data/detect_table_plane.py --all
"""

import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

PD         = os.path.join(config.DATA_HUB, 'ProcessedData')
DEPTH_BASE = os.path.join(PD, 'depth', 'ThirdPerson')


# ── RANSAC 平面拟合 ───────────────────────────────────────────────────────────

def fit_plane_ransac(pts, n_iter=200, thr=0.01, min_ratio=0.3):
    """
    RANSAC 拟合平面，返回法线向量 (3,)。
    pts: (N, 3) 点云，单位：米
    """
    best_normal = None
    best_inliers = 0
    N = len(pts)

    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = pts[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n = n / norm

        # 点到平面距离
        dists = np.abs((pts - p0) @ n)
        inliers = int((dists < thr).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = n

    ratio = best_inliers / N if N > 0 else 0
    if ratio < min_ratio:
        return None, ratio
    return best_normal, ratio


def backproject_depth_frame(depth_frame, K, max_depth=2.5, stride=4):
    """
    将单帧深度图反投影为 3D 点云。
    depth_frame: (H, W) 米制深度
    K: (3, 3) 相机内参
    stride: 下采样步长（加速）
    返回: (N, 3) 点云
    """
    H, W = depth_frame.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    d = depth_frame[vs, us]
    valid = (d > 0.1) & (d < max_depth)
    us, vs, d = us[valid], vs[valid], d[valid]

    x = (us - cx) / fx * d
    y = (vs - cy) / fy * d
    z = d
    return np.stack([x, y, z], axis=1)


def rotation_align_z(n_cam):
    """
    构造旋转矩阵 R，使 R @ n_cam = [0, 0, 1]（世界 Z 轴）。
    即将桌面法线方向对齐到世界 Z 轴（Z↑ = 重力反方向）。
    """
    n = np.array(n_cam, dtype=float)
    n /= np.linalg.norm(n)

    # 确保法线朝向摄像机（Z>0）
    if n[2] < 0:
        n = -n

    z_world = np.array([0.0, 0.0, 1.0])

    # 旋转轴 = cross(n, z_world)，旋转角 = angle between them
    axis = np.cross(n, z_world)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        # 已经对齐
        return np.eye(3)

    axis /= axis_norm
    angle = float(np.arccos(np.clip(np.dot(n, z_world), -1, 1)))

    # Rodrigues 公式
    K_mat = np.array([
        [ 0,       -axis[2],  axis[1]],
        [ axis[2],  0,       -axis[0]],
        [-axis[1],  axis[0],  0      ],
    ])
    R = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * (K_mat @ K_mat)
    return R


# ── 主处理函数 ────────────────────────────────────────────────────────────────

def process_dataset(dataset, n_seqs=30):
    """
    对数据集检测桌面平面，输出 R_cam_to_world.json。
    用 n_seqs 个序列的深度帧聚合，提高鲁棒性。
    """
    depth_dir = os.path.join(DEPTH_BASE, dataset)
    if not os.path.isdir(depth_dir):
        print(f'❌ depth 目录不存在: {depth_dir}')
        return

    out_path = os.path.join(depth_dir, 'R_cam_to_world.json')

    seqs = sorted(os.listdir(depth_dir))
    seqs = [s for s in seqs if os.path.isdir(os.path.join(depth_dir, s))]

    # 均匀采样序列
    if len(seqs) > n_seqs:
        idx = np.linspace(0, len(seqs) - 1, n_seqs).astype(int)
        seqs = [seqs[i] for i in idx]

    print(f'Dataset: {dataset}  |  使用 {len(seqs)} 个序列检测桌面...')

    all_normals = []
    for seq in seqs:
        depths_path = os.path.join(depth_dir, seq, 'depths.npz')
        k_path      = os.path.join(depth_dir, seq, 'K.txt')
        if not (os.path.exists(depths_path) and os.path.exists(k_path)):
            continue

        try:
            depths = np.load(depths_path)['depths']   # (N, H, W)
            K = np.loadtxt(k_path)                     # (3, 3)

            # 取中间帧
            frame = depths[len(depths) // 2]
            pts = backproject_depth_frame(frame, K, max_depth=2.5, stride=4)

            if len(pts) < 200:
                continue

            normal, ratio = fit_plane_ransac(pts, n_iter=150, thr=0.015)
            if normal is not None and ratio > 0.25:
                # 法线朝向相机（Z>0）
                if normal[2] < 0:
                    normal = -normal
                all_normals.append(normal)

        except Exception as e:
            pass

    if not all_normals:
        print(f'❌ {dataset}: 未能检测到有效桌面平面')
        return

    # 对所有序列的法线取平均（四元数无需用此处，向量平均然后归一化即可）
    n_mean = np.array(all_normals).mean(axis=0)
    n_mean /= np.linalg.norm(n_mean)

    R = rotation_align_z(n_mean)

    # 验证：检查旋转后 Z 轴
    z_after = R @ n_mean
    print(f'✅ {dataset}: 桌面法线 n_cam = [{n_mean[0]:.3f}, {n_mean[1]:.3f}, {n_mean[2]:.3f}]')
    print(f'   旋转后 Z 对齐误差: {np.linalg.norm(z_after - [0,0,1]):.4f}')
    print(f'   来自 {len(all_normals)} 个序列')

    result = {
        'dataset':        dataset,
        'n_cam_table':    n_mean.tolist(),      # 桌面法线（相机系）
        'R_cam_to_world': R.tolist(),           # 3×3，使 n_cam → Z_world
        'n_seqs_used':    len(all_normals),
        'note': (
            'R_cam_to_world maps camera-frame vectors to world-frame. '
            'Apply as: v_world = R_cam_to_world @ v_cam. '
            'World Z = gravity-up (table normal).'
        )
    }

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'   保存 → {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--all',     action='store_true')
    parser.add_argument('--n_seqs',  type=int, default=30,
                        help='用于平面拟合的序列数量')
    args = parser.parse_args()

    if args.all:
        datasets = ['oakink', 'dexycb']
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        return

    for ds in datasets:
        print(f'\n{"="*50}')
        process_dataset(ds, n_seqs=args.n_seqs)


if __name__ == '__main__':
    main()
