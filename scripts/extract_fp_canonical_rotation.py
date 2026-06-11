#!/usr/bin/env python3
"""
extract_fp_canonical_rotation.py
=================================
从 FoundationPose (FP) 的 ob_in_cam 姿态数据中提取每个物体的
"真实桌面放置朝向"，生成与 Human Prior 数据一致的 canonical_rotation。

原理：
  FP 给出每帧 T_ob_in_cam（物体→相机的 4×4 变换，真实米制）。
  当物体静止放在桌面时，世界 Z 轴（向上）在相机空间的方向 = R_cam_world @ [0,0,1]。
  我们通过 R_ob_in_cam 求出物体坐标系里的"向上"方向，
  然后计算把该方向转到 Z 轴所需的旋转 → canonical_rotation。

  这样仿真里物体的朝向与 FP 看到的朝向一致，
  确保 HP label (标注在 SAM3D mesh frame) 和 grasp_point (仿真 canonical frame) 天然对齐。

输出：
  sim/canonical_rotation_fp.json   — FP 导出的旋转
  sim/canonical_rotation.json      — 可选：更新已有文件

用法：
  python3 scripts/extract_fp_canonical_rotation.py --dataset dexycb
  python3 scripts/extract_fp_canonical_rotation.py --dataset oakink
  python3 scripts/extract_fp_canonical_rotation.py --all
  python3 scripts/extract_fp_canonical_rotation.py --obj ycb_dex_09 --dataset dexycb --visualize
"""

import os, sys, json, argparse, glob
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from natsort import natsorted

PROJ       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE_BASE  = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'poses', 'ThirdPerson')
MESH_BASE  = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
OUT_JSON   = os.path.join(PROJ, 'sim', 'canonical_rotation_fp.json')
CAN_JSON   = os.path.join(PROJ, 'sim', 'canonical_rotation.json')


# ── FP pose 读取 ─────────────────────────────────────────────────────────────

def load_ob_in_cam_poses(pose_dir: str) -> np.ndarray:
    """读取 ob_in_cam/*.txt，每个文件是 4×4 矩阵，返回 (N, 4, 4)。"""
    files = natsorted(glob.glob(os.path.join(pose_dir, 'ob_in_cam', '*.txt')))
    if not files:
        return np.zeros((0, 4, 4))
    poses = []
    for fp in files:
        try:
            T = np.loadtxt(fp)
            if T.shape == (4, 4):
                poses.append(T)
        except Exception:
            pass
    return np.array(poses) if poses else np.zeros((0, 4, 4))


# ── 序列 → 物体映射（复用 batch_align_mano_fp 的逻辑）────────────────────────

sys.path.insert(0, PROJ)
try:
    from data.batch_align_mano_fp import dexycb_seq_to_ycb, seq_to_obj
    _HAS_SEQ_MAP = True
except ImportError:
    _HAS_SEQ_MAP = False

def get_seq_obj_mapping(dataset: str) -> dict:
    """返回 {seq_id: obj_name} 映射。"""
    ds_pose_dir = os.path.join(POSE_BASE, dataset)
    if not os.path.isdir(ds_pose_dir):
        print(f'  ⚠ pose dir not found: {ds_pose_dir}')
        return {}

    mapping = {}
    for seq_id in os.listdir(ds_pose_dir):
        seq_dir = os.path.join(ds_pose_dir, seq_id)
        if not os.path.isdir(seq_dir):
            continue
        if _HAS_SEQ_MAP:
            try:
                obj_ds, obj_name = seq_to_obj(dataset, seq_id)
                if obj_name:
                    mapping.setdefault(obj_name, []).append((seq_id, obj_ds))
            except Exception:
                pass
        else:
            # 简单回退：序列名直接作为 obj 名
            mapping.setdefault(seq_id, []).append((seq_id, dataset))
    return mapping


# ── 相机朝向估计 ─────────────────────────────────────────────────────────────

def estimate_camera_up(poses: np.ndarray) -> np.ndarray:
    """
    DexYCB / OakInk 使用固定三脚架相机，大致俯视桌面。
    相机坐标系：X=右, Y=下, Z=前（OpenCV convention）。
    世界 Z 轴（向上）在相机空间 ≈ -Y 轴方向（因为 Y=down）。
    返回 (3,) 单位向量：世界"向上"在相机空间的方向。
    """
    # 从 FP pose 的 t 分量粗估相机朝向：
    # 物体 translation (t) 大约在相机前方，Z > 0，Y > 0（桌面在相机下方）
    # → 相机 Y 轴（向下）对应世界 -Z（向下）
    # → 世界向上 = 相机 -Y = [0, -1, 0]
    return np.array([0.0, -1.0, 0.0])


def up_in_mesh_frame(R_ob_in_cam: np.ndarray, cam_up: np.ndarray) -> np.ndarray:
    """
    给定 ob_in_cam 旋转 R，以及相机空间里世界"向上"的方向，
    返回物体坐标系（mesh frame）里的"向上"方向。

    R_ob_in_cam: (3,3)  mesh_frame → camera_frame
    cam_up: (3,)        世界向上，camera frame
    → up_mesh = R^T @ cam_up = camera_frame → mesh_frame
    """
    return R_ob_in_cam.T @ cam_up


def rotation_to_align_up_to_z(up_vec: np.ndarray) -> np.ndarray:
    """
    计算旋转矩阵 R，使得 R @ up_vec ≈ [0, 0, 1]。
    返回 (3,3) 旋转矩阵。
    """
    up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-8)
    target = np.array([0.0, 0.0, 1.0])

    # 用 Rodrigues 公式
    v = np.cross(up_vec, target)
    s = np.linalg.norm(v)
    c = np.dot(up_vec, target)

    if s < 1e-6:
        # 已经对齐或完全反转
        if c > 0:
            return np.eye(3)
        else:
            # 旋转 180° 绕 X
            return np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)

    Vx = np.array([
        [ 0,   -v[2],  v[1]],
        [ v[2],  0,   -v[0]],
        [-v[1],  v[0],  0  ]
    ])
    R = np.eye(3) + Vx + Vx @ Vx * ((1 - c) / (s * s))
    return R


def median_rotation(Rs: np.ndarray) -> np.ndarray:
    """
    对多帧旋转矩阵取"中位旋转"：
    转为 quaternion → 平均 → 归一化 → 回矩阵。
    """
    quats = Rotation.from_matrix(Rs).as_quat()   # (N, 4) xyzw
    # 确保符号一致（避免 antipodal ambiguity）
    ref = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], ref) < 0:
            quats[i] *= -1
    mean_q = quats.mean(axis=0)
    mean_q /= np.linalg.norm(mean_q)
    return Rotation.from_quat(mean_q).as_matrix()


# ── 选取"静止"帧 ─────────────────────────────────────────────────────────────

def select_stable_frames(poses: np.ndarray, max_frames: int = 50) -> np.ndarray:
    """
    选取物体平移变化最小的帧（物体静止放在桌面）。
    先过滤掉平移跳变帧，再均匀采样 max_frames 帧。
    """
    if len(poses) == 0:
        return poses

    ts = poses[:, :3, 3]   # (N, 3) translations
    # 计算逐帧位移
    diffs = np.linalg.norm(np.diff(ts, axis=0), axis=1)   # (N-1,)
    # 过滤跳变帧（位移 > 5cm）
    mask = np.ones(len(poses), dtype=bool)
    bad = np.where(diffs > 0.05)[0]
    mask[bad + 1] = False

    stable = poses[mask]
    if len(stable) == 0:
        stable = poses

    # 均匀采样
    if len(stable) > max_frames:
        idx = np.linspace(0, len(stable) - 1, max_frames, dtype=int)
        stable = stable[idx]

    return stable


# ── 单物体处理 ────────────────────────────────────────────────────────────────

def process_object(obj_name: str, seqs: list, dataset: str) -> dict | None:
    """
    从该物体的所有序列里提取 canonical up_in_mesh，
    返回 {'euler_xyz': [rx, ry, rz], 'matrix': 3×3 list, 'n_frames': int}。
    """
    all_up_vecs = []
    n_frames_total = 0
    cam_up = estimate_camera_up(None)

    for seq_id, obj_ds in seqs:
        pose_dir = os.path.join(POSE_BASE, dataset, seq_id)
        poses = load_ob_in_cam_poses(pose_dir)
        if len(poses) == 0:
            continue

        stable = select_stable_frames(poses)
        n_frames_total += len(stable)

        for T in stable:
            R = T[:3, :3]
            # 检查旋转矩阵合法性
            if abs(np.linalg.det(R) - 1.0) > 0.1:
                continue
            up_mesh = up_in_mesh_frame(R, cam_up)
            all_up_vecs.append(up_mesh)

    if len(all_up_vecs) < 5:
        print(f'  ⚠ {obj_name}: 只有 {len(all_up_vecs)} 帧，跳过')
        return None

    all_up_vecs = np.array(all_up_vecs)   # (N, 3)
    # 取各帧 up_in_mesh 的中位方向
    median_up = np.median(all_up_vecs, axis=0)
    median_up /= np.linalg.norm(median_up) + 1e-8

    # 计算把 median_up 转到 Z 轴的旋转
    R_canonical = rotation_to_align_up_to_z(median_up)

    # 转为 euler 角（xyz，degrees）用于 canonical_rotation.json
    euler = Rotation.from_matrix(R_canonical).as_euler('xyz', degrees=True)

    # 验证：旋转后 up_in_mesh 是否确实接近 Z 轴
    check = R_canonical @ median_up
    cos_err = np.dot(check, [0, 0, 1])

    print(f'  {obj_name}: {n_frames_total} frames, '
          f'up_in_mesh={median_up.round(3)}, '
          f'euler={euler.round(1)}, '
          f'alignment={cos_err:.3f}')

    return {
        'euler_xyz': euler.tolist(),
        'matrix': R_canonical.tolist(),
        'up_in_mesh': median_up.tolist(),
        'n_frames': n_frames_total,
        'cos_alignment': float(cos_err),
        'source': 'fp_derived',
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Extract FP-derived canonical rotations')
    ap.add_argument('--dataset', default=None, choices=['dexycb', 'oakink'],
                    help='单一 dataset')
    ap.add_argument('--all',  action='store_true', help='全部 dataset')
    ap.add_argument('--obj',  default=None, help='只处理单个物体')
    ap.add_argument('--update-canonical', action='store_true',
                    help='同时更新 sim/canonical_rotation.json')
    ap.add_argument('--visualize', action='store_true',
                    help='可视化旋转前后的 mesh（需要 Open3D）')
    ap.add_argument('--out', default=OUT_JSON, help='输出 JSON 路径')
    args = ap.parse_args()

    datasets = []
    if args.all:
        datasets = ['dexycb', 'oakink']
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ['dexycb', 'oakink']

    # 加载已有 canonical_rotation.json
    existing = {}
    if os.path.exists(CAN_JSON):
        existing = json.load(open(CAN_JSON))

    results = {}

    for dataset in datasets:
        print(f'\n=== {dataset} ===')
        obj_seqs = get_seq_obj_mapping(dataset)
        if not obj_seqs:
            print(f'  No sequences found for {dataset}')
            continue

        for obj_name, seqs in sorted(obj_seqs.items()):
            if args.obj and obj_name != args.obj:
                continue

            result = process_object(obj_name, seqs, dataset)
            if result is not None:
                results[obj_name] = result

                if args.visualize:
                    _visualize(obj_name, dataset, result)

    # 保存 FP 导出的旋转
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✅ FP canonical rotations → {args.out}  ({len(results)} objects)')

    # 可选：更新主 canonical_rotation.json
    if args.update_canonical:
        # 合并：FP 导出的优先，existing 作为 fallback
        merged = dict(existing)
        updated = 0
        for obj_name, r in results.items():
            euler = r['euler_xyz']
            # 只更新旋转角度显著的（>5°）
            if any(abs(e) > 5 for e in euler):
                merged[obj_name] = euler
                updated += 1
        with open(CAN_JSON, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f'✅ Updated canonical_rotation.json  ({updated} objects changed)')

    return results


def _visualize(obj_name: str, dataset: str, result: dict):
    """可视化旋转前后的 mesh（仅用于调试）。"""
    try:
        import open3d as o3d, trimesh
    except ImportError:
        print('  Open3D/trimesh not available for visualization')
        return

    ds_dir = 'dexycb' if dataset == 'dexycb' else 'oakink'
    mesh_path = os.path.join(MESH_BASE, ds_dir, obj_name, 'mesh.ply')
    if not os.path.exists(mesh_path):
        return

    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    sf_path = os.path.join(MESH_BASE, ds_dir, obj_name, 'scale.json')
    if os.path.exists(sf_path):
        sf = json.load(open(sf_path))['scale_factor']
        mesh.vertices *= sf

    R = np.array(result['matrix'])
    rotated = mesh.copy()
    rotated.vertices = (R @ mesh.vertices.T).T

    def to_o3d(m, color):
        pcd = o3d.geometry.PointCloud()
        pts, _ = trimesh.sample.sample_surface(m, 4096)
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(color, (len(pts), 1)))
        return pcd

    orig    = to_o3d(mesh,    [0.6, 0.6, 0.6])  # gray
    rotd    = to_o3d(rotated, [0.2, 0.6, 1.0])  # blue
    # Z 轴箭头
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.003, cone_radius=0.006,
        cylinder_height=0.08, cone_height=0.02)
    arrow.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries(
        [orig, rotd, arrow],
        window_name=f'{obj_name} — gray=original, blue=canonical',
        width=800, height=600)


if __name__ == '__main__':
    main()
