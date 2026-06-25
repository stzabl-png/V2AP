#!/usr/bin/env python3
"""
estimate_obj_rotation.py
========================
对每个物体估计 canonical rotation，使 mesh 对齐到真实世界中物体的自然姿态。

方法 (按优先级):
  stable_pose ★首选: trimesh.compute_stable_poses() 凸包稳定姿态（桌上自然摆放）
  pca_z    纯几何 PCA，把最长/特征轴对齐到世界 Z（stable_pose 失败时 fallback）
  world_z  FP ob_in_cam + R_cam_to_world → 世界坐标系平均旋转 (fallback)
  icp_hp   OakInk fallback: ICP 对齐 training_fp point_cloud
  fp_avg   YCB fallback: 直接平均 ob_in_cam 旋转（相机系，非世界系）
  identity 最终 fallback: 不旋转

输出:
  ProcessedData/obj_meshes/{dataset}/{obj_id}/rotation.json
  {
    "euler_xyz_deg": [rx, ry, rz],   # 应用到 mesh 的旋转 (degrees)
    "method": "stable_pose" | "pca_z" | "world_z" | "icp_hp" | "fp_avg" | "identity",
    "obj": "A01001",
    "dataset": "oakink"
  }

用法:
  python3 data/estimate_obj_rotation.py --dataset oakink           # stable_pose 优先
  python3 data/estimate_obj_rotation.py --dataset oakink --method icp_hp
  python3 data/estimate_obj_rotation.py --dataset ycb
  python3 data/estimate_obj_rotation.py --all --force
  python3 data/estimate_obj_rotation.py --obj A01001
"""

import os, sys, json, argparse, glob
import numpy as np
from scipy.spatial.transform import Rotation

PROJ         = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBJ_MESHES   = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
HP_INFER_DIR = os.path.join(PROJ, 'data_hub', 'human_prior', 'infer')
HP_DIR       = os.path.join(PROJ, 'data_hub', 'human_prior')
TRAINING_FP  = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp')
FP_POSES_DIR = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'poses', 'ThirdPerson')  # 正确路径
FP_EGO_DIR   = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'poses', 'Egocentric')
DEPTH_BASE   = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'depth', 'ThirdPerson')


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_mesh_scaled(obj_id, dataset):
    """加载 SAM3D mesh 并应用 scale_factor → 米制."""
    mesh_path  = os.path.join(OBJ_MESHES, dataset, obj_id, 'mesh.ply')
    scale_path = os.path.join(OBJ_MESHES, dataset, obj_id, 'scale.json')
    if not os.path.exists(mesh_path):
        return None, None
    import trimesh
    mesh = trimesh.load(mesh_path, force='mesh')
    scale = 1.0
    if os.path.exists(scale_path):
        scale = float(json.load(open(scale_path))['scale_factor'])
    mesh.vertices = mesh.vertices * scale
    return mesh, scale


def load_hp_pointcloud(obj_id, dataset):
    """
    加载 human_prior point_cloud (object canonical frame, 米制).
    搜索顺序:
      1. training_fp/{dataset}/{obj_id}.hdf5  ← HuggingFace 下载的
      2. human_prior/infer/oakink_{obj_id}.hdf5
      3. human_prior/infer/{obj_id}.hdf5
      4. human_prior/{obj_id}.hdf5
    """
    import h5py
    candidates = [
        os.path.join(TRAINING_FP, dataset, f'{obj_id}.hdf5'),          # ← 首选
        os.path.join(HP_INFER_DIR, f'oakink_{obj_id}.hdf5'),
        os.path.join(HP_INFER_DIR, f'{obj_id}.hdf5'),
        os.path.join(HP_DIR, f'oakink_{obj_id}.hdf5'),
        os.path.join(HP_DIR, f'{obj_id}.hdf5'),
    ]
    for path in candidates:
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                pc = f['point_cloud'][()].astype(np.float32)
            return pc, path
    return None, None


def load_fp_poses(obj_id, dataset):
    """
    加载 FoundationPose ob_in_cam poses (4×4 矩阵列表).
    搜索: poses/ThirdPerson/{dataset}/{seq}/ob_in_cam/*.txt
    只加载 obj_id 对应的序列（seq 名称以 obj_id 开头）。
    """
    ds_key = 'dexycb' if dataset in ('ycb', 'dexycb') else dataset
    fp_dir = os.path.join(FP_POSES_DIR, ds_key)
    if not os.path.isdir(fp_dir):
        return []

    poses = []
    for seq in sorted(os.listdir(fp_dir)):
        # 只取属于该物体的序列
        if obj_id and not seq.startswith(obj_id):
            continue
        ob_dir = os.path.join(fp_dir, seq, 'ob_in_cam')
        if not os.path.isdir(ob_dir):
            continue
        for txt in sorted(glob.glob(os.path.join(ob_dir, '*.txt'))):
            try:
                T = np.loadtxt(txt)
                if T.shape == (4, 4):
                    poses.append(T)
            except Exception:
                pass
    return poses


def load_r_cam_to_world(dataset):
    """
    加载桌面检测得到的 R_cam_to_world (3×3)。
    由 detect_table_plane.py 生成。
    """
    ds_key = 'dexycb' if dataset in ('ycb', 'dexycb') else dataset
    path = os.path.join(DEPTH_BASE, ds_key, 'R_cam_to_world.json')
    if not os.path.exists(path):
        return None
    data = json.load(open(path))
    return np.array(data['R_cam_to_world'])


def pca_z_rotation(mesh_verts):
    """
    PCA-based canonical rotation:
    - 找 mesh 的 3 个主轴（PCA 分解）
    - 根据长宽比决定对齐策略:
        elongated (s0/s1 > 1.4):  最长轴 → +Z  (瓶子竖立)
        flat      (s1/s2 > 2.5):  最短轴 → +Z  (盘子平放)
        otherwise: 第二长轴 → +Z
    - 确保 Z+ 方向是物体较轻的一侧（重心在下）
    """
    verts = np.array(mesh_verts)
    centered = verts - verts.mean(0)
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)

    r0, r1, r2 = s[0], s[1], s[2]

    if r0 / max(r1, 1e-8) > 1.4:
        target_axis = Vt[0]   # 细长: 最长轴 -> Z
    elif r1 / max(r2, 1e-8) > 2.5:
        target_axis = Vt[2]   # 扁平: 最短轴 -> Z
    else:
        target_axis = Vt[1]   # 中等: 第二轴 -> Z

    if target_axis[2] < 0:
        target_axis = -target_axis

    # 重心在下: 下半部分质心应在 -Z 侧
    proj = centered @ target_axis
    lower_half_mask = proj < 0
    if lower_half_mask.any():
        lower_center = centered[lower_half_mask].mean(0)
        if np.dot(lower_center, target_axis) > 0:
            target_axis = -target_axis

    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(target_axis, z)
    sin_a = float(np.linalg.norm(cross))
    cos_a = float(np.dot(target_axis, z))
    if sin_a < 1e-8:
        return np.eye(3)
    rot_vec = cross / sin_a * np.arctan2(sin_a, cos_a)
    return Rotation.from_rotvec(rot_vec).as_matrix()


def stable_pose_rotation(mesh):
    """
    trimesh 稳定姿态：凸包候选底面 + 均匀密度 CoM 投影，取概率最高的姿态。
    返回 3×3 旋转（不含平移），使 mesh 顶点 v' = R @ v 与 compute_stable_poses 的旋转一致。
    """
    try:
        transforms, probs = mesh.compute_stable_poses()
    except Exception:
        return None, None

    if transforms is None or len(transforms) == 0:
        return None, None

    T = np.asarray(transforms[0], dtype=np.float64)
    if T.shape != (4, 4):
        return None, None

    R = T[:3, :3]
    if abs(np.linalg.det(R) - 1.0) > 0.05:
        return None, None

    prob = float(probs[0]) if probs is not None and len(probs) > 0 else None
    return R, prob


def icp_rotation_only(src_pts, tgt_pts, n_iter=50):
    """
    只估计旋转的简化 ICP（固定质心对齐，只优化旋转）。
    src_pts: (N, 3) SAM3D mesh 采样点 (已 scale)
    tgt_pts: (M, 3) HP point_cloud (canonical frame)
    返回: R (3×3) 使 R @ src 最接近 tgt
    """
    from scipy.spatial import cKDTree

    # 质心对齐（去除平移影响）
    src = src_pts - src_pts.mean(axis=0)
    tgt = tgt_pts - tgt_pts.mean(axis=0)

    # 归一化到相同尺度
    src_scale = np.sqrt((src ** 2).sum(axis=1).mean())
    tgt_scale = np.sqrt((tgt ** 2).sum(axis=1).mean())
    if src_scale < 1e-8 or tgt_scale < 1e-8:
        return np.eye(3)
    src = src / src_scale * tgt_scale

    R = np.eye(3)
    for _ in range(n_iter):
        # 旋转 src
        src_rot = (R @ src.T).T

        # 找最近邻
        tree = cKDTree(tgt)
        dists, idxs = tree.query(src_rot, k=1)
        tgt_matched = tgt[idxs]

        # SVD 求最优旋转
        H = src_rot.T @ tgt_matched
        U, _, Vt = np.linalg.svd(H)
        R_new = Vt.T @ U.T
        # 保证行列式为 1（纯旋转）
        if np.linalg.det(R_new) < 0:
            Vt[-1, :] *= -1
            R_new = Vt.T @ U.T
        R = R_new @ R
        # 收敛判断
        if np.allclose(R_new, np.eye(3), atol=1e-5):
            break

    return R


def average_rotations(R_list):
    """
    对多个旋转矩阵取平均（四元数平均法）。
    """
    quats = np.array([Rotation.from_matrix(R).as_quat() for R in R_list])
    # 确保半球一致性
    ref = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], ref) < 0:
            quats[i] = -quats[i]
    q_mean = quats.mean(axis=0)
    q_mean /= np.linalg.norm(q_mean)
    return Rotation.from_quat(q_mean).as_matrix()


# ─────────────────────────────────────────────────────────────────────────────
# 核心处理函数
# ─────────────────────────────────────────────────────────────────────────────

def estimate_one(obj_id, dataset, force=False, n_mesh_pts=2048, method_override=None):
    out_path = os.path.join(OBJ_MESHES, dataset, obj_id, 'rotation.json')

    if os.path.exists(out_path) and not force:
        existing = json.load(open(out_path))
        print(f'  ⏭  {obj_id}: 已存在 {existing["euler_xyz_deg"]} (method={existing.get("method","?")})')
        return True

    mesh, scale = load_mesh_scaled(obj_id, dataset)
    if mesh is None:
        print(f'  ⚠️  {obj_id}: mesh 未找到')
        return False

    R_final = None
    method  = None

    # ── 方法0: stable_pose ★ trimesh 稳定姿态（首选）────────────────────────
    if method_override in (None, 'stable_pose'):
        try:
            R_sp, prob = stable_pose_rotation(mesh)
            if R_sp is not None:
                R_final = R_sp
                euler_test = Rotation.from_matrix(R_final).as_euler('xyz', degrees=True)
                prob_str = f' prob={prob:.3f}' if prob is not None else ''
                method = 'stable_pose'
                print(f'  🪑 {obj_id}: stable_pose{prob_str} -> euler={[round(e,1) for e in euler_test]}')
            else:
                print(f'  ⚠️  {obj_id}: stable_pose 无可用姿态')
        except Exception as e:
            print(f'  ⚠️  {obj_id}: stable_pose 失败: {e}')
            R_final = None

    # ── 方法1: pca_z 纯几何主轴对齐（fallback）────────────────────────────
    if R_final is None and method_override in (None, 'pca_z'):
        try:
            R_final = pca_z_rotation(mesh.vertices)
            euler_test = Rotation.from_matrix(R_final).as_euler('xyz', degrees=True)
            method = 'pca_z'
            print(f'  📐 {obj_id}: pca_z -> euler={[round(e,1) for e in euler_test]}')
        except Exception as e:
            print(f'  ⚠️  {obj_id}: pca_z 失败: {e}')
            R_final = None

    # ── 方法2: world_z (FP位姿+桌面检测，fallback) ──────────────────────────
    if R_final is None and method_override in (None, 'world_z'):
        R_cw = load_r_cam_to_world(dataset)
        if R_cw is not None:
            fp_poses = load_fp_poses(obj_id, dataset)
            if len(fp_poses) >= 3:
                R_world_list = []
                for T in fp_poses:
                    R_cam_obj = T[:3, :3]
                    t = T[:3, 3]
                    if np.linalg.norm(t) < 0.01 or np.linalg.norm(t) > 3.0 or t[2] < 0.01:
                        continue
                    if abs(np.linalg.det(R_cam_obj) - 1.0) > 0.05:
                        continue
                    R_world_obj = R_cw @ R_cam_obj
                    R_world_list.append(R_world_obj)
                if len(R_world_list) >= 3:
                    if len(R_world_list) > 500:
                        idx = np.linspace(0, len(R_world_list)-1, 500).astype(int)
                        R_world_list = [R_world_list[i] for i in idx]
                    R_final = average_rotations(R_world_list)
                    method  = f'world_z ({len(R_world_list)} frames)'
                    print(f'  🌍 {obj_id}: world_z ({len(R_world_list)} frames)')

    # ── 方法3: ICP 对齐 HP point_cloud (OakInk fallback) ────────────────────
    if R_final is None and method_override in (None, 'icp_hp'):
        hp_pc, hp_path = load_hp_pointcloud(obj_id, dataset)
        if hp_pc is not None:
            try:
                pts_idx = np.random.choice(len(mesh.vertices), min(n_mesh_pts, len(mesh.vertices)), replace=False)
                src_pts = mesh.vertices[pts_idx]
                tgt_pts = hp_pc[:min(n_mesh_pts, len(hp_pc))]
                R_final = icp_rotation_only(src_pts, tgt_pts)
                method  = 'icp_hp'
                print(f'  🔄 {obj_id}: ICP 对齐 HP ({hp_path.split("/")[-1]}, {len(tgt_pts)} pts)')
            except Exception as e:
                print(f'  ⚠️  {obj_id}: ICP 失败: {e}')
                R_final = None

    # ── 方法4: FP 平均旋转（相机系，硬编码 Y→Z 假设） ───────────────────────
    if R_final is None and method_override in (None, 'fp_avg'):
        fp_poses = load_fp_poses(obj_id, dataset)
        if fp_poses:
            R_list = [T[:3, :3] for T in fp_poses]
            if len(R_list) > 500:
                idx = np.linspace(0, len(R_list)-1, 500).astype(int)
                R_list = [R_list[i] for i in idx]
            R_cam_obj = average_rotations(R_list)
            # 假设相机 Y↓ → 世界 Z↑
            R_cam_world = np.array([[1,0,0],[0,0,-1],[0,1,0]])
            R_final = R_cam_world @ R_cam_obj
            method  = f'fp_avg ({len(fp_poses)} frames)'
            print(f'  🔄 {obj_id}: FP 平均旋转 ({len(fp_poses)} frames)')

    # ── Fallback: identity ────────────────────────────────────────────────────
    if R_final is None:
        print(f'  ⚠️  {obj_id}: 无可用数据，使用 identity')
        euler = [0.0, 0.0, 0.0]
        method = 'identity'
    else:
        euler = Rotation.from_matrix(R_final).as_euler('xyz', degrees=True).tolist()

    result = {
        'euler_xyz_deg': [round(e, 4) for e in euler],
        'method': method or 'identity',
        'obj': obj_id,
        'dataset': dataset,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'  ✅ {obj_id}: euler={[round(e,1) for e in euler]}  method={method}')
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def list_dataset_objs(dataset):
    ds_dir = os.path.join(OBJ_MESHES, dataset)
    if not os.path.isdir(ds_dir):
        return []
    return sorted(
        o for o in os.listdir(ds_dir)
        if os.path.exists(os.path.join(ds_dir, o, 'mesh.ply'))
        and '_' not in o.split('_')[0]  # 过滤掉 A01001_0001_0002 这类子序列
    )


def main():
    parser = argparse.ArgumentParser(description='估计每个物体的 canonical rotation')
    parser.add_argument('--obj',     help='单个物体 ID')
    parser.add_argument('--dataset', help='数据集: oakink / ycb / dexycb')
    parser.add_argument('--all',     action='store_true')
    parser.add_argument('--force',   action='store_true', help='覆盖已有 rotation.json')
    parser.add_argument('--method',  default=None,
                        choices=['stable_pose', 'pca_z', 'world_z', 'icp_hp', 'fp_avg'],
                        help='强制使用指定方法（默认: stable_pose → pca_z → world_z → …）')
    args = parser.parse_args()

    datasets_to_run = ['oakink', 'dexycb'] if args.all else ([args.dataset] if args.dataset else None)

    if args.obj:
        for ds in ['oakink', 'ycb', 'arctic', 'dexycb']:
            if os.path.exists(os.path.join(OBJ_MESHES, ds, args.obj, 'mesh.ply')):
                estimate_one(args.obj, ds, force=args.force, method_override=args.method)
                return
        print(f'❌ 未找到 {args.obj}')
        return

    if datasets_to_run is None:
        parser.print_help()
        return

    total, ok = 0, 0
    for ds in datasets_to_run:
        obj_ids = list_dataset_objs(ds)
        # 检查 world_z 可用性
        r_cw = load_r_cam_to_world(ds)
        method_note = 'stable_pose → pca_z → world_z …'
        if r_cw is not None:
            method_note += '  (world_z ✅)'
        else:
            method_note += '  (world_z ❌ 需 detect_table_plane.py)'
        print(f'\n=== {ds} ({len(obj_ids)} 物体)  方法: {method_note} ===')
        for obj_id in obj_ids:
            total += 1
            if estimate_one(obj_id, ds, force=args.force, method_override=args.method):
                ok += 1

    print(f'\n完成: {ok}/{total}')


if __name__ == '__main__':
    main()
