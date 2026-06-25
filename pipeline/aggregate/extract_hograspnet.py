#!/usr/bin/env python3
"""
HOGraspNet → V2AP Human Prior 转换脚本

从 HOGraspNet 数据集 (ECCV 2024) 的标注 JSON 中提取 MANO 手部接触信息,
将接触区域投影到物体表面, 生成 human_prior HDF5 文件。

HOGraspNet 数据格式:
    - contact: (778,) float [0,1] — MANO 手部顶点的接触概率
    - Mesh[0]: {mano_pose, mano_betas, mano_trans, object_mat, object_file, ...}
    - hand: {mano_scale, mano_xyz_root, 3D_pose_per_cam}
    - 30 个扫描物体 (01-30), mesh 在 data/{id}_{name}/

方案: 利用 MANO 参数重建手部 mesh → 找接触顶点 → 变换到物体坐标系
    → KDTree 投影到物体表面

依赖: manopth (MANO PyTorch layer), trimesh, h5py, numpy

用法:
    # Run from project root
    python data/extract_hograspnet.py
    python data/extract_hograspnet.py --hog_dir /path/to/HOGraspNet/data
"""

import os
import sys
import json
import glob
import argparse
import re
import time
import numpy as np
import trimesh
import h5py
from collections import defaultdict
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# ---- Constants ----
CONTACT_THRESHOLD = 0.3   # 手部顶点接触概率阈值
OBJ_CONTACT_RADIUS = 2.0  # 投影到物体表面的接触半径 (cm, HOGraspNet 坐标系)
CM_TO_M = 0.01  # HOGraspNet 数据是 cm, 最终输出转为 m

# 物体编号 → (目录名, 显示名)
OBJ_MAP = {}  # 运行时从文件系统构建


def try_import_mano():
    """尝试导入 manopth, 如果失败则回退到 joint-based 方法."""
    try:
        import torch
        from manopth.manolayer import ManoLayer
        return ManoLayer, torch
    except ImportError:
        return None, None


def build_mano_hand(mano_layer, torch, mano_pose, mano_betas, mano_trans, mano_scale):
    """使用 manopth 重建 MANO 手部 mesh 顶点.

    Args:
        mano_layer: ManoLayer instance
        mano_pose: (1, 45) or (1, 48) — 手部姿态参数
        mano_betas: (1, 10) — 形状参数
        mano_trans: (1, 3) — 平移
        mano_scale: float — 缩放系数

    Returns:
        verts: (778, 3) numpy array — 手部顶点坐标
    """
    pose = torch.FloatTensor(mano_pose)
    betas = torch.FloatTensor(mano_betas)
    trans = torch.FloatTensor(mano_trans)

    # manopth 期望 pose 是 (B, ncomps+3), 全局旋转 + PCA 系数
    # HOGraspNet 提供的是 axis-angle (1, 45) = 15 joints × 3
    if pose.shape[-1] == 45:
        # 全部 axis-angle 参数
        verts, _ = mano_layer(pose, betas, trans)
    elif pose.shape[-1] == 48:
        verts, _ = mano_layer(pose[:, :48], betas, trans)
    else:
        verts, _ = mano_layer(pose, betas, trans)

    verts = verts[0].detach().numpy()  # (778, 3)

    # 应用 mano_scale (HOGraspNet 使用 cm/mm 空间)
    # mano_scale 通常在 ~7-8 范围, 表示 mm→m 或类似缩放
    # 我们需要最终在物体坐标系里工作
    return verts


def get_hand_contact_points_mano(annotation, mano_layer, torch_module):
    """使用 MANO 重建手 → 提取接触点 (世界坐标).

    Returns:
        contact_pts: (K, 3) 接触点坐标 (世界系)
        or None if failed
    """
    contact = np.array(annotation.get('contact', []))
    if len(contact) != 778:
        return None

    mesh_info = annotation.get('Mesh', [{}])[0]
    mano_pose = mesh_info.get('mano_pose')
    mano_betas = mesh_info.get('mano_betas')
    mano_trans = mesh_info.get('mano_trans')
    hand_info = annotation.get('hand', {})
    mano_scale = hand_info.get('mano_scale', 1.0)

    if not (mano_pose and mano_betas and mano_trans):
        return None

    try:
        verts = build_mano_hand(mano_layer, torch_module,
                                mano_pose, mano_betas, mano_trans, mano_scale)
    except Exception:
        return None

    # 选择接触顶点
    mask = contact > CONTACT_THRESHOLD
    if mask.sum() == 0:
        return None

    return verts[mask]


def get_hand_contact_points_joints(annotation):
    """回退方法: 使用 21 个手部关节位置近似接触区域.

    方法: 取 contact map 非零的 MANO 区域,
         利用手指关节位置作为接触点的代理.

    Returns:
        contact_pts: (K, 3) 手部关节坐标中有接触的关节
        or None
    """
    contact = np.array(annotation.get('contact', []))
    if len(contact) != 778:
        return None

    hand_info = annotation.get('hand', {})
    # 3D_pose_per_cam 是 (21, 3) 世界坐标
    poses_3d = hand_info.get('3D_pose_per_cam')
    if not poses_3d:
        return None

    joints_3d = np.array(poses_3d)  # (21, 3)
    if joints_3d.ndim == 3:
        joints_3d = joints_3d[0]  # 取第一个相机
    if joints_3d.shape[0] != 21:
        return None

    # MANO 778 顶点到 21 关节的粗略映射
    # 5 根手指, 每根 4 个关节 + 手腕
    # 对每个关节, 如果对应区域的平均接触 > 阈值, 则标记该关节为接触
    # 粗略分组: 每根手指约 156 个顶点 (778/5)
    # Thumb: 0-155, Index: 156-311, Middle: 312-467, Ring: 468-623, Pinky: 624-777
    finger_regions = [
        (0, 156),    # Thumb
        (156, 312),  # Index
        (312, 468),  # Middle
        (468, 624),  # Ring
        (624, 778),  # Pinky
    ]
    # 关节索引: [Wrist, TMCP, TPIP, TDIP, TTIP, IMCP, IPIP, ...]
    finger_joints = [
        [1, 2, 3, 4],      # Thumb
        [5, 6, 7, 8],      # Index
        [9, 10, 11, 12],   # Middle
        [13, 14, 15, 16],  # Ring
        [17, 18, 19, 20],  # Pinky
    ]

    contact_joints = [0]  # 手腕总是包含
    for (start, end), joint_idxs in zip(finger_regions, finger_joints):
        region_contact = contact[start:end]
        avg_contact = region_contact.mean()
        if avg_contact > CONTACT_THRESHOLD:
            contact_joints.extend(joint_idxs)

    if len(contact_joints) <= 1:
        # 如果没有手指有接触, 使用所有非零接触关节
        total_contact = contact.mean()
        if total_contact > CONTACT_THRESHOLD:
            contact_joints = list(range(21))
        else:
            return None

    return joints_3d[contact_joints]


def load_object_mesh(hog_dir, obj_id_str):
    """加载 HOGraspNet 扫描物体 mesh.

    Args:
        hog_dir: HOGraspNet data 目录
        obj_id_str: "16" → 查找 "16_golf_ball" 目录

    Returns:
        trimesh.Trimesh or None
    """
    obj_id = int(obj_id_str)
    # 查找匹配的目录
    for d in os.listdir(hog_dir):
        if os.path.isdir(os.path.join(hog_dir, d)):
            match = re.match(r'^(\d+)_(.+)$', d)
            if match and int(match.group(1)) == obj_id:
                obj_dir = os.path.join(hog_dir, d)
                obj_files = glob.glob(os.path.join(obj_dir, '*.obj'))
                if obj_files:
                    return trimesh.load(obj_files[0], force='mesh', process=False)
    return None


def transform_points_to_object_frame(points, obj_pose_mat):
    """将世界坐标的点变换到物体坐标系.

    Args:
        points: (N, 3) world coordinates
        obj_pose_mat: (4, 4) object pose matrix (world→object transform)

    Returns:
        (N, 3) points in object frame
    """
    # obj_pose_mat: T_world_to_obj or T_obj_to_world?
    # HOGraspNet 的 object pose_data 是物体在世界坐标系的位姿 (T_obj2world)
    # 需要求逆得到 T_world2obj
    T = np.array(obj_pose_mat)
    if T.shape == (4, 4):
        T_inv = np.linalg.inv(T)
        pts_h = np.hstack([points, np.ones((len(points), 1))])
        pts_obj = (T_inv @ pts_h.T).T[:, :3]
        return pts_obj.astype(np.float32)
    return points


def discover_sequences(hog_dir):
    """发现所有标注序列.

    Returns:
        sequences: dict {obj_id: [(seq_dir, [(cam, json_path), ...]), ...]}
    """
    sequences = defaultdict(list)

    for d in sorted(os.listdir(hog_dir)):
        if not os.path.isdir(os.path.join(hog_dir, d)):
            continue
        # 匹配: YYMMDD_SNN_obj_XX_grasp_YY
        match = re.match(r'^\d+_S\d+_obj_(\d+)_grasp_\d+$', d)
        if not match:
            continue

        obj_id = match.group(1)
        seq_dir = os.path.join(hog_dir, d)

        # 遍历所有 trial 和相机
        json_files = []
        for trial_dir in sorted(glob.glob(os.path.join(seq_dir, 'trial_*'))):
            anno_dir = os.path.join(trial_dir, 'annotation')
            if not os.path.isdir(anno_dir):
                continue
            for cam_dir in sorted(glob.glob(os.path.join(anno_dir, '*'))):
                cam = os.path.basename(cam_dir)
                # 只用 mas (主相机) 避免重复
                if cam != 'mas':
                    continue
                for jf in sorted(glob.glob(os.path.join(cam_dir, '*.json'))):
                    json_files.append((cam, jf))

        if json_files:
            sequences[obj_id].append((d, json_files))

    return dict(sequences)


def process_object_hog(hog_dir, obj_id, seq_list, num_points, use_mano,
                       mano_layer=None, torch_module=None, max_frames=50):
    """处理一个物体的所有序列, 生成 human_prior.

    Args:
        hog_dir: data root
        obj_id: "16" etc
        seq_list: [(seq_name, [(cam, json_path), ...]), ...]
        num_points: 采样点数
        use_mano: bool
        mano_layer: optional ManoLayer
        torch_module: optional torch
        max_frames: 每个序列最多取的帧数

    Returns:
        (mesh, points, normals, labels, force_center, n_frames) or None
    """
    # 加载物体 mesh
    obj_mesh = load_object_mesh(hog_dir, obj_id)
    if obj_mesh is None:
        return None

    # 收集所有接触点 (物体坐标系)
    all_contact_obj = []
    total_frames = 0

    for seq_name, json_files in seq_list:
        # 限制每个序列的帧数 (取均匀间隔)
        if len(json_files) > max_frames:
            indices = np.linspace(0, len(json_files) - 1, max_frames, dtype=int)
            json_files = [json_files[i] for i in indices]

        for cam, jp in json_files:
            try:
                with open(jp, encoding='utf-8-sig') as f:
                    anno = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # 获取物体位姿
            obj_info = anno.get('object', {})
            obj_pose = obj_info.get('6D_pose_per_cam') or obj_info.get('pose_data')
            if not obj_pose:
                continue

            # 获取手部接触点 (世界系)
            if use_mano and mano_layer is not None:
                contact_pts = get_hand_contact_points_mano(anno, mano_layer, torch_module)
            else:
                contact_pts = get_hand_contact_points_joints(anno)

            if contact_pts is None or len(contact_pts) == 0:
                continue

            # 变换到物体坐标系
            obj_mat = np.array(obj_pose).reshape(4, 4)
            contact_obj = transform_points_to_object_frame(contact_pts, obj_mat)
            all_contact_obj.append(contact_obj)
            total_frames += 1

    if not all_contact_obj or total_frames == 0:
        return None

    all_contact_obj = np.vstack(all_contact_obj)

    # 表面采样
    points, face_idx = trimesh.sample.sample_surface(obj_mesh, num_points)
    points = np.array(points, dtype=np.float32)
    normals = np.array(obj_mesh.face_normals[face_idx], dtype=np.float32)

    # KDTree: 找每个采样点最近的接触点
    tree = cKDTree(all_contact_obj)
    dists, _ = tree.query(points)
    labels = (dists < OBJ_CONTACT_RADIUS).astype(np.float32)

    # 如果接触比例太低, 尝试放大半径
    contact_ratio = labels.mean()
    if contact_ratio < 0.01 and len(all_contact_obj) > 10:
        for radius in [3.0, 5.0, 8.0]:
            labels_try = (dists < radius).astype(np.float32)
            if labels_try.mean() > 0.02:
                labels = labels_try
                break

    contact_mask = labels > 0.5
    if contact_mask.sum() == 0:
        return None

    force_center = points[contact_mask].mean(axis=0).astype(np.float32)

    # 转为米 (HOGraspNet 数据是 cm)
    points_m = points * CM_TO_M
    normals_m = normals  # 法线方向不受缩放影响
    force_center_m = force_center * CM_TO_M

    return obj_mesh, points_m, normals_m, labels, force_center_m, total_frames


def main():
    parser = argparse.ArgumentParser(
        description="HOGraspNet → V2AP human prior 转换")
    parser.add_argument("--hog_dir", type=str,
                        default=os.environ.get("HOGRASPNET_DIR", ""),
                        help="HOGraspNet data 目录")
    parser.add_argument("--num_points", type=int, default=config.NUM_POINTS)
    parser.add_argument("--use_mano", action="store_true", default=False,
                        help="使用 MANO 重建 (需要 manopth)")
    parser.add_argument("--max_frames", type=int, default=30,
                        help="每个序列最多处理的帧数")
    args = parser.parse_args()

    if not os.path.isdir(args.hog_dir):
        print(f"❌ HOGraspNet 目录未找到: {args.hog_dir}")
        return

    config.ensure_dirs()
    mesh_out_dir = os.path.join(config.DATA_HUB, "meshes", "hograspnet")
    os.makedirs(mesh_out_dir, exist_ok=True)

    # MANO 检查
    mano_layer, torch_module = None, None
    if args.use_mano:
        ManoLayer, torch_module = try_import_mano()
        if ManoLayer:
            mano_layer = ManoLayer(
                mano_root=os.path.join(os.path.dirname(__file__), '..', 'thirdparty', 'mano'),
                use_pca=False, ncomps=45, flat_hand_mean=False
            )
            print("  ✅ MANO loaded")
        else:
            print("  ⚠️ manopth 未安装, 使用 joint-based 方法")
            args.use_mano = False

    print("=" * 60)
    print("HOGraspNet → V2AP Human Prior")
    print("=" * 60)
    print(f"  Data dir:      {args.hog_dir}")
    print(f"  Output:        {config.HUMAN_PRIOR_DIR}")
    print(f"  Mesh copy:     {mesh_out_dir}")
    print(f"  Points:        {args.num_points}")
    print(f"  Method:        {'MANO' if args.use_mano else 'Joint-based'}")
    print(f"  Contact thr:   {CONTACT_THRESHOLD}")
    print(f"  Max frames:    {args.max_frames}")
    sys.stdout.flush()

    # Step 1: 发现所有序列
    sequences = discover_sequences(args.hog_dir)
    total_seqs = sum(len(v) for v in sequences.values())
    print(f"\n  Found {len(sequences)} objects, {total_seqs} sequences")
    sys.stdout.flush()

    # Step 2: 处理
    success = 0
    skipped = 0
    registry_updates = {}
    total_start = time.time()

    for obj_id in sorted(sequences.keys()):
        seq_list = sequences[obj_id]

        # 构建 obj_name
        obj_mesh_test = load_object_mesh(args.hog_dir, obj_id)
        if obj_mesh_test is None:
            print(f"  ⚠️ obj_{obj_id}: mesh not found, skip")
            skipped += 1
            continue

        # 查找物体名
        obj_name = None
        for d in os.listdir(args.hog_dir):
            m = re.match(r'^(\d+)_(.+)$', d)
            if m and int(m.group(1)) == int(obj_id):
                obj_name = m.group(2)
                break

        hog_id = f"hog_{obj_id}_{obj_name}" if obj_name else f"hog_{obj_id}"

        result = process_object_hog(
            args.hog_dir, obj_id, seq_list, args.num_points,
            args.use_mano, mano_layer, torch_module, args.max_frames
        )

        if result is None:
            print(f"  ⚠️ {hog_id}: no valid contacts, skip")
            skipped += 1
            continue

        mesh, points, normals, labels, force_center, n_frames = result
        n_contact = int(labels.sum())
        ratio = n_contact / args.num_points * 100

        # 保存 HDF5
        out_path = os.path.join(config.HUMAN_PRIOR_DIR, f"{hog_id}.hdf5")
        with h5py.File(out_path, 'w') as f:
            f.create_dataset("point_cloud", data=points, compression="gzip")
            f.create_dataset("normals", data=normals, compression="gzip")
            f.create_dataset("human_prior", data=labels)
            f.create_dataset("force_center", data=force_center)
            f.attrs["obj_id"] = hog_id
            f.attrs["source"] = "hograspnet"
            f.attrs["object_name"] = obj_name or obj_id
            f.attrs["n_frames"] = n_frames
            f.attrs["n_sequences"] = len(seq_list)

        # 复制 mesh (转为米)
        mesh_dst = os.path.join(mesh_out_dir, f"{hog_id}.obj")
        if not os.path.exists(mesh_dst):
            mesh_m = mesh.copy()
            mesh_m.vertices *= CM_TO_M
            mesh_m.export(mesh_dst)

        # Registry
        registry_updates[hog_id] = {
            "source": "hograspnet",
            "format": "obj",
            "category": obj_name or obj_id,
            "mesh_path": f"meshes/hograspnet/{hog_id}.obj",
        }

        success += 1
        print(f"  ✅ {hog_id}: {n_contact}/{args.num_points} ({ratio:.1f}%), "
              f"{n_frames} frames, {len(seq_list)} seqs")
        sys.stdout.flush()

    # Step 3: 更新 registry
    if registry_updates:
        registry_path = config.REGISTRY_PATH
        if os.path.exists(registry_path):
            with open(registry_path) as f:
                registry = json.load(f)
        else:
            registry = {}

        n_new = sum(1 for k in registry_updates if k not in registry)
        registry.update(registry_updates)

        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2, sort_keys=True)
        print(f"\n  📝 Registry: {n_new} new, {len(registry_updates) - n_new} updated")

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"DONE in {total_time:.1f}s")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Human Prior: {config.HUMAN_PRIOR_DIR}")
    print(f"  Meshes:      {mesh_out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
