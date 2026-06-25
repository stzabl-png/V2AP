#!/usr/bin/env python3
"""
3D AffordanceNet → V2AP Human Prior 转换脚本

从 3D AffordanceNet 的 PKL 数据 (ShapeNet 物体, 18 种 affordance) 中提取
grasp/wrap_grasp 标签, 转换为 V2AP 的 human_prior HDF5 格式。

3D AffordanceNet PKL 格式 (list of dicts):
    {
        "shape_id": "xxxx",
        "semantic class": "Mug",
        "affordance": ["grasp", "contain", ...],
        "full_shape": {
            "coordinate": (N, 3) float,       # 点云 xyz
            "label": {
                "grasp": (N,) float [0,1],    # 逐点 affordance
                "wrap_grasp": (N,) float [0,1],
                ...
            }
        }
    }

输出格式 (与现有 human_prior 一致):
    point_cloud:  (N, 3)  float32
    normals:      (N, 3)  float32 (从点云估计)
    human_prior:  (N,)    float32 (0/1)
    force_center: (3,)    float32

用法:
    # Run from project root
    python data/extract_affordancenet.py
    python data/extract_affordancenet.py --affordance_types grasp,wrap_grasp,lift
    python data/extract_affordancenet.py --max_objects 100
"""

import os
import sys
import json
import pickle
import argparse
import time
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


# ============================================================
# 3D AffordanceNet 23 个物体类别 → V2AP 类别映射
# ============================================================
CATEGORY_MAP = {
    "Bag": "bag",
    "Bed": "bed",
    "Bowl": "bowl",
    "Clock": "clock",
    "Dishwasher": "dishwasher",
    "Display": "display",
    "Door": "door",
    "Earphone": "earphone",
    "Faucet": "faucet",
    "Hat": "hat",
    "StorageFurniture": "storage",
    "Keyboard": "keyboard",
    "Knife": "knife",
    "Laptop": "laptop",
    "Microwave": "microwave",
    "Mug": "mug",
    "Refrigerator": "refrigerator",
    "Chair": "chair",
    "Scissors": "scissors",
    "Table": "table",
    "TrashCan": "trashcan",
    "Vase": "vase",
    "Bottle": "bottle",
}

# 与抓取相关的 affordance 类型
GRASP_AFFORDANCES = ["grasp", "wrap_grasp"]


# ============================================================
# 点云法线估计 (简化版: 基于 KNN)
# ============================================================

def estimate_normals_knn(points, k=10):
    """用 KNN 估计点云法线.

    Args:
        points: (N, 3) 点云
        k: 近邻数

    Returns:
        normals: (N, 3) 法线
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)

    normals = np.zeros_like(points)
    for i in range(len(points)):
        neighbors = points[idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        try:
            cov = centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]  # 最小特征值对应的特征向量
        except np.linalg.LinAlgError:
            normals[i] = [0, 0, 1]

    # 朝外翻转 (假设质心在内部)
    centroid = points.mean(axis=0)
    outward = points - centroid
    flip = np.sum(normals * outward, axis=1) < 0
    normals[flip] *= -1

    return normals.astype(np.float32)


def resample_to_n_points(points, labels, n_points):
    """重采样到指定点数.

    Args:
        points: (M, 3) 原始点云
        labels: (M,) 标签
        n_points: 目标点数

    Returns:
        new_points: (n_points, 3)
        new_labels: (n_points,)
    """
    M = len(points)
    if M == n_points:
        return points, labels
    elif M > n_points:
        idx = np.random.choice(M, n_points, replace=False)
    else:
        idx = np.random.choice(M, n_points, replace=True)
    return points[idx], labels[idx]


def process_shape(shape_data, affordance_types, contact_threshold, n_points):
    """处理单个 shape.

    Args:
        shape_data: dict from PKL
        affordance_types: list of affordance types to use as contact
        contact_threshold: threshold for affordance value → contact label
        n_points: target number of points

    Returns:
        points, normals, labels, force_center or None if no contact
    """
    full_shape = shape_data["full_shape"]
    coords = full_shape["coordinate"].astype(np.float32)  # (N, 3)
    label_dict = full_shape["label"]

    # 合并多种 affordance: max(grasp, wrap_grasp, ...)
    combined = np.zeros(len(coords), dtype=np.float32)
    for aff_type in affordance_types:
        if aff_type in label_dict:
            aff_val = label_dict[aff_type].astype(np.float32).ravel()
            combined = np.maximum(combined, aff_val)

    # 二值化
    binary_labels = (combined > contact_threshold).astype(np.float32)

    # 跳过完全没有接触的
    if binary_labels.sum() == 0:
        return None

    # 重采样
    points, labels = resample_to_n_points(coords, binary_labels, n_points)

    # 估计法线 (简化: 用 KNN)
    normals = estimate_normals_knn(points, k=10)

    # Force center: 接触点均值
    contact_mask = labels > 0.5
    if contact_mask.sum() > 0:
        force_center = points[contact_mask].mean(axis=0).astype(np.float32)
    else:
        force_center = np.zeros(3, dtype=np.float32)

    return points, normals, labels, force_center


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="3D AffordanceNet → V2AP human prior 转换")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(config.PROJECT_DIR),
                                             "3D_AffordanceNet"),
                        help="3D AffordanceNet 数据目录 (含 full_shape_*.pkl)")
    parser.add_argument("--affordance_types", type=str,
                        default="grasp,wrap_grasp",
                        help="用哪些 affordance 做 contact label (逗号分隔)")
    parser.add_argument("--contact_threshold", type=float, default=0.5,
                        help="Affordance 值阈值 → contact label")
    parser.add_argument("--num_points", type=int, default=config.NUM_POINTS,
                        help="每个物体采样点数")
    parser.add_argument("--max_objects", type=int, default=None,
                        help="最多处理几个物体 (debug 用)")
    parser.add_argument("--update_registry", action="store_true", default=True,
                        help="更新 registry.json")
    args = parser.parse_args()

    affordance_types = args.affordance_types.split(",")

    config.ensure_dirs()

    print("=" * 60)
    print("3D AffordanceNet → V2AP Human Prior")
    print("=" * 60)
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Output:          {config.HUMAN_PRIOR_DIR}")
    print(f"  Affordance types: {affordance_types}")
    print(f"  Contact thresh:  {args.contact_threshold}")
    print(f"  Points:          {args.num_points}")
    if args.max_objects:
        print(f"  Max objects:     {args.max_objects}")
    sys.stdout.flush()

    # Load PKL files
    pkl_files = []
    for split in ["train", "val", "test"]:
        path = os.path.join(args.data_dir, f"full_shape_{split}_data.pkl")
        if os.path.exists(path):
            pkl_files.append((split, path))

    if not pkl_files:
        print(f"\n  ❌ No PKL files found in {args.data_dir}")
        return

    print(f"\n  Loading PKL files: {[s for s, _ in pkl_files]}")
    sys.stdout.flush()

    # 聚合同一 shape_id 的所有数据 (train+val+test 可能有重复)
    all_shapes = {}  # shape_id → shape_data
    total_loaded = 0

    for split, path in pkl_files:
        t0 = time.time()
        print(f"  Loading {split}...", end=" ", flush=True)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"{len(data)} shapes ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

        for item in data:
            sid = item["shape_id"]
            if sid not in all_shapes:
                all_shapes[sid] = item
                total_loaded += 1

    print(f"  Total unique shapes: {total_loaded}")
    sys.stdout.flush()

    # Process each shape
    success = 0
    skipped_no_contact = 0
    skipped_no_affordance = 0
    registry_updates = {}
    category_counts = {}

    total_start = time.time()
    shape_items = list(all_shapes.items())

    if args.max_objects:
        shape_items = shape_items[:args.max_objects]

    for i, (shape_id, shape_data) in enumerate(shape_items):
        sem_class = shape_data["semantic class"]
        category = CATEGORY_MAP.get(sem_class, sem_class.lower())
        obj_id = f"an_{shape_id}"  # an_ = AffordanceNet prefix

        # 检查是否有目标 affordance
        avail_aff = shape_data.get("affordance", [])
        has_target = any(a in avail_aff for a in affordance_types)
        if not has_target:
            skipped_no_affordance += 1
            continue

        # 处理
        result = process_shape(
            shape_data, affordance_types,
            args.contact_threshold, args.num_points)

        if result is None:
            skipped_no_contact += 1
            continue

        points, normals, labels, force_center = result
        n_contact = int(labels.sum())

        # 保存 HDF5
        out_path = os.path.join(config.HUMAN_PRIOR_DIR, f"{obj_id}.hdf5")
        with h5py.File(out_path, 'w') as f:
            f.create_dataset("point_cloud", data=points, compression="gzip")
            f.create_dataset("normals", data=normals, compression="gzip")
            f.create_dataset("human_prior", data=labels)
            f.create_dataset("force_center", data=force_center)
            f.attrs["obj_id"] = obj_id
            f.attrs["source"] = "affordancenet"
            f.attrs["category"] = category
            f.attrs["semantic_class"] = sem_class
            f.attrs["shape_id"] = shape_id
            f.attrs["num_points"] = args.num_points
            f.attrs["contact_threshold"] = args.contact_threshold
            f.attrs["affordance_types"] = ",".join(affordance_types)
            f.attrs["n_contact_verts"] = n_contact

        # Registry
        registry_updates[obj_id] = {
            "source": "affordancenet",
            "format": "pointcloud",
            "category": category,
            "mesh_path": f"_pointcloud_only_/{shape_id}",
        }

        # 统计
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1
        success += 1

        if (i + 1) % 2000 == 0 or i == len(shape_items) - 1:
            ratio = n_contact / args.num_points * 100
            print(f"  [{i+1}/{len(shape_items)}] {success} saved, "
                  f"last: {obj_id} ({sem_class}) {n_contact}/{args.num_points} ({ratio:.1f}%)")
            sys.stdout.flush()

    # 更新 registry
    if args.update_registry and registry_updates:
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

        print(f"\n  📝 Registry updated: {n_new} new, "
              f"{len(registry_updates) - n_new} updated")

    total_time = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"DONE in {total_time:.1f}s")
    print(f"  Success:              {success}")
    print(f"  Skipped (no contact): {skipped_no_contact}")
    print(f"  Skipped (no afford):  {skipped_no_affordance}")
    print(f"  Categories:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {cnt}")
    print(f"  Output: {config.HUMAN_PRIOR_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
