#!/usr/bin/env python3
"""
OakInk2 → V2AP Human Prior 转换脚本

从 OakInk2 数据集中提取:
1. affordance_part 子 mesh → 标记物体表面的可抓取区域 
2. object_repair 完整 mesh → 采样点云

OakInk2 数据结构:
    object_repair/align_ds/{obj_id}/model.obj — 完整物体 mesh
    object_affordance/affordance_part/{part_id}/model.obj — 功能部件子 mesh
    
    命名规则: C02@0010@00001 (affordance part), O02@0010@00003 (full object)
              中间的 @XXXX@ 是物体类别编号

方案: affordance part 的顶点 → KDTree 投影到完整物体表面 → 标记接触区域

依赖: trimesh, h5py, numpy, scipy

用法:
    # Run from project root
    python data/extract_oakink2.py
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
PART_CONTACT_RADIUS = 0.005  # affordance part 投影到完整 mesh 的距离 (m)


def find_object_mapping(oakink2_dir):
    """建立 affordance part → full object 的映射.
    
    affordance_part: C02@0010@00001, C02@0010@00002
    object_repair:   O02@0010@00003
    
    通过中间的类别编号 @XXXX@ 来匹配.
    
    Returns:
        mapping: dict { full_obj_id: [part_id, ...] }
        obj_mesh_paths: dict { full_obj_id: mesh_path }
        part_mesh_paths: dict { part_id: mesh_path }
    """
    repair_dir = os.path.join(oakink2_dir, "object_repair", "align_ds")
    aff_dir = os.path.join(oakink2_dir, "object_affordance", "affordance_part")
    
    if not os.path.isdir(repair_dir) or not os.path.isdir(aff_dir):
        return {}, {}, {}
    
    # 收集所有物体 mesh
    obj_mesh_paths = {}
    obj_categories = {}  # category_key → [obj_id, ...]
    for d in sorted(os.listdir(repair_dir)):
        mesh_path = os.path.join(repair_dir, d, "model.obj")
        if os.path.isfile(mesh_path):
            obj_mesh_paths[d] = mesh_path
            # 提取类别 key: O02@0010@00003 → "0010"
            parts = d.split("@")
            if len(parts) >= 2:
                cat_key = parts[1]
            else:
                cat_key = d
            obj_categories.setdefault(cat_key, []).append(d)
    
    # 收集所有 affordance parts
    part_mesh_paths = {}
    part_categories = {}  # category_key → [part_id, ...]
    for d in sorted(os.listdir(aff_dir)):
        mesh_path = os.path.join(aff_dir, d, "model.obj")
        if os.path.isfile(mesh_path):
            part_mesh_paths[d] = mesh_path
            parts = d.split("@")
            if len(parts) >= 2:
                cat_key = parts[1]
            else:
                cat_key = d
            part_categories.setdefault(cat_key, []).append(d)
    
    # 匹配: 通过类别 key
    mapping = {}
    for cat_key, part_ids in part_categories.items():
        obj_ids = obj_categories.get(cat_key, [])
        if not obj_ids:
            continue
        # 每个完整物体都关联所有同类别的 affordance parts
        for obj_id in obj_ids:
            mapping[obj_id] = part_ids
    
    return mapping, obj_mesh_paths, part_mesh_paths


def process_object_oakink2(full_mesh_path, part_mesh_paths_list, num_points):
    """处理一个物体: 加载完整 mesh + 所有部件 mesh, 生成 human prior.
    
    Args:
        full_mesh_path: 完整物体 mesh 路径
        part_mesh_paths_list: [part_mesh_path, ...] 部件 mesh 路径列表
        num_points: 采样点数
        
    Returns:
        (mesh, points, normals, labels, force_center) or None
    """
    # 加载完整物体 mesh
    try:
        full_mesh = trimesh.load(full_mesh_path, force='mesh', process=False)
    except Exception as e:
        print(f"    ❌ 加载 mesh 失败: {e}")
        return None
    
    if len(full_mesh.vertices) < 10:
        return None
    
    # 合并所有部件 mesh 的顶点作为接触区域
    all_part_verts = []
    for pp in part_mesh_paths_list:
        try:
            part_mesh = trimesh.load(pp, force='mesh', process=False)
            all_part_verts.append(part_mesh.vertices)
        except Exception:
            continue
    
    if not all_part_verts:
        return None
    
    all_part_verts = np.vstack(all_part_verts)
    
    # 表面采样
    points, face_idx = trimesh.sample.sample_surface(full_mesh, num_points)
    points = np.array(points, dtype=np.float32)
    normals = np.array(full_mesh.face_normals[face_idx], dtype=np.float32)
    
    # KDTree: 找每个采样点最近的部件顶点
    tree = cKDTree(all_part_verts)
    dists, _ = tree.query(points)
    labels = (dists < PART_CONTACT_RADIUS).astype(np.float32)
    
    # 如果接触比例太低, 尝试放大半径
    contact_ratio = labels.mean()
    if contact_ratio < 0.02 and len(all_part_verts) > 10:
        for radius in [0.008, 0.01, 0.02, 0.03]:
            labels_try = (dists < radius).astype(np.float32)
            if labels_try.mean() > 0.02:
                labels = labels_try
                break
    
    contact_mask = labels > 0.5
    if contact_mask.sum() == 0:
        return None
    
    force_center = points[contact_mask].mean(axis=0).astype(np.float32)
    
    return full_mesh, points, normals, labels, force_center


def main():
    parser = argparse.ArgumentParser(
        description="OakInk2 → V2AP human prior 转换")
    parser.add_argument("--oakink2_dir", type=str,
                        default=os.environ.get("OAKINK2_DIR", ""),
                        help="OakInk2 hub 目录")
    parser.add_argument("--num_points", type=int, default=config.NUM_POINTS)
    args = parser.parse_args()
    
    if not os.path.isdir(args.oakink2_dir):
        print(f"❌ OakInk2 目录未找到: {args.oakink2_dir}")
        return
    
    config.ensure_dirs()
    mesh_out_dir = os.path.join(config.DATA_HUB, "meshes", "oakink2")
    os.makedirs(mesh_out_dir, exist_ok=True)
    
    print("=" * 60)
    print("OakInk2 → V2AP Human Prior")
    print("=" * 60)
    print(f"  Data dir:      {args.oakink2_dir}")
    print(f"  Output:        {config.HUMAN_PRIOR_DIR}")
    print(f"  Mesh copy:     {mesh_out_dir}")
    print(f"  Points:        {args.num_points}")
    print(f"  Contact radius: {PART_CONTACT_RADIUS}")
    sys.stdout.flush()
    
    # Step 1: 建立映射
    mapping, obj_paths, part_paths = find_object_mapping(args.oakink2_dir)
    
    # 也处理没有 affordance part 的独立物体
    # 一些物体 ID 以 C 开头 (C10001 etc.) 在 repair 中
    standalone_objects = {}
    for obj_id in obj_paths:
        if obj_id not in mapping:
            # 检查是否有对应的 affordance part (可能命名不同)
            standalone_objects[obj_id] = obj_paths[obj_id]
    
    total_objects = len(mapping)
    total_parts = sum(len(v) for v in mapping.values())
    print(f"\n  Objects with affordance parts: {total_objects}")
    print(f"  Total affordance parts: {total_parts}")
    print(f"  Standalone objects (no parts): {len(standalone_objects)}")
    sys.stdout.flush()
    
    # Step 2: 处理有 affordance part 的物体
    success = 0
    skipped = 0
    registry_updates = {}
    total_start = time.time()
    
    for obj_id in sorted(mapping.keys()):
        part_ids = mapping[obj_id]
        part_mesh_list = [part_paths[pid] for pid in part_ids if pid in part_paths]
        
        if not part_mesh_list:
            skipped += 1
            continue
        
        oi2_id = f"oi2_{obj_id}"
        
        result = process_object_oakink2(
            obj_paths[obj_id], part_mesh_list, args.num_points
        )
        
        if result is None:
            print(f"  ⚠️ {oi2_id}: no valid contacts, skip")
            skipped += 1
            continue
        
        mesh, points, normals, labels, force_center = result
        n_contact = int(labels.sum())
        ratio = n_contact / args.num_points * 100
        
        # 保存 HDF5
        out_path = os.path.join(config.HUMAN_PRIOR_DIR, f"{oi2_id}.hdf5")
        with h5py.File(out_path, 'w') as f:
            f.create_dataset("point_cloud", data=points, compression="gzip")
            f.create_dataset("normals", data=normals, compression="gzip")
            f.create_dataset("human_prior", data=labels)
            f.create_dataset("force_center", data=force_center)
            f.attrs["obj_id"] = oi2_id
            f.attrs["source"] = "oakink2"
            f.attrs["n_parts"] = len(part_ids)
        
        # 复制 mesh
        mesh_dst = os.path.join(mesh_out_dir, f"{oi2_id}.obj")
        if not os.path.exists(mesh_dst):
            mesh.export(mesh_dst)
        
        registry_updates[oi2_id] = {
            "source": "oakink2",
            "format": "obj",
            "category": obj_id,
            "mesh_path": f"meshes/oakink2/{oi2_id}.obj",
        }
        
        success += 1
        print(f"  ✅ {oi2_id}: {n_contact}/{args.num_points} ({ratio:.1f}%), "
              f"{len(part_ids)} parts")
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
