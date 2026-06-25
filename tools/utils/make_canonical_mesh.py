#!/usr/bin/env python3
"""
make_canonical_mesh.py — SAM3D Raw Mesh → Canonical Mesh (正确大小+正确朝向)
=============================================================================

流程:
  1. 加载 SAM3D raw mesh (data_hub/meshes/SAM3DMesh/{dataset}/{obj}/mesh.ply)
  2. × scale_factor (from scale.json) → 真实米制大小
  3. × canonical_rotation (from sim/canonical_rotation.json) → 正确朝向 (Z=向上)
  4. 平移: 底部贴 Z=0, XY 居中
  → 输出: data_hub/meshes/canonical/{dataset}/{obj}/mesh.ply

canonical_rotation 含义:
  - 旋转使物体在 Z-up 世界中自然竖立 (与 Isaac Sim 坐标系一致)
  - 由人工在 sim 中目视验证后记录

用法:
  python3 tools/make_canonical_mesh.py --dataset oakink --obj A01001 --force
  python3 tools/make_canonical_mesh.py --dataset oakink
  python3 tools/make_canonical_mesh.py --all
"""

import os, sys, json, argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pathlib import Path

PROJ          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM3D_ROOT    = os.path.join(PROJ, 'data_hub', 'meshes', 'SAM3DMesh', 'meshes')
OBJ_MESHES    = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
CANONICAL_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'canonical')
CANONICAL_ROT = os.path.join(PROJ, 'sim', 'canonical_rotation.json')
DATASETS      = ['oakink', 'ycb', 'egodex']


def load_canonical_rotations():
    if os.path.exists(CANONICAL_ROT):
        with open(CANONICAL_ROT) as f:
            d = json.load(f)
        return {k: v for k, v in d.items() if not k.startswith('_')}
    return {}


def process_one(dataset: str, obj_id: str, canonical_rotations: dict, force: bool = False):
    """
    处理单个物体:
      SAM3D raw → scale → canonical_rotation → center → canonical PLY
    """
    raw_path   = os.path.join(SAM3D_ROOT, dataset, obj_id, 'mesh.ply')
    scale_path = os.path.join(OBJ_MESHES, dataset, obj_id, 'scale.json')
    out_dir    = os.path.join(CANONICAL_DIR, dataset, obj_id)
    out_path   = os.path.join(out_dir, 'mesh.ply')

    if not os.path.exists(raw_path):
        print(f"  ⚠️  raw mesh not found: {raw_path}")
        return False
    if not os.path.exists(scale_path):
        print(f"  ⚠️  scale.json not found: {scale_path}")
        return False
    if os.path.exists(out_path) and not force:
        print(f"  ⏭  {obj_id}: already exists (use --force to redo)")
        return True

    # ── 读取 scale_factor ────────────────────────────────────────────
    with open(scale_path) as f:
        scale_info = json.load(f)
    scale_factor = float(scale_info['scale_factor'])

    # ── 加载 raw mesh ────────────────────────────────────────────────
    mesh = trimesh.load(raw_path, force='mesh')

    # ── Step 1: scale → metric (meters) ─────────────────────────────
    mesh.apply_scale(scale_factor)

    # ── Step 2: canonical rotation ───────────────────────────────────
    rot_euler = canonical_rotations.get(obj_id, None)
    if rot_euler is not None:
        R = Rotation.from_euler('xyz', rot_euler, degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        mesh.apply_transform(T)

    # ── Step 3: center — 底部贴 Z=0, XY 居中 ─────────────────────────
    z_min = mesh.bounds[0][2]
    cx    = (mesh.bounds[0][0] + mesh.bounds[1][0]) / 2
    cy    = (mesh.bounds[0][1] + mesh.bounds[1][1]) / 2
    mesh.vertices[:, 0] -= cx
    mesh.vertices[:, 1] -= cy
    mesh.vertices[:, 2] -= z_min

    # ── 保存 ─────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    mesh.export(out_path)

    bbox = mesh.bounding_box.extents
    rot_str = f", rot={rot_euler}" if rot_euler else ""
    print(f"  ✅ {obj_id}: {bbox[0]*100:.1f}×{bbox[1]*100:.1f}×{bbox[2]*100:.1f} cm"
          f"  (scale={scale_factor:.4f}{rot_str})")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--obj',     help='只处理指定物体 ID')
    parser.add_argument('--all',     action='store_true')
    parser.add_argument('--force',   action='store_true')
    args = parser.parse_args()

    canonical_rotations = load_canonical_rotations()
    print(f"已加载 canonical 旋转配置: {list(canonical_rotations.keys())}")

    datasets = DATASETS if args.all else ([args.dataset] if args.dataset else ['oakink'])

    total, ok = 0, 0
    for ds in datasets:
        ds_dir = os.path.join(SAM3D_ROOT, ds)
        if not os.path.isdir(ds_dir):
            print(f"⚠️  dataset dir not found: {ds_dir}")
            continue

        obj_ids = sorted(os.listdir(ds_dir))
        if args.obj:
            obj_ids = [o for o in obj_ids if args.obj in o]
        if not obj_ids:
            print(f"⚠️  no matching objects in {ds}")
            continue

        print(f"\n=== {ds} ({len(obj_ids)} 物体) ===")
        for obj_id in obj_ids:
            total += 1
            if process_one(ds, obj_id, canonical_rotations, args.force):
                ok += 1

    print(f"\n完成: {ok}/{total}")
    print(f"canonical meshes → {CANONICAL_DIR}")


if __name__ == '__main__':
    main()
