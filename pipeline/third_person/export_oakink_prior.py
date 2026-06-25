#!/usr/bin/env python3
"""
export_oakink_prior.py — OakInk 接触聚合 → Human Prior HDF5 (Step 5)
======================================================================
将同一 obj_id 下的所有序列接触数据（vert_contact_count.npy）聚合，
输出: data_hub/human_prior/oakink_{obj_id}.hdf5

格式与 aggregate_prior.py 产出的 HDF5 兼容：
  point_cloud  (N,3)  - 物体表面采样点
  normals      (N,3)  - 法线
  human_prior  (N,)   - 0/1 接触标签
  force_center (3,)   - 接触重心

用法:
  conda activate hawor
  cd $PROJ
  python data/export_oakink_prior.py
  python data/export_oakink_prior.py --obj_id A01001
"""
import os, sys, glob, json, argparse
import numpy as np

# NumPy 2.0 兼容补丁
if not hasattr(np.ndarray, 'ptp'):
    np.ndarray.ptp = lambda self, *a, **kw: np.ptp(self, *a, **kw)

import trimesh
import h5py
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# ── 路径配置 ─────────────────────────────────────────────────────────────────
ANNOT_JSON    = os.path.join(config.OUTPUT_DIR, 'oakink_keyframe_annot.json')
CONTACT_DIR   = os.path.join(config.OUTPUT_DIR, 'contact_oakink')
GT_MESH_DIR   = os.path.join(config.DATA_HUB, 'meshes', 'v1')
OUT_DIR       = os.path.join(config.DATA_HUB, 'human_prior')
os.makedirs(OUT_DIR, exist_ok=True)

NUM_POINTS     = config.NUM_POINTS      # 1024
CONTACT_RADIUS = config.CONTACT_RADIUS  # 0.005m (5mm)

ap = argparse.ArgumentParser()
ap.add_argument('--obj_id',         default=None, help='只处理单个 obj_id (不含 _0004 变体)')
ap.add_argument('--num_points',     default=NUM_POINTS,     type=int)
ap.add_argument('--contact_radius', default=CONTACT_RADIUS, type=float)
ap.add_argument('--force',          action='store_true')
ap.add_argument('--out-dir',        default=None, dest='out_dir',
                help='覆盖输出目录（默认: data_hub/human_prior）')
args = ap.parse_args()

if args.out_dir:
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)


def get_base_obj_id(obj_id):
    """把 A01001_0004 → A01001（GT mesh 不包含 _0004 变体）"""
    if obj_id in os.listdir(GT_MESH_DIR):
        return obj_id
    parts = obj_id.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return obj_id


def load_mesh(obj_id):
    """加载 GT mesh，返回 (trimesh, scale_factor)。"""
    p = os.path.join(GT_MESH_DIR, f'{obj_id}.obj')
    if not os.path.exists(p):
        return None
    mesh = trimesh.load(p, force='mesh', process=False)
    if mesh.vertices.max() > 1.0:
        mesh.vertices /= 1000.0  # mm → m
    return mesh


def aggregate_to_prior(mesh, all_vert_counts, num_points, contact_radius):
    """
    将多序列的 vert_contact_count 聚合，生成表面采样点和接触标签。

    vert_contact_count: (V,) 浮点数组，V = mesh.vertices 数量
    """
    # 将所有序列的接触计数对齐（同一 obj_id 同一 mesh，数量相同）
    combined = np.zeros(len(mesh.vertices), dtype=np.float32)
    for cnt in all_vert_counts:
        if len(cnt) == len(mesh.vertices):
            combined += cnt
        else:
            # 允许轻微不一致（降采样后 mesh 顶点数会变化）
            n = min(len(cnt), len(mesh.vertices))
            combined[:n] += cnt[:n]

    # 已接触的顶点坐标（作为 contact_pts 传给 KDTree）
    contact_mask  = combined > 0
    contact_verts = mesh.vertices[contact_mask]

    if len(contact_verts) == 0:
        return None

    # 均匀采样物体表面点
    points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    points  = np.array(points,                   dtype=np.float32)
    normals = np.array(mesh.face_normals[face_idx], dtype=np.float32)

    # 标注: 到已接触顶点距离 < contact_radius → 接触 (1)
    tree   = cKDTree(contact_verts)
    dists, _ = tree.query(points)
    labels = (dists < contact_radius).astype(np.float32)

    # 接触重心（force center 近似）
    force_center = contact_verts.mean(axis=0).astype(np.float32)

    return points, normals, labels, force_center


def main():
    annot = json.load(open(ANNOT_JSON))

    # 按 base obj_id 分组
    obj_to_seqs = {}
    for seq_id, ann in annot.items():
        raw_obj   = ann.get('obj_id', seq_id.rsplit('_', 2)[0])
        base_obj  = get_base_obj_id(raw_obj)
        obj_to_seqs.setdefault(base_obj, []).append(seq_id)

    if args.obj_id:
        obj_ids = [args.obj_id]
    else:
        obj_ids = sorted(obj_to_seqs.keys())

    print(f'待聚合: {len(obj_ids)} 个 obj_id\n')
    success = skip = error = 0

    for i, obj_id in enumerate(obj_ids):
        out_f = os.path.join(OUT_DIR, f'oakink_{obj_id}.hdf5')
        if os.path.exists(out_f) and not args.force:
            skip += 1
            continue

        # 加载 mesh
        mesh = load_mesh(obj_id)
        if mesh is None:
            print(f'  [{i+1}/{len(obj_ids)}] {obj_id}: ❌ mesh 未找到')
            error += 1
            continue

        # 收集所有序列的 vert_contact_count
        all_counts = []
        for seq_id in obj_to_seqs.get(obj_id, []):
            cnt_path = os.path.join(CONTACT_DIR, seq_id, 'vert_contact_count.npy')
            if os.path.exists(cnt_path):
                all_counts.append(np.load(cnt_path))

        if not all_counts:
            skip += 1
            continue

        result = aggregate_to_prior(mesh, all_counts, args.num_points, args.contact_radius)
        if result is None:
            print(f'  [{i+1}/{len(obj_ids)}] {obj_id}: ⚠️ 无接触数据 ({len(all_counts)} 序列)')
            skip += 1
            continue

        points, normals, labels, force_center = result
        n_contact = int(labels.sum())

        with h5py.File(out_f, 'w') as h:
            h.create_dataset('point_cloud',  data=points,       compression='gzip')
            h.create_dataset('normals',      data=normals,       compression='gzip')
            h.create_dataset('human_prior',  data=labels)
            h.create_dataset('force_center', data=force_center)
            h.attrs['obj_id']            = obj_id
            h.attrs['source']            = 'oakink'
            h.attrs['num_points']        = args.num_points
            h.attrs['contact_radius']    = args.contact_radius
            h.attrs['n_contact_pts']     = n_contact
            h.attrs['n_seqs']            = len(all_counts)

        print(f'  [{i+1}/{len(obj_ids)}] {obj_id}: ✅ '
              f'{n_contact}/{args.num_points} 接触  '
              f'({len(all_counts)} 序列)')
        success += 1

    print(f'\n{"="*60}')
    print(f'  成功:{success}  跳过:{skip}  失败:{error}')
    print(f'  输出: {OUT_DIR}')


if __name__ == '__main__':
    main()
