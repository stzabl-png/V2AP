#!/usr/bin/env python3
"""
export_arctic_prior.py — 将 ARCTIC 接触检测结果转为 human_prior HDF5

上游: output/affordance_batch/s05_{obj}_grab_01/vert_contact_count.npy
下游: data_hub/human_prior/{obj}.hdf5  (与 OakInk 格式完全一致)

格式:
    point_cloud:   (N, 3) float32  # 物体 mesh 采样点 (object local frame, 单位: m)
    normals:       (N, 3) float32
    human_prior:   (N,)  float32   # [0, 1] 归一化接触频率

用法:
    conda activate hawor
    python tools/export_arctic_prior.py
    python tools/export_arctic_prior.py --obj box      # 单个物体
    python tools/export_arctic_prior.py --n_points 2048  # 调整采样数
"""
import os, sys, argparse
import numpy as np
import trimesh
import h5py
from scipy.spatial import cKDTree

PROJ   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config as _cfg
ARCTIC = _cfg.ARCTIC_ROOT
AFF    = os.path.join(PROJ, 'output', 'affordance_batch')
HP_DIR = os.path.join(PROJ, 'data_hub', 'human_prior')
os.makedirs(HP_DIR, exist_ok=True)

OBJS = ('box capsulemachine espressomachine ketchup microwave '
        'mixer notebook phone scissors waffleiron').split()


def export_one(obj, n_points=1024, force=False):
    out_path = os.path.join(HP_DIR, f'{obj}.hdf5')
    if os.path.exists(out_path) and not force:
        print(f'  ⏭  {obj}: already exists, skipping')
        return False

    # ── 找 contact count ──────────────────────────────────────
    cnt_path = os.path.join(AFF, f's05_{obj}_grab_01', 'vert_contact_count.npy')
    if not os.path.exists(cnt_path):
        print(f'  ❌ {obj}: vert_contact_count.npy not found at {cnt_path}')
        return False

    cnt = np.load(cnt_path)          # (V,) per-vertex contact frequency

    # ── 加载 ARCTIC GT mesh (同 batch_contact.py 一致) ────────
    mesh_path = os.path.join(ARCTIC, 'meta', 'object_vtemplates', obj, 'mesh_tex.obj')
    if not os.path.exists(mesh_path):
        print(f'  ❌ {obj}: mesh not found at {mesh_path}')
        return False

    mesh = trimesh.load(mesh_path, force='mesh')
    mesh.vertices /= 1000.0          # mm → m  (与 batch_contact.py 一致)

    if len(cnt) != len(mesh.vertices):
        print(f'  ⚠  {obj}: cnt({len(cnt)}) ≠ verts({len(mesh.vertices)}), skipping')
        return False

    # ── Gaussian diffusion: 扩散接触信号 ────────────────────
    # raw cnt 只在被接触顶点上有值，需要扩散到邻近区域
    # sigma=8mm, radius=3cm (比 batch_contact 的可视化参数更宽松)
    sigma  = 0.008   # 8mm
    radius = 0.030   # 3cm
    contact_mask  = cnt > 0
    contact_verts = mesh.vertices[contact_mask]
    contact_scores = cnt[contact_mask]
    diffused = np.zeros(len(mesh.vertices), dtype=np.float32)

    if contact_verts.shape[0] > 0:
        tree_v = cKDTree(mesh.vertices)
        for cv, cs in zip(contact_verts, contact_scores):
            idxs = tree_v.query_ball_point(cv, radius)
            dists = np.linalg.norm(mesh.vertices[idxs] - cv, axis=1)
            diffused[idxs] += cs * np.exp(-0.5 * (dists / sigma) ** 2)
        diffused /= (diffused.max() + 1e-8)
    else:
        diffused[:] = 0.0   # box: 无接触 → 全零 → 下游 100% 随机

    prior_verts = diffused
    coverage = float(np.mean(prior_verts > 0.1))

    # ── 采样 N 个点 (与 OakInk 训练格式一致) ─────────────────
    pts, face_idx = mesh.sample(n_points, return_index=True)
    norms         = mesh.face_normals[face_idx]
    pts           = pts.astype(np.float32)
    norms         = norms.astype(np.float32)

    # ── KNN: vertex prior → sampled points ───────────────────
    tree = cKDTree(mesh.vertices)
    _, idx = tree.query(pts)
    hp = prior_verts[idx].astype(np.float32)

    # ── 保存 HDF5 ─────────────────────────────────────────────
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('point_cloud', data=pts)
        f.create_dataset('normals',     data=norms)
        f.create_dataset('human_prior', data=hp)
        f.attrs['source']   = 'arctic_haptic'
        f.attrs['object']   = obj
        f.attrs['n_points'] = n_points
        f.attrs['coverage'] = coverage

    print(f'  ✓ {obj}: coverage={coverage:.1%}  '
          f'max_prior={float(hp.max()):.3f}  → {os.path.basename(out_path)}')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj',       default=None,  help='单个物体名')
    ap.add_argument('--n_points',  default=1024,  type=int)
    ap.add_argument('--force',     action='store_true', help='覆盖已有文件')
    args = ap.parse_args()

    objs = [args.obj] if args.obj else OBJS

    print('=' * 55)
    print('  ARCTIC Prior → data_hub/human_prior/{obj}.hdf5')
    print('=' * 55)

    ok = 0
    for obj in objs:
        if export_one(obj, args.n_points, args.force):
            ok += 1

    print(f'\n  Done: {ok}/{len(objs)} exported to {HP_DIR}')
    print('  Next: python3 tools/random_grasp_sampler.py --all')


if __name__ == '__main__':
    main()
