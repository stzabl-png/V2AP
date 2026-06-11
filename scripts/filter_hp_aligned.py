#!/usr/bin/env python3
"""
filter_hp_aligned.py
对 RobotPosteriorTotal_combined 里的每条抓取做 HP 对齐过滤。

判定逻辑:
  1. 从 grasp_point + finger_dir + width 估算两个接触点
  2. 若该物体有 canonical rotation，将接触点旋转回 HP 坐标系
  3. 在 HP 点云里找最近点，查其 human_prior label
  4. 至少一个接触点的最近 HP 点 label >= HP_THRESH → hp_match=True

输出:
  output/RobotPosteriorFiltered/
    {obj_id}_robot_gt.hdf5   — 原始结构 + 每条抓取新增 attrs: hp_match, hp_score
    filter_summary.csv       — 物体级别统计
"""
import os, sys, glob, json, h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED_DIR = os.path.join(PROJ, 'output', 'RobotPosteriorTotal_combined')
HP_DIRS = {
    'oakink': os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp', 'oakink'),
    'dexycb': os.path.join(PROJ, 'data_hub', 'ProcessedData', 'training_fp', 'dexycb'),
}
CAN_ROT_JSON = os.path.join(PROJ, 'sim', 'canonical_rotation.json')
OUT_DIR = os.path.join(PROJ, 'output', 'RobotPosteriorFiltered')
os.makedirs(OUT_DIR, exist_ok=True)

HP_THRESH = 0.3   # HP label 阈值，>= 此值视为 HP 区域

# CANONICAL_ROTS 已不需要：删除 rotation.json 后，grasp candidates 和 HP
# 都在 SAM3D 原始坐标系，无需坐标系变换。


def load_hp(obj_id):
    ds = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
    fp = os.path.join(HP_DIRS[ds], f'{obj_id}.hdf5')
    if not os.path.exists(fp):
        return None, None
    with h5py.File(fp, 'r') as f:
        pc = f['point_cloud'][()].astype(np.float32)
        hp = f['human_prior'][()].astype(np.float32)
    return pc, hp


def get_scale_factor(obj_id):
    """Read scale_factor from scale.json (mesh file units → real meters)."""
    ds = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
    MESH_BASE = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')
    p = os.path.join(MESH_BASE, ds, obj_id, 'scale.json')
    if os.path.exists(p):
        import json as _json
        return float(_json.load(open(p)).get('scale_factor', 1.0))
    return 1.0


def contact_points(grasp_point, finger_dir, width):
    """从抓取中心估算两个接触点"""
    fd = np.asarray(finger_dir, float)
    fd = fd / (np.linalg.norm(fd) + 1e-8)
    hw = float(width) / 2.0
    c_L = np.asarray(grasp_point, float) + fd * hw
    c_R = np.asarray(grasp_point, float) - fd * hw
    return c_L, c_R


def rotate_to_hp_frame(points, obj_id):
    """SAM3D 原始坐标系已统一，不再需要旋转变换。保留此函数仅展示对照。"""
    return points   # no-op

def get_mesh_scale(obj_id):
    """Load canonical mesh and return (center, max_extent) in GT/sim meter space."""
    if HAS_MESH_UTILS:
        ds = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
        ply, ds2 = find_ply(obj_id, ds)
        if ply:
            try:
                mesh = load_mesh_canonical(obj_id, ds2, verbose=False)
                center = mesh.centroid.astype(np.float64)
                ext    = mesh.bounds[1] - mesh.bounds[0]
                return center, float(np.linalg.norm(ext))
            except Exception:
                pass
    return None, None


def scale_contacts_to_hp(c_L, c_R, obj_id, hp_pc):
    """
    Scale contact points from GT meter space → HP normalized space.
    hp_pc: HP point cloud (used to derive hp center+scale)
    """
    # HP space stats
    hp_center = hp_pc.mean(axis=0).astype(np.float64)
    hp_ext    = hp_pc.max(axis=0) - hp_pc.min(axis=0)
    hp_scale  = float(np.linalg.norm(hp_ext))

    # GT/mesh space stats
    mesh_center, mesh_scale = get_mesh_scale(obj_id)
    if mesh_center is None or mesh_scale < 1e-6:
        # Fallback: assume contact points are already in HP space
        return c_L, c_R

    sf = hp_scale / mesh_scale   # scale factor GT→HP

    c_L_hp = (np.asarray(c_L, float) - mesh_center) * sf + hp_center
    c_R_hp = (np.asarray(c_R, float) - mesh_center) * sf + hp_center
    return c_L_hp, c_R_hp


def check_hp_match(c_L, c_R, hp_pc, hp_labels, hp_tree):
    """Check if either contact point is within the HP region."""
    scores = []
    for cp in [c_L, c_R]:
        dist, idx = hp_tree.query(cp)
        scores.append(float(hp_labels[idx]))
    hp_score = max(scores)
    hp_match = hp_score >= HP_THRESH
    return hp_match, hp_score, scores


def process_object(obj_id, gt_path):
    # Load HP and scale to real meters
    hp_pc_raw, hp_labels = load_hp(obj_id)
    if hp_pc_raw is None:
        return None

    sf = get_scale_factor(obj_id)          # mesh file units → real meters
    hp_pc = hp_pc_raw * sf                 # now in real meters, same as GT

    hp_tree = cKDTree(hp_pc)

    results = []
    with h5py.File(gt_path, 'r') as src:
        n_total = int(src.attrs.get('n_successful', 0))
        sources = str(src.attrs.get('sources', ''))
        sg = src.get('successful_grasps', {})
        for key in sg.keys():
            g = sg[key]
            gp = g['grasp_point'][:]
            fd = g['finger_dir'][:]
            w  = float(g.attrs.get('gripper_width', 0.05))

            # Estimate contact points
            c_L, c_R = contact_points(gp, fd, w)

            # 坐标系已统一（SAM3D 原始帧），无需旋转
            c_L_hp = c_L
            c_R_hp = c_R

            # Query HP (same coordinate space as mesh.ply)
            hp_match, hp_score, per_finger = check_hp_match(
                c_L_hp, c_R_hp, hp_pc, hp_labels, hp_tree)

            results.append({
                'key':      key,
                'gp':       gp,
                'rot':      g['rotation'][:] if 'rotation' in g else np.eye(3, dtype=np.float32),
                'ad':       g['approach_dir'][:],
                'fd':       fd,
                'w':        w,
                'score':    float(g.attrs.get('score', 0)),
                'approach_type': str(g.attrs.get('approach_type', 'unknown')),
                'contact_pts_local': g['contact_points_local'][:] if 'contact_points_local' in g else None,
                'hp_match': hp_match,
                'hp_score': hp_score,
                'hp_L':     float(per_finger[0]),
                'hp_R':     float(per_finger[1]),
            })

    return results, n_total, sources


def write_output(obj_id, results, sources, out_path):
    n_total   = len(results)
    n_match   = sum(1 for r in results if r['hp_match'])
    n_nomatch = n_total - n_match

    with h5py.File(out_path, 'w') as f:
        f.attrs['obj_id']        = obj_id
        f.attrs['n_successful']  = n_total
        f.attrs['n_hp_match']    = n_match
        f.attrs['n_hp_mismatch'] = n_nomatch
        f.attrs['sources']       = sources
        f.attrs['schema']        = 'RobotPosteriorFiltered_v1'
        f.attrs['hp_threshold']  = HP_THRESH

        sg = f.create_group('successful_grasps')
        sg.attrs['count'] = n_total

        for i, r in enumerate(results):
            gi = sg.create_group(f'grasp_{i}')
            gi.create_dataset('grasp_point',  data=r['gp'])
            gi.create_dataset('rotation',     data=r['rot'])
            gi.create_dataset('approach_dir', data=r['ad'])
            gi.create_dataset('finger_dir',   data=r['fd'])
            gi.attrs['gripper_width']  = r['w']
            gi.attrs['score']          = r['score']
            gi.attrs['approach_type']  = r['approach_type']
            gi.attrs['hp_match']       = bool(r['hp_match'])
            gi.attrs['hp_score']       = r['hp_score']
            gi.attrs['hp_score_L']     = r['hp_L']
            gi.attrs['hp_score_R']     = r['hp_R']
            if r['contact_pts_local'] is not None:
                gi.create_dataset('contact_points_local', data=r['contact_pts_local'])

    return n_total, n_match, n_nomatch


def main():
    import csv

    gt_files = sorted(glob.glob(os.path.join(COMBINED_DIR, '*_robot_gt.hdf5')))
    print(f'处理 {len(gt_files)} 个物体 (HP_THRESH={HP_THRESH})\n')

    summary_rows = []
    total_grasps = total_match = total_mismatch = 0

    for gt_path in gt_files:
        obj_id = os.path.basename(gt_path).replace('_robot_gt.hdf5', '')
        out_path = os.path.join(OUT_DIR, f'{obj_id}_robot_gt.hdf5')

        ret = process_object(obj_id, gt_path)
        if ret is None:
            print(f'  ⚠️  {obj_id}: 无 HP 数据，跳过')
            continue

        results, n_raw, sources = ret
        n_total, n_match, n_nomatch = write_output(obj_id, results, sources, out_path)

        pct = n_match / n_total * 100 if n_total else 0
        print(f'  {obj_id:<14} 总{n_total:>3} | HP✅{n_match:>3} ({pct:>5.1f}%) | HP❌{n_nomatch:>3}')

        summary_rows.append({
            'obj_id': obj_id, 'n_total': n_total,
            'n_hp_match': n_match, 'n_hp_mismatch': n_nomatch,
            'hp_match_pct': round(pct, 1), 'sources': sources,
        })
        total_grasps   += n_total
        total_match    += n_match
        total_mismatch += n_nomatch

    # 写 CSV
    csv_path = os.path.join(OUT_DIR, 'filter_summary.csv')
    with open(csv_path, 'w', newline='') as cf:
        w = csv.DictWriter(cf, fieldnames=['obj_id','n_total','n_hp_match','n_hp_mismatch','hp_match_pct','sources'])
        w.writeheader()
        w.writerows(summary_rows)

    print(f'\n{"="*60}')
    print(f'过滤完成: {len(summary_rows)} 个物体')
    print(f'  总抓取:      {total_grasps}')
    print(f'  HP 匹配 ✅:  {total_match}  ({total_match/total_grasps*100:.1f}%)')
    print(f'  HP 不匹配 ❌: {total_mismatch}  ({total_mismatch/total_grasps*100:.1f}%)')
    print(f'  输出: {OUT_DIR}')
    print(f'  摘要: {csv_path}')


if __name__ == '__main__':
    main()
