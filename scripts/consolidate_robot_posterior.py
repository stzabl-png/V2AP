#!/usr/bin/env python3
"""
consolidate_robot_posterior.py
合并所有成功抓取数据到 RobotPosteriorTotal/，每个物体一个 HDF5

数据源 (按时间顺序):
  R1  → robot_gt_oakink
  R2  → robot_gt_oakink_r2
  R3  → robot_gt_r3          (含部分 DexYCB)
  D1  → robot_gt_dexycb
  D2  → robot_gt_dexycb_r2
  NEW → grasp_collect/merged (今天 Round0+1)
"""
import os, sys, glob, h5py
import numpy as np

BASE    = '/home/lyh/Project/V2AP/output'
OUT_DIR = '/home/lyh/Project/V2AP/output/RobotPosteriorTotal'
os.makedirs(OUT_DIR, exist_ok=True)

# 源目录及标签（优先级：后面的覆盖前面的不重要，但成功次数全部累加）
SOURCES = [
    ('R1',  f'{BASE}/robot_gt_oakink',        '_robot_gt.hdf5'),
    ('R2',  f'{BASE}/robot_gt_oakink_r2',     '_robot_gt.hdf5'),
    ('R3',  f'{BASE}/robot_gt_r3',            '_robot_gt.hdf5'),
    ('D1',  f'{BASE}/robot_gt_dexycb',        '_robot_gt.hdf5'),
    ('D2',  f'{BASE}/robot_gt_dexycb_r2',     '_robot_gt.hdf5'),
    ('NEW', f'{BASE}/grasp_collect/merged',   '_robot_gt_merged.hdf5'),
]


def load_source(src_dir, suffix):
    """返回 {obj_id: [(grasp_dict, round_label), ...]}"""
    result = {}
    for fp in sorted(glob.glob(f'{src_dir}/*{suffix}')):
        obj_id = os.path.basename(fp)[:-len(suffix)]
        grasps = []
        try:
            with h5py.File(fp, 'r') as f:
                n = int(f.attrs.get('n_successful', 0))
                if n == 0:
                    continue
                sg = f.get('successful_grasps', {})
                for k in sg.keys():
                    g = sg[k]
                    entry = {
                        'grasp_point':  g['grasp_point'][:],
                        'rotation':     g['rotation'][:] if 'rotation' in g else np.eye(3, dtype=np.float32),
                        'approach_dir': g['approach_dir'][:],
                        'finger_dir':   g['finger_dir'][:],
                        'gripper_width': float(g.attrs.get('gripper_width', 0.05)),
                        'score':        float(g.attrs.get('score', 0.0)),
                        'approach_type': str(g.attrs.get('approach_type', 'unknown')),
                    }
                    if 'contact_points_local' in g:
                        entry['contact_points_local'] = g['contact_points_local'][:]
                    grasps.append(entry)
        except Exception as e:
            print(f'  ⚠️  {obj_id} @ {src_dir}: {e}')
            continue
        if grasps:
            if obj_id not in result:
                result[obj_id] = []
            result[obj_id].extend(grasps)
    return result


def write_merged(obj_id, all_grasps, out_path, sources_used):
    with h5py.File(out_path, 'w') as f:
        f.attrs['obj_id']       = obj_id
        f.attrs['n_successful'] = len(all_grasps)
        f.attrs['sources']      = ','.join(sources_used)
        f.attrs['schema']       = 'RobotPosteriorTotal_v1'

        sg = f.create_group('successful_grasps')
        sg.attrs['count'] = len(all_grasps)
        for i, g in enumerate(all_grasps):
            gi = sg.create_group(f'grasp_{i}')
            gi.create_dataset('grasp_point',  data=g['grasp_point'])
            gi.create_dataset('rotation',     data=g['rotation'])
            gi.create_dataset('approach_dir', data=g['approach_dir'])
            gi.create_dataset('finger_dir',   data=g['finger_dir'])
            gi.attrs['gripper_width']  = g['gripper_width']
            gi.attrs['score']          = g['score']
            gi.attrs['approach_type']  = g['approach_type']
            if 'contact_points_local' in g:
                gi.create_dataset('contact_points_local',
                                  data=g['contact_points_local'])
                gi.attrs['has_contact_points'] = True
            else:
                gi.attrs['has_contact_points'] = False


def main():
    print(f'输出目录: {OUT_DIR}\n')

    # 收集所有源的数据
    all_data   = {}   # obj_id → [grasps]
    obj_source = {}   # obj_id → [源标签]

    for label, src_dir, suffix in SOURCES:
        if not os.path.exists(src_dir):
            print(f'  SKIP (not found): {src_dir}')
            continue
        src_data = load_source(src_dir, suffix)
        for obj_id, grasps in src_data.items():
            if obj_id not in all_data:
                all_data[obj_id]   = []
                obj_source[obj_id] = []
            all_data[obj_id].extend(grasps)
            obj_source[obj_id].append(f'{label}(×{len(grasps)})')
        print(f'  {label}: {len(src_data)} 个物体')

    print(f'\n合并后: {len(all_data)} 个物体')
    total_grasps = sum(len(v) for v in all_data.values())
    print(f'总抓取次数: {total_grasps} 次\n')

    # 写出每个物体的 HDF5
    for obj_id in sorted(all_data.keys()):
        grasps = all_data[obj_id]
        out_path = os.path.join(OUT_DIR, f'{obj_id}_robot_gt.hdf5')
        write_merged(obj_id, grasps, out_path, obj_source[obj_id])

    print(f'✅ 全部写入 {OUT_DIR}')
    print(f'   共 {len(all_data)} 个 HDF5，{total_grasps} 次成功抓取')

    # 输出摘要 CSV
    import csv
    csv_path = os.path.join(OUT_DIR, 'summary.csv')
    with open(csv_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['obj_id', 'n_grasps', 'sources'])
        for obj_id in sorted(all_data.keys()):
            w.writerow([obj_id, len(all_data[obj_id]),
                        ' | '.join(obj_source[obj_id])])
    print(f'   摘要: {csv_path}')


if __name__ == '__main__':
    main()
