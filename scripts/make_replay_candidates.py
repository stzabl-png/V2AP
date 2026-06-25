#!/usr/bin/env python3
"""
make_replay_candidates.py
把 grasp_collect/merged/*_robot_gt_merged.hdf5 里的成功抓取
转成 run_grasp_sim.py 能读的候选 HDF5 格式。

输出到 output/grasp_collect/candidates/replay/ 目录。
每个物体一个 HDF5，内容是它所有成功的 grasp_point + rotation。

用法:
    python3 scripts/make_replay_candidates.py
    python3 scripts/make_replay_candidates.py --obj C28001   # 单个物体
"""
import os, sys, glob, h5py, argparse
import numpy as np

PROJ      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED    = os.path.join(PROJ, 'output', 'grasp_collect', 'merged')
OUT_DIR   = os.path.join(PROJ, 'output', 'grasp_collect', 'candidates', 'replay')
MESH_BASE = os.path.join(PROJ, 'data_hub', 'ProcessedData', 'obj_meshes')

os.makedirs(OUT_DIR, exist_ok=True)


def get_mesh_path(obj_id):
    ds = 'dexycb' if obj_id.startswith('ycb_') else 'oakink'
    p  = os.path.join(MESH_BASE, ds, obj_id, 'mesh.ply')
    return p if os.path.exists(p) else ''


def convert_one(obj_id, merged_path, out_path):
    with h5py.File(merged_path, 'r') as src:
        n_ok = int(src.attrs.get('n_successful', 0))
        if n_ok == 0:
            return 0
        sg = src.get('successful_grasps', {})
        grasps = []
        for k in sg.keys():
            g = sg[k]
            grasps.append({
                'position':      g['grasp_point'][:],
                'rotation':      g['rotation'][:] if 'rotation' in g else np.eye(3, dtype=np.float32),
                'approach_dir':  g['approach_dir'][:],
                'finger_dir':    g['finger_dir'][:],
                'gripper_width': float(g.attrs.get('gripper_width', 0.05)),
                'score':         float(g.attrs.get('score', 100.0)),
                'name':          k,
            })

    mesh_path = get_mesh_path(obj_id)

    with h5py.File(out_path, 'w') as dst:
        dst.attrs['obj_id']   = obj_id
        dst.attrs['n_replay'] = len(grasps)
        dst.attrs['source']   = merged_path

        # metadata（与 Stage A 格式一致）
        meta = dst.create_group('metadata')
        meta.attrs['obj_id']    = obj_id
        meta.attrs['method']    = 'replay_from_merged'
        meta.attrs['mesh_path'] = mesh_path
        meta.attrs['mesh_prerotation_euler']     = np.zeros(3)
        meta.attrs['canonical_rotation_applied'] = False
        meta.attrs['no_rotation']                = True
        meta.attrs['canonical_euler_info']       = np.zeros(3)

        # candidates group（与 Stage A 格式完全一致）
        cg = dst.create_group('candidates')
        for i, g in enumerate(grasps):
            ci = cg.create_group(f'candidate_{i}')
            ci.attrs['name']          = g['name']
            ci.attrs['score']         = g['score']
            ci.attrs['gripper_width'] = g['gripper_width']
            ci.attrs['approach_type'] = 'replay'
            ci.attrs['is_manual']     = False

            ci.create_dataset('position',    data=g['position'])
            ci.create_dataset('grasp_point', data=g['position'])
            ci.create_dataset('rotation',    data=g['rotation'])

        # 最佳抓取（score 最高）
        best = max(grasps, key=lambda x: x['score'])
        wg = dst.create_group('grasp')
        wg.attrs['gripper_width'] = best['gripper_width']
        wg.create_dataset('grasp_point',     data=best['position'])
        wg.create_dataset('position',        data=best['position'])
        wg.create_dataset('rotation',        data=best['rotation'])
        # quaternion 占位（run_grasp_sim 不从这里读）
        wg.create_dataset('quaternion_wxyz', data=np.array([1,0,0,0], dtype=np.float32))

    return len(grasps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', default=None, help='只处理单个物体 ID')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(MERGED, '*_robot_gt_merged.hdf5')))
    if args.obj:
        files = [f for f in files if os.path.basename(f).startswith(args.obj)]

    print(f"处理 {len(files)} 个物体 → {OUT_DIR}")
    total = 0
    obj_list = []
    for fp in files:
        obj_id   = os.path.basename(fp).replace('_robot_gt_merged.hdf5', '')
        out_path = os.path.join(OUT_DIR, f'{obj_id}_candidates.hdf5')
        n = convert_one(obj_id, fp, out_path)
        if n > 0:
            print(f"  ✅ {obj_id}: {n} 个抓取 → {os.path.basename(out_path)}")
            obj_list.append((obj_id, n, out_path))
            total += n
        else:
            print(f"  ⬛ {obj_id}: 0次成功，跳过")

    print(f"\n完成! {len(obj_list)} 个物体, 共 {total} 个抓取候选")
    print(f"输出目录: {OUT_DIR}")
    print()
    print("=== Sim Replay 指令 ===")
    print("# 逐个物体运行（复制下面的指令）:")
    ISAAC = '/home/lyh/isaac-sim-5.0/python.sh'
    for obj_id, n, path in obj_list:
        usd = f'/home/lyh/Project/V2AP/data_hub/usd/{obj_id}.usd'
        print(f"{ISAAC} sim/run_grasp_sim.py \\")
        print(f"    --hdf5 {path} \\")
        print(f"    --object_scale 1.0 \\")
        print(f"    --max-candidates {n}  # {obj_id}")
        print()


if __name__ == '__main__':
    main()
