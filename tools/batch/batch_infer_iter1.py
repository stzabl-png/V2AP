#!/usr/bin/env python3
"""
GT-Free Self-Improving Loop — Iteration 1
==========================================
对指定物体列表（通常是 val 集 10 个未见物体）:
  1. 用训练好的模型预测接触区域
  2. 生成 20 个抓取候选位姿 → output/grasps_iter1/
  3. 后续由 batch_auto_sim.sh 验证，结果写 output/robot_gt_iter1/

用法:
    # 全部 10 个 val 物体
    python tools/batch_infer_iter1.py \
        --ckpt output/checkpoints_gtfree_v2/best_model.pth

    # 指定物体
    python tools/batch_infer_iter1.py \
        --ckpt output/checkpoints_gtfree_v2/best_model.pth \
        --obj A01006 S10021

    # 自定义输出目录
    python tools/batch_infer_iter1.py \
        --ckpt output/checkpoints_gtfree_v2/best_model.pth \
        --out-dir output/grasps_iter1
"""

import os
import sys
import argparse
import json
import subprocess

# 项目根目录
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import config

MESH_DIR = os.path.join(PROJ, 'data_hub', 'meshes', 'v1')

# 读取旋转覆盖 (sim/object_rotation_overrides.json)
_OVERRIDE_FILE = os.path.join(PROJ, 'sim', 'object_rotation_overrides.json')
try:
    import json as _json
    with open(_OVERRIDE_FILE) as _f:
        _raw = _json.load(_f)
    ROTATION_OVERRIDES = {
        k: (v['rotation'] if isinstance(v, dict) else v)
        for k, v in _raw.items() if not k.startswith('_')
    }
except Exception:
    ROTATION_OVERRIDES = {}

# 默认 val 物体（从 checkpoints_gtfree_v2/split_info.json 读取）
DEFAULT_SPLIT = os.path.join(PROJ, 'output', 'checkpoints_gtfree_v2', 'split_info.json')


def find_mesh(obj_id):
    """查找物体 mesh 路径（支持 .obj / .ply）。"""
    for ext in ['.obj', '.ply']:
        p = os.path.join(MESH_DIR, f'{obj_id}{ext}')
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="GT-Free Iter1: model-driven grasp generation")
    parser.add_argument('--ckpt', required=True,
                        help='训练好的 checkpoint 路径 (e.g. output/checkpoints_gtfree_v2/best_model.pth)')
    parser.add_argument('--obj', nargs='*', default=None,
                        help='物体 ID 列表（默认读 split_info.json 的 val_objects）')
    parser.add_argument('--out-dir', default=os.path.join(PROJ, 'output', 'grasps_iter1'),
                        help='抓取候选输出目录 (default: output/grasps_iter1)')
    parser.add_argument('--n-candidates', type=int, default=20,
                        help='每个物体生成候选位姿数 (default: 20)')
    parser.add_argument('--num-points', type=int, default=1024,
                        help='点云采样数 (default: 1024)')
    args = parser.parse_args()

    # ── 确定物体列表 ───────────────────────────────────────────────
    if args.obj:
        obj_ids = args.obj
    elif os.path.exists(DEFAULT_SPLIT):
        with open(DEFAULT_SPLIT) as f:
            split = json.load(f)
        obj_ids = sorted(split.get('val_objects', []))
        print(f"  [auto] 从 split_info.json 读取 {len(obj_ids)} 个 val 物体")
    else:
        print("❌ 未指定 --obj，且找不到 split_info.json")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"GT-Free Iter1: 模型驱动抓取生成")
    print(f"{'='*60}")
    print(f"  Checkpoint:  {args.ckpt}")
    print(f"  物体数:      {len(obj_ids)}")
    print(f"  输出目录:    {args.out_dir}")
    print(f"  每物体候选:  {args.n_candidates}")
    print(f"{'='*60}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    ok = fail = skip = 0
    for i, obj_id in enumerate(obj_ids):
        print(f"[{i+1}/{len(obj_ids)}] {obj_id} ...", flush=True)

        mesh_path = find_mesh(obj_id)
        if mesh_path is None:
            print(f"  ⚠️  mesh 不存在，跳过")
            skip += 1
            continue

        # 输出 HDF5 路径
        out_hdf5 = os.path.join(args.out_dir, f'{obj_id}_grasps.hdf5')
        if os.path.exists(out_hdf5):
            print(f"  ✅ 已存在，跳过")
            ok += 1
            continue

        # 调用 inference/grasp_pose.py（支持 --ckpt 和 --mesh-euler 参数）
        cmd = [
            sys.executable, '-m', 'inference.grasp_pose',
            '--mesh', mesh_path,
            '--output', out_hdf5,
            '--ckpt', args.ckpt,
            '--num_points', str(args.num_points),
            '--device', 'cuda',
        ]
        # 如果这个物体在 Sim 里有旋转覆盖, 预旋转 mesh 使生成方向与 Sim 一致
        if obj_id in ROTATION_OVERRIDES:
            rot = ROTATION_OVERRIDES[obj_id]
            cmd += ['--mesh-euler'] + [str(r) for r in rot]
            print(f"  ↻ mesh-euler 预旋转: {rot}°")

        try:
            result = subprocess.run(
                cmd, cwd=PROJ,
                capture_output=False, timeout=300
            )
            if result.returncode == 0:
                print(f"  ✅ 候选已生成: {out_hdf5}")
                ok += 1
            else:
                print(f"  ❌ grasp_pose 失败 (code={result.returncode})")
                fail += 1
        except subprocess.TimeoutExpired:
            print(f"  ❌ 超时")
            fail += 1
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            fail += 1

    print(f"\n{'='*60}")
    print(f"完成  ✅ {ok}  ❌ {fail}  ⏭️ {skip}  Total: {len(obj_ids)}")
    print(f"抓取候选: {args.out_dir}")
    print(f"\n下一步 — Sim 验证:")
    print(f"  GRASP_DIR={args.out_dir} \\")
    print(f"  GT_DIR={PROJ}/output/robot_gt_iter1 \\")
    print(f"  bash scripts/batch_auto_sim.sh")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
