#!/usr/bin/env python3
"""
Baseline2 — Step 2: HDF5 → Zarr
=================================
把 record_trajectory.py 输出的 HDF5 轨迹合并转换为 DP3 所需的 Zarr 格式。

用法:
  cd ~/Project/V2AP
  python Baseline2/data_pipeline/hdf5_to_zarr.py
  python Baseline2/data_pipeline/hdf5_to_zarr.py --dataset dexycb
  python Baseline2/data_pipeline/hdf5_to_zarr.py --hdf5_dir Baseline2/data/hdf5/oakink \
                                                   --zarr_dir Baseline2/data/zarr/oakink
"""
import os, sys, glob, argparse
import numpy as np
import h5py
from tqdm import tqdm

# 共用 replay_buffer (来自 Sim_VideoPolicy)
sys.path.insert(0, "/home/lyh/Project/Sim_VideoPolicy")
from lib.utils.replay_buffer import ReplayBuffer

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",  type=str, default="oakink")
parser.add_argument("--hdf5_dir", type=str, default=None,
                    help="HDF5 根目录 (默认 Baseline2/data/hdf5/{dataset})")
parser.add_argument("--zarr_dir", type=str, default=None,
                    help="Zarr 输出目录 (默认 Baseline2/data/zarr/{dataset})")
args = parser.parse_args()

hdf5_dir = args.hdf5_dir or os.path.join(
    PROJ, "Baseline2", "data", "hdf5", args.dataset)
zarr_dir = args.zarr_dir or os.path.join(
    PROJ, "Baseline2", "data", "zarr", args.dataset)

os.makedirs(os.path.dirname(zarr_dir), exist_ok=True)

# ── 收集所有 HDF5 文件 ────────────────────────────────────────────────────
h5_files = sorted(glob.glob(os.path.join(hdf5_dir, "**", "*.hdf5"), recursive=True))
print(f"找到 {len(h5_files)} 个 HDF5 episode")
print(f"输出 Zarr: {zarr_dir}")

if len(h5_files) == 0:
    print("❌ 无 HDF5 文件，请先运行 sim/record_trajectory.py")
    sys.exit(1)

# ── 转换 ──────────────────────────────────────────────────────────────────
replay_buffer = ReplayBuffer.create_from_path(zarr_dir, mode="w")

total_frames = 0
skipped = 0

for h5_path in tqdm(h5_files, desc="Converting"):
    try:
        with h5py.File(h5_path, "r") as f:
            state  = f["state"][:].astype(np.float32)       # (T, 8)
            action = f["action"][:].astype(np.float32)      # (T, 8)
            pc     = f["point_cloud"][:].astype(np.float32) # (T, 4096, 3)
    except Exception as e:
        print(f"  ⚠️  跳过 {h5_path}: {e}")
        skipped += 1
        continue

    T = state.shape[0]
    if T < 5:
        skipped += 1
        continue

    episode = {
        "state":       state,
        "action":      action,
        "point_cloud": pc,
    }
    replay_buffer.add_episode(episode, compressors="disk")
    total_frames += T

print(f"\n✅ 转换完成")
print(f"   Episodes: {len(h5_files) - skipped}  (跳过 {skipped})")
print(f"   总帧数:   {total_frames}")
print(f"   Zarr:     {zarr_dir}")
print(f"\n下一步: bash Baseline2/scripts/train_dp3.sh")
