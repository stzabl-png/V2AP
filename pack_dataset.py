#!/usr/bin/env python3
"""
打包 training_m5/ 中的 per-object HDF5 → affordance_train.h5 / affordance_val.h5
格式与 model/train.py 期待的完全一致:
  data/points       (N, 4096, 3)
  data/normals      (N, 4096, 3)
  data/labels       (N, 4096)       ← robot_gt
  data/force_centers (N, 3)
  data/obj_ids      (N,)            ← bytes
"""
import os, sys, glob, h5py, numpy as np, random

PROJ = os.path.dirname(os.path.abspath(__file__))
IN_DIR  = os.path.join(PROJ, "data_hub", "training_m5")
OUT_DIR = os.path.join(PROJ, "output", "dataset_new")
os.makedirs(OUT_DIR, exist_ok=True)

VAL_RATIO = 0.2
RANDOM_SEED = 42

files = sorted(glob.glob(os.path.join(IN_DIR, "*.hdf5")))
print(f"找到 {len(files)} 个物体 HDF5")

random.seed(RANDOM_SEED)
random.shuffle(files)
n_val = max(1, int(len(files) * VAL_RATIO))
val_files  = files[:n_val]
train_files = files[n_val:]
print(f"Train: {len(train_files)}  Val: {len(val_files)}")

def pack(file_list, out_path):
    all_pts, all_nrm, all_lbl, all_fc, all_ids = [], [], [], [], []
    for fp in file_list:
        obj_id = os.path.splitext(os.path.basename(fp))[0]
        with h5py.File(fp, 'r') as f:
            pts = f['point_cloud'][:]
            nrm = f['normals'][:]
            lbl = f['robot_gt'][:]
            fc  = f['force_center'][:] if 'force_center' in f else np.zeros(3, dtype=np.float32)
        all_pts.append(pts)
        all_nrm.append(nrm)
        all_lbl.append(lbl)
        all_fc.append(fc)
        all_ids.append(obj_id.encode())
        print(f"  ✅ {obj_id}  label>0.5: {(lbl>0.5).mean()*100:.1f}%")

    with h5py.File(out_path, 'w') as f:
        grp = f.create_group('data')
        grp.create_dataset('points',        data=np.stack(all_pts),  compression='gzip')
        grp.create_dataset('normals',       data=np.stack(all_nrm),  compression='gzip')
        grp.create_dataset('labels',        data=np.stack(all_lbl),  compression='gzip')
        grp.create_dataset('force_centers', data=np.stack(all_fc),   compression='gzip')
        grp.create_dataset('obj_ids',       data=np.array(all_ids))
    print(f"  → 写出: {out_path}  ({len(file_list)} 物体)")

print("\n=== 打包 Train ===")
pack(train_files, os.path.join(OUT_DIR, "affordance_train.h5"))
print("\n=== 打包 Val ===")
pack(val_files,   os.path.join(OUT_DIR, "affordance_val.h5"))
print("\n✅ 数据集打包完成")
