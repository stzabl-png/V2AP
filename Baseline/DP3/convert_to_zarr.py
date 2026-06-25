#!/usr/bin/env python3
"""
Baseline1 — concatenate episode HDF5 files → DP3 zarr (ReplayBuffer format).

Same on-disk layout as Baseline2/convert_to_zarr.py so both DP baselines train
with an identical task config:
    data/point_cloud  (N_total, 4096, 3)  float32
    data/state        (N_total, 8)        float32   [x,y,z, qw,qx,qy,qz, gripper]
    data/action       (N_total, 8)        float32   = state shifted by 1
    meta/episode_ends (n_episodes,)        int64

The per-episode `finger_angles` / `wrist_pose` datasets (GT MANO metadata) are
ignored here — DP3 only consumes point_cloud / state / action.

Usage:
    python Baseline1/convert_to_zarr.py \\
        --input_dir   Baseline1/data/episodes \\
        --output_zarr Baseline1/data/human_dp_baseline.zarr
"""
import argparse, os, glob
import numpy as np
import h5py
import zarr

ap = argparse.ArgumentParser()
ap.add_argument("--input_dir",   required=True, nargs="+",
                help="one or more dirs with *.hdf5 episode files; files concatenated in dir-then-name order")
ap.add_argument("--output_zarr", default="Baseline1/data/human_dp_baseline.zarr")
ap.add_argument("--pattern",     default="*.hdf5", help="glob pattern for episode files")
args = ap.parse_args()


def main():
    files = []
    for d in args.input_dir:
        files.extend(sorted(glob.glob(os.path.join(d, args.pattern))))
    if not files:
        print(f"❌ no {args.pattern} files in {args.input_dir}"); return
    print(f"Found {len(files)} episode files")

    # First pass: probe one episode for shapes (N_pts, state_dim).
    n_pts, state_dim = None, None
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                n_pts = f["point_cloud"].shape[1]; state_dim = f["state"].shape[1]
            break
        except Exception:
            continue
    if n_pts is None:
        print("❌ no readable episodes"); return

    # Incremental write to zarr: append per episode → memory stays at one-episode size,
    # avoids OOM when concatenating many GB of point clouds (e.g. DexYCB+OakInk merged).
    root = zarr.open(args.output_zarr, mode="w")
    data = root.require_group("data")
    pc_arr     = data.empty("point_cloud", shape=(0, n_pts, 3), chunks=(100, n_pts, 3), dtype=np.float32)
    state_arr  = data.empty("state",       shape=(0, state_dim), chunks=(2000, state_dim), dtype=np.float32)
    action_arr = data.empty("action",      shape=(0, state_dim), chunks=(2000, state_dim), dtype=np.float32)
    ep_ends, cum, n_bad = [], 0, 0
    g_min, g_max = float("inf"), float("-inf")
    xyz_min = np.array([ float("inf")]*3); xyz_max = np.array([float("-inf")]*3)
    for path in files:
        try:
            with h5py.File(path, "r") as f:
                pc  = f["point_cloud"][:].astype(np.float32)
                st  = f["state"][:].astype(np.float32)
                ac  = f["action"][:].astype(np.float32)
        except Exception as e:
            print(f"  ⚠️  {os.path.basename(path)}: read error {e}"); n_bad += 1; continue
        T = len(st)
        if not (len(pc) == T == len(ac)) or T < 1:
            print(f"  ⚠️  {os.path.basename(path)}: shape mismatch / empty"); n_bad += 1; continue
        pc_arr.append(pc); state_arr.append(st); action_arr.append(ac)
        cum += T; ep_ends.append(cum)
        g_min = min(g_min, float(st[:, 7].min())); g_max = max(g_max, float(st[:, 7].max()))
        xyz_min = np.minimum(xyz_min, st[:, :3].min(0)); xyz_max = np.maximum(xyz_max, st[:, :3].max(0))

    if not ep_ends:
        print("❌ no usable episodes"); return

    ep_ends = np.array(ep_ends, dtype=np.int64)
    print(f"\nepisodes : {len(ep_ends)}  ({n_bad} skipped)")
    print(f"steps    : {cum}")
    print(f"point_cloud {pc_arr.shape}  state {state_arr.shape}  action {action_arr.shape}")
    print(f"gripper range: [{g_min:.3f}, {g_max:.3f}]  xyz extent (m): {np.round(xyz_max-xyz_min, 3)}")

    meta = root.require_group("meta")
    meta.create_dataset("episode_ends", data=ep_ends, dtype=np.int64)

    print(f"\n✅ wrote {args.output_zarr}")
    print(f"   verify: python -c \"import zarr; r=zarr.open('{args.output_zarr}'); "
          f"print(r['data/point_cloud'].shape, r['data/state'].shape, r['meta/episode_ends'][:5])\"")


if __name__ == "__main__":
    main()
