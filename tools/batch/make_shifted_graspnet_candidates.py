#!/usr/bin/env python3
"""
make_shifted_graspnet_candidates.py
====================================
从现有 GraspNet 候选目录读取所有候选，
沿 approach 方向插入更深 SHIFT_M（默认 2.5cm），
输出格式与 graspnet_best_shifted_2.5cm 完全相同（{obj_id}_grasp.hdf5）。

数据源优先级:
  1. graspnet_candidates_titan/{obj_id}/trial_XX_grasp.hdf5  (88 个物体，每个 40 trials)
  2. graspnet_candidates/{obj_id}/trial_XX_grasp.hdf5        (补充 unseen 等缺口)
"""

from __future__ import annotations
import argparse
import csv
import glob
import os
import numpy as np
import h5py

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TITAN_DIR = os.path.join(PROJ, "output", "graspnet_candidates_titan")
FALLBACK_DIR = os.path.join(PROJ, "output", "graspnet_candidates")


def load_all_candidates_for_obj(obj_id: str) -> list[dict]:
    """从 titan 或 fallback 目录读取该物体的所有候选。"""
    candidates = []
    for base_dir in [TITAN_DIR, FALLBACK_DIR]:
        obj_dir = os.path.join(base_dir, obj_id)
        if not os.path.isdir(obj_dir):
            continue
        trial_files = sorted(glob.glob(os.path.join(obj_dir, "trial_*.hdf5")))
        for trial_path in trial_files:
            try:
                with h5py.File(trial_path, "r") as f:
                    if "candidates" not in f:
                        continue
                    cg = f["candidates"]
                    n = int(cg.attrs.get("n_candidates", 0))
                    meta = f.get("metadata", {})
                    scale = float(meta.attrs.get("scale_factor", 1.0)) if hasattr(meta, "attrs") else 1.0
                    dataset = str(meta.attrs.get("dataset", "oakink")) if hasattr(meta, "attrs") else "oakink"
                    no_rot = bool(meta.attrs.get("no_rotation", True)) if hasattr(meta, "attrs") else True
                    for i in range(n):
                        key = f"candidate_{i}"
                        if key not in cg:
                            continue
                        ci = cg[key]
                        candidates.append({
                            "score": float(ci.attrs.get("score", 0.0)),
                            "position": np.array(ci["position"]),
                            "rotation": np.array(ci["rotation"]),
                            "gripper_width": float(ci.attrs.get("gripper_width", 0.08)),
                            "scale_factor": scale,
                            "dataset": dataset,
                            "no_rotation": no_rot,
                        })
            except Exception as e:
                print(f"    [warn] {trial_path}: {e}")
        if candidates:
            break  # titan 有数据就不用 fallback
    return candidates


def shift_candidates(candidates: list[dict], shift_m: float) -> list[dict]:
    """沿 approach 方向（rotation[:, 2]）向物体内部移动 shift_m 米。"""
    result = []
    for c in candidates:
        c2 = dict(c)
        approach = c2["rotation"][:, 2]
        c2["position"] = c2["position"] + approach * shift_m
        result.append(c2)
    return result


def write_candidates_hdf5(out_path: str, obj_id: str, candidates: list[dict]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scale = float(candidates[0].get("scale_factor", 1.0))
    dataset = str(candidates[0].get("dataset", "oakink"))
    no_rot = bool(candidates[0].get("no_rotation", True))
    with h5py.File(out_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["obj_id"] = obj_id
        meta.attrs["scale_factor"] = scale
        meta.attrs["dataset"] = dataset
        meta.attrs["no_rotation"] = no_rot
        meta.attrs["source"] = "graspnet_baseline"
        cg = f.create_group("candidates")
        cg.attrs["n_candidates"] = len(candidates)
        for idx, c in enumerate(candidates):
            ci = cg.create_group(f"candidate_{idx}")
            ci.create_dataset("position", data=c["position"].astype(np.float32))
            ci.create_dataset("rotation", data=c["rotation"].astype(np.float32))
            ci.attrs["name"] = f"graspnet_{idx}"
            ci.attrs["score"] = float(c["score"])
            ci.attrs["gripper_width"] = float(c.get("gripper_width", 0.08))
            ci.attrs["approach_type"] = "graspnet_baseline"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-list", default="evaluation/configs/eval_objects_all.csv")
    parser.add_argument("--outdir", default="output/graspnet_shifted_2.5cm_all")
    parser.add_argument("--shift", type=float, default=0.025, help="沿 approach 插入深度（米）")
    parser.add_argument("--top-n", type=int, default=10, help="每物体保留 top-N 候选")
    args = parser.parse_args()

    obj_list_path = args.obj_list if os.path.isabs(args.obj_list) else os.path.join(PROJ, args.obj_list)
    with open(obj_list_path, newline="") as f:
        obj_ids = [row["obj_id"].strip() for row in csv.DictReader(f) if row.get("obj_id", "").strip()]

    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(PROJ, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print(f"共 {len(obj_ids)} 个物体 | shift={args.shift*100:.1f}cm | top-N={args.top_n}")
    print(f"输出目录: {outdir}\n")

    ok, skipped = 0, 0
    for obj_id in obj_ids:
        candidates = load_all_candidates_for_obj(obj_id)
        if not candidates:
            print(f"  [SKIP] {obj_id}: 无候选数据")
            skipped += 1
            continue
        candidates.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
        top = candidates[: args.top_n]
        shifted = shift_candidates(top, args.shift)
        out_path = os.path.join(outdir, f"{obj_id}_grasp.hdf5")
        write_candidates_hdf5(out_path, obj_id, shifted)
        print(f"  [OK] {obj_id}: {len(top)} candidates (best={top[0]['score']:.3f})")
        ok += 1

    print(f"\n完成: {ok} 成功, {skipped} 跳过")


if __name__ == "__main__":
    main()
