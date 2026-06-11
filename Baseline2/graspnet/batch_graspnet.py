#!/usr/bin/env python3
"""
Batch GraspNet Inference + HDF5 Generation
============================================
Run GraspNet-Baseline on all evaluation objects and produce candidate HDF5
files compatible with eval_pool.py.

Data sources (from HuggingFace eval_assets):
  - Meshes:     data_hub/meshes/SAM3DMesh/rotated_mesh/{oakink,ycb}/{obj_id}/mesh.ply
  - Scale:      data_hub/ProcessedData/obj_meshes/{oakink,ycb}/{obj_id}/scale.json
  - Eval CSV:   evaluation/configs/eval_objects_merged_success_ge30.csv

Usage:
    # Standard: 87 objects × 20 trials, each trial = independent point cloud → top-1
    python Baseline2/graspnet/batch_graspnet.py \
        --eval-csv evaluation/configs/eval_objects_merged_success_ge30.csv \
        --n-trials 20 --n-top 1

    # Quick sanity check: 1 object, 3 trials
    python Baseline2/graspnet/batch_graspnet.py \
        --obj-ids A02015 --n-trials 3 --n-top 1
"""

import os
import sys
import csv
import time
import json
import argparse
import traceback
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "Baseline2", "graspnet"))

from graspnet_infer import (
    load_model,
    mesh_to_pointcloud,
    infer_grasps,
    find_rotated_mesh,
    find_scale_json,
    read_scale_factor,
)
from graspnet_to_hdf5 import (
    graspgroup_to_candidates,
    write_candidate_hdf5,
)


def load_eval_objects(eval_csv: str) -> list[dict]:
    """Load evaluation object list from CSV."""
    rows = []
    with open(eval_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def infer_obj_dataset(obj_id: str) -> str:
    """Guess dataset from obj_id prefix."""
    if obj_id.startswith("ycb_dex_"):
        return "ycb"
    if obj_id.startswith("unseen_"):
        return "unseen"
    return "oakink"


def main():
    parser = argparse.ArgumentParser(
        description="Batch GraspNet inference on evaluation objects"
    )
    parser.add_argument(
        "--eval-csv", type=str, default=None,
        help="Path to eval_objects CSV",
    )
    parser.add_argument(
        "--obj-ids", nargs="+", default=None,
        help="Explicit list of object IDs (overrides --eval-csv)",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(PROJ, "Baseline2", "graspnet", "checkpoints", "checkpoint-rs.tar"),
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(PROJ, "output", "graspnet_candidates"),
    )
    parser.add_argument("--n-points", type=int, default=20000)
    parser.add_argument("--n-top", type=int, default=1,
                        help="Top-K grasps to keep per trial (default: 1 for top-1 experiment)")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of independent trials per object (each with different point cloud)")
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    parser.add_argument("--z-approach-max", type=float, default=0.3)
    parser.add_argument("--max-retries", type=int, default=0,
                        help="Max retries per trial when result is empty (0=no retry)")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip trials that already have output HDF5",
    )
    args = parser.parse_args()

    # ── Resolve object list ──────────────────────────────────
    if args.obj_ids:
        obj_ids = args.obj_ids
    elif args.eval_csv:
        rows = load_eval_objects(args.eval_csv)
        obj_ids = [r["obj_id"] for r in rows]
    else:
        default_csv = os.path.join(
            PROJ, "evaluation", "configs", "eval_objects_merged_success_ge30.csv"
        )
        if os.path.isfile(default_csv):
            rows = load_eval_objects(default_csv)
            obj_ids = [r["obj_id"] for r in rows]
        else:
            print(f"❌ No --eval-csv or --obj-ids specified, and default CSV not found:")
            print(f"   {default_csv}")
            sys.exit(1)

    print(f"{'=' * 60}")
    print(f"GraspNet-Baseline Batch Inference (Multi-Trial)")
    print(f"{'=' * 60}")
    print(f"  Objects:    {len(obj_ids)}")
    print(f"  Trials:     {args.n_trials} per object")
    print(f"  Top-K:      {args.n_top} per trial")
    print(f"  Max retries:{args.max_retries} per empty trial")
    print(f"  Total runs: {len(obj_ids) * args.n_trials}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output:     {args.output_dir}")
    print(f"{'=' * 60}")

    if not os.path.isfile(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # ── Load model once ──────────────────────────────────────
    net = load_model(args.checkpoint)

    # ── Process each object × trial ──────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    t0 = time.time()
    total_trials = len(obj_ids) * args.n_trials
    done = 0

    for idx, obj_id in enumerate(obj_ids):
        dataset = infer_obj_dataset(obj_id)
        obj_dir = os.path.join(args.output_dir, obj_id)
        os.makedirs(obj_dir, exist_ok=True)

        # Find mesh + scale (once per object)
        mesh_path = find_rotated_mesh(obj_id, dataset)
        if mesh_path is None:
            print(f"\n[{idx + 1}/{len(obj_ids)}] {obj_id} ❌ Mesh not found")
            for t in range(args.n_trials):
                results.append({"obj_id": obj_id, "trial": t, "status": "no_mesh", "n_candidates": 0})
                done += 1
            continue

        scale_json = find_scale_json(obj_id, dataset)
        scale_factor = read_scale_factor(scale_json) if scale_json else 1.0

        print(f"\n[{idx + 1}/{len(obj_ids)}] {obj_id} (scale={scale_factor:.4f}, {args.n_trials} trials)")

        for trial in range(args.n_trials):
            out_path = os.path.join(obj_dir, f"trial_{trial:02d}_grasp.hdf5")

            if args.skip_existing and os.path.isfile(out_path):
                done += 1
                results.append({"obj_id": obj_id, "trial": trial, "status": "skipped", "n_candidates": -1})
                continue

            try:
                candidates = []
                attempt = 0
                max_attempts = 1 + args.max_retries

                while len(candidates) == 0 and attempt < max_attempts:
                    if attempt > 0:
                        print(f"  trial {trial:2d}: retry {attempt}/{args.max_retries} (re-sampling point cloud)")

                    # Each attempt: fresh random point cloud sampling
                    points, mesh = mesh_to_pointcloud(mesh_path, scale_factor, args.n_points)

                    # Inference
                    gg = infer_grasps(
                        net, points,
                        n_top=args.n_top,
                        collision_thresh=args.collision_thresh,
                    )

                    # Convert + write HDF5
                    candidates = graspgroup_to_candidates(
                        gg,
                        scale_factor=scale_factor,
                        z_approach_max=args.z_approach_max,
                    )
                    attempt += 1

                write_candidate_hdf5(
                    candidates,
                    obj_id=obj_id,
                    output_path=out_path,
                    dataset=dataset,
                    scale_factor=scale_factor,
                )

                done += 1
                n_cand = len(candidates)
                retries_used = attempt - 1
                retry_tag = f" (after {retries_used} retries)" if retries_used > 0 else ""
                results.append({
                    "obj_id": obj_id, "trial": trial,
                    "status": "ok", "n_candidates": n_cand,
                    "retries": retries_used,
                })
                print(
                    f"  trial {trial:2d}: {n_cand} candidate(s){retry_tag} "
                    f"[{done}/{total_trials}]"
                )

            except Exception as e:
                done += 1
                print(f"  trial {trial:2d}: ❌ {e}")
                traceback.print_exc()
                results.append({"obj_id": obj_id, "trial": trial, "status": "error", "n_candidates": 0})

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = sum(1 for r in results if r["status"] in ("no_mesh", "error"))
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_empty = sum(1 for r in results if r["status"] == "ok" and r["n_candidates"] == 0)

    # Per-object stats
    obj_stats = {}
    for r in results:
        if r["status"] != "ok":
            continue
        oid = r["obj_id"]
        obj_stats.setdefault(oid, []).append(r["n_candidates"])

    print(f"\n{'=' * 60}")
    print(f"Batch Complete in {elapsed:.1f}s ({elapsed / max(1, n_ok):.2f}s/trial)")
    print(f"  ✅ Success:   {n_ok}/{total_trials}")
    print(f"  ⏭️  Skipped:   {n_skip}")
    print(f"  ❌ Failed:    {n_fail}")
    print(f"  ⚠️  Empty:    {n_empty}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"{'=' * 60}")

    # Write summary JSON
    summary = {
        "n_objects": len(obj_ids),
        "n_trials": args.n_trials,
        "n_top": args.n_top,
        "total_runs": total_trials,
        "n_success": n_ok,
        "n_failed": n_fail,
        "n_skipped": n_skip,
        "n_empty": n_empty,
        "elapsed_s": round(elapsed, 1),
        "per_object": {
            oid: {
                "mean_candidates": round(float(np.mean(counts)), 2),
                "trials_ok": len(counts),
            }
            for oid, counts in obj_stats.items()
        },
    }
    summary_path = os.path.join(args.output_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  📊 Summary:  {summary_path}")


if __name__ == "__main__":
    main()
