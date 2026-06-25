#!/usr/bin/env python3
"""
populate_ego_meshes.py
======================
从 RawMesh/egodex/{task}/mesh.ply  →  obj_meshes/egocentric/{obj_name}/mesh.ply
(以 symlink 方式，不复制文件，节省磁盘)

5 个物体有重命名 (task_name != obj_name):
  add_remove_lid                -> container_lid
  arrange_topple_dominoes       -> domino
  assemble_disassemble_furniture_bench_chair -> furniture_chair
  assemble_disassemble_furniture_bench_desk  -> furniture_desk
  assemble_disassemble_furniture_bench_drawer-> furniture_drawer

用法:
  python tools/populate_ego_meshes.py          # dry-run 默认
  python tools/populate_ego_meshes.py --apply  # 真正执行
"""

import os, sys, json, argparse
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
RAWMESH_BASE = PROJ / "data_hub/ProcessedData/RawMesh/egodex"
OUT_BASE     = PROJ / "data_hub/ProcessedData/obj_meshes/egocentric"
REGISTRY_JSON = PROJ / "tools/egodex_sequence_registry.json"


def build_task_to_obj():
    with open(REGISTRY_JSON) as f:
        reg = json.load(f)
    task_to_obj = {}
    for k, v in reg.items():
        if v.get("skipped") or "obj_name" not in v:
            continue
        task = v["seq_id"].split("/")[0]
        task_to_obj[task] = v["obj_name"]
    return task_to_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually create symlinks (default: dry-run only)")
    ap.add_argument("--copy", action="store_true",
                    help="Copy files instead of symlinking")
    args = ap.parse_args()

    task_to_obj = build_task_to_obj()

    done = skipped = missing = 0
    renamed_count = 0

    print(f"{'='*60}")
    print(f" populate_ego_meshes — {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f" RawMesh  : {RAWMESH_BASE}")
    print(f" Out      : {OUT_BASE}")
    print(f"{'='*60}")

    for task, obj_name in sorted(task_to_obj.items()):
        src_ply = RAWMESH_BASE / task / "mesh.ply"
        dst_dir = OUT_BASE / obj_name
        dst_ply = dst_dir / "mesh.ply"

        label = " (renamed)" if task != obj_name else ""

        if not src_ply.exists():
            print(f"  ❌ MISSING  {task}/mesh.ply{label}")
            missing += 1
            continue

        if dst_ply.exists() or dst_ply.is_symlink():
            print(f"  ⏭  exists  {obj_name}/mesh.ply{label}")
            skipped += 1
            continue

        print(f"  {'✅' if args.apply else '🔵'} {'link' if not args.copy else 'copy'}"
              f"  {task}/mesh.ply  →  {obj_name}/mesh.ply{label}")
        if task != obj_name:
            renamed_count += 1

        if args.apply:
            dst_dir.mkdir(parents=True, exist_ok=True)
            if args.copy:
                import shutil
                shutil.copy2(src_ply, dst_ply)
            else:
                # Absolute symlink to source
                dst_ply.symlink_to(src_ply.resolve())
        done += 1

    print(f"\n{'='*60}")
    print(f"  {'Would link' if not args.apply else 'Linked'}: {done}  "
          f"(of which {renamed_count} renamed)  "
          f"| Already exists: {skipped}  | Missing src: {missing}")
    if not args.apply and done > 0:
        print(f"\n  ▶ Run with --apply to create the symlinks.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
