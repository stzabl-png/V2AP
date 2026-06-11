#!/usr/bin/env python3
"""
expand_ego_registry.py — 扩展 egodex_sequence_registry.json

把已有 mesh 的 obj_name 对应的所有 HaWoR 完成序列加入 registry。
只新增条目，不修改现有条目。

前提：
  - HaWoR 已经跑完（mano/Egocentric/Egodex/egodex/{task}/{ep}.npz 含 right_verts）
  - FoundationPose 会在之后补跑新增序列

用法：
  python tools/expand_ego_registry.py [--dry-run]
"""
import os, sys, json, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

REGISTRY_JSON = os.path.join(config.PROJECT_DIR, "tools", "egodex_sequence_registry.json")
MANO_BASE     = os.path.join(config.DATA_HUB, "ProcessedData", "mano", "Egocentric", "Egodex", "egodex")
MESH_BASE     = os.path.join(config.DATA_HUB, "ProcessedData", "obj_meshes", "egocentric")
DEPTH_BASE    = os.path.join(config.DATA_HUB, "ProcessedData", "egocentric_depth", "egodex")


def has_valid_hawor(npz_path):
    try:
        d = np.load(npz_path, allow_pickle=True)
        return "right_verts" in d
    except:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印要新增的条目，不写入文件")
    args = ap.parse_args()

    # 1. 加载现有 registry
    with open(REGISTRY_JSON) as f:
        reg = json.load(f)

    # 2. 建立 task → obj_name 映射（来自现有 registry）
    task_to_obj = {}
    existing_seqs = set()   # 已在 registry 里的 "task/ep"
    for k, v in reg.items():
        if v.get("skipped") or "obj_name" not in v:
            continue
        sid = v["seq_id"]              # e.g. "add_remove_lid/22"
        task = sid.rsplit("/", 1)[0]
        task_to_obj[task] = v["obj_name"]
        existing_seqs.add(sid)

    print(f"现有 registry: {len(reg)} 条  |  覆盖 task: {len(task_to_obj)}  |  已有序列: {len(existing_seqs)}")

    # 3. 扫描 MANO 目录，找新增序列
    new_entries = {}
    skipped_no_hawor = 0
    skipped_no_depth = 0
    skipped_no_mesh  = 0

    for task, obj_name in sorted(task_to_obj.items()):
        task_mano_dir = os.path.join(MANO_BASE, task)
        if not os.path.isdir(task_mano_dir):
            continue

        # 检查 mesh 存在
        mesh_path = os.path.join(MESH_BASE, obj_name, "mesh.ply")
        if not os.path.exists(mesh_path):
            skipped_no_mesh += 1
            continue

        for npz_f in sorted(glob.glob(os.path.join(task_mano_dir, "*.npz"))):
            ep = os.path.splitext(os.path.basename(npz_f))[0]
            seq_id = f"{task}/{ep}"

            if seq_id in existing_seqs:
                continue  # 已在 registry

            # HaWoR 完整性检查
            if not has_valid_hawor(npz_f):
                skipped_no_hawor += 1
                continue

            # MegaSAM 深度检查
            depth_dir = os.path.join(DEPTH_BASE, task, ep)
            if not os.path.exists(os.path.join(depth_dir, "depth.npz")) and \
               not os.path.exists(os.path.join(depth_dir, "depths.npz")):
                skipped_no_depth += 1
                continue

            key = f"egodex__{seq_id.replace('/', '/')}"
            new_entries[key] = {
                "dataset":  "egodex",
                "seq_id":   seq_id,
                "obj_name": obj_name,
                "skipped":  False,
                "source":   "expand_ego_registry"
            }

    print(f"\n新增条目: {len(new_entries)}")
    print(f"跳过(无 HaWoR): {skipped_no_hawor}  跳过(无深度): {skipped_no_depth}  跳过(无mesh): {skipped_no_mesh}")

    if not new_entries:
        print("\n没有新条目需要添加。")
        return

    print("\n新增列表:")
    for k, v in sorted(new_entries.items()):
        print(f"  {v['seq_id']:<45} → {v['obj_name']}")

    if args.dry_run:
        print("\n[dry-run] 未写入文件。")
        return

    # 4. 写入 registry
    reg.update(new_entries)
    with open(REGISTRY_JSON, "w") as f:
        json.dump(reg, f, indent=2)
    print(f"\n✅ 已写入 {REGISTRY_JSON}  (总条目: {len(reg)})")


if __name__ == "__main__":
    main()
