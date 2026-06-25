#!/usr/bin/env python3
"""batch_sam3d_recon.py — 用 Meta SAM3D 重建数据集物体 mesh

Sources:
  • ycb        ← obj_recon_input/ycb/{obj}/                           (28 objects)
  • egodex     ← obj_recon_input/egocentric/{task}/                   (filtered to EgoDex
                 (filtered by ls of RawData/EgoRawData/egodex/test/)   task list, 91 tasks)
  • oakink     ← obj_recon_input/oakink/{obj}/                        (100 objects)

Outputs:  obj_meshes/{dataset}/{key}/mesh.ply  +  meta.json

特性:
  • 模型只加载一次，对象间循环
  • 已有 mesh.ply 默认跳过（--redo 强制重做）
  • 失败的 obj 记录到 batch_log.jsonl
  • 进度条显示当前 obj / ETA

用法:
    cd /home/lyh/Project/V2AP
    conda run -n sam3d-objects python tools/batch/batch_sam3d_recon.py
    python tools/batch_sam3d_recon.py --datasets egodex --limit 3
    python tools/batch_sam3d_recon.py --redo
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Set BEFORE importing sam3d_objects
os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", os.environ.get("CUDA_HOME", ""))
# Blackwell / RTX 5090 (sm_120): spconv's implicit_gemm autotuner SIGFPEs (no sm_120
# kernel), so force the native sparse-conv algo. xformers build is mismatched too →
# fall back to torch sdpa. setdefault so callers can still override from the env.
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("XFORMERS_DISABLED", "1")

PROJ = str(Path(__file__).resolve().parent.parent.parent)  # V2AP root
SAM3D_ROOT = os.environ.get("SAM3D_DIR",
             os.path.join(PROJ, "thirdparty", "sam3d"))
sys.path.insert(0, os.path.join(SAM3D_ROOT, "notebook"))
sys.path.insert(0, SAM3D_ROOT)  # for obj_recon_core

INPUT_BASE      = os.path.join(PROJ, "data_hub", "ProcessedData", "obj_recon_input")
OUTPUT_BASE     = os.path.join(PROJ, "data_hub", "ProcessedData", "obj_meshes")
LOG_PATH        = os.path.join(PROJ, "output", "sam3d_batch_log.jsonl")
EGODEX_RAW_BASE = os.path.join(PROJ, "data_hub", "RawData", "EgoRawData", "egodex", "test")

DEFAULT_DATASETS = ["ycb", "egodex", "oakink"]


def egodex_task_whitelist():
    """Return set of valid EgoDex task names (dirs in RawData/EgoRawData/egodex/test/)."""
    if not os.path.isdir(EGODEX_RAW_BASE):
        return set()
    return {d for d in os.listdir(EGODEX_RAW_BASE)
            if os.path.isdir(os.path.join(EGODEX_RAW_BASE, d))}


def discover(datasets):
    """Yield (dataset, key, image_path, mask_dir, out_dir) tuples.

    For each dataset, source is INPUT_BASE/{src_dir}/{key}/{image.png, 0.png}
    Output goes to OUTPUT_BASE/{dataset}/{key}/.
    For "egodex", we source from "egocentric/" but filter keys by EgoDex task whitelist.
    """
    egodex_tasks = egodex_task_whitelist()

    for ds in datasets:
        if ds == "egodex":
            src_subdir = "egocentric"
            keep_filter = lambda name: name in egodex_tasks
        elif ds == "hoi4d":
            # HOI4D masks are stored under egocentric/ with ZY... naming
            src_subdir = "egocentric"
            keep_filter = lambda name: name.startswith("ZY")
        elif ds in ("ycb", "oakink", "arctic", "taco"):
            src_subdir = ds
            keep_filter = lambda name: True
        else:
            print(f"[skip] unknown dataset: {ds}")
            continue

        src_root = Path(INPUT_BASE) / src_subdir
        if not src_root.is_dir():
            print(f"[skip] {src_root} not found")
            continue

        n_total = n_kept = 0
        for obj_dir in sorted(src_root.iterdir()):
            if not obj_dir.is_dir():
                continue
            n_total += 1
            if not keep_filter(obj_dir.name):
                continue
            img = obj_dir / "image.png"
            mask = obj_dir / "0.png"
            if not (img.exists() and mask.exists()):
                continue
            out = Path(OUTPUT_BASE) / ds / obj_dir.name
            yield ds, obj_dir.name, str(img), str(obj_dir), str(out)
            n_kept += 1
        print(f"[discover] {ds}: {n_kept}/{n_total} from {src_subdir}/")


def run_one(inference, image_path, mask_dir, out_dir, seed=42):
    from inference import load_image, load_single_mask
    t0 = time.time()
    img = load_image(image_path)
    mask = load_single_mask(mask_dir, index=0)
    out = inference(img, mask, seed=seed)
    glb = out["glb"]
    os.makedirs(out_dir, exist_ok=True)
    mesh_path = os.path.join(out_dir, "mesh.ply")
    glb.export(mesh_path)
    extent = (glb.bounds[1] - glb.bounds[0]).tolist()
    meta = {
        "n_verts": len(glb.vertices),
        "n_faces": len(glb.faces),
        "bbox_extent": extent,
        "source_image": image_path,
        "source_mask_dir": mask_dir,
        "time_s": round(time.time() - t0, 2),
        "ts": time.time(),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return {"status": "ok", "mesh_path": mesh_path, **meta}


def append_log(entry):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                    help="comma-separated subset of {ycb,egodex,oakink,arctic,taco}")
    ap.add_argument("--redo", action="store_true", help="reprocess even if mesh.ply exists")
    ap.add_argument("--obj", default=None, help="filter to this obj name (substring)")
    ap.add_argument("--limit", type=int, default=0, help="cap total objects (0=all)")
    ap.add_argument("--dry-run", action="store_true", help="list work, don't run SAM3D")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    print(f"Datasets: {datasets}")

    queue = []
    for ds, key, img, mdir, out in discover(datasets):
        if args.obj and args.obj not in key:
            continue
        if not args.redo and (Path(out) / "mesh.ply").exists():
            continue
        queue.append((ds, key, img, mdir, out))
        if args.limit and len(queue) >= args.limit:
            break

    print(f"\nPending: {len(queue)} objects")
    if args.dry_run:
        for ds, k, _, _, _ in queue[:30]:
            print(f"  {ds}/{k}")
        if len(queue) > 30:
            print(f"  ... +{len(queue)-30} more")
        return
    if not queue:
        print("Nothing to do.")
        return

    print("Loading SAM3D pipeline (one-time, ~30s)...")
    from inference import Inference
    cfg_path = f"{SAM3D_ROOT}/checkpoints/hf/pipeline.yaml"
    inference = Inference(cfg_path, compile=False)
    print("✅ Pipeline ready\n")

    done = failed = 0
    t_start = time.time()
    for i, (ds, key, img, mdir, out) in enumerate(queue, 1):
        elapsed = time.time() - t_start
        eta = elapsed / max(done + failed, 1) * (len(queue) - i + 1) if done + failed > 0 else 0
        print(f"[{i:4d}/{len(queue)}] {ds}/{key}  "
              f"(✅{done} ❌{failed}  elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m)",
              flush=True)
        try:
            r = run_one(inference, img, mdir, out)
            r.update({"dataset": ds, "key": key})
            append_log(r)
            done += 1
            print(f"      ✅ {r['n_verts']} v / {r['n_faces']} f  "
                  f"bbox=[{r['bbox_extent'][0]:.2f},{r['bbox_extent'][1]:.2f},{r['bbox_extent'][2]:.2f}]  "
                  f"{r['time_s']}s")
        except Exception as e:
            err = {"status": "fail", "dataset": ds, "key": key,
                   "error": str(e)[:300], "ts": time.time()}
            append_log(err)
            failed += 1
            print(f"      ❌ {type(e).__name__}: {str(e)[:150]}")
            traceback.print_exc(limit=2)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done: {done}  Failed: {failed}  Total: {total/60:.1f} min")
    print(f"Log: {LOG_PATH}")
    print(f"Output: {OUTPUT_BASE}/{{ycb,egodex,oakink}}/{{obj}}/mesh.ply")


if __name__ == "__main__":
    main()
