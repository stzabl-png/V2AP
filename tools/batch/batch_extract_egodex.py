#!/usr/bin/env python3
"""
batch_extract_egodex.py — 批量提帧 EgoDex .mp4 → extracted_images/

对每一条 {task}/{subject}.mp4 提取 JPEG 帧，存到:
  {task}/{subject}/extracted_images/XXXX.jpg

用法:
    python tools/batch_extract_egodex.py                  # 全部
    python tools/batch_extract_egodex.py --task add_remove_lid
    python tools/batch_extract_egodex.py --limit 10       # 前10条测试
    python tools/batch_extract_egodex.py --workers 8      # 并行
"""

import os, sys, subprocess, argparse, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import sys as _sys, os as _os; _sys.path.insert(0, str(Path(__file__).parent.parent))
import config as _cfg
EGODEX_ROOT = Path(_cfg.DATA_HUB) / "RawData/"  
                   "EgoRawData/egodex/test")


def extract_one(mp4_path: Path, fps: float = 5.0, skip_existing: bool = True):
    """
    提取单个 .mp4 为 JPEG 序列。
    fps=5 → 每 200ms 一帧，约 1000 帧/200s 视频。
    Returns (mp4_path, n_frames, status)
    """
    task   = mp4_path.parent.name
    subj   = mp4_path.stem        # e.g. "0", "1", ...
    out_dir = mp4_path.parent / subj / "extracted_images"

    if skip_existing and out_dir.exists() and len(list(out_dir.glob("*.jpg"))) > 0:
        n = len(list(out_dir.glob("*.jpg")))
        return str(mp4_path), n, "cached"

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(mp4_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",          # JPEG quality (2=high, 31=low)
        str(out_dir / "%04d.jpg")
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        return str(mp4_path), 0, "failed"

    n = len(list(out_dir.glob("*.jpg")))
    return str(mp4_path), n, "done"


def main():
    parser = argparse.ArgumentParser(description="EgoDex 批量提帧")
    parser.add_argument("--task",    default=None, help="只处理此 task 子串")
    parser.add_argument("--limit",   type=int, default=0, help="前 N 条 (0=全部)")
    parser.add_argument("--fps",     type=float, default=5.0,
                        help="提帧帧率 (default=5fps → ~1000帧/200s)")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行 worker 数 (default=4)")
    parser.add_argument("--redo",    action="store_true", help="重新提帧")
    args = parser.parse_args()

    # 收集所有 .mp4
    mp4_files = sorted(EGODEX_ROOT.rglob("*.mp4"))
    if args.task:
        mp4_files = [f for f in mp4_files if args.task in str(f)]
    if args.limit > 0:
        mp4_files = mp4_files[:args.limit]

    print(f"{'='*60}")
    print(f" EgoDex Frame Extraction")
    print(f" Videos  : {len(mp4_files)}")
    print(f" FPS     : {args.fps}  Workers: {args.workers}")
    print(f" Output  : {{task}}/{{subj}}/extracted_images/")
    print(f"{'='*60}\n")

    done = cached = failed = 0
    log = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_one, f, args.fps, not args.redo): f
            for f in mp4_files
        }
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Extract"):
            path, n, status = fut.result()
            if status == "done":
                done += 1
            elif status == "cached":
                cached += 1
            else:
                failed += 1
                tqdm.write(f"  ❌ {path}")
            log.append({"path": path, "n_frames": n, "status": status})

    print(f"\n✅ Done: {done}   ⏭ Cached: {cached}   ❌ Failed: {failed}")

    out_log = EGODEX_ROOT.parent / "extract_log.json"
    with open(out_log, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log → {out_log}")

    print("\n下一步: 对提帧完的序列运行 MegaSAM")
    print("  bash tools/batch_megasam_egodex.sh")


if __name__ == "__main__":
    main()
