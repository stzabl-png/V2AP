#!/usr/bin/env python3
"""
Multi-GPU Parallel Launcher for V2AP pipeline.

Automatically shards sequences across all available GPUs.
Falls back to single-GPU mode if only one GPU is present.

Usage (single GPU — same as running directly):
    python launch_parallel.py data/batch_depth_pro.py --dataset dexycb --two-pass

Usage (8 GPUs — auto-sharded):
    python launch_parallel.py data/batch_depth_pro.py --dataset dexycb --two-pass

Usage (use only 4 of 8 GPUs):
    python launch_parallel.py --gpus 0,1,2,3 data/batch_depth_pro.py --dataset dexycb

Usage (specify GPU IDs explicitly):
    python launch_parallel.py --gpus 4,5,6,7 data/batch_depth_pro.py --dataset dexycb
"""

import argparse
import subprocess
import sys
import os
import time
import math


def get_available_gpus():
    """Return list of available CUDA device IDs."""
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
    except Exception:
        pass
    return [0]  # fallback: assume 1 GPU


def get_total_sequences(script, extra_args):
    """
    Dry-run the script with --dry-run-count to get total sequence count.
    Falls back to a large number if not supported.
    """
    try:
        result = subprocess.run(
            [sys.executable, script] + extra_args + ["--dry-run-count"],
            capture_output=True, text=True, timeout=60
        )
        for line in result.stdout.splitlines():
            if line.strip().startswith("SEQUENCE_COUNT:"):
                return int(line.strip().split(":")[1])
    except Exception:
        pass
    return None   # None → launch without --start/--end (script handles it)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU parallel launcher. Auto-detects GPUs and shards sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU IDs to use (e.g. '0,1,2,3'). "
                             "Default: all available GPUs.")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Parallel workers per GPU (default: 1).")
    parser.add_argument("script", help="Script to run (e.g. data/batch_depth_pro.py)")
    parser.add_argument("script_args", nargs=argparse.REMAINDER,
                        help="All arguments to pass to the script.")
    args = parser.parse_args()

    # Resolve GPU list
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    else:
        gpu_ids = get_available_gpus()

    n_gpus = len(gpu_ids)
    print(f"[launcher] GPUs available: {gpu_ids} ({n_gpus} total)")

    script = args.script
    script_args = args.script_args

    # Single GPU: just run directly, no sharding overhead
    if n_gpus == 1:
        print(f"[launcher] Single GPU mode — running directly on GPU {gpu_ids[0]}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        proc = subprocess.run(
            [sys.executable, script] + script_args,
            env=env
        )
        sys.exit(proc.returncode)

    # Multi-GPU: try to get total sequences for clean sharding
    total = get_total_sequences(script, script_args)
    n_workers = n_gpus * args.workers_per_gpu

    if total is not None:
        # Clean even sharding
        chunk = math.ceil(total / n_workers)
        shards = []
        for i in range(n_workers):
            start = i * chunk
            end   = min(start + chunk, total)
            if start >= total:
                break
            shards.append((start, end))
        print(f"[launcher] {total} sequences → {len(shards)} shards of ~{chunk} each")
    else:
        # Script doesn't support --dry-run-count; use start/end with estimated large N
        # Each worker gets a non-overlapping slice; extra workers will just exit early
        shards = [(i, 0) for i in range(n_workers)]  # end=0 means "to end of list"
        print(f"[launcher] Could not determine total sequences — launching {n_workers} workers with --start N --end 0")
        print(f"           (workers will naturally stop when their slice is empty)")

    # Launch one process per shard
    processes = []
    log_dir = os.path.join(os.path.dirname(script), "..", "logs", "parallel")
    os.makedirs(log_dir, exist_ok=True)

    for i, (start, end) in enumerate(shards):
        gpu_id = gpu_ids[i % n_gpus]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [sys.executable, script] + script_args + [f"--start={start}"]
        if end > 0:
            cmd.append(f"--end={end}")

        log_path = os.path.join(log_dir, f"worker_{i:02d}_gpu{gpu_id}.log")
        log_file = open(log_path, "w")
        print(f"[launcher] Worker {i:02d} → GPU {gpu_id} | seqs [{start}:{end or '→end'}] | log: {log_path}")

        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((i, gpu_id, proc, log_file))

    print(f"\n[launcher] {len(processes)} workers launched. Waiting...\n")

    # Monitor and wait
    start_time = time.time()
    try:
        while True:
            alive = [p for _, _, p, _ in processes if p.poll() is None]
            done  = [p for _, _, p, _ in processes if p.poll() is not None]
            errors = [(i, g, p) for i, g, p, _ in processes if p.poll() not in (None, 0)]

            elapsed = time.time() - start_time
            print(f"\r[launcher] {len(done)}/{len(processes)} done | "
                  f"{len(errors)} errors | elapsed {elapsed/60:.1f}min", end="", flush=True)

            if not alive:
                break
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n[launcher] Interrupted — terminating workers...")
        for _, _, p, _ in processes:
            p.terminate()
        sys.exit(1)

    # Close log files
    for _, _, _, lf in processes:
        lf.close()

    # Final report
    print(f"\n\n[launcher] All done in {(time.time()-start_time)/60:.1f} min")
    exit_codes = [(i, g, p.returncode) for i, g, p, _ in processes]
    failed = [(i, g, rc) for i, g, rc in exit_codes if rc != 0]

    if failed:
        print(f"[launcher] ⚠️  {len(failed)} workers failed:")
        for i, g, rc in failed:
            log_path = os.path.join(log_dir, f"worker_{i:02d}_gpu{g}.log")
            print(f"  Worker {i:02d} (GPU {g}) exit={rc} → tail: {log_path}")
            os.system(f"tail -5 {log_path}")
        sys.exit(1)
    else:
        print(f"[launcher] ✅ All {len(processes)} workers succeeded.")


if __name__ == "__main__":
    main()
