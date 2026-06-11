#!/usr/bin/env python3
"""
run_egodex_pipeline.py — EgoDex 全自动处理流水线

按顺序执行:
  Step 1.5: MegaSAM cam_c2w → HaWoR SLAM 格式 (megasam_to_hawor_slam.py)
  Step 2:   HaWoR → MANO 手部参数 (third_party/hawor/demo.py)
  Step 3:   FoundationPose → 物体位姿 (batch_obj_pose_ego.py)
  Step 4:   MANO + FP → HumanPrior 接触图 (gen_ego_contact_map.py)

用法:
  # 全量 (从 registry 读取所有 active 序列)
  python scripts/run_egodex_pipeline.py

  # 只跑 Step 1.5 + 2 (SLAM + HaWoR)
  python scripts/run_egodex_pipeline.py --steps 1.5,2

  # 限制前 N 条
  python scripts/run_egodex_pipeline.py --limit 5

  # 从指定 task 开始
  python scripts/run_egodex_pipeline.py --start-task pour

注意:
  Step 1.5 + 2 需要 hawor 环境:    conda run -n hawor python ...
  Step 3 需要 bundlesdf 环境:       conda run -n bundlesdf python ...
  Step 4 需要 hawor 环境:           conda run -n hawor python ...

  本脚本使用 subprocess 调用不同 conda 环境的脚本。
"""
import os, sys, json, subprocess, time, argparse
import numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

DATA_HUB = PROJ / "data_hub"
DEPTH_BASE = DATA_HUB / "ProcessedData" / "depth" / "Egocentric" / "Egodex"
RAW_BASE = DATA_HUB / "RawData" / "EgoRawData" / "egodex" / "test"
REGISTRY_JSON = PROJ / "tools" / "egodex_sequence_registry.json"
LOG_FILE = PROJ / "output" / "egodex_pipeline_log.jsonl"


def load_registry():
    """Load active (non-skipped) entries from registry."""
    with open(REGISTRY_JSON) as f:
        reg = json.load(f)
    active = {k: v for k, v in reg.items() if not v.get("skipped")}
    return active


def get_focal_from_K(depth_dir):
    """Extract focal length from K.npy, scale to original video resolution."""
    K_path = os.path.join(depth_dir, "K.npy")
    meta_path = os.path.join(depth_dir, "meta.json")
    K = np.load(K_path)
    fx_depth = K[0, 0]

    # K is at depth resolution (e.g. 640x360). We need focal at original resolution (1920x1080).
    # Read depth resolution from meta if available, else infer from depth.npz
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if "calibrated_fx" in meta:
            return meta["calibrated_fx"]
        hw = meta.get("hw", [360, 640])
        W_depth = hw[1]
    else:
        depths = np.load(os.path.join(depth_dir, "depth.npz"))["depths"]
        W_depth = depths.shape[2]

    # Original EgoDex video is 1920x1080 (most common)
    W_orig = 1920  # default
    scale = W_orig / W_depth
    return fx_depth * scale


def log_result(entry):
    os.makedirs(LOG_FILE.parent, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─── Step 1.5: cam_c2w → SLAM ────────────────────────────────────────────────
def run_step_1_5(task, seq_id, depth_dir, video_path, seq_dir):
    """Convert MegaSAM cam_c2w → HaWoR SLAM npz."""
    cam_c2w_path = os.path.join(depth_dir, "cam_c2w.npy")
    if not os.path.exists(cam_c2w_path):
        return False, "cam_c2w.npy missing"

    # Check if SLAM already exists
    slam_dir = os.path.join(seq_dir, "SLAM")
    existing = [f for f in os.listdir(slam_dir) if f.startswith("hawor_slam")] if os.path.exists(slam_dir) else []
    if existing:
        return True, f"cached ({existing[0]})"

    focal = get_focal_from_K(depth_dir)

    cmd = [
        "python", str(PROJ / "tools" / "megasam_to_hawor_slam.py"),
        "--cam_c2w", cam_c2w_path,
        "--video_path", video_path,
        "--img_focal", str(focal),
        "--out_dir", slam_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return False, result.stderr[-500:]
    return True, "done"


# ─── Step 2: HaWoR → MANO ────────────────────────────────────────────────────
def run_step_2(task, seq_id, depth_dir, video_path, seq_dir):
    """Run HaWoR to get MANO hand parameters."""
    # Check if already done
    res_path = os.path.join(seq_dir, "world_space_res.pth")
    if os.path.exists(res_path):
        return True, "cached"

    focal = get_focal_from_K(depth_dir)
    hawor_dir = PROJ / "third_party" / "hawor"

    cmd = [
        "conda", "run", "-n", "hawor",
        "python", str(hawor_dir / "demo.py"),
        "--video_path", video_path,
        "--img_focal", str(focal),
        "--vis_mode", "world",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(hawor_dir), timeout=600
    )
    if result.returncode != 0:
        return False, result.stderr[-500:]

    # Check output exists
    if os.path.exists(res_path):
        return True, "done"
    return False, "world_space_res.pth not created"


# ─── Step 3: FoundationPose ──────────────────────────────────────────────────
def run_step_3():
    """Run FoundationPose batch on all registered sequences."""
    cmd = [
        "conda", "run", "-n", "bundlesdf",
        "python", str(PROJ / "tools" / "batch_obj_pose_ego.py"),
        "--dataset", "egodex",
        "--debug", "0",
    ]
    result = subprocess.run(
        cmd, capture_output=False, text=True,
        cwd=str(PROJ), timeout=86400  # 24h
    )
    return result.returncode == 0


# ─── Step 4: Contact Map ─────────────────────────────────────────────────────
def run_step_4():
    """Run contact map generation on all sequences."""
    cmd = [
        "conda", "run", "-n", "hawor",
        "python", str(PROJ / "tools" / "gen_ego_contact_map.py"),
    ]
    result = subprocess.run(
        cmd, capture_output=False, text=True,
        cwd=str(PROJ), timeout=86400
    )
    return result.returncode == 0


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EgoDex 全自动处理流水线")
    parser.add_argument("--steps", type=str, default="1.5,2,3,4",
                        help="Comma-separated steps to run (default: 1.5,2,3,4)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N sequences")
    parser.add_argument("--start-task", type=str, default=None,
                        help="Start from this task name")
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",")]
    print(f"{'='*60}")
    print(f" EgoDex Pipeline — Steps: {steps}")
    print(f"{'='*60}\n")

    reg = load_registry()
    entries = sorted(reg.items())
    print(f"Registry: {len(entries)} active entries\n")

    # Filter
    if args.start_task:
        idx = next((i for i, (k, v) in enumerate(entries) if args.start_task in v["seq_id"]), 0)
        entries = entries[idx:]
        print(f"Starting from task containing '{args.start_task}' (idx={idx})")
    if args.limit > 0:
        entries = entries[:args.limit]
        print(f"Limited to {args.limit} entries")

    # ─── Step 1.5 + 2: per-sequence (SLAM + HaWoR) ─────────────────────────
    if "1.5" in steps or "2" in steps:
        done_slam = skip_slam = fail_slam = 0
        done_hawor = skip_hawor = fail_hawor = 0
        total = len(entries)

        for i, (key, cfg) in enumerate(entries):
            task = cfg["seq_id"].split("/")[0]
            seq_num = cfg["seq_id"].split("/")[1]
            depth_dir = cfg["depth_dir"]
            video_path = str(RAW_BASE / task / f"{seq_num}.mp4")
            seq_dir = str(RAW_BASE / task / seq_num)

            print(f"\n[{i+1}/{total}] {cfg['seq_id']}  (obj: {cfg['obj_name']})")

            # Step 1.5
            if "1.5" in steps:
                ok, msg = run_step_1_5(task, cfg["seq_id"], depth_dir, video_path, seq_dir)
                if ok:
                    if "cached" in msg:
                        skip_slam += 1
                    else:
                        done_slam += 1
                    print(f"  SLAM: ✅ {msg}")
                else:
                    fail_slam += 1
                    print(f"  SLAM: ❌ {msg[:100]}")
                    log_result({"step": "1.5", "seq": cfg["seq_id"], "status": "fail", "error": msg[:200]})

            # Step 2
            if "2" in steps:
                ok, msg = run_step_2(task, cfg["seq_id"], depth_dir, video_path, seq_dir)
                if ok:
                    if "cached" in msg:
                        skip_hawor += 1
                    else:
                        done_hawor += 1
                    print(f"  HaWoR: ✅ {msg}")
                else:
                    fail_hawor += 1
                    print(f"  HaWoR: ❌ {msg[:100]}")
                    log_result({"step": "2", "seq": cfg["seq_id"], "status": "fail", "error": msg[:200]})

        if "1.5" in steps:
            print(f"\n--- Step 1.5 SLAM: Done={done_slam} Cached={skip_slam} Failed={fail_slam} ---")
        if "2" in steps:
            print(f"--- Step 2 HaWoR: Done={done_hawor} Cached={skip_hawor} Failed={fail_hawor} ---")

    # ─── Step 3: FoundationPose (batch) ──────────────────────────────────────
    if "3" in steps:
        print(f"\n{'='*60}")
        print(f" Step 3: FoundationPose (batch)")
        print(f"{'='*60}")
        ok = run_step_3()
        print(f"\nStep 3: {'✅ Done' if ok else '❌ Failed'}")

    # ─── Step 4: Contact Map (batch) ─────────────────────────────────────────
    if "4" in steps:
        print(f"\n{'='*60}")
        print(f" Step 4: Contact Map Generation")
        print(f"{'='*60}")
        ok = run_step_4()
        print(f"\nStep 4: {'✅ Done' if ok else '❌ Failed'}")

    print(f"\n{'='*60}")
    print(f" Pipeline Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
