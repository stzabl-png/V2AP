#!/usr/bin/env python3
"""
label_pick_task.py — 人工标注视频是否为 Pick 任务

操作:
  ENTER   → ✅ Pick（是抓取任务）
  SPACE   → ❌ Not Pick
  S       → ⏭  跳过（暂不判断）
  ← →     → 上一个 / 下一个视频（不改变已有标签）
  Q       → 保存并退出

用法:
  # HOI4D 全量（从头开始）
  conda activate base
  python tools/anno/label_pick_task.py --dataset hoi4d

  # 断点续标（跳过已标注的）
  python tools/anno/label_pick_task.py --dataset hoi4d --resume

  # 只标特定子集
  python tools/anno/label_pick_task.py --dataset hoi4d --seq ZY20210800001

  # 单个 mp4 文件
  python tools/anno/label_pick_task.py --mp4 /path/to/video.mp4

输出:
  output/labels/pick_labels_{dataset}.json
  {
    "hoi4d/ZY20210800001_H1_C1_N19_S100_s02_T1": "pick",
    "hoi4d/ZY20210800001_H1_C2_N24_S214_s04_T1": "not_pick",
    ...
  }
"""

import os, sys, json, argparse, time
import cv2
import numpy as np
from glob import glob
from natsort import natsorted
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELS_DIR  = os.path.join(config.PROJECT_DIR, "output", "labels")
HOI4D_ROOT  = os.path.join(config.DATA_HUB, "RawData", "EgoRawData", "hoi4d", "HOI4D_release")
WIN         = "Pick Task Labeler"
DISP_H      = 540    # display height (width auto)
FPS_DISPLAY = 15     # playback fps in viewer

LABEL_PICK     = "pick"
LABEL_NOT_PICK = "not_pick"
LABEL_SKIP     = "skip"

# ── Collect videos ────────────────────────────────────────────────────────────

def collect_hoi4d(seq_filter=None):
    """Yields (seq_id, mp4_path) for all HOI4D sequences."""
    if not os.path.isdir(HOI4D_ROOT):
        print(f"⚠  HOI4D_ROOT not found: {HOI4D_ROOT}")
        return []
    results = []
    for mp4 in natsorted(glob(os.path.join(HOI4D_ROOT, "**", "image.mp4"), recursive=True)):
        rel = os.path.relpath(mp4, HOI4D_ROOT)
        seq = rel.replace(os.sep + "align_rgb" + os.sep + "image.mp4", "")
        seq_id = "hoi4d/" + seq.replace(os.sep, "_")
        if seq_filter and seq_filter not in seq_id:
            continue
        results.append((seq_id, mp4))
    return results


# ── Labels I/O ────────────────────────────────────────────────────────────────

def load_labels(path):
    if os.path.exists(path):
        return json.loads(open(path).read())
    return {}


def save_labels(labels, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)


# ── Video loader ──────────────────────────────────────────────────────────────

def load_video_frames(mp4_path, max_frames=300, resize_h=DISP_H):
    """Load all frames from an mp4 into a list of BGR images."""
    cap = cv2.VideoCapture(mp4_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 15
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        scale = resize_h / h
        frame = cv2.resize(frame, (int(w * scale), resize_h),
                           interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return frames, orig_fps, total


# ── UI rendering ──────────────────────────────────────────────────────────────

LABEL_COLORS = {
    LABEL_PICK:     (50, 200, 80),   # green
    LABEL_NOT_PICK: (50, 80, 220),   # red
    LABEL_SKIP:     (120, 120, 120), # grey
    None:           (200, 200, 50),  # yellow (unlabeled)
}

def draw_overlay(frame, seq_id, fi, total_frames, idx, total_seqs,
                 current_label, orig_fps, ms_per_display_frame):
    out = frame.copy()
    h, w = out.shape[:2]

    # Header bar
    cv2.rectangle(out, (0, 0), (w, 70), (15, 15, 25), -1)

    # Progress dot
    col = LABEL_COLORS.get(current_label, LABEL_COLORS[None])
    cv2.circle(out, (18, 22), 9, col, -1)

    # Seq info
    short = seq_id.split("/")[-1] if "/" in seq_id else seq_id
    cv2.putText(out, f"[{idx+1}/{total_seqs}]  {short}",
                (36, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)

    # Label status
    label_txt = f"Label: {current_label.upper()}" if current_label else "Label: —"
    cv2.putText(out, label_txt, (36, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # Controls hint
    hint = "ENTER=Pick  SPACE=NotPick  S=Skip  ← →=Prev/Next  Q=Quit"
    cv2.putText(out, hint, (w // 2 - 290, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 150, 120), 1)

    # Frame progress bar
    prog = int(w * fi / max(total_frames - 1, 1))
    cv2.rectangle(out, (0, 67), (prog, 71), col, -1)
    cv2.rectangle(out, (0, 67), (w, 71), (40, 40, 40), 1)

    # Frame counter
    ts = f"{fi+1}/{total_frames}"
    cv2.putText(out, ts, (w - 80, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 180, 200), 1)

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Human labeling: is this video a Pick task?")
    parser.add_argument("--dataset", default="hoi4d",
                        choices=["hoi4d"],
                        help="Dataset to label")
    parser.add_argument("--mp4",    default=None,
                        help="Label a single mp4 file instead of a full dataset")
    parser.add_argument("--seq",    default=None,
                        help="Substring filter on seq_id")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-labeled sequences (default: show all)")
    parser.add_argument("--start",  type=int, default=0,
                        help="Start from sequence index N")
    parser.add_argument("--out",    default=None,
                        help="Custom output JSON path")
    args = parser.parse_args()

    os.makedirs(LABELS_DIR, exist_ok=True)

    # ── Collect sequences ─────────────────────────────────────────────────────
    if args.mp4:
        seq_id = os.path.splitext(os.path.basename(args.mp4))[0]
        videos = [(seq_id, args.mp4)]
        label_path = args.out or os.path.join(LABELS_DIR, "pick_labels_custom.json")
    else:
        if args.dataset == "hoi4d":
            videos = collect_hoi4d(seq_filter=args.seq)
        else:
            print(f"Unknown dataset: {args.dataset}")
            return
        label_path = args.out or os.path.join(LABELS_DIR, f"pick_labels_{args.dataset}.json")

    labels = load_labels(label_path)

    if args.resume:
        videos = [(sid, mp4) for sid, mp4 in videos
                  if labels.get(sid) not in (LABEL_PICK, LABEL_NOT_PICK)]

    total_seqs = len(videos)
    if total_seqs == 0:
        print("No sequences to label.")
        return

    print(f"\n  Dataset  : {args.dataset}")
    print(f"  Total    : {total_seqs} sequences")
    print(f"  Labels   : {label_path}")
    already = sum(1 for _, mp4 in videos
                  if labels.get(next((s for s, m in videos if m == mp4), "")) in
                  (LABEL_PICK, LABEL_NOT_PICK))
    print(f"  Progress : {already}/{total_seqs} labeled")
    print(f"\n  ENTER=Pick  SPACE=NotPick  S=Skip  ← →=Prev/Next  Q=Quit\n")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, DISP_H + 80)

    seq_idx = args.start
    cached_frames = {}    # seq_id → (frames, orig_fps, total)

    while 0 <= seq_idx < total_seqs:
        seq_id, mp4_path = videos[seq_idx]
        current_label = labels.get(seq_id)

        # Load frames (cached)
        if seq_id not in cached_frames:
            print(f"  Loading [{seq_idx+1}/{total_seqs}] {seq_id}...")
            cached_frames[seq_id] = load_video_frames(mp4_path)
        frames, orig_fps, total_frames = cached_frames[seq_id]

        if not frames:
            print(f"  ⚠  No frames: {mp4_path}")
            seq_idx += 1
            continue

        # Keep cache small (max 5 sequences)
        if len(cached_frames) > 5:
            oldest = next(iter(cached_frames))
            del cached_frames[oldest]

        # ── Playback loop ─────────────────────────────────────────────────────
        fi = 0
        ms_per_frame = max(1, int(1000 / FPS_DISPLAY))
        action = None   # 'pick', 'not_pick', 'skip', 'prev', 'next', 'quit'

        while action is None:
            frame = frames[fi]
            disp = draw_overlay(frame, seq_id, fi, len(frames), seq_idx, total_seqs,
                                current_label, orig_fps, ms_per_frame)
            cv2.imshow(WIN, disp)

            key = cv2.waitKey(ms_per_frame) & 0xFF

            if key == 13 or key == 10:        # ENTER
                action = 'pick'
            elif key == ord(' '):             # SPACE
                action = 'not_pick'
            elif key == ord('s') or key == ord('S'):
                action = 'skip'
            elif key == ord('q') or key == ord('Q'):
                action = 'quit'
            elif key == 81 or key == 2424832: # ← left arrow
                action = 'prev'
            elif key == 83 or key == 2555904: # → right arrow
                action = 'next'

            # Advance frame (loop)
            fi = (fi + 1) % len(frames)

        # ── Handle action ─────────────────────────────────────────────────────
        if action == 'quit':
            save_labels(labels, label_path)
            print(f"\n  Saved and quit. Labels: {label_path}")
            break

        elif action == 'prev':
            seq_idx = max(0, seq_idx - 1)
            continue

        elif action == 'next':
            seq_idx += 1
            continue

        elif action in ('pick', 'not_pick', 'skip'):
            label_val = {
                'pick':     LABEL_PICK,
                'not_pick': LABEL_NOT_PICK,
                'skip':     LABEL_SKIP,
            }[action]
            labels[seq_id] = label_val
            save_labels(labels, label_path)

            icon = "✅" if label_val == LABEL_PICK else ("❌" if label_val == LABEL_NOT_PICK else "⏭")
            print(f"  {icon}  [{seq_idx+1}/{total_seqs}]  {seq_id}  → {label_val}")
            seq_idx += 1

    cv2.destroyAllWindows()

    # ── Final summary ─────────────────────────────────────────────────────────
    pick_count     = sum(1 for v in labels.values() if v == LABEL_PICK)
    not_pick_count = sum(1 for v in labels.values() if v == LABEL_NOT_PICK)
    skip_count     = sum(1 for v in labels.values() if v == LABEL_SKIP)

    print(f"\n{'='*50}")
    print(f"  ✅ Pick     : {pick_count}")
    print(f"  ❌ Not pick : {not_pick_count}")
    print(f"  ⏭  Skipped  : {skip_count}")
    print(f"  Total done  : {pick_count + not_pick_count}")
    print(f"  Output      : {label_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
