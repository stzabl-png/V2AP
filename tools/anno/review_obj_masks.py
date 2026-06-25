#!/usr/bin/env python3
"""
review_obj_masks.py — 快速审阅全部 obj_recon_input mask

显示每个序列的 image.png + 0.png overlay，支持快速翻页和修正

操作:
  SPACE / → / D   下一个（确认当前 OK）
  ← / A           上一个
  R               对当前序列重新画 bbox（激活画框模式）
  ENTER           画框后保存新 mask
  ESC / Q         退出

标注状态:
  绿色边框 = 手动标注
  黄色边框 = 自动生成（第3帧+深度前景）

用法:
  conda activate base
  python tools/review_obj_masks.py                    # 全部
  python tools/review_obj_masks.py --dataset dexycb
  python tools/review_obj_masks.py --dataset arctic --only-auto  # 只看自动生成的
"""

import os, sys, argparse
import cv2, numpy as np
from natsort import natsorted
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

OUT_BASE = os.path.join(config.DATA_HUB, "ProcessedData", "obj_recon_input")
DATASETS = ["arctic", "oakink", "ho3d_v3", "dexycb"]
WIN      = "Mask Review"
DISP_H   = 720

# ── Mouse state ───────────────────────────────────────────────────────────────
bbox_state = {'drawing': False, 'x0':0,'y0':0,'x1':0,'y1':0, 'done': False, 'active': False}

def mouse_cb(event, x, y, flags, param):
    if not bbox_state['active']: return
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox_state.update(drawing=True, x0=x,y0=y,x1=x,y1=y, done=False)
    elif event == cv2.EVENT_MOUSEMOVE and bbox_state['drawing']:
        bbox_state.update(x1=x, y1=y)
    elif event == cv2.EVENT_LBUTTONUP:
        bbox_state.update(drawing=False, x1=x, y1=y, done=True)


def load_mask_overlay(img_bgr, mask_path, scale):
    """Return display frame with mask overlay."""
    disp = img_bgr.copy()
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, 0)
        if mask is not None:
            mask = cv2.resize(mask, (disp.shape[1], disp.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            overlay = disp.copy()
            overlay[mask > 0] = (0, 100, 255)          # blue = mask region
            disp = cv2.addWeighted(disp, 0.65, overlay, 0.35, 0)
            # Draw mask contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(disp, contours, -1, (0, 255, 100), 2)
    return disp


def draw_ui(disp, idx, total, ds, seq_id, is_manual, redraw_mode):
    H, W = disp.shape[:2]
    # Top bar
    bar_color = (10, 10, 10)
    cv2.rectangle(disp, (0, 0), (W, 58), bar_color, -1)

    # Status indicator
    status_color = (0, 200, 80) if is_manual else (0, 200, 220)
    status_text  = "MANUAL" if is_manual else "AUTO"
    cv2.circle(disp, (16, 18), 8, status_color, -1)
    cv2.putText(disp, status_text, (28, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

    cv2.putText(disp, f'[{idx+1}/{total}]  {ds} / {seq_id}',
                (100, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)

    if redraw_mode:
        cv2.putText(disp, '✏ REDRAW MODE — drag bbox, ENTER to save, ESC cancel',
                    (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1)
    else:
        cv2.putText(disp, 'SPACE/→:next  ←:prev  R:redraw  Q:quit',
                    (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

    # Progress bar
    prog = int(W * (idx / max(total-1, 1)))
    cv2.rectangle(disp, (0, 56), (prog, 60), status_color, -1)
    cv2.rectangle(disp, (0, 56), (W,   60), (40,40,40), 1)

    # Draw live bbox
    if bbox_state['active'] and (bbox_state['done'] or bbox_state['drawing']):
        x0 = min(bbox_state['x0'], bbox_state['x1'])
        y0 = min(bbox_state['y0'], bbox_state['y1'])
        x1 = max(bbox_state['x0'], bbox_state['x1'])
        y1 = max(bbox_state['y0'], bbox_state['y1'])
        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 80), 2)
        if bbox_state['done']:
            cv2.putText(disp, 'ENTER to save',
                        (x0, max(y0-6, 65)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,80), 1)

    return disp


def save_new_mask(img_path, mask_out, scale, H, W):
    x0 = int(min(bbox_state['x0'], bbox_state['x1']) / scale)
    y0 = int(min(bbox_state['y0'], bbox_state['y1']) / scale)
    x1 = int(max(bbox_state['x0'], bbox_state['x1']) / scale)
    y1 = int(max(bbox_state['y0'], bbox_state['y1']) / scale)
    x0,x1 = max(0,x0), min(W,x1)
    y0,y1 = max(0,y0), min(H,y1)
    if x1-x0 < 5 or y1-y0 < 5:
        return False, "Bbox too small"
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    cv2.imwrite(mask_out, mask)
    # Mark as manual
    src_file = mask_out.replace("0.png", "source_frame.txt")
    if os.path.exists(src_file):
        content = open(src_file).read().split('\n')
        content = [l for l in content if 'auto=' not in l]
        content.append('reviewed=True')
        open(src_file, 'w').write('\n'.join(content))
    return True, f"bbox=[{x0},{y0},{x1},{y1}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default=None, choices=DATASETS)
    parser.add_argument("--only-auto", action="store_true", help="Only show auto-generated masks")
    parser.add_argument("--start",     type=int, default=0, help="Start from sequence N")
    args = parser.parse_args()

    # Collect all
    all_seqs = []
    datasets = [args.dataset] if args.dataset else DATASETS
    for ds in datasets:
        ds_dir = os.path.join(OUT_BASE, ds)
        if not os.path.isdir(ds_dir): continue
        for seq_id in natsorted(os.listdir(ds_dir)):
            img_path  = os.path.join(ds_dir, seq_id, "image.png")
            mask_path = os.path.join(ds_dir, seq_id, "0.png")
            src_path  = os.path.join(ds_dir, seq_id, "source_frame.txt")
            if not (os.path.exists(img_path) and os.path.exists(mask_path)):
                continue
            # Detect manual vs auto
            is_manual = False
            if os.path.exists(src_path):
                txt = open(src_path).read()
                is_manual = 'auto=True' not in txt or 'reviewed=True' in txt
            if args.only_auto and is_manual:
                continue
            all_seqs.append((ds, seq_id, img_path, mask_path, is_manual))

    total = len(all_seqs)
    print(f"Found {total} sequences to review\n")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, mouse_cb)

    idx       = args.start
    redraw_mode = False

    while 0 <= idx < total:
        ds, seq_id, img_path, mask_path, is_manual = all_seqs[idx]

        # Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            idx += 1; continue
        H, W = img_bgr.shape[:2]
        scale = DISP_H / H
        img_d = cv2.resize(img_bgr, (int(W*scale), DISP_H))

        if not redraw_mode:
            bbox_state.update(active=False, done=False, drawing=False)

        cv2.resizeWindow(WIN, img_d.shape[1], DISP_H)

        while True:
            # Draw overlay
            disp = load_mask_overlay(img_d.copy(), mask_path, scale)
            disp = draw_ui(disp, idx, total, ds, seq_id, is_manual, redraw_mode)
            cv2.imshow(WIN, disp)
            key = cv2.waitKey(30) & 0xFF

            if not redraw_mode:
                # Navigation
                if key in (32, 83, ord('d'), ord('D'), 3):  # SPACE/→/D
                    idx += 1; break
                elif key in (81, ord('a'), ord('A'), 2):     # ←/A
                    idx = max(0, idx - 1); break
                elif key in (ord('r'), ord('R')):
                    redraw_mode = True
                    bbox_state.update(active=True, done=False, drawing=False)
                elif key in (ord('q'), ord('Q'), 27):         # Q/ESC
                    print(f"\nExited at {idx+1}/{total}")
                    cv2.destroyAllWindows()
                    return

            else:  # Redraw mode
                if key in (13, 10):  # ENTER
                    if bbox_state['done']:
                        ok, msg = save_new_mask(img_path, mask_path, scale, H, W)
                        if ok:
                            print(f"  ✅ {ds}/{seq_id}: {msg}")
                            is_manual = True
                            redraw_mode = False
                            bbox_state['active'] = False
                            break  # refresh with new mask
                        else:
                            print(f"  ⚠️  {msg}")
                    else:
                        print("  Draw a bbox first!")
                elif key == 27:  # ESC → cancel redraw
                    redraw_mode = False
                    bbox_state.update(active=False, done=False, drawing=False)
                    break

    cv2.destroyAllWindows()
    print(f"\n✅ Review complete. Reviewed {idx} / {total} sequences.")


if __name__ == "__main__":
    main()
