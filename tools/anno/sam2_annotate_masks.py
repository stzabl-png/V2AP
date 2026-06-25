#!/usr/bin/env python3
"""
sam2_annotate_masks.py — SAM2 点击提示半自动 mask 标注（GUI 进程）

工作流:
  1. 显示序列帧（可翻帧找到最佳帧）
  2. 左键点击  = 前景点（绿色）→ SAM2 实时生成 mask
  3. 右键点击  = 背景点（红色）→ 精化 mask
  4. ENTER     = 保存当前帧 + mask → image.png + 0.png
  5. C         = 清除所有点重新标
  6. ← →       = 翻帧（同时清除点）
  7. S         = 跳过此序列
  8. Q         = 保存并退出

环境:
  conda activate base   ← GUI 环境，需有 opencv 显示支持
  python tools/sam2_annotate_masks.py
  python tools/sam2_annotate_masks.py --dataset arctic
  python tools/sam2_annotate_masks.py --only-auto   # 只标自动生成的
  python tools/sam2_annotate_masks.py --redo         # 全部重标
"""

import os, sys, json, argparse, subprocess
import numpy as np
import cv2
from natsort import natsorted
from PIL import Image
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_BASE    = os.path.join(config.DATA_HUB, "ProcessedData", "obj_recon_input")
RAW_BASE    = os.path.join(config.DATA_HUB, "RawData", "ThirdPersonRawData")
HOI4D_ROOT  = os.path.join(config.DATA_HUB, "RawData", "EgoRawData", "hoi4d", "HOI4D_release")
DATASETS    = ["arctic", "oakink", "ho3d_v3", "dexycb", "hoi4d"]
WIN       = "SAM2 Mask Annotator"
DISP_H    = 720

# hawor env python (has SAM2)
# Set HAWOR_PY env var to override, or set HAWOR_DIR to the hawor conda env root
def _find_hawor_python():
    if "HAWOR_PY" in os.environ:
        return os.environ["HAWOR_PY"]
    if "HAWOR_DIR" in os.environ:
        return os.path.join(os.environ["HAWOR_DIR"], "bin", "python")
    # Try common conda locations
    import shutil
    for candidate in [
        os.path.expanduser("~/anaconda3/envs/hawor/bin/python"),
        os.path.expanduser("~/miniconda3/envs/hawor/bin/python"),
        os.path.expanduser("~/miniforge3/envs/hawor/bin/python"),
        "/opt/conda/envs/hawor/bin/python",
    ]:
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "Cannot find hawor Python. Set HAWOR_PY=/path/to/hawor/bin/python "
        "or HAWOR_DIR=/path/to/hawor/env in your .env or environment."
    )
HAWOR_PY  = _find_hawor_python()
SAM2_SRV  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2_server.py")

# ── SAM2 subprocess client ────────────────────────────────────────────────────
class SAM2Client:
    def __init__(self):
        print("Starting SAM2 server (hawor env)...")
        self.proc = subprocess.Popen(
            [HAWOR_PY, SAM2_SRV],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, bufsize=1
        )
        resp = self._read()
        if resp.get("status") != "ready":
            raise RuntimeError(f"SAM2 server failed to start: {resp}")
        print("✅ SAM2 server ready")

    def _read(self):
        line = self.proc.stdout.readline()
        return json.loads(line) if line else {"status": "error"}

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def set_image(self, path):
        self._send({"cmd": "set_image", "path": path})
        return self._read()

    def predict(self, fg, bg):
        self._send({"cmd": "predict", "fg": fg, "bg": bg})
        return self._read()

    def quit(self):
        try:
            self._send({"cmd": "quit"})
            self.proc.wait(timeout=5)
        except:
            self.proc.kill()


# ── Dataset frame finders ─────────────────────────────────────────────────────
def get_all_frames(ds, seq_id):
    if ds == "arctic":
        subj, seq = seq_id.split("__", 1)
        d = os.path.join(RAW_BASE, "arctic", subj, seq, "1")
        return natsorted(glob(os.path.join(d, "*.jpg")) + glob(os.path.join(d, "*.png")))
    elif ds == "oakink":
        d = os.path.join(RAW_BASE, "oakink_v1", seq_id)
        return natsorted(glob(os.path.join(d, "*", "north_west_color_*.png")) +
                         glob(os.path.join(d, "north_west_color_*.png")))
    elif ds == "ho3d_v3":
        split, sid = seq_id.split("__", 1)
        d = os.path.join(RAW_BASE, "ho3d_v3", split, sid, "rgb")
        return natsorted(glob(os.path.join(d, "*.jpg")))
    elif ds == "dexycb":
        parts = seq_id.split("__")
        if len(parts) == 3:
            d = os.path.join(RAW_BASE, "dexycb", *parts)
            return natsorted(glob(os.path.join(d, "color_*.jpg")))
    elif ds == "hoi4d":
        # seq_id = ZY20210800001_H1_C1_N19_S100_s02_T1
        # mp4 at HOI4D_release/ZY.../H.../align_rgb/image.mp4
        path_parts = seq_id.split("_")
        # Reconstruct path: ZY... / H... / C... / N... / S... / s... / T...
        # The naming is ZY20210800001_H1_C1_N19_S100_s02_T1
        # → ZY20210800001 / H1 / C1 / N19 / S100 / s02 / T1
        zy = path_parts[0]   # ZY20210800001
        rest = path_parts[1:]  # ['H1','C1','N19','S100','s02','T1']
        mp4 = os.path.join(HOI4D_ROOT, zy, *rest, "align_rgb", "image.mp4")
        if not os.path.exists(mp4):
            return []
        # Extract frames to temp dir and return paths
        import tempfile, subprocess as _sp
        tmp = os.path.join(config.DATA_HUB, "ProcessedData", "_anno_tmp", seq_id)
        os.makedirs(tmp, exist_ok=True)
        existing = natsorted(glob(os.path.join(tmp, "*.jpg")))
        if not existing:
            _sp.run(["ffmpeg", "-y", "-i", mp4, "-q:v", "2",
                     os.path.join(tmp, "%06d.jpg")],
                    capture_output=True)
            existing = natsorted(glob(os.path.join(tmp, "*.jpg")))
        return existing
    return []


# ── Mouse state ───────────────────────────────────────────────────────────────
mouse_state = {'fg': [], 'bg': [], 'changed': False, 'mode': 'fg'}  # mode: 'fg' or 'bg'

def mouse_cb(event, x, y, flags, param):
    scale = param['scale']   # scale_orig = DISP_H / H
    if event == cv2.EVENT_LBUTTONDOWN:
        ox, oy = int(x / scale), int(y / scale)
        if mouse_state['mode'] == 'fg':
            mouse_state['fg'].append([ox, oy])
        else:
            mouse_state['bg'].append([ox, oy])
        mouse_state['changed'] = True


# ── UI rendering ──────────────────────────────────────────────────────────────
def draw_ui(frame_d, fi, total_frames, idx, total_seqs, ds, seq_id,
            is_manual, mask_img, fg_pts, bg_pts, scale, H, W, waiting, mode):
    out = frame_d.copy()
    dH, dW = out.shape[:2]

    # Mask overlay
    if mask_img is not None:
        mx = cv2.resize(mask_img, (dW, dH), interpolation=cv2.INTER_NEAREST)
        overlay = out.copy()
        overlay[mx > 0] = (0, 80, 255)
        out = cv2.addWeighted(out, 0.6, overlay, 0.4, 0)
        cnts, _ = cv2.findContours(mx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 100), 2)

    # Points
    for px, py in fg_pts:
        cv2.circle(out, (int(px*scale), int(py*scale)), 7, (0, 255, 80), -1)
        cv2.circle(out, (int(px*scale), int(py*scale)), 7, (255, 255, 255), 2)
    for px, py in bg_pts:
        cv2.circle(out, (int(px*scale), int(py*scale)), 7, (0, 0, 220), -1)
        cv2.circle(out, (int(px*scale), int(py*scale)), 7, (255, 255, 255), 2)

    # Header
    cv2.rectangle(out, (0, 0), (dW, 62), (12, 14, 22), -1)
    sc = (0, 200, 80) if is_manual else (0, 200, 220)
    cv2.circle(out, (14, 20), 8, sc, -1)
    cv2.putText(out, f'[{idx+1}/{total_seqs}] {ds} / {seq_id}',
                (30, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    # Mode indicator
    mode_color = (0, 255, 80) if mode == 'fg' else (0, 80, 220)
    mode_text  = '● FG (左键)' if mode == 'fg' else '● BG (左键)'
    cv2.putText(out, mode_text, (6, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1)
    cv2.putText(out, f'Frame {fi+1}/{total_frames}  B:切换FG/BG  C:清除  ←→:换帧  ENTER:保存  S:跳过  Q:退出',
                (160, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 160, 130), 1)

    # Stats
    info = f'FG:{len(fg_pts)} BG:{len(bg_pts)}'
    if mask_img is not None:
        cov = (mask_img > 0).mean() * 100
        info += f'  mask={cov:.1f}%'
    if waiting:
        info += '  [SAM2...]'
    cv2.putText(out, info, (dW - 220, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 220, 100), 1)

    # Frame bar
    prog = int(dW * fi / max(total_frames - 1, 1))
    cv2.rectangle(out, (0, 59), (prog, 63), sc, -1)
    cv2.rectangle(out, (0, 59), (dW, 63), (40, 40, 40), 1)

    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default=None, choices=DATASETS)
    parser.add_argument("--seq",       default=None,
                        help="Comma-separated seq_ids or substring filter")
    parser.add_argument("--only-auto", action="store_true")
    parser.add_argument("--redo",      action="store_true")
    parser.add_argument("--start",     type=int, default=0)
    args = parser.parse_args()

    sam2 = SAM2Client()

    # ── Sequence discovery ────────────────────────────────────────────────────
    def _discover_third_person(ds, raw_subdir, fn_glob_pairs):
        """Generic third-person frame discoverer."""
        input_dir = os.path.join(RAW_BASE, raw_subdir)
        if not os.path.isdir(input_dir):
            return []
        results = []
        for entry in natsorted(os.listdir(input_dir)):
            for pattern in fn_glob_pairs:
                frames = natsorted(glob(os.path.join(input_dir, entry, pattern)))
                if frames:
                    results.append((entry, frames))
                    break
        return results

    def discover_hoi4d_seqs(seq_filter=None):
        """HOI4D: yields (seq_id, [frame_jpg_paths])."""
        if not os.path.isdir(HOI4D_ROOT):
            return []
        results = []
        for mp4 in natsorted(glob(os.path.join(HOI4D_ROOT, "**", "image.mp4"),
                                  recursive=True)):
            rel = os.path.relpath(mp4, HOI4D_ROOT)
            seq = rel.replace(os.sep + "align_rgb" + os.sep + "image.mp4", "")
            seq_id = seq.replace(os.sep, "_")
            if seq_filter and seq_filter not in seq_id:
                continue
            results.append((seq_id, mp4))   # mp4 path; frames extracted lazily
        return results

    all_seqs = []
    datasets = [args.dataset] if args.dataset else DATASETS

    # Explicit seq_id list (comma-separated) takes priority
    seq_filter_list = None
    if args.seq and "," in args.seq:
        seq_filter_list = [s.strip() for s in args.seq.split(",")]

    for ds in datasets:
        if ds == "arctic":
            input_dir = os.path.join(RAW_BASE, "arctic")
            if os.path.isdir(input_dir):
                for subj in natsorted(os.listdir(input_dir)):
                    for seq in natsorted(os.listdir(os.path.join(input_dir, subj))):
                        cam_dir = os.path.join(input_dir, subj, seq, "1")
                        imgs = natsorted(glob(os.path.join(cam_dir, "*.jpg")) +
                                         glob(os.path.join(cam_dir, "*.png")))
                        if imgs:
                            all_seqs.append((ds, f"{subj}__{seq}", imgs))
        elif ds == "oakink":
            input_dir = os.path.join(RAW_BASE, "oakink_v1")
            if os.path.isdir(input_dir):
                for seq_id in natsorted(os.listdir(input_dir)):
                    imgs = natsorted(glob(os.path.join(input_dir, seq_id, "*",
                                                       "north_west_color_*.png")))
                    if imgs:
                        all_seqs.append((ds, seq_id, imgs))
        elif ds == "ho3d_v3":
            for split in ["train", "evaluation"]:
                d = os.path.join(RAW_BASE, "ho3d_v3", split)
                if not os.path.isdir(d):
                    continue
                for sid in natsorted(os.listdir(d)):
                    imgs = natsorted(glob(os.path.join(d, sid, "rgb", "*.jpg")))
                    if imgs:
                        all_seqs.append((ds, f"{split}__{sid}", imgs))
        elif ds == "dexycb":
            input_dir = os.path.join(RAW_BASE, "dexycb")
            if os.path.isdir(input_dir):
                for subj in natsorted(os.listdir(input_dir)):
                    for date in natsorted(os.listdir(os.path.join(input_dir, subj))):
                        for cam in natsorted(os.listdir(
                                os.path.join(input_dir, subj, date))):
                            d = os.path.join(input_dir, subj, date, cam)
                            imgs = natsorted(glob(os.path.join(d, "color_*.jpg")))
                            if imgs:
                                all_seqs.append((ds, f"{subj}__{date}__{cam}", imgs))
        elif ds == "hoi4d":
            sf = args.seq if args.seq and "," not in args.seq else None
            for seq_id, mp4_or_frames in discover_hoi4d_seqs(sf):
                # Store mp4 path; frames extracted on demand in get_all_frames
                all_seqs.append((ds, seq_id, [mp4_or_frames]))

    # Apply seq_id filter
    if seq_filter_list:
        all_seqs = [(ds, sid, f) for ds, sid, f in all_seqs
                    if any(flt in sid for flt in seq_filter_list)]
    elif args.seq and "," not in args.seq:
        all_seqs = [(ds, sid, f) for ds, sid, f in all_seqs if args.seq in sid]

    # Skip already-annotated unless --redo
    filtered = []
    for ds, seq_id, frames in all_seqs:
        out_dir = os.path.join(OUT_BASE, "egocentric" if ds == "hoi4d" else ds, seq_id)
        img_out = os.path.join(out_dir, "image.png")
        src_out = os.path.join(out_dir, "source_frame.txt")
        if not args.redo and os.path.exists(img_out):
            src_txt = open(src_out).read() if os.path.exists(src_out) else ""
            if 'review=sam2' in src_txt:
                continue
        filtered.append((ds, seq_id, frames))
    all_seqs = filtered

    total_seqs = len(all_seqs)
    print(f"Sequences to annotate: {total_seqs}\n")
    if total_seqs == 0:
        print("All sequences already annotated. Use --redo to re-annotate.")
        sam2.quit(); return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    seq_idx = args.start
    while seq_idx < total_seqs:
        ds, seq_id, _frames_hint = all_seqs[seq_idx]
        # For HOI4D, frames are extracted from MP4 on demand via get_all_frames
        all_frames = get_all_frames(ds, seq_id) if ds == "hoi4d" else _frames_hint
        out_subdir = "egocentric" if ds == "hoi4d" else ds
        out_dir  = os.path.join(OUT_BASE, out_subdir, seq_id)
        img_out  = os.path.join(out_dir, "image.png")
        mask_out = os.path.join(out_dir, "0.png")
        src_out  = os.path.join(out_dir, "source_frame.txt")

        if not all_frames:
            seq_idx += 1; continue

        # Default to frame from source_frame.txt
        fi = 2
        if os.path.exists(src_out):
            for line in open(src_out):
                if line.startswith("frame_idx="):
                    try: fi = int(line.split("=")[1])
                    except: pass
        fi = max(0, min(fi, len(all_frames) - 1))

        is_manual = os.path.exists(src_out) and 'auto=True' not in open(src_out).read()

        # State
        current_path = None
        current_mask = None   # uint8 H,W
        H = W = scale = None
        frame_d = None
        waiting = False

        mouse_state['fg'] = []
        mouse_state['bg'] = []
        mouse_state['changed'] = False

        print(f"\n  [{seq_idx+1}/{total_seqs}] {ds}/{seq_id}  ({len(all_frames)} frames)")
        action = None

        while True:
            fi = max(0, min(fi, len(all_frames) - 1))
            fpath = all_frames[fi]

            # Load new frame
            if fpath != current_path:
                img_pil  = Image.open(fpath).convert("RGB")
                H, W     = img_pil.height, img_pil.width
                scale    = DISP_H / H
                frame_d  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                frame_d  = cv2.resize(frame_d, (int(W * scale), DISP_H))

                cv2.resizeWindow(WIN, int(W * scale), DISP_H)
                cv2.setMouseCallback(WIN, mouse_cb, {'scale': scale})  # pass scale_orig

                # Tell SAM2 server about new image
                sam2.set_image(fpath)
                current_path = fpath
                current_mask = None
                mouse_state['fg'] = []
                mouse_state['bg'] = []
                mouse_state['changed'] = False
                waiting = False

            # Run SAM2 if user clicked
            if mouse_state['changed'] and mouse_state['fg']:
                waiting = True
                resp = sam2.predict(mouse_state['fg'], mouse_state['bg'])
                waiting = False
                mouse_state['changed'] = False
                if resp.get("status") == "ok" and resp.get("mask_path"):
                    current_mask = cv2.imread(resp["mask_path"], 0)
                else:
                    current_mask = None

            disp = draw_ui(frame_d, fi, len(all_frames), seq_idx, total_seqs,
                           ds, seq_id, is_manual, current_mask,
                           mouse_state['fg'], mouse_state['bg'],
                           scale, H, W, waiting, mouse_state['mode'])
            cv2.imshow(WIN, disp)
            key = cv2.waitKey(30) & 0xFF

            if   key in (81, 2):  fi -= 1; continue         # ←
            elif key in (83, 3):  fi += 1; continue          # →
            elif key == 85:       fi = max(0, fi - 10); continue
            elif key == 86:       fi = min(len(all_frames)-1, fi+10); continue
            elif key in (ord('b'), ord('B')):   # toggle fg/bg mode
                mouse_state['mode'] = 'bg' if mouse_state['mode'] == 'fg' else 'fg'
                print(f"  Mode: {mouse_state['mode'].upper()}")
            elif key in (ord('c'), ord('C')):
                mouse_state['fg'] = []; mouse_state['bg'] = []
                mouse_state['mode'] = 'fg'
                current_mask = None
            elif key in (ord('s'), ord('S')):
                action = 'skip'; break
            elif key in (ord('q'), ord('Q'), 27):
                action = 'quit'; break
            elif key in (13, 10):   # ENTER
                if current_mask is None:
                    print("  ⚠️  先点击物体设置前景点")
                    continue
                os.makedirs(out_dir, exist_ok=True)
                Image.open(fpath).convert("RGB").save(img_out)
                cv2.imwrite(mask_out, current_mask)
                with open(src_out, "w") as f:
                    f.write(f"{fpath}\nframe_idx={fi}\nreview=sam2\n")
                cov = (current_mask > 0).mean() * 100
                print(f"  ✅ {ds}/{seq_id}: frame={fi+1} "
                      f"fg={len(mouse_state['fg'])} mask={cov:.1f}%")
                action = 'next'; break

        if action == 'quit':
            print(f"\n  Quit at {seq_idx+1}/{total_seqs}")
            break
        elif action == 'skip':
            print(f"  ⏭  {ds}/{seq_id}")
        seq_idx += 1

    cv2.destroyAllWindows()
    sam2.quit()
    print(f"\n✅ 标注完成: {seq_idx}/{total_seqs}")


if __name__ == "__main__":
    main()
