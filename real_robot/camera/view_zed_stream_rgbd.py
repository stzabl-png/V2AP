#!/usr/bin/env python3
"""
Live preview of zed_streamer left RGB + depth (TCP ZS01).

Prerequisite on dexmate-nano (depth enabled — omit --no-depth):
  cd zed_stream/
  sudo ./build/zed_streamer --clean --jpeg-quality 100 --max-fps 30 \\
    --resolution HD1080 --no-right --no-pc --no-imu

Usage:
  python camera/view_zed_stream_rgbd.py
  python camera/view_zed_stream_rgbd.py --host 192.168.50.22 --timeout 20
  python camera/view_zed_stream_rgbd.py --no-display   # FPS stats only

Requires: pip install lz4 opencv-python
Press q or Esc to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from camera.zed_stream_receiver import ZedStreamRgbdReceiver  # noqa: E402


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(d) & (d > 0)
    vis = np.zeros((*d.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return vis
    d_min = float(np.min(d[valid]))
    d_max = float(np.max(d[valid]))
    span = max(d_max - d_min, 1e-6)
    norm = np.zeros_like(d, dtype=np.float32)
    norm[valid] = ((d[valid] - d_min) / span).astype(np.float32)
    cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cm[~valid] = 0
    return cm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live zed_stream RGB + depth preview")
    p.add_argument("--host", type=str, default="192.168.50.22")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--width", type=int, default=640, help="Display width (resize)")
    p.add_argument("--height", type=int, default=360, help="Display height (resize)")
    p.add_argument("--timeout", type=float, default=15.0, help="Seconds to wait for first frame")
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open OpenCV windows (print FPS only)",
    )
    p.add_argument("--quiet", action="store_true", help="Less receiver logging")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    rx = ZedStreamRgbdReceiver(
        args.host,
        args.port,
        out_width=args.width,
        out_height=args.height,
        verbose=not args.quiet,
    )

    try:
        rx.start(timeout=args.timeout)
        print(f"Connected — streaming from tcp://{args.host}:{args.port}")

        last_ts: float | None = None
        frame_count = 0
        fps_timer = time.time()

        while True:
            frame = rx.get_rgbd()
            if frame.ts_sec != last_ts:
                last_ts = frame.ts_sec
                frame_count += 1

            elapsed = time.time() - fps_timer
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                valid = int((np.isfinite(frame.depth) & (frame.depth > 0)).sum())
                print(
                    f"[view] {fps:.1f} fps | ts={frame.ts_sec:.6f} | "
                    f"valid depth px={valid}/{frame.depth.size}"
                )
                frame_count = 0
                fps_timer = time.time()

            if not args.no_display:
                rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                depth_bgr = depth_to_colormap(frame.depth)
                cv2.imshow("zed_stream RGB", rgb_bgr)
                cv2.imshow("zed_stream depth", depth_bgr)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        rx.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
