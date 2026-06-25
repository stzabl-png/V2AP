#!/usr/bin/env python3
"""
Interactive ROI + depth segmentation tool.

1. 打开 RGB 图, 用鼠标画框选中物体
2. 框内自动用深度区分 物体 vs 桌面
3. 可视化结果, 按 Enter 确认 / r 重画
4. 保存 mask.png → 供 graspnet_from_rgbd --seg mask 使用

Usage:
    python -m s2r.draw_roi_mask \
        --session-dir data_hub/sessions/sessions/20260602_192346_chips
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent


# ── Mouse callback state ──────────────────────────────────

class RoiDrawer:
    """Mouse-driven rectangle drawer on an OpenCV window."""

    def __init__(self, img: np.ndarray, window_name: str = "Draw ROI"):
        self.img = img.copy()
        self.display = img.copy()
        self.window_name = window_name
        self.drawing = False
        self.x0 = self.y0 = self.x1 = self.y1 = 0
        self.roi_set = False

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x0, self.y0 = x, y
            self.x1, self.y1 = x, y
            self.roi_set = False
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.x1, self.y1 = x, y
            self.display = self.img.copy()
            cv2.rectangle(self.display, (self.x0, self.y0), (self.x1, self.y1),
                          (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.x1, self.y1 = x, y
            self.roi_set = True
            self.display = self.img.copy()
            cv2.rectangle(self.display, (self.x0, self.y0), (self.x1, self.y1),
                          (0, 255, 0), 2)

    def get_roi(self) -> tuple[int, int, int, int] | None:
        """Show window, let user draw. Returns (x, y, w, h) or None."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 540)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

        print("\n  🖱️  在图像上拖拽鼠标画框选中物体")
        print("     Enter = 确认  |  r = 重画  |  q = 退出\n")

        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                return None
            elif key == 13 and self.roi_set:  # Enter
                break
            elif key == ord("r"):
                self.display = self.img.copy()
                self.roi_set = False

        cv2.destroyAllWindows()

        # Normalize coordinates
        x0 = min(self.x0, self.x1)
        y0 = min(self.y0, self.y1)
        x1 = max(self.x0, self.x1)
        y1 = max(self.y0, self.y1)
        w = x1 - x0
        h = y1 - y0
        if w < 5 or h < 5:
            print("  ⚠️  框太小，请重试")
            return None
        return (x0, y0, w, h)


# ── Depth-based object/table separation ───────────────────

def segment_object_in_roi(
    depth: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    roi: tuple[int, int, int, int],
    table_height_m: float,
    *,
    margin_above_table_m: float = 0.005,
    max_obj_height_m: float = 0.40,
) -> tuple[np.ndarray, dict]:
    """Separate object from table within ROI using base-frame Z height.

    Returns:
        mask: (H, W) uint8, 255 = object pixel
        info: dict with stats
    """
    H, W = depth.shape
    x0, y0, rw, rh = roi

    # Create pixel grid for full image
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

    # ROI mask
    in_roi = np.zeros((H, W), dtype=bool)
    in_roi[y0:y0+rh, x0:x0+rw] = True
    valid = in_roi & (depth > 0.01)

    # Backproject ROI pixels to 3D (camera frame)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth[valid]
    px = (u_grid[valid].astype(np.float32) - cx) * z / fx
    py = (v_grid[valid].astype(np.float32) - cy) * z / fy
    pts_cam = np.column_stack([px, py, z])

    # Camera → base frame
    R = T_base_cam[:3, :3]
    t = T_base_cam[:3, 3]
    pts_base = (R @ pts_cam.T).T + t

    # Height-based separation
    z_base = pts_base[:, 2]
    is_object = z_base > (table_height_m + margin_above_table_m)
    is_object &= z_base < (table_height_m + max_obj_height_m)

    # Build pixel mask
    # Map the boolean mask back to image coordinates
    obj_mask = np.zeros((H, W), dtype=np.uint8)
    roi_pixels = np.column_stack(np.where(valid))  # (N, 2) row, col
    obj_pixels = roi_pixels[is_object]
    if len(obj_pixels) > 0:
        obj_mask[obj_pixels[:, 0], obj_pixels[:, 1]] = 255

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel)

    # Stats
    obj_pts = pts_base[is_object]
    table_pts = pts_base[~is_object]
    info = {
        "n_roi_pixels": int(valid.sum()),
        "n_object_pixels": int(is_object.sum()),
        "n_table_pixels": int((~is_object).sum()),
    }
    if len(obj_pts) > 0:
        centroid = obj_pts.mean(axis=0)
        extents = obj_pts.max(axis=0) - obj_pts.min(axis=0)
        info.update({
            "centroid_base": centroid.tolist(),
            "extents_cm": (extents * 100).tolist(),
            "z_range": [float(obj_pts[:, 2].min()), float(obj_pts[:, 2].max())],
        })

    return obj_mask, info


# ── Visualization ─────────────────────────────────────────

def show_segmentation(
    rgb: np.ndarray,
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
    info: dict,
) -> bool:
    """Show segmentation overlay. Returns True if user confirms."""
    overlay = rgb.copy()

    # Color object pixels green, table pixels in ROI red
    x0, y0, rw, rh = roi
    roi_region = np.zeros_like(mask)
    roi_region[y0:y0+rh, x0:x0+rw] = 255

    # Object = green overlay
    obj_pixels = mask > 0
    overlay[obj_pixels] = (overlay[obj_pixels] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)

    # Table in ROI = red overlay (everything in ROI that's not object)
    table_pixels = (roi_region > 0) & (~obj_pixels)
    overlay[table_pixels] = (overlay[table_pixels] * 0.7 + np.array([0, 0, 200]) * 0.3).astype(np.uint8)

    # Draw ROI rectangle
    cv2.rectangle(overlay, (x0, y0), (x0+rw, y0+rh), (0, 255, 0), 2)

    # Text info
    n_obj = info.get("n_object_pixels", 0)
    ext = info.get("extents_cm", [0, 0, 0])
    text_lines = [
        f"Object: {n_obj} px",
        f"Size: {ext[0]:.0f}x{ext[1]:.0f}x{ext[2]:.0f} cm",
    ]
    for i, line in enumerate(text_lines):
        cv2.putText(overlay, line, (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.namedWindow("Segmentation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Segmentation", 960, 540)

    print(f"\n  绿色 = 物体 ({n_obj} pixels)")
    print(f"  红色 = 桌面")
    print(f"  物体大小: X={ext[0]:.1f} Y={ext[1]:.1f} Z={ext[2]:.1f} cm")
    if "centroid_base" in info:
        c = info["centroid_base"]
        print(f"  物体中心 (base): [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]")
    print(f"\n  Enter = 确认保存  |  r = 重画  |  q = 放弃")

    while True:
        cv2.imshow("Segmentation", overlay)
        key = cv2.waitKey(30) & 0xFF
        if key == 13:  # Enter
            cv2.destroyAllWindows()
            return True
        elif key == ord("r"):
            cv2.destroyAllWindows()
            return False
        elif key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit(0)

    return False


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive: draw ROI → depth-segment object → save mask"
    )
    parser.add_argument("--session-dir", type=str, required=True,
                        help="Path to session directory")
    parser.add_argument("--table-height", type=float, default=None,
                        help="Override table height (meters)")
    parser.add_argument("--run-graspnet", action="store_true",
                        help="After mask, auto-run graspnet_from_rgbd")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for GraspNet (if --run-graspnet)")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    input_dir = session_dir / "input"

    # Load data
    rgb = cv2.imread(str(input_dir / "rgb" / "left_rgb.png"))
    depth = np.load(str(input_dir / "depth" / "depth.npy"))
    K = np.load(str(input_dir / "calib" / "K.npy"))

    with open(input_dir / "calib" / "extrinsics.json") as f:
        T_base_cam = np.array(json.load(f)["T_base_cam"])

    if args.table_height is not None:
        table_height = args.table_height
    else:
        table_path = input_dir / "scene" / "table.json"
        if table_path.exists():
            with open(table_path) as f:
                table_height = json.load(f).get("table_height_m", 0.98)
        else:
            table_height = 0.98

    print(f"\n  Session: {session_dir.name}")
    print(f"  Image:   {rgb.shape[1]}×{rgb.shape[0]}")
    print(f"  Table:   {table_height:.2f}m")

    # Interactive loop: draw → segment → confirm
    while True:
        drawer = RoiDrawer(rgb)
        roi = drawer.get_roi()
        if roi is None:
            print("  取消")
            return

        print(f"  ROI: ({roi[0]},{roi[1]},{roi[2]},{roi[3]})")

        # Segment
        mask, info = segment_object_in_roi(
            depth, K, T_base_cam, roi, table_height
        )

        if info.get("n_object_pixels", 0) < 50:
            print("  ⚠️  物体点太少! 试着画大一点的框, 或检查 table_height")
            continue

        # Show result
        confirmed = show_segmentation(rgb, mask, roi, info)
        if confirmed:
            break
        # else: loop back to draw again

    # Save mask
    seg_dir = session_dir / "output" / "segment"
    seg_dir.mkdir(parents=True, exist_ok=True)
    mask_path = seg_dir / "mask.png"
    cv2.imwrite(str(mask_path), mask)
    print(f"\n  ✅ Mask 已保存: {mask_path}")
    print(f"     物体: {info['n_object_pixels']} pixels")
    print(f"     大小: {info['extents_cm'][0]:.1f}×{info['extents_cm'][1]:.1f}×{info['extents_cm'][2]:.1f} cm")

    # Save ROI info
    roi_info = {
        "roi_xywh": list(roi),
        "table_height_m": table_height,
        **info,
    }
    with open(seg_dir / "roi_info.json", "w") as f:
        json.dump(roi_info, f, indent=2)

    # Run GraspNet if requested
    if args.run_graspnet:
        print(f"\n  正在运行 GraspNet...")
        from s2r.graspnet_from_rgbd import process_rgbd_session
        process_rgbd_session(
            session_dir,
            seg_mode="mask",
            mask_path=str(mask_path),
            device=args.device,
        )
    else:
        print(f"\n  下一步: 运行 GraspNet:")
        print(f"    python -m s2r.graspnet_from_rgbd \\")
        print(f"        --session-dir {session_dir} \\")
        print(f"        --seg mask --mask-path {mask_path} \\")
        print(f"        --device cuda")


if __name__ == "__main__":
    main()
