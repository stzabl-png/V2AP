"""
Unified ZED stream client — receives any combination of left/right/depth/pointcloud/IMU.

Usage:
    python zed_stream_client.py --host <ip> --port 30000
    python zed_stream_client.py --host <ip> --viz-image          # matplotlib left/right/depth
    python zed_stream_client.py --host <ip> --viz-pc             # open3d point cloud
    python zed_stream_client.py --host <ip> --viz-imu            # matplotlib live IMU plot
    python zed_stream_client.py --host <ip> --viz-image --viz-pc --viz-imu  # all three
"""

raise NotImplementedError("This file is for reference.")
import argparse
import io
import socket
import struct
import threading
import time
from collections import deque

import lz4.block
import numpy as np

# ── Protocol constants ──
FRAME_HDR_FMT = "<4sQHH"
FRAME_HDR_SZ = struct.calcsize(FRAME_HDR_FMT)
SEG_HDR_FMT = "<BBIIII"  # type, enc, dim0, dim1, comp_sz, raw_sz
SEG_HDR_SZ = struct.calcsize(SEG_HDR_FMT)
MAGIC = b"ZS01"

TYPE_LEFT, TYPE_RIGHT, TYPE_DEPTH, TYPE_PC, TYPE_IMU = 0, 1, 2, 3, 4
ENC_JPEG, ENC_LZ4, ENC_RAW = 0, 1, 2

IMU_FMT = "<ffffffffff"  # ax ay az gx gy gz ox oy oz ow
IMU_SZ = struct.calcsize(IMU_FMT)


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Server closed")
        buf.extend(chunk)
    return bytes(buf)


def decode_jpeg(data):
    from PIL import Image
    return np.array(Image.open(io.BytesIO(data)))


def decode_pointcloud_lz4(raw: bytes, dim0: int, dim1: int, raw_sz: int) -> np.ndarray:
    """
    Point cloud segment (TYPE_PC, LZ4): server sends dim0 = number of points, dim1 = 4.
    Each point is four float32: x, y, z, w — same layout as ZED MEASURE::XYZRGBA (sl::float4).
    The fourth float holds RGBA packed in its 32-bit pattern (use floatBitsToUint in C/GL).
    """
    if dim1 != 4:
        raise ValueError(f"pointcloud dim1 must be 4 (floats per point), got {dim1}")
    expected = dim0 * 4 * np.dtype(np.float32).itemsize
    if raw_sz != expected:
        raise ValueError(f"pointcloud raw_sz mismatch: header {raw_sz} vs dim0*16={expected}")
    if len(raw) != raw_sz:
        raise ValueError(f"pointcloud payload length {len(raw)} != raw_sz {raw_sz}")
    # copy(): independent of decompress buffer; contiguous (N,4) for safe column views
    return np.frombuffer(raw, dtype=np.float32).reshape(dim0, 4).copy()


def xyzrgba_w_to_rgb_open3d(w: np.ndarray) -> np.ndarray:
    """
    ZED SDK packs color in the 4th float of XYZRGBA. Unpack like ZED GLViewer vertex shader:
    uint c = floatBitsToUint(w); R=c&0xFF; G=(c>>8)&0xFF; B=(c>>16)&0xFF
    Returns (N, 3) float64 in 0..1 for Open3D.
    """
    w = np.asarray(w, dtype=np.float32)
    c = w.view(np.uint32)
    r = (c & np.uint32(0xFF)).astype(np.float64) * (1.0 / 255.0)
    g = ((c >> 8) & np.uint32(0xFF)).astype(np.float64) * (1.0 / 255.0)
    b = ((c >> 16) & np.uint32(0xFF)).astype(np.float64) * (1.0 / 255.0)
    return np.stack([r, g, b], axis=1)


def decode_segment(seg_type, encoding, dim0, dim1, payload, raw_sz):
    if encoding == ENC_JPEG:
        return decode_jpeg(payload)
    elif encoding == ENC_LZ4:
        raw = lz4.block.decompress(payload, uncompressed_size=raw_sz)
        if seg_type == TYPE_DEPTH:
            return np.frombuffer(raw, dtype=np.float32).reshape(dim1, dim0)
        elif seg_type == TYPE_PC:
            return decode_pointcloud_lz4(raw, dim0, dim1, raw_sz)
    elif encoding == ENC_RAW:
        if seg_type == TYPE_IMU:
            vals = struct.unpack(IMU_FMT, payload[:IMU_SZ])
            return {
                "accel": np.array(vals[0:3]),
                "gyro": np.array(vals[3:6]),
                "orient": np.array(vals[6:10]),
            }
    return payload


def depth_to_rgb_u8(d):
    """Depth float32 (m) → uint8 RGB for imshow; avoids NaN/inf divide warnings."""
    d = np.asarray(d, dtype=np.float64)
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    d_min = float(np.min(d[valid]))
    d_max = float(np.max(d[valid]))
    span = d_max - d_min
    if span <= 1e-8 or not np.isfinite(span):
        d_norm = np.zeros_like(d, dtype=np.float64)
        d_norm[valid] = 0.5
    else:
        d_norm = np.zeros_like(d, dtype=np.float64)
        d_norm[valid] = (d[valid] - d_min) / span
    d_norm = np.clip(d_norm, 0.0, 1.0)
    from matplotlib import cm
    return (cm.viridis(d_norm)[:, :, :3] * 255).astype(np.uint8)


def receive_loop(sock, latest, lock, running):
    """Background thread: receive frames and store latest data."""
    frame_count = 0
    fps_timer = time.time()

    while running[0]:
        try:
            hdr = recv_exact(sock, FRAME_HDR_SZ)
        except ConnectionError:
            break
        magic, ts_ns, ch_mask, n_seg = struct.unpack(FRAME_HDR_FMT, hdr)
        if magic != MAGIC:
            continue

        frame = {"timestamp_ns": ts_ns, "channel_mask": ch_mask}
        for _ in range(n_seg):
            sh = recv_exact(sock, SEG_HDR_SZ)
            seg_type, enc, d0, d1, comp_sz, raw_sz = struct.unpack(SEG_HDR_FMT, sh)
            payload = recv_exact(sock, comp_sz)
            data = decode_segment(seg_type, enc, d0, d1, payload, raw_sz)

            if seg_type == TYPE_LEFT:
                frame["left"] = data
            elif seg_type == TYPE_RIGHT:
                frame["right"] = data
            elif seg_type == TYPE_DEPTH:
                frame["depth"] = data
            elif seg_type == TYPE_PC:
                frame["pc"] = data
            elif seg_type == TYPE_IMU:
                frame["imu"] = data

        with lock:
            latest[0] = frame

        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 2.0:
            fps = frame_count / elapsed
            keys = [k for k in frame if k not in ("timestamp_ns", "channel_mask")]
            print(f"[RX] {fps:.1f} fps | channels: {', '.join(keys)}")
            frame_count = 0
            fps_timer = time.time()


def matplotlib_main_loop(latest, lock, running, show_image, show_imu):
    """
    Run Matplotlib on the main thread only (TkAgg / Tcl are not thread-safe).
    Receives updates via `latest` filled by receive_loop in a background thread.
    """
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig_img = axes = displays = None
    if show_image:
        fig_img, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig_img.suptitle("ZED Stream — Left | Right | Depth")
        displays = [ax.imshow(np.zeros((100, 100, 3), dtype=np.uint8)) for ax in axes]
        for ax, t in zip(axes, ["Left", "Right", "Depth"]):
            ax.set_title(t)
            ax.axis("off")
        fig_img.tight_layout()

    fig_imu = ax1 = ax2 = lines_a = lines_g = None
    accel_hist = gyro_hist = t_hist = None
    imu_idx = 0
    if show_imu:
        N = 200
        accel_hist = deque(maxlen=N)
        gyro_hist = deque(maxlen=N)
        t_hist = deque(maxlen=N)
        fig_imu, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        fig_imu.suptitle("ZED IMU")
        ax1.set_ylabel("Accel (m/s²)")
        ax2.set_ylabel("Gyro (deg/s)")
        ax2.set_xlabel("Sample")
        lines_a = [ax1.plot([], [], label=l)[0] for l in ("ax", "ay", "az")]
        lines_g = [ax2.plot([], [], label=l)[0] for l in ("gx", "gy", "gz")]
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper left")
        fig_imu.tight_layout()

    plt.ion()
    # Non-blocking: GUI must stay on this (main) thread.
    plt.show(block=False)

    try:
        while running[0]:
            with lock:
                frame = latest[0]

            if show_image and frame is not None and fig_img is not None:
                if "left" in frame:
                    displays[0].set_data(frame["left"])
                    displays[0].set_extent(
                        [0, frame["left"].shape[1], frame["left"].shape[0], 0]
                    )
                if "right" in frame:
                    displays[1].set_data(frame["right"])
                    displays[1].set_extent(
                        [0, frame["right"].shape[1], frame["right"].shape[0], 0]
                    )
                if "depth" in frame:
                    d_color = depth_to_rgb_u8(frame["depth"])
                    displays[2].set_data(d_color)
                    displays[2].set_extent([0, d_color.shape[1], d_color.shape[0], 0])
                fig_img.canvas.draw_idle()
                fig_img.canvas.flush_events()

            if show_imu and frame is not None and "imu" in frame and fig_imu is not None:
                imu = frame["imu"]
                accel_hist.append(imu["accel"])
                gyro_hist.append(imu["gyro"])
                t_hist.append(imu_idx)
                imu_idx += 1
                if len(t_hist) >= 2:
                    t = list(t_hist)
                    a = np.array(list(accel_hist))
                    g = np.array(list(gyro_hist))
                    for i, line in enumerate(lines_a):
                        line.set_data(t, a[:, i])
                    for i, line in enumerate(lines_g):
                        line.set_data(t, g[:, i])
                    for ax in (ax1, ax2):
                        ax.set_xlim(t[0], t[-1])
                        ax.relim()
                        ax.autoscale_view(scalex=False)
                fig_imu.canvas.draw_idle()
                fig_imu.canvas.flush_events()

            plt.pause(0.03)
    except KeyboardInterrupt:
        raise
    finally:
        if fig_img is not None:
            plt.close(fig_img)
        if fig_imu is not None:
            plt.close(fig_imu)


def viz_pc_loop(latest, lock, running):
    """Open3D point cloud viewer."""
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window("ZED Point Cloud", width=1024, height=768)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    added = True

    while running[0]:
        with lock:
            frame = latest[0]
        if frame is None or "pc" not in frame:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.03)
            continue

        pts = frame["pc"]  # (N, 4): x, y, z, w — w is color packed as float (ZED XYZRGBA)
        xyz = pts[:, :3].astype(np.float64)
        rgb = xyzrgba_w_to_rgb_open3d(pts[:, 3])

        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        if added:
            vis.update_geometry(pcd)
            vis.reset_view_point(True)
            added = False
        else:
            vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="ZED unified stream client")
    parser.add_argument("--host", type=str, default=os.environ.get("CAMERA_IP", "<your-camera-ip>"))
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--viz-image", action="store_true", help="Visualize left/right/depth (matplotlib)")
    parser.add_argument("--viz-pc", action="store_true", help="Visualize point cloud (open3d)")
    parser.add_argument("--viz-imu", action="store_true", help="Visualize IMU data (matplotlib)")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect((args.host, args.port))
    print(f"Connected to {args.host}:{args.port}")

    lock = threading.Lock()
    latest = [None]
    running = [True]

    rx_thread = threading.Thread(target=receive_loop, args=(sock, latest, lock, running), daemon=True)
    rx_thread.start()

    viz_threads = []
    if args.viz_pc:
        t = threading.Thread(target=viz_pc_loop, args=(latest, lock, running), daemon=True)
        t.start()
        viz_threads.append(t)

    try:
        if args.viz_image or args.viz_imu:
            # Matplotlib + Tk: must run on main thread (not worker thread).
            matplotlib_main_loop(latest, lock, running, args.viz_image, args.viz_imu)
        else:
            while running[0]:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        running[0] = False
        sock.close()
        for t in viz_threads:
            t.join(timeout=2)
        print("Done.")


if __name__ == "__main__":
    main()
