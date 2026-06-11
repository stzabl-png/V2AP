#!/usr/bin/env python3
"""
Receive left RGB (JPEG) + depth (LZ4 float32) from dexmate-nano zed_streamer (TCP ZS01).

Requires on robot (dexmate-nano):
  cd zed_stream/
  sudo ./build/zed_streamer ... --no-right --no-pc --no-imu
  # omit --no-depth so depth is included

Default: tcp://<CAMERA_IP>:30000
"""

from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import lz4.block
except ImportError as e:
    lz4 = None  # type: ignore
    _LZ4_IMPORT_ERROR = e
else:
    _LZ4_IMPORT_ERROR = None

FRAME_HDR_FMT = "<4sQHH"
FRAME_HDR_SZ = struct.calcsize(FRAME_HDR_FMT)
SEG_HDR_FMT = "<BBIIII"
SEG_HDR_SZ = struct.calcsize(SEG_HDR_FMT)
MAGIC = b"ZS01"

TYPE_LEFT, TYPE_DEPTH = 0, 2
ENC_JPEG, ENC_LZ4 = 0, 1


@dataclass
class RgbdFrame:
    ts_sec: float
    rgb: np.ndarray  # (H,W,3) uint8 RGB
    depth: np.ndarray  # (H,W) float32 meters


class ZedStreamRgbdReceiver:
    """Background TCP client for zed_streamer left RGB + depth."""

    def __init__(
        self,
        host: str,
        port: int = 30000,
        *,
        out_width: int = 640,
        out_height: int = 360,
        verbose: bool = False,
    ) -> None:
        if _LZ4_IMPORT_ERROR is not None:
            raise ImportError(
                "lz4 required for zed_stream depth (pip install lz4)"
            ) from _LZ4_IMPORT_ERROR

        self.host = host
        self.port = int(port)
        self.out_width = int(out_width)
        self.out_height = int(out_height)
        self.verbose = verbose

        self._lock = threading.Lock()
        self._latest: RgbdFrame | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None

    def start(self, timeout: float = 15.0) -> None:
        if self._thread is not None:
            raise RuntimeError("Receiver already started")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(0.5)
        sock.connect((self.host, self.port))
        self._sock = sock

        if self.verbose:
            print(f"[zed_stream] connected to tcp://{self.host}:{self.port}")

        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        if not self._ready.wait(timeout=timeout):
            self.stop()
            raise TimeoutError(
                f"Timeout waiting for RGB+depth from zed_streamer at "
                f"{self.host}:{self.port} (is --no-depth removed on server?)"
            )

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_rgbd(self) -> RgbdFrame:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("No RGB-D frame yet")
            frame = self._latest
        return RgbdFrame(
            ts_sec=frame.ts_sec,
            rgb=frame.rgb.copy(),
            depth=frame.depth.copy(),
        )

    def _recv_exact(self, n_bytes: int) -> bytes | None:
        if self._sock is None:
            return None
        buf = bytearray()
        while len(buf) < n_bytes and not self._stop.is_set():
            try:
                chunk = self._sock.recv(n_bytes - len(buf))
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return None
                raise
            if not chunk:
                raise ConnectionError("zed_streamer closed connection")
            buf.extend(chunk)
        if self._stop.is_set():
            return None
        return bytes(buf)

    @staticmethod
    def _decode_depth(payload: bytes, dim0: int, dim1: int, raw_sz: int) -> np.ndarray:
        raw = lz4.block.decompress(payload, uncompressed_size=raw_sz)
        return np.frombuffer(raw, dtype=np.float32).reshape(dim1, dim0).copy()

    @staticmethod
    def _decode_left_jpeg(payload: bytes) -> np.ndarray:
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode left JPEG")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _resize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        if h == self.out_height and w == self.out_width:
            return rgb
        return cv2.resize(
            rgb, (self.out_width, self.out_height), interpolation=cv2.INTER_LINEAR
        )

    def _resize_depth(self, depth: np.ndarray, rgb_shape: tuple[int, int]) -> np.ndarray:
        if depth.shape[:2] == rgb_shape:
            return depth.astype(np.float32, copy=False)
        return cv2.resize(
            depth.astype(np.float32),
            (rgb_shape[1], rgb_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    def _publish_frame(self, ts_ns: int, left_rgb: np.ndarray, depth: np.ndarray) -> None:
        rgb = self._resize_rgb(left_rgb)
        depth_aligned = self._resize_depth(depth, rgb.shape[:2])
        frame = RgbdFrame(ts_sec=ts_ns * 1e-9, rgb=rgb, depth=depth_aligned)
        with self._lock:
            self._latest = frame
            self._ready.set()

    def _recv_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                hdr = self._recv_exact(FRAME_HDR_SZ)
                if hdr is None:
                    break
                magic, ts_ns, _ch_mask, n_seg = struct.unpack(FRAME_HDR_FMT, hdr)
                if magic != MAGIC:
                    continue

                left_rgb: np.ndarray | None = None
                depth: np.ndarray | None = None

                for _ in range(n_seg):
                    seg_hdr = self._recv_exact(SEG_HDR_SZ)
                    if seg_hdr is None:
                        return
                    seg_type, enc, dim0, dim1, comp_sz, raw_sz = struct.unpack(
                        SEG_HDR_FMT, seg_hdr
                    )
                    payload = self._recv_exact(comp_sz)
                    if payload is None:
                        return

                    if seg_type == TYPE_LEFT and enc == ENC_JPEG:
                        left_rgb = self._decode_left_jpeg(payload)
                    elif seg_type == TYPE_DEPTH and enc == ENC_LZ4:
                        depth = self._decode_depth(payload, dim0, dim1, raw_sz)

                if left_rgb is None or depth is None:
                    continue

                self._publish_frame(ts_ns, left_rgb, depth)
            except ConnectionError:
                break
            except Exception as exc:
                if self.verbose and not self._stop.is_set():
                    print(f"[zed_stream] decode error: {exc}")
