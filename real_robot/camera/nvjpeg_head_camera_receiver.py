#!/usr/bin/env python3
import argparse
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


FRAME_HDR_FMT = "<4sQHH"
FRAME_HDR_SZ = struct.calcsize(FRAME_HDR_FMT)
SEG_HDR_FMT = "<BBIIII"  # type, enc, dim0, dim1, comp_sz, raw_sz
SEG_HDR_SZ = struct.calcsize(SEG_HDR_FMT)
MAGIC = b"ZS01"

TYPE_LEFT = 0
ENC_JPEG = 0


@dataclass
class StreamFrame:
    ts_sec: float
    image_rgb: np.ndarray


class NVJPEGHeadCameraReceiver:
    """
    Receiver for the unified ZED TCP stream that extracts only the left RGB image.

    The public API mirrors `camera/head_camera_receiver.py` so this can be used as
    a near drop-in replacement for the existing teleop camera receiver.
    """

    def __init__(
        self,
        im_h: int,
        im_w: int,
        sender_ip: str,
        ports: Dict[str, int],
        verbose: bool = True,
    ):
        if not ports:
            raise ValueError("ports must contain exactly one head camera stream")
        if len(ports) != 1:
            raise ValueError(
                "NVJPEGHeadCameraReceiver expects exactly one stream name -> TCP port entry"
            )

        self.im_h = int(im_h)
        self.im_w = int(im_w)
        self.sender_ip = sender_ip
        self.ports = ports
        self.verbose = verbose

        self.stream_name, self.port = next(iter(ports.items()))

        self._lock = threading.Lock()
        self._buffer: Optional[StreamFrame] = None
        self._ready_event = threading.Event()

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._sock: Optional[socket.socket] = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(0.5)
        self._sock.connect((self.sender_ip, self.port))

    def start_receiving(self, timeout: float | None = None):
        if self._thread is not None:
            raise RuntimeError("Receiver already started")

        if self.verbose:
            print(
                f"[TCP RX HEAD] {self.stream_name} connecting to "
                f"tcp://{self.sender_ip}:{self.port}"
            )

        self._stop_evt.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        if self.verbose:
            print(f"[TCP RX HEAD] Waiting for first frame from {self.stream_name}...")

        if not self._ready_event.wait(timeout=timeout):
            self.stop()
            raise TimeoutError(f"Timeout waiting for first frame from {self.stream_name}")

        if self.verbose:
            print("[TCP RX HEAD] Stream is live")

    def stop(self):
        self._stop_evt.set()

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

    def _prepare_output_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convert a decoded BGR frame into the RGB output expected by callers."""
        if frame_bgr.shape[0] != self.im_h or frame_bgr.shape[1] != self.im_w:
            frame_bgr = cv2.resize(
                frame_bgr,
                (self.im_w, self.im_h),
                interpolation=cv2.INTER_LINEAR,
            )
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def get_data(self, name: str) -> StreamFrame:
        if name != self.stream_name:
            raise KeyError(f"Unknown stream '{name}', expected '{self.stream_name}'")

        with self._lock:
            frame = self._buffer
            if frame is None:
                raise RuntimeError(f"No frame received yet for {name}")
            return StreamFrame(ts_sec=frame.ts_sec, image_rgb=frame.image_rgb)

    def _recv_exact(self, n_bytes: int) -> bytes | None:
        if self._sock is None:
            return None

        buf = bytearray()
        while len(buf) < n_bytes and not self._stop_evt.is_set():
            try:
                chunk = self._sock.recv(n_bytes - len(buf))
            except socket.timeout:
                continue
            except OSError:
                if self._stop_evt.is_set():
                    return None
                raise

            if not chunk:
                raise ConnectionError("Server closed")
            buf.extend(chunk)

        if self._stop_evt.is_set():
            return None
        return bytes(buf)

    def _decode_left_frame(self, encoding: int, payload: bytes) -> np.ndarray:
        if encoding != ENC_JPEG:
            raise ValueError(f"Unsupported left-image encoding: {encoding}")

        # Keep decoded frames in BGR to stay consistent with the existing
        # receiver's get_data() path, which converts BGR -> RGB before returning.
        frame_bgr = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame_bgr is None:
            raise ValueError("Failed to decode left JPEG payload")
        return frame_bgr

    def _recv_loop(self):
        if self.verbose:
            print("[TCP RX HEAD] Receive loop started")

        frame_count = 0
        fps_timer = time.time()

        while not self._stop_evt.is_set():
            try:
                hdr = self._recv_exact(FRAME_HDR_SZ)
                if hdr is None:
                    break

                magic, ts_ns, _channel_mask, n_seg = struct.unpack(FRAME_HDR_FMT, hdr)
                if magic != MAGIC:
                    if self.verbose:
                        print(f"[TCP RX HEAD] Invalid frame magic: {magic!r}")
                    continue

                left_frame_bgr = None

                for _ in range(n_seg):
                    seg_hdr = self._recv_exact(SEG_HDR_SZ)
                    if seg_hdr is None:
                        return

                    seg_type, encoding, _dim0, _dim1, comp_sz, _raw_sz = struct.unpack(
                        SEG_HDR_FMT, seg_hdr
                    )
                    payload = self._recv_exact(comp_sz)
                    if payload is None:
                        return

                    if seg_type != TYPE_LEFT:
                        continue

                    left_frame_bgr = self._decode_left_frame(encoding, payload)

                if left_frame_bgr is None:
                    continue

                output_frame = self._prepare_output_frame(left_frame_bgr)
                sf = StreamFrame(ts_sec=ts_ns * 1e-9, image_rgb=output_frame)

                with self._lock:
                    self._buffer = sf
                    self._ready_event.set()

                frame_count += 1
                elapsed = time.time() - fps_timer
                if self.verbose and elapsed >= 2.0:
                    fps = frame_count / elapsed
                    print(f"[TCP RX HEAD] {fps:.1f} fps | channel: left")
                    frame_count = 0
                    fps_timer = time.time()

            except ConnectionError as exc:
                if self.verbose and not self._stop_evt.is_set():
                    print(f"[TCP RX HEAD] Connection closed: {exc}")
                break
            except Exception as exc:
                if self.verbose and not self._stop_evt.is_set():
                    print(f"[TCP RX HEAD] Error decoding frame: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Receive only the left RGB image from the unified ZED TCP stream"
    )
    parser.add_argument("--host", type=str, default=os.environ.get("CAMERA_IP", "<your-camera-ip>"))
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--stream-name", type=str, default="HEAD_CAM")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Receive frames without opening an OpenCV window",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce receiver-side logging",
    )
    args = parser.parse_args()

    rx = NVJPEGHeadCameraReceiver(
        im_h=args.height,
        im_w=args.width,
        sender_ip=args.host,
        ports={args.stream_name: args.port},
        verbose=not args.quiet,
    )

    try:
        rx.start_receiving(timeout=args.timeout)
        print(
            f"Receiver fully initialized, {args.stream_name} left RGB stream is live"
        )

        last_ts = None
        frame_count = 0
        fps_timer = time.time()

        while True:
            head = rx.get_data(args.stream_name)

            if head.ts_sec != last_ts:
                last_ts = head.ts_sec
                frame_count += 1

            elapsed = time.time() - fps_timer
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                print(f"[MAIN] {fps:.1f} fps | ts={head.ts_sec:.6f}")
                frame_count = 0
                fps_timer = time.time()

            if not args.no_display:
                head_image_bgr = cv2.cvtColor(head.image_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow(args.stream_name, head_image_bgr)
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
