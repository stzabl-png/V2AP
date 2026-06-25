"""Viewport capture → MP4 for evaluation episodes."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess

from termcolor import cprint


class EpisodeVideoRecorder:
    """Record one evaluation episode via world.step hooks (PNG frames → ffmpeg)."""

    def __init__(
        self,
        mp4_path: str,
        *,
        record_every: int = 3,
        fps: int = 30,
        keep_frames: bool = False,
    ):
        self.mp4_path = os.path.abspath(mp4_path)
        self.record_every = max(1, int(record_every))
        self.fps = max(1, int(fps))
        self.keep_frames = bool(keep_frames)
        self._active = False
        self._frames_dir = os.path.join(os.path.dirname(self.mp4_path), ".video_frames_tmp")
        self._frame_idx = 0
        self._step_counter = 0
        self._viewport = None
        self._vu = None
        self._orig_world_step = None
        self.n_frames = 0
        self.encode_error: str | None = None
        os.makedirs(os.path.dirname(self.mp4_path) or ".", exist_ok=True)

    def _ensure_viewport(self) -> None:
        if self._viewport is not None:
            return
        import omni.kit.viewport.utility as vu

        self._vu = vu
        self._viewport = vu.get_active_viewport()
        if self._viewport is None:
            raise RuntimeError(
                "No active viewport for recording. Use headed mode or Xvfb (DISPLAY=:99)."
            )

    def attach_world(self, world) -> None:
        self._orig_world_step = world.step

        def step_with_capture(render=True):
            use_render = render or self._active
            result = self._orig_world_step(render=use_render)
            if self._active:
                self._maybe_capture()
            return result

        world.step = step_with_capture

    def _maybe_capture(self) -> None:
        self._step_counter += 1
        if self._step_counter % self.record_every != 0:
            return
        self._ensure_viewport()
        path = os.path.join(self._frames_dir, f"f_{self._frame_idx:05d}.png")
        self._vu.capture_viewport_to_file(self._viewport, path)
        self._frame_idx += 1

    def start(self) -> None:
        self._ensure_viewport()
        if os.path.isdir(self._frames_dir):
            shutil.rmtree(self._frames_dir, ignore_errors=True)
        os.makedirs(self._frames_dir, exist_ok=True)
        self._frame_idx = 0
        self._step_counter = 0
        self._active = True
        self.encode_error = None
        cprint(f"  [eval] REC start → {self.mp4_path}", "magenta")
        for _ in range(3):
            self._maybe_capture()

    def stop(self) -> str | None:
        if not self._active:
            return None
        self._active = False
        self.n_frames = self._frame_idx
        if self.n_frames > 0:
            self._encode_mp4()
        else:
            cprint(f"  [eval] REC skip (0 frames): {self.mp4_path}", "yellow")
            return None
        if not self.keep_frames and os.path.isdir(self._frames_dir):
            shutil.rmtree(self._frames_dir, ignore_errors=True)
        if os.path.isfile(self.mp4_path):
            cprint(f"  [eval] REC done {self.n_frames} frames → {self.mp4_path}", "magenta")
            return self.mp4_path
        return None

    def _encode_mp4(self) -> None:
        pattern = os.path.join(self._frames_dir, "f_%05d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(self.fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            self.mp4_path,
        ]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            self.encode_error = "ffmpeg not found"
            cprint(f"  [eval] ffmpeg not found; frames kept in {self._frames_dir}", "red")
        except subprocess.CalledProcessError as exc:
            self.encode_error = str(exc)
            cprint(f"  [eval] ffmpeg failed: {exc}; frames in {self._frames_dir}", "red")
