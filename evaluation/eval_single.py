#!/usr/bin/env python3
"""Single-object evaluation runner.

Example:
    $ISAAC_SIM_PATH/python.sh evaluation/eval_single.py --obj-id A16013 \\
        --candidate-hdf5 output/grasp_collect_no_rot/candidates/pool/A16013_grasp.hdf5 \\
        --headless --selection sample
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from evaluation.affordance_ckpt import add_affordance_checkpoint_args, resolve_affordance_checkpoint
from evaluation.placement import add_random_obj_xy_args
from evaluation.randomness import add_eval_seed_args

DEFAULT_CANDIDATE_PYTHON = "/home/vision/miniconda3/envs/bundlesdf/bin/python"


def default_candidate_python() -> str:
    return DEFAULT_CANDIDATE_PYTHON if os.path.isfile(DEFAULT_CANDIDATE_PYTHON) else "python"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-object modular evaluation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--obj-id", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--policy", choices=("a2g_pdm",), default="a2g_pdm")
    parser.add_argument("--candidate-hdf5", default=None)
    parser.add_argument(
        "--generate-candidate",
        action="store_true",
        help="Generate A2G/PDM candidates before IsaacSim starts; uses --mesh or auto-resolves rotated SAM3D mesh by --obj-id.",
    )
    parser.add_argument("--mesh", default=None, help="Mesh path for --generate-candidate.")
    parser.add_argument(
        "--mesh-root",
        default=str(PROJ / "data_hub" / "meshes" / "SAM3DMesh" / "rotated_mesh"),
        help="Rotated SAM3D mesh root used by --generate-candidate when --mesh is omitted.",
    )
    parser.add_argument(
        "--sam3d-rotated-mesh",
        action="store_true",
        help="Treat --mesh as an already rotated SAM3D mesh; skip +X rotation and preserve frame.",
    )
    parser.add_argument(
        "--candidate-dir",
        default=str(PROJ / "output" / "evaluation" / "candidates"),
        help="Where generated candidate HDF5 files are written.",
    )
    parser.add_argument(
        "--candidate-output",
        default=None,
        help="Explicit output HDF5 path when generating candidates.",
    )
    parser.add_argument(
        "--candidate-python",
        default=default_candidate_python(),
        help="Python command used for --generate-candidate, e.g. 'python' or 'conda run -n bundlesdf python'.",
    )
    parser.add_argument("--selection", choices=("top", "index", "sample"), default="top")
    parser.add_argument("--candidate-index", type=int, default=0)
    add_eval_seed_args(parser)
    parser.add_argument("--trial", type=int, default=0, help="Trial index (episode id suffix).")
    parser.add_argument("--object-scale", type=float, default=1.0)
    add_random_obj_xy_args(parser)
    parser.add_argument("--z-yaw-deg", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-open", action="store_true",
                        help="Keep sim window open after grasp for screen recording. Ctrl+C to exit.")
    parser.add_argument("--wait-before-grasp", action="store_true",
                        help="Pause after scene is ready (object visible), wait for Enter, then execute grasp.")
    parser.add_argument(
        "--result-dir",
        default=str(PROJ / "output" / "evaluation" / "single"),
    )
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--save-hdf5", action="store_true", help="Also write robot_gt-compatible HDF5.")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record viewport MP4 to {result_dir}/{episode_id}.mp4",
    )
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--record-every", type=int, default=3)
    parser.add_argument("--record-keep-frames", action="store_true")
    parser.add_argument(
        "--no-auto-xvfb",
        action="store_true",
        help="Disable automatic xvfb-run re-exec for --record-video when DISPLAY is missing.",
    )
    parser.add_argument(
        "--xvfb-screen",
        default="-screen 0 1280x720x24",
        help="xvfb-run -s argument used by --record-video when DISPLAY is missing.",
    )
    parser.add_argument(
        "--log-only",
        action="store_true",
        help="Write full stdout/stderr to a log file and print nothing to terminal.",
    )
    parser.add_argument(
        "--loud",
        action="store_true",
        help="Print full stdout/stderr and also write it to a log file.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for --log-only/--loud logs (default: <result-dir>/logs).",
    )
    add_affordance_checkpoint_args(parser)
    return parser


def candidate_output_name(obj_id: str, z_yaw_deg: float | None) -> str:
    """Legacy/default candidate name lookup.

    Pool candidates usually use `{obj}_grasp.hdf5` for yaw=0, while freshly
    yaw-conditioned PDM generation may write `{obj}_yaw000_grasp.hdf5`.
    """
    if z_yaw_deg is None:
        return f"{obj_id}_grasp.hdf5"
    tag = int(round(float(z_yaw_deg))) % 360
    if tag == 0:
        return f"{obj_id}_grasp.hdf5"
    return f"{obj_id}_yaw{tag:03d}_grasp.hdf5"


def generated_candidate_output_name(obj_id: str, z_yaw_deg: float) -> str:
    """Name written by tools/glb_to_pdm_grasp.py when --z-yaw-deg is passed."""
    tag = int(round(float(z_yaw_deg))) % 360
    return f"{obj_id}_yaw{tag:03d}_grasp.hdf5"


def infer_rotated_mesh_dataset(obj_id: str, dataset: str | None) -> str:
    if dataset:
        return "ycb" if dataset == "dexycb" else dataset
    if obj_id.startswith("unseen_"):
        return "unseen"
    if obj_id.startswith("ycb_dex_"):
        return "ycb"
    if obj_id.startswith("arctic_"):
        return "arctic"
    return "oakink"


def resolve_generate_mesh(args: argparse.Namespace) -> tuple[Path, bool]:
    """Resolve the mesh used for online candidate generation.

    Returns (mesh_path, is_rotated_sam3d_mesh).
    """
    if args.mesh:
        return Path(args.mesh).expanduser().resolve(), bool(args.sam3d_rotated_mesh)

    mesh_root = Path(args.mesh_root).expanduser().resolve()
    ds = infer_rotated_mesh_dataset(args.obj_id, args.dataset)
    candidates = [
        mesh_root / ds / args.obj_id / "mesh.ply",
        mesh_root / args.obj_id / "mesh.ply",
    ]
    for extra_ds in ("unseen", "oakink", "ycb", "arctic", "dexycb", "egocentric", "ho3d_v3"):
        candidates.append(mesh_root / extra_ds / args.obj_id / "mesh.ply")
    for path in candidates:
        if path.is_file():
            return path.resolve(), True
    raise FileNotFoundError(
        "No --mesh provided and rotated SAM3D mesh was not found for "
        f"{args.obj_id} under {mesh_root}"
    )


def candidate_generation_env() -> dict[str, str]:
    """Clean IsaacSim python.sh variables before launching external ML Python."""
    env = os.environ.copy()
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONEXECUTABLE",
        "PYTHONNOUSERSITE",
        "PYTHONUSERBASE",
    ):
        env.pop(key, None)
    return env


def maybe_generate_candidate(args: argparse.Namespace) -> str | None:
    if not args.generate_candidate:
        return args.candidate_hdf5
    mesh_path, is_rotated_sam3d = resolve_generate_mesh(args)

    out_dir = Path(args.candidate_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.candidate_output:
        out_path = Path(args.candidate_output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = out_dir / candidate_output_name(args.obj_id, args.z_yaw_deg)
    generated_path = out_dir / generated_candidate_output_name(args.obj_id, args.z_yaw_deg)

    cmd = [
        *shlex.split(args.candidate_python),
        str(PROJ / "tools" / "glb_to_pdm_grasp.py"),
        "--mesh",
        str(mesh_path),
        "--obj-id",
        args.obj_id,
        "--output-dir",
        str(out_dir),
        "--dataset",
        args.dataset or "evaluation",
        "--z-yaw-deg",
        str(float(args.z_yaw_deg)),
        "--no-vis",
        "--no-affordance-output",
        "--seed",
        str(int(getattr(args, "seed", 42))),
    ]
    if is_rotated_sam3d:
        cmd.append("--sam3d-rotated-mesh")
    aff_ckpt = resolve_affordance_checkpoint(
        hp_affordance=bool(getattr(args, "hp_affordance", False)),
        affordance_checkpoint=getattr(args, "affordance_checkpoint", None),
    )
    cmd.extend(["--affordance-checkpoint", str(aff_ckpt)])
    print(
        "[eval] generating candidate HDF5 "
        f"obj={args.obj_id} yaw={float(args.z_yaw_deg):.1f} "
        f"aff_ckpt={aff_ckpt.name} -> {out_path}"
    )
    proc = subprocess.run(
        cmd,
        cwd=str(PROJ),
        env=candidate_generation_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        lines = (proc.stdout or "").splitlines()
        tail = "\n".join(lines[-80:])
        raise RuntimeError(
            "candidate generation failed "
            f"(returncode={proc.returncode}). Use --candidate-python with a Python env "
            "that has torch/trimesh/scipy/h5py and the A2G model deps.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output tail:\n{tail}"
        )
    if args.candidate_output and generated_path.is_file() and generated_path != out_path:
        shutil.move(str(generated_path), str(out_path))
    elif not out_path.is_file() and generated_path.is_file():
        out_path = generated_path
    if not out_path.is_file():
        raise FileNotFoundError(f"candidate generation finished but output is missing: {out_path}")
    return str(out_path)


def default_episode_id(args: argparse.Namespace) -> str:
    if args.episode_id:
        return args.episode_id
    yaw_tag = int(round(float(args.z_yaw_deg))) % 360
    return f"{args.obj_id}_{args.policy}_yaw{yaw_tag:03d}_t{int(args.trial):03d}"


class OutputController:
    """FD-level stdout/stderr routing for noisy IsaacSim startup logs."""

    def __init__(self, args: argparse.Namespace, episode_id: str):
        if args.log_only and args.loud:
            raise ValueError("Use only one of --log-only or --loud")
        self.mode = "loud" if args.loud else ("log-only" if args.log_only else "normal")
        self.episode_id = episode_id
        self.log_path: str | None = None
        self._orig_stdout = None
        self._orig_stderr = None
        self._targets: list[tuple[int, int | None, threading.Thread | None]] = []
        self._log_file = None
        self._devnull = None

        if self.mode in {"log-only", "loud"}:
            log_dir = Path(args.log_dir or (Path(args.result_dir) / "logs")).expanduser().resolve()
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = str(log_dir / f"{episode_id}.log")

    @property
    def prints_important(self) -> bool:
        return self.mode != "log-only"

    def start(self) -> None:
        self._orig_stdout = os.dup(1)
        self._orig_stderr = os.dup(2)
        if self.mode == "normal":
            self._devnull = open(os.devnull, "wb")
            os.dup2(self._devnull.fileno(), 1)
            os.dup2(self._devnull.fileno(), 2)
            return

        assert self.log_path is not None
        self._log_file = open(self.log_path, "ab", buffering=0)
        self._redirect_fd_with_tee(1, self._orig_stdout if self.mode == "loud" else None)
        self._redirect_fd_with_tee(2, self._orig_stderr if self.mode == "loud" else None)

    def _redirect_fd_with_tee(self, fd: int, console_fd: int | None) -> None:
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, fd)
        os.close(write_fd)

        def pump() -> None:
            while True:
                try:
                    chunk = os.read(read_fd, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                if self._log_file is not None:
                    self._log_file.write(chunk)
                if console_fd is not None:
                    os.write(console_fd, chunk)
            try:
                os.close(read_fd)
            except OSError:
                pass

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        self._targets.append((fd, read_fd, thread))

    def important(self, message: str = "") -> None:
        if not self.prints_important:
            return
        fd = self._orig_stdout if self._orig_stdout is not None else 1
        os.write(fd, (message + "\n").encode("utf-8", errors="replace"))

    def stop(self) -> None:
        if self._orig_stdout is not None:
            os.dup2(self._orig_stdout, 1)
        if self._orig_stderr is not None:
            os.dup2(self._orig_stderr, 2)
        for _fd, _read_fd, thread in self._targets:
            if thread is not None:
                thread.join(timeout=1.0)
        if self._devnull is not None:
            self._devnull.close()
        if self._log_file is not None:
            self._log_file.close()
        if self._orig_stdout is not None:
            os.close(self._orig_stdout)
            self._orig_stdout = None
        if self._orig_stderr is not None:
            os.close(self._orig_stderr)
            self._orig_stderr = None


def maybe_reexec_with_xvfb(args: argparse.Namespace, *, headless: bool) -> None:
    """For video recording on headless machines, restart under xvfb-run once."""
    if not args.record_video or headless:
        return
    if args.no_auto_xvfb:
        return
    if os.environ.get("A2G_EVAL_XVFB_ACTIVE") == "1":
        return

    display = os.environ.get("DISPLAY")
    if display:
        # DISPLAY can be set but unusable (e.g. stale SSH forwarding). In that
        # case Isaac may start and then immediately shut down when the first
        # UI tick runs. Prefer to fall back to Xvfb automatically.
        try:
            subprocess.run(
                ["xdpyinfo"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.0,
                check=True,
            )
            return
        except Exception:
            pass

    xvfb_run = shutil.which("xvfb-run")
    if xvfb_run is None:
        raise RuntimeError(
            "--record-video needs a DISPLAY. Install xvfb or run manually with "
            "`xvfb-run -s \"-screen 0 1280x720x24\" ...`, or pass --no-auto-xvfb "
            "if you have another display setup."
        )

    isaac_root = os.environ.get("ISAAC_SIM_PATH", "").strip()
    isaac_python = os.path.join(isaac_root, "python.sh") if isaac_root else ""
    runner = isaac_python if os.path.isfile(isaac_python) else sys.executable
    cmd = [xvfb_run, "-a", "-s", args.xvfb_screen, runner, *sys.argv]
    env = os.environ.copy()
    env["A2G_EVAL_XVFB_ACTIVE"] = "1"
    if not args.log_only:
        print("[eval] --record-video: DISPLAY is missing; restarting under xvfb-run", flush=True)
        if args.loud:
            print("[eval] " + " ".join(cmd), flush=True)
    os.execvpe(xvfb_run, cmd, env)


def main() -> None:
    parser = build_parser()
    args, _ = parser.parse_known_args()

    # If DISPLAY is missing, Isaac may start without creating a default window,
    # while still loading viewport UI extensions; that can immediately trigger
    # UI-related errors and an early shutdown. Default to headless in that case.
    headless = bool(args.headless) or (not os.environ.get("DISPLAY") and not args.record_video)
    if args.record_video and headless:
        if not args.log_only:
            print("[eval] --record-video: forcing headed mode (viewport capture needs DISPLAY / Xvfb)")
        headless = False
    maybe_reexec_with_xvfb(args, headless=headless)

    episode_id = default_episode_id(args)
    output = OutputController(args, episode_id)
    output.start()
    output.important(f"[eval] episode={episode_id} obj={args.obj_id} yaw={float(args.z_yaw_deg):.1f}")
    if output.log_path:
        output.important(f"[eval] full log: {output.log_path}")

    candidate_hdf5 = None
    simulation_app = None
    recorder = None
    try:
        candidate_hdf5 = maybe_generate_candidate(args)
        if not candidate_hdf5:
            raise ValueError("provide --candidate-hdf5 or use --generate-candidate --mesh")
        output.important(f"[eval] candidate_hdf5={candidate_hdf5}")

        # IsaacSim must be created before importing modules that touch isaacsim APIs.
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": headless})
        output.important("[eval] simulation app started")
        from evaluation.randomness import resolve_policy_seed
        from evaluation.policies.a2g_pdm import A2GPDMPolicy, A2GPDMPolicyConfig
        from evaluation.results import append_episode_jsonl, build_episode_record, write_episode_json
        from sim.evaluation.curobo_executor import execute_open_loop_grasp, write_robot_gt_hdf5
        from sim.evaluation.scene_builder import build_scene_spec, setup_scene
        from sim.evaluation.video_recorder import EpisodeVideoRecorder

        policy_seed_val = resolve_policy_seed(
            eval_seed=int(args.seed),
            policy_seed=args.policy_seed,
            trial=args.trial,
        )

        scene_spec = build_scene_spec(
            obj_id=args.obj_id,
            episode_id=episode_id,
            dataset=args.dataset,
            object_scale=args.object_scale,
            sim_z_yaw_deg=args.z_yaw_deg,
            seed=int(args.trial),
            candidate_hdf5=candidate_hdf5,
            random_obj_xy=bool(args.random_obj_xy),
            obj_xy_jitter_m=float(args.obj_xy_jitter_m),
            eval_seed=int(args.seed),
        )
        render = not headless
        scene = setup_scene(scene_spec, render=render)
        if args.random_obj_xy:
            dx, dy = scene_spec.obj_xy_offset
            output.important(
                f"[eval] random_obj_xy jitter={args.obj_xy_jitter_m:.3f}m "
                f"offset=({dx:+.4f}, {dy:+.4f}) "
                f"spawn=({scene_spec.object_position_world[0]:.4f}, "
                f"{scene_spec.object_position_world[1]:.4f}, "
                f"{scene_spec.object_position_world[2]:.4f})"
            )
        output.important("[eval] scene ready")

        if args.record_video:
            mp4_path = os.path.join(args.result_dir, f"{episode_id}.mp4")
            recorder = EpisodeVideoRecorder(
                mp4_path,
                record_every=args.record_every,
                fps=args.record_fps,
                keep_frames=args.record_keep_frames,
            )
            recorder.attach_world(scene.world)
            recorder.start()
            output.important(f"[eval] recording video -> {mp4_path}")

        policy = A2GPDMPolicy(
            A2GPDMPolicyConfig(
                candidate_hdf5=candidate_hdf5,
                selection=args.selection,
                candidate_index=args.candidate_index,
                seed=policy_seed_val,
            )
        )

        policy_output = policy.predict(scene)
        if policy_output.kind != "open_loop_grasp" or policy_output.command is None:
            raise RuntimeError(f"unsupported policy output for first runner: {policy_output.kind}")

        if args.wait_before_grasp and not headless:
            output.important("[eval] scene ready — press Enter to execute grasp...")
            # Keep rendering while waiting
            import select
            sys.stdin = open('/dev/stdin')
            while True:
                scene.world.step(render=True)
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    sys.stdin.readline()
                    break
            output.important("[eval] executing grasp...")

        execution = execute_open_loop_grasp(scene, policy_output.command)
        plan = execution.planning
        output.important(
            "[eval] plans "
            f"pregrasp={plan.get('pregrasp_plan_success')} "
            f"direct={plan.get('direct_plan_success')} "
            f"final={plan.get('final_plan_success')} "
            f"lift={plan.get('lift_plan_success')}"
        )

        video_path = None
        if recorder is not None:
            video_path = recorder.stop()
            execution.video_path = video_path
            execution.video_n_frames = recorder.n_frames
            if recorder.encode_error:
                execution.metadata["video_encode_error"] = recorder.encode_error

        record = build_episode_record(
            scene=scene_spec,
            policy_name=policy.name,
            policy_output=policy_output,
            execution=execution,
            video_path=video_path,
        )
        json_path = write_episode_json(record, args.result_dir)
        jsonl_path = append_episode_jsonl(record, args.result_dir)
        output.important(f"[eval] wrote episode JSON: {json_path}")
        output.important(f"[eval] appended JSONL: {jsonl_path}")
        if video_path:
            output.important(f"[eval] wrote video: {video_path}")

        if args.save_hdf5:
            h5_path = write_robot_gt_hdf5(
                result_dir=args.result_dir,
                scene=scene_spec,
                command=policy_output.command,
                execution=execution,
                policy_name=policy.name,
            )
            output.important(f"[eval] wrote robot_gt HDF5: {h5_path}")

        output.important(
            f"[eval] result: {'SUCCESS' if execution.success else 'FAILED'}"
            f" z_delta={execution.z_delta_m}"
            f" failure_stage={execution.failure_stage}"
        )

        if args.keep_open and simulation_app is not None and not headless:
            output.important("[eval] --keep-open: window stays open. Press Ctrl+C to exit.")
            try:
                while simulation_app.is_running():
                    scene.world.step(render=True)
            except KeyboardInterrupt:
                output.important("[eval] Ctrl+C received, closing.")
    finally:
        if recorder is not None and recorder._active:
            recorder.stop()
        if simulation_app is not None:
            simulation_app.close()
        output.stop()


if __name__ == "__main__":
    main()
