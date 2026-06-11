#!/usr/bin/env python3
"""
Phase 2 demo server–client pipeline (Razor client → Titan server → Razor grasp).

Requires SSH + rsync to Titan (see demo/phase2/SERVER_CLIENT_PLAN.md).

Environment:
  PIPELINE_TITAN_ROOT=/path/to/V2AP
  PIPELINE_TITAN_SSH_HOST=vision@<your-gpu-server>
  PIPELINE_TITAN_MODE=daemon          # default: Titan segment_daemon + SAM2 web UI
  PIPELINE_TITAN_SEGMENT_TUNNEL_PORT=7860

Usage:
  source setup.sh
  source demo/phase2/server_client_env.example

  # Titan (once): python -m demo.pipeline.segment_daemon
  # SAM2 web UI auto-opens via background SSH tunnel when waiting_segment

  python demo/phase2/run_server_client_pipeline.py --object-name chips
  python demo/phase2/run_server_client_pipeline.py --session-id 20260602_192346_chips --skip-capture
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase2.constants import DEFAULT_GRASP_ATTEMPTS, DEFAULT_TITAN_MAX_CANDIDATES  # noqa: E402
from demo.phase2.pack_session import make_session_id  # noqa: E402
from demo.phase2.session_titan_options import patch_input_titan_options  # noqa: E402
from demo.phase2.server_client_config import load_server_client_pipeline_config  # noqa: E402
from demo.phase2.server_client_transport import (  # noqa: E402
    rsync_download_output,
    rsync_upload_input,
    ssh_mark_upload_complete,
    ssh_run_titan_pipeline,
    wait_for_titan_success,
    write_orchestrator_state,
)
from demo.phase2.visualize_grasp import show_titan_pipeline_vis  # noqa: E402

# Flags that must not be swallowed by --capture-extra / --grasp-extra (REMAINDER).
_PROMOTED_PIPELINE_FLAGS: tuple[tuple[str, str], ...] = (
    ("--grasp-attempts", "grasp_attempts"),
    ("--titan-max-candidates", "titan_max_candidates"),
    ("--no-candidate-fallback", "no_candidate_fallback"),
)


def _promote_pipeline_flags_from_remainder(args: argparse.Namespace) -> None:
    """
    Recover pipeline flags accidentally placed after ``--capture-extra`` / ``--grasp-extra``.

    Example (broken without this)::

        ... --capture-extra --sam-point 320 180 --grasp-attempts 5
        → grasp_attempts stayed 1, ``5`` landed in capture_extra.
    """
    for attr in ("capture_extra", "grasp_extra"):
        remainder = getattr(args, attr, None)
        if not remainder:
            continue
        cleaned: list[str] = []
        i = 0
        promoted: list[str] = []
        while i < len(remainder):
            tok = remainder[i]
            matched = False
            for flag, dest in _PROMOTED_PIPELINE_FLAGS:
                if tok != flag:
                    continue
                matched = True
                if dest == "no_candidate_fallback":
                    setattr(args, dest, True)
                    promoted.append(flag)
                else:
                    if i + 1 >= len(remainder):
                        raise SystemExit(f"{flag} in --{attr} requires a value")
                    setattr(args, dest, type(getattr(args, dest))(remainder[i + 1]))
                    promoted.extend([flag, remainder[i + 1]])
                    i += 1
                break
            if not matched:
                cleaned.append(tok)
            i += 1
        if promoted:
            logger.warning(
                f"Promoted pipeline flag(s) from --{attr}: {' '.join(promoted)} "
                f"(put --grasp-attempts before --capture-extra to avoid this)"
            )
        setattr(args, attr, cleaned or None)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 demo server–client pipeline (Razor orchestrator)"
    )
    p.add_argument("--object-name", type=str, help="Object slug (e.g. chips)")
    p.add_argument("--session-id", type=str, default=None)
    p.add_argument("--skip-capture", action="store_true")
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument("--skip-titan", action="store_true", help="Do not trigger Titan")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-grasp", action="store_true")
    p.add_argument(
        "--ssh-pipeline",
        action="store_true",
        help="Legacy: ssh-run process_razor_session instead of segment_daemon",
    )
    p.add_argument(
        "--no-sam-popup",
        action="store_true",
        help="Do not auto-start SSH tunnel or open SAM2 browser "
        "(manual ssh -L required)",
    )
    p.add_argument(
        "--no-titan-vis",
        action="store_true",
        help="Skip blocking T3–T6 PNG review after Titan download",
    )
    p.add_argument(
        "--dry-run-transport",
        action="store_true",
        help="Print steps only; no rsync/ssh/grasp",
    )
    p.add_argument(
        "--grasp-attempts",
        type=int,
        default=DEFAULT_GRASP_ATTEMPTS,
        help="Real robot grasp runs after one perceive (default 1). "
        "Default: each run searches Titan candidates until IK passes. "
        "With --no-candidate-fallback: run k fixed ranks via --attempt-index 0..k-1.",
    )
    p.add_argument(
        "--titan-max-candidates",
        type=int,
        default=DEFAULT_TITAN_MAX_CANDIDATES,
        help="Ask Titan PDM to export N grasp candidates (written to input/session.json; default 50)",
    )
    p.add_argument(
        "--capture-extra",
        nargs=argparse.REMAINDER,
        help="Extra args for capture_session.py (must follow all pipeline flags, "
        "or place --grasp-attempts before this flag)",
    )
    p.add_argument(
        "--grasp-extra",
        nargs=argparse.REMAINDER,
        help="Extra args for run_auto_grasp.py (must follow --grasp-attempts etc.)",
    )
    p.add_argument(
        "--no-candidate-fallback",
        action="store_true",
        help="Multi-grasp test mode: grasp attempt i uses Titan sorted index i only "
        "(--attempt-index i); IK fail skips to next attempt without trying other ranks. "
        "Default off: each attempt runs run_auto_grasp candidate pool fallback until IK ok.",
    )
    p.add_argument(
        "--ik-palm-soft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward to run_auto_grasp: palm lateral in IK soft cost (default: off). "
        "Palm hard check after IK is always on.",
    )
    return p.parse_args()


def _parse_pipeline_args() -> argparse.Namespace:
    args = _parse_args()
    _promote_pipeline_flags_from_remainder(args)
    return args


def _run_capture(session_id: str, object_name: str, extra: list[str] | None) -> None:
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "demo" / "phase2" / "capture_session.py"),
        "--object-name",
        object_name,
        "--session-id",
        session_id,
    ]
    if extra:
        cmd.extend(extra)
    logger.info(f"Capture: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _merge_grasp_extra(
    grasp_extra: list[str] | None,
    *,
    ik_palm_soft: bool,
) -> list[str]:
    """Merge pipeline grasp flags into ``run_auto_grasp`` argv (``grasp_extra`` wins)."""
    extra = list(grasp_extra or [])
    if any(t in extra for t in ("--ik-palm-soft", "--no-ik-palm-soft")):
        return extra
    if ik_palm_soft:
        extra.append("--ik-palm-soft")
    return extra


def _run_grasp(
    session_id: str,
    extra: list[str] | None,
    *,
    check: bool = True,
) -> int:
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "demo" / "phase2" / "run_auto_grasp.py"),
        "--session-id",
        session_id,
        "--no-prompts",
    ]
    if extra:
        cmd.extend(extra)
    logger.info(f"Grasp: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return int(proc.returncode)


def main() -> None:
    args = _parse_pipeline_args()
    cfg = load_server_client_pipeline_config()

    logger.info(
        f"Pipeline options: grasp_attempts={args.grasp_attempts}, "
        f"titan_max_candidates={args.titan_max_candidates}, "
        f"no_candidate_fallback={args.no_candidate_fallback}, "
        f"ik_palm_soft={args.ik_palm_soft}"
    )

    if not args.skip_capture and not args.object_name and not args.session_id:
        raise SystemExit("--object-name required unless --skip-capture with --session-id")

    object_slug = args.object_name or "object"
    session_id = args.session_id or make_session_id(object_slug)
    session_dir = cfg.local_session_dir(session_id)

    state: dict = {
        "session_id": session_id,
        "object_slug": object_slug,
        "updated_at_iso": datetime.now(timezone.utc).isoformat(),
        "steps": {},
    }

    logger.info(
        f"=== Server–client pipeline session={session_id} titan={cfg.titan_ssh_host} ==="
    )

    if args.dry_run_transport:
        logger.info("[dry-run] capture → upload → titan → download → grasp")
        logger.info(f"  local:  {session_dir}")
        logger.info(f"  remote: {cfg.remote_session_dir(session_id)}")
        return

    try:
        if not args.skip_capture:
            _run_capture(session_id, object_slug, args.capture_extra)
            state["steps"]["capture"] = "ok"

        if not args.skip_upload:
            input_dir = session_dir / "input"
            if not (input_dir / "session.json").is_file():
                raise FileNotFoundError(
                    f"Missing {input_dir / 'session.json'} — run capture first or fix session"
                )
            patch_input_titan_options(
                input_dir, max_candidates=args.titan_max_candidates
            )
            state["titan_max_candidates"] = args.titan_max_candidates
            rsync_upload_input(cfg, session_id)
            state["steps"]["upload"] = "ok"

        if not args.skip_titan:
            use_ssh_pipeline = args.ssh_pipeline or cfg.titan_mode == "ssh_pipeline"
            if use_ssh_pipeline:
                rc = ssh_run_titan_pipeline(cfg, session_id)
                state["steps"]["titan_ssh_exit_code"] = rc
                if rc != 0:
                    logger.warning(
                        f"Titan pipeline exit {rc}; polling status.json anyway..."
                    )
            else:
                ssh_mark_upload_complete(cfg, session_id)
                state["steps"]["titan_mark"] = "ok"
            sam_popup = cfg.titan_auto_sam_popup and not args.no_sam_popup
            status = wait_for_titan_success(
                cfg, session_id, sam_popup=sam_popup
            )
            state["steps"]["titan"] = "ok"
            state["titan_status"] = {
                "success": status.get("success"),
                "pipeline_version": status.get("pipeline_version"),
                "n_candidates": status.get("titan", {}).get("n_candidates"),
            }

        if not args.skip_download:
            rsync_download_output(cfg, session_id)
            state["steps"]["download"] = "ok"
            if not args.no_titan_vis:
                show_titan_pipeline_vis(session_dir)

        if not args.skip_grasp:
            if args.grasp_attempts < 1:
                raise SystemExit("--grasp-attempts must be >= 1")
            logger.info(
                f"Grasp stage: {args.grasp_attempts} attempt(s), "
                f"mode={'fixed-rank (--no-candidate-fallback)' if args.no_candidate_fallback else 'pool IK fallback per attempt'}"
            )
            grasp_steps: list[dict] = []
            strict_one_rank = args.no_candidate_fallback
            multi_attempt = args.grasp_attempts > 1
            for attempt in range(args.grasp_attempts):
                grasp_extra = _merge_grasp_extra(
                    args.grasp_extra,
                    ik_palm_soft=args.ik_palm_soft,
                )
                if strict_one_rank:
                    logger.info(
                        f"=== Grasp attempt {attempt + 1}/{args.grasp_attempts} "
                        f"(no-candidate-fallback: Titan sorted index {attempt} only) ==="
                    )
                    if "--attempt-index" not in grasp_extra and "--rank" not in grasp_extra:
                        grasp_extra.extend(["--attempt-index", str(attempt)])
                    if "--no-candidate-fallback" not in grasp_extra:
                        grasp_extra.append("--no-candidate-fallback")
                else:
                    logger.info(
                        f"=== Grasp attempt {attempt + 1}/{args.grasp_attempts} "
                        f"(candidate pool IK fallback until one passes) ==="
                    )
                if "--grasp-preview-label" not in grasp_extra:
                    grasp_extra.extend(
                        ["--grasp-preview-label", f"attempt-{attempt + 1}"]
                    )
                rc = _run_grasp(
                    session_id,
                    grasp_extra,
                    check=(not multi_attempt and not strict_one_rank),
                )
                step = {
                    "attempt": attempt + 1,
                    "attempt_index": attempt if strict_one_rank else None,
                    "ok": rc == 0,
                    "exit_code": rc,
                    "mode": (
                        "no_candidate_fallback"
                        if strict_one_rank
                        else "pool_fallback"
                    ),
                }
                grasp_steps.append(step)
                if rc != 0:
                    if multi_attempt or strict_one_rank:
                        logger.warning(
                            f"Grasp attempt {attempt + 1} failed (exit {rc}); "
                            f"continuing ({args.grasp_attempts - attempt - 1} left)"
                        )
                        continue
                    raise SystemExit(rc)
            if args.grasp_attempts > 1:
                state["steps"]["grasp"] = grasp_steps
                n_ok = sum(1 for s in grasp_steps if s.get("ok"))
                if n_ok == 0:
                    raise SystemExit(
                        f"All {args.grasp_attempts} grasp attempts failed"
                    )
                if n_ok < args.grasp_attempts:
                    logger.warning(
                        f"Grasp: {n_ok}/{args.grasp_attempts} attempts succeeded"
                    )
                logger.info(
                    f"Grasp summary: {n_ok}/{args.grasp_attempts} attempts succeeded"
                )
            else:
                state["steps"]["grasp"] = (
                    grasp_steps[0] if grasp_steps else "ok"
                )
                if grasp_steps and not grasp_steps[0].get("ok"):
                    raise SystemExit(grasp_steps[0].get("exit_code", 1))

        state["success"] = True
        logger.info(f"=== Server–client pipeline complete: {session_id} ===")

    except Exception as exc:
        state["success"] = False
        state["error"] = str(exc)
        write_orchestrator_state(session_dir, state)
        raise

    write_orchestrator_state(session_dir, state)


if __name__ == "__main__":
    main()
