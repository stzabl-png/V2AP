#!/usr/bin/env python3
"""
Write input/.upload_complete for a session (local copy + optional remote via SSH).

After rsync to Titan, the orchestrator calls the remote Titan script; this CLI is
for manual/debug use on Razor (writes local marker only).

Usage:
  python demo/phase2/mark_upload_complete.py --session-id 20260602_192346_chips
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.phase2.constants import SESSIONS_DIR  # noqa: E402
from demo.phase2.session_markers import write_upload_complete  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write input/.upload_complete locally")
    ap.add_argument("--session-id", type=str, required=True)
    ap.add_argument("--sessions-dir", type=Path, default=SESSIONS_DIR)
    ap.add_argument("--source", type=str, default="razor")
    args = ap.parse_args(argv)

    session_dir = Path(args.sessions_dir) / args.session_id
    if not (session_dir / "input").is_dir():
        print(f"No input/ under {session_dir}", file=sys.stderr)
        return 1

    out = write_upload_complete(session_dir, source=args.source)
    print(f"Marked upload complete: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
