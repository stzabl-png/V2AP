#!/usr/bin/env python3
"""
Per-object triptych: Human prior | Robot posterior GT | Affordance (v6 pred).

Reads existing PNGs under output/paper_vis/::

    hp/vis/{obj_id}.png
    affordance/robot_posterior/vis/{obj_id}.png
    affordance/vis/{obj_id}.png

Writes::

    compare/{obj_id}.png
    compare/overview.png   (grid, optional)

Usage::

    python tools/paper_vis_triptych.py
    python tools/paper_vis_triptych.py --output-dir output/paper_vis --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJ / "output" / "paper_vis"
DEFAULT_CSV = PROJ / "evaluation" / "configs" / "eval_objects_merged_success_ge30.csv"

sys.path.insert(0, str(PROJ))
from tools.paper_vis_batch import load_obj_ids_from_csv  # noqa: E402
from model.inference_v6 import compose_png_grid  # noqa: E402

BG = (26, 26, 46)
LABEL_COLOR = (220, 220, 220)
PANEL_LABELS = ("Human prior", "Robot posterior", "Affordance (v6)")


def _load_rgb(path: Path) -> Image.Image | None:
    if not path.is_file():
        return None
    return Image.open(path).convert("RGB")


def compose_triptych(
    panels: list[tuple[str, Image.Image]],
    out_path: Path,
    *,
    panel_height: int = 480,
    gap: int = 8,
    label_h: int = 28,
    font_size: int = 16,
) -> bool:
    """Horizontally stitch labeled panels; return False if any panel missing."""
    if len(panels) != 3:
        raise ValueError("expected 3 panels")

    resized: list[tuple[str, Image.Image]] = []
    for label, im in panels:
        scale = panel_height / max(im.height, 1)
        new_w = max(1, int(im.width * scale))
        resized.append((label, im.resize((new_w, panel_height), Image.Resampling.LANCZOS)))

    total_w = sum(im.width for _, im in resized) + gap * (len(resized) - 1)
    canvas_h = label_h + panel_height
    canvas = Image.new("RGB", (total_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    x = 0
    for label, im in resized:
        canvas.paste(im, (x, label_h))
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        tx = x + (im.width - tw) // 2
        draw.text((tx, 4), label, fill=LABEL_COLOR, font=font)
        x += im.width + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)
    return True


def panel_paths(root: Path, obj_id: str) -> tuple[Path, Path, Path]:
    return (
        root / "hp" / "vis" / f"{obj_id}.png",
        root / "affordance" / "robot_posterior" / "vis" / f"{obj_id}.png",
        root / "affordance" / "vis" / f"{obj_id}.png",
    )


def obj_ids_from_summary(root: Path) -> list[str]:
    p = root / "summary.json"
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as f:
        doc = json.load(f)
    return [r["obj_id"] for r in doc.get("objects", []) if "obj_id" in r]


def main() -> None:
    p = argparse.ArgumentParser(description="HP | robot posterior | affordance triptych per object")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--obj", nargs="*", default=None)
    p.add_argument("--panel-height", type=int, default=480)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-overview", action="store_true")
    args = p.parse_args()

    root = args.output_dir.expanduser().resolve()
    compare_dir = root / "compare"

    if args.obj:
        obj_ids = list(dict.fromkeys(args.obj))
    else:
        obj_ids = obj_ids_from_summary(root)
        if not obj_ids:
            obj_ids = load_obj_ids_from_csv(args.csv.expanduser().resolve())

    print("=" * 72)
    print("paper_vis_triptych")
    print(f"  objects: {len(obj_ids)}")
    print(f"  output:  {compare_dir}")
    print("=" * 72)

    ok_paths: list[str] = []
    skipped: list[dict] = []

    for i, oid in enumerate(obj_ids, 1):
        out_path = compare_dir / f"{oid}.png"
        if not args.overwrite and out_path.is_file():
            ok_paths.append(str(out_path))
            continue

        hp_p, rp_p, aff_p = panel_paths(root, oid)
        images = [_load_rgb(hp_p), _load_rgb(rp_p), _load_rgb(aff_p)]
        names = ("hp", "robot_posterior", "affordance")
        missing = [names[j] for j, im in enumerate(images) if im is None]
        if missing:
            print(f"[{i}/{len(obj_ids)}] skip {oid}: missing {missing}")
            skipped.append({"obj_id": oid, "missing": missing})
            continue

        panels = list(zip(PANEL_LABELS, images))
        compose_triptych(
            panels,
            out_path,
            panel_height=args.panel_height,
        )
        ok_paths.append(str(out_path))
        print(f"[{i}/{len(obj_ids)}] {oid} -> {out_path.name}")

    if not args.no_overview and len(ok_paths) > 1:
        compose_png_grid(
            ok_paths,
            str(compare_dir / "overview.png"),
            cols=min(3, len(ok_paths)),
            max_cell_width=900,
        )
        print(f"overview -> {compare_dir / 'overview.png'}")

    summary_path = root / "summary.json"
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            doc = json.load(f)
    else:
        doc = {}
    doc["compare_triptych_dir"] = str(compare_dir)
    doc["compare_triptych_count"] = len(ok_paths)
    doc["compare_triptych_skipped"] = skipped
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    print(f"\nDone: {len(ok_paths)} triptychs, {len(skipped)} skipped.")


if __name__ == "__main__":
    main()
