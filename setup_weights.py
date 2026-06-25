#!/usr/bin/env python3
"""
setup_weights.py — Download all model weights and datasets from HuggingFace
============================================================================
Run once after cloning the repo to pull all required model weights.

Usage:
    python setup_weights.py              # download all weights (NOT datasets)
    python setup_weights.py --tool fp    # download only FoundationPose weights
    python setup_weights.py --tool thirdmasks  # Phase 1A FP init masks

Tools (model weights — ~10 GB total):
    fp          — FoundationPose        (248 MB)
    hawor       — HaWoR                 (3.6 GB, checkpoints + external)
    haptic      — HaPTIC vitpose        (3.8 GB)
    megasam     — MegaSAM               (21 MB)
    depthpro    — Apple Depth Pro       (1.9 GB, avoids Apple CDN firewall issues)

Tools (data assets — download as needed):
    egomasks    — Egocentric FP init masks (EgoDex+TACO ego, ~70 MB)
                  Source: UCBProject/EgoDataMask
    thirdmasks  — Third-person FP init masks (DexYCB/HO3D/OakInk/TACO, ~30 MB)
                  Source: UCBProject/ThirdDataMask
    objmeshes   — Object meshes for FP (YCB+EgoDex+OakInk, ~1 GB)
                  Source: UCBProject/V2AP-Mesh
    egodex      — EgoDex raw videos (~30 GB, 3051 sequences)
                  Source: UCBProject/V2AP-EgoDex
    taco        — TACO Allocentric+Ego raw videos (~120+24 GB)
                  Source: UCBProject/V2AP-TACO

Notes:
    - MANO model weights require manual registration at https://mano.is.tue.mpg.de/
      Place MANO_RIGHT.pkl and MANO_LEFT.pkl under third_party/haptic/assets/mano/
    - DexYCB / HO3D / OakInk must be downloaded from their official sites (license).
    - Requires: pip install huggingface_hub
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

WEIGHTS_REPO = "UCBProject/V2AP-Weights"   # model weights
REPO_TYPE    = "dataset"
PROJECT      = Path(__file__).parent

# Legacy alias used by README inline examples
REPO_ID = WEIGHTS_REPO


def download_and_place(patterns, local_prefix, dest, repo_id=None):
    """Download files matching patterns from repo_id and move them to dest."""
    repo_id = repo_id or WEIGHTS_REPO
    tmp = PROJECT / ".hf_tmp"
    tmp.mkdir(exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        allow_patterns=patterns,
        local_dir=str(tmp),
    )
    src = tmp / local_prefix
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(src), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)


def download_fp():
    print("\n📥 FoundationPose weights (~248 MB)...")
    fp_root = Path(os.environ.get("FP_ROOT",
                   str(PROJECT / "third_party" / "FoundationPose"))) / "weights"
    fp_root.mkdir(parents=True, exist_ok=True)
    for folder in ["2023-10-28-18-33-37", "2024-01-11-20-02-45"]:
        dest = fp_root / folder
        if dest.exists():
            print(f"   ⏭️  {folder} already exists, skipping")
            continue
        download_and_place(
            patterns=[f"FoundationPose/weights/{folder}/*"],
            local_prefix=f"FoundationPose/weights/{folder}",
            dest=dest,
        )
        print(f"   ✅ {folder}")
    print(f"✅ FoundationPose done → {fp_root}")


def download_hawor():
    print("\n📥 HaWoR weights (~3.6 GB, this may take a while)...")
    hawor_root = PROJECT / "third_party" / "hawor" / "weights"

    # checkpoints (main model)
    ckpt_dest = hawor_root / "hawor" / "checkpoints"
    if not (ckpt_dest / "hawor.ckpt").exists():
        print("   Downloading hawor/checkpoints...")
        download_and_place(
            patterns=["HaWoR/weights/hawor/checkpoints/hawor.ckpt",
                      "HaWoR/weights/hawor/checkpoints/infiller.pt",
                      "HaWoR/weights/hawor/checkpoints/model_config.yaml"],
            local_prefix="HaWoR/weights/hawor/checkpoints",
            dest=ckpt_dest,
        )
    else:
        print("   ⏭️  hawor/checkpoints already exists, skipping")

    # external (detector + droid)
    ext_dest = hawor_root / "external"
    if not (ext_dest / "detector.pt").exists():
        print("   Downloading external weights...")
        download_and_place(
            patterns=["HaWoR/weights/external/*"],
            local_prefix="HaWoR/weights/external",
            dest=ext_dest,
        )
    else:
        print("   ⏭️  external already exists, skipping")

    print("✅ HaWoR done")


def download_haptic():
    print("\n📥 HaPTIC vitpose weights (~3.8 GB, this may take a while)...")
    dest = PROJECT / "third_party" / "haptic" / "_DATA" / "vitpose_ckpts"
    if dest.exists() and any(dest.iterdir()):
        print("   ⏭️  HaPTIC vitpose already exists, skipping")
        return
    download_and_place(
        patterns=["HaPTIC/vitpose_ckpts/*"],
        local_prefix="HaPTIC/vitpose_ckpts",
        dest=dest,
    )
    print("✅ HaPTIC done")


def download_megasam():
    print("\n📥 MegaSAM checkpoint (~21 MB)...")
    dest = PROJECT / "mega-sam" / "checkpoints" / "megasam_final.pth"
    if dest.exists():
        print("   ⏭️  megasam_final.pth already exists, skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    download_and_place(
        patterns=["MegaSAM/checkpoints/megasam_final.pth"],
        local_prefix="MegaSAM/checkpoints",
        dest=dest.parent,
    )
    print("✅ MegaSAM done")


def download_depthpro():
    print("\n📥 Depth Pro checkpoint (~1.9 GB)...")
    dest = PROJECT / "third_party" / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"
    if dest.exists():
        print("   ⏭️  depth_pro.pt already exists, skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    download_and_place(
        patterns=["DepthPro/checkpoints/depth_pro.pt"],
        local_prefix="DepthPro/checkpoints",
        dest=dest.parent,
    )
    print("✅ Depth Pro done")


def download_egomasks():
    """Download egocentric FP init masks from UCBProject/EgoDataMask (~70 MB)."""
    print("\n📥 Egocentric object masks — UCBProject/EgoDataMask (~70 MB)...")
    dest = PROJECT / "data_hub" / "ProcessedData" / "obj_recon_input" / "egocentric"
    if dest.exists() and any(dest.iterdir()):
        print(f"   ⏭️  {dest} already populated, skipping")
        return
    dest.mkdir(parents=True, exist_ok=True)
    tmp = PROJECT / ".hf_tmp_egomasks"
    snapshot_download(
        repo_id="UCBProject/EgoDataMask",
        repo_type="dataset",
        local_dir=str(tmp),
    )
    src = tmp / "egocentric"
    if src.exists():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(src), str(dest))
        shutil.rmtree(str(tmp), ignore_errors=True)
        n = sum(1 for _ in dest.iterdir() if _.is_dir())
        print(f"✅ Egocentric masks: {n} task categories → {dest}")
    else:
        # Flat structure — copy everything directly
        shutil.copytree(str(tmp), str(dest), dirs_exist_ok=True)
        shutil.rmtree(str(tmp), ignore_errors=True)
        print(f"✅ Egocentric masks → {dest}")


def download_third_masks():
    """Download third-person FP init masks from UCBProject/ThirdDataMask (~30 MB).

    Provides obj_recon_input/{ycb,oakink,taco}/ masks needed by Phase 1A
    FoundationPose (batch_obj_pose.py). Without these, FP cannot initialize.
    """
    print("\n📥 Third-person object masks — UCBProject/ThirdDataMask (~30 MB)...")
    base = PROJECT / "data_hub" / "ProcessedData" / "obj_recon_input"
    # Check if any subdataset masks already exist
    existing = [d for d in ["ycb", "oakink", "taco"] if (base / d).exists()]
    if existing:
        print(f"   ⏭️  Masks already present for: {', '.join(existing)}")
        missing = [d for d in ["ycb", "oakink", "taco"] if d not in existing]
        if not missing:
            return
        print(f"   Downloading missing: {', '.join(missing)}")
    base.mkdir(parents=True, exist_ok=True)
    tmp = PROJECT / ".hf_tmp_thirdmasks"
    snapshot_download(
        repo_id="UCBProject/ThirdDataMask",
        repo_type="dataset",
        local_dir=str(tmp),
    )
    for ds in ["ycb", "oakink", "taco"]:
        src = tmp / ds
        dst = base / ds
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            n = sum(1 for _ in dst.iterdir()) if dst.exists() else 0
            print(f"   ✅ {ds}: {n} objects → {dst}")
    # Also handle flat structure
    for item in tmp.iterdir():
        if item.is_dir() and not (base / item.name).exists():
            shutil.move(str(item), str(base / item.name))
    shutil.rmtree(str(tmp), ignore_errors=True)
    print(f"✅ Third-person masks done → {base}")


def download_obj_meshes():
    """Download object meshes from UCBProject/V2AP-Mesh (~1 GB).

    Provides obj_meshes/{ycb,egocentric,oakink}/ needed by Phase 1A+1B
    FoundationPose and scale estimation.
    """
    print("\n📥 Object meshes — UCBProject/V2AP-Mesh (~1 GB)...")
    dest = PROJECT / "data_hub" / "ProcessedData" / "obj_meshes"
    if dest.exists() and any(dest.iterdir()):
        n = sum(1 for _ in dest.iterdir() if _.is_dir())
        print(f"   ⏭️  {dest} already has {n} subdirs, skipping")
        return
    dest.mkdir(parents=True, exist_ok=True)
    tmp = PROJECT / ".hf_tmp_meshes"
    snapshot_download(
        repo_id="UCBProject/V2AP-Mesh",
        repo_type="dataset",
        local_dir=str(tmp),
    )
    # Move contents into obj_meshes/
    for item in tmp.iterdir():
        if item.name.startswith("."):
            continue
        dst = dest / item.name
        if not dst.exists():
            shutil.move(str(item), str(dst))
    shutil.rmtree(str(tmp), ignore_errors=True)
    n = sum(1 for _ in dest.iterdir() if _.is_dir())
    print(f"✅ Object meshes: {n} categories → {dest}")


def download_oakink():
    """Download OakInk v1 raw data from UCBProject/V2AP-OakInk (~25 GB).

    Places data at: data_hub/RawData/ThirdPersonRawData/oakink/
    """
    print("\n📥 OakInk v1 raw data — UCBProject/V2AP-OakInk (~25 GB)...")
    dest = PROJECT / "data_hub" / "RawData" / "ThirdPersonRawData" / "oakink"
    if dest.exists() and any(dest.iterdir()):
        n = sum(1 for _ in dest.iterdir() if _.is_dir())
        print(f"   ⏭️  {dest} already has {n} seq dirs, skipping")
        return
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="UCBProject/V2AP-OakInk",
        repo_type="dataset",
        local_dir=str(dest),
    )
    n = sum(1 for _ in dest.iterdir() if _.is_dir())
    print(f"✅ OakInk: {n} sequence directories → {dest}")


def download_egodex():
    """Download EgoDex raw dataset from UCBProject/V2AP-EgoDex (~30 GB)."""
    print("\n📥 EgoDex raw dataset (~30 GB, 101 tasks, 3051 sequences)...")
    dest = PROJECT / "data_hub" / "RawData" / "EgoRawData" / "egodex"
    if dest.exists() and any(dest.iterdir()):
        n = sum(1 for _ in dest.iterdir() if _.is_dir())
        print(f"   ⏭️  {dest} already has {n} task dirs, skipping")
        return
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="UCBProject/V2AP-EgoDex",
        repo_type="dataset",
        local_dir=str(dest),
    )
    n = sum(1 for _ in dest.iterdir() if _.is_dir())
    print(f"✅ EgoDex: {n} task categories → {dest}")


def download_taco():
    """Download TACO dataset from UCBProject/V2AP-TACO.

    Contains both Allocentric (~120 GB) and Egocentric (~24 GB) splits.
    """
    print("\n📥 TACO dataset — UCBProject/V2AP-TACO (~144 GB total)...")
    raw_base = PROJECT / "data_hub" / "RawData" / "ThirdPersonRawData" / "taco"
    ego_base = PROJECT / "data_hub" / "RawData" / "EgoRawData" / "taco"

    alloc_dest = raw_base / "Allocentric_RGB_Videos"
    ego_dest   = ego_base / "Egocentric_RGB_Videos"

    if alloc_dest.exists() and ego_dest.exists():
        print(f"   ⏭️  TACO already present, skipping")
        return

    print("   This is a large download (~144 GB). Use Ctrl+C to cancel and")
    print("   download manually from taco-group.github.io if preferred.")
    tmp = PROJECT / ".hf_tmp_taco"
    snapshot_download(
        repo_id="UCBProject/V2AP-TACO",
        repo_type="dataset",
        local_dir=str(tmp),
    )
    # Move Allocentric split
    src_alloc = tmp / "Allocentric_RGB_Videos"
    if src_alloc.exists():
        alloc_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_alloc), str(alloc_dest))
        print(f"   ✅ Allocentric → {alloc_dest}")
    # Move Egocentric split
    src_ego = tmp / "Egocentric_RGB_Videos"
    if src_ego.exists():
        ego_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_ego), str(ego_dest))
        print(f"   ✅ Egocentric  → {ego_dest}")
    shutil.rmtree(str(tmp), ignore_errors=True)
    print("✅ TACO done")


# ── Tool registry ──────────────────────────────────────────────────────────────
# Default (no --tool): only downloads model weights, NOT large datasets
WEIGHT_TOOLS = {
    "fp":          download_fp,
    "hawor":       download_hawor,
    "haptic":      download_haptic,
    "megasam":     download_megasam,
    "depthpro":    download_depthpro,
}

DATA_TOOLS = {
    "egomasks":    download_egomasks,    # ~70 MB
    "thirdmasks":  download_third_masks, # ~30 MB — needed for Phase 1A FP
    "objmeshes":   download_obj_meshes,  # ~1 GB  — needed for Phase 1A+1B FP
    "oakink":      download_oakink,      # ~25 GB — Phase 1A (third-person)
    "egodex":      download_egodex,      # ~30 GB — Phase 1B (egocentric)
    "taco":        download_taco,        # ~144 GB
}

TOOLS = {**WEIGHT_TOOLS, **DATA_TOOLS}


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights and datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_weights.py                    # all model weights (~10 GB)
  python setup_weights.py --tool fp          # FoundationPose only
  python setup_weights.py --tool thirdmasks  # Phase 1A FP init masks
  python setup_weights.py --tool objmeshes   # object meshes
  python setup_weights.py --tool egodex      # EgoDex raw dataset (~30 GB)
  python setup_weights.py --all-data         # all datasets (WARNING: ~200+ GB)
"""
    )
    parser.add_argument("--tool", choices=list(TOOLS.keys()),
                        help="Download a specific tool only")
    parser.add_argument("--all-data", action="store_true",
                        help="Also download all large datasets (egodex + taco, ~200 GB)")
    args = parser.parse_args()

    print("=" * 60)
    print("  V2AP — Setup")
    print(f"  Weights: {WEIGHTS_REPO}")
    print("=" * 60)

    if args.tool:
        TOOLS[args.tool]()
    else:
        # Default: model weights + small essential data assets
        for name, fn in WEIGHT_TOOLS.items():
            fn()
        # Always download masks and meshes (small, essential for FP)
        download_egomasks()
        download_third_masks()
        download_obj_meshes()
        if args.all_data:
            download_egodex()
            download_taco()

    print("\n" + "=" * 60)
    print("  Setup complete.")
    print()
    print("  ⚠️  MANO requires manual download (license restriction):")
    print("     1. Register at https://mano.is.tue.mpg.de/")
    print("     2. Download mano_v1_2.zip → extract MANO_RIGHT.pkl / MANO_LEFT.pkl")
    print("     3. Place under: third_party/haptic/assets/mano/")
    print("=" * 60)


if __name__ == "__main__":
    main()
