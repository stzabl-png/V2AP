#!/usr/bin/env python3
"""
Download training data and checkpoints from HuggingFace.

Usage:
    pip install huggingface-hub
    python download_data.py
"""

import os
import tarfile
import argparse


def main():
    parser = argparse.ArgumentParser(description="Download V2AP data from HuggingFace")
    parser.add_argument("--repo", default="UCBProject/V2AP-Data",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--subset", choices=["all", "train", "inference"],
                        default="all",
                        help="What to download: all, train-only, or inference-only")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface-hub not installed.")
        print("  pip install huggingface-hub")
        return

    project_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Downloading from {args.repo} ({args.subset})...")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=project_dir,
    )

    # Extract compressed archives
    for archive_name, extract_to in [
        ("data_hub/human_prior.tar.gz", "data_hub"),
        ("data_hub/training_m5.tar.gz", "data_hub"),
    ]:
        archive_path = os.path.join(project_dir, archive_name)
        extract_dir = os.path.join(project_dir, extract_to)
        if os.path.exists(archive_path):
            print(f"Extracting {archive_name}...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
            os.remove(archive_path)
            print(f"  Done, removed {archive_name}")

    print()
    print("Download complete!")
    print(f"  Data:        {os.path.join(project_dir, 'data_hub')}")
    print(f"  Dataset:     {os.path.join(project_dir, 'dataset')}")
    print(f"  Checkpoints: {os.path.join(project_dir, 'checkpoints')}")
    print()
    print("Quick start:")
    print("  python run.py --train --epochs 200        # Train")
    print("  python run.py --mesh path/to/obj --no-sim # Inference")


if __name__ == "__main__":
    main()
