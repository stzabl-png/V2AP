#!/usr/bin/env python3
"""
Single-sequence ViPE worker — runs in an isolated process so each call
gets a fresh CUDA context.

Called by eval_hoi4d_benchmark.py via subprocess.
"""
import argparse, sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   required=True)
    parser.add_argument("--outdir",  required=True)
    parser.add_argument("--biv2ap",  required=True)
    args = parser.parse_args()

    if args.biv2ap not in sys.path:
        sys.path.insert(0, args.biv2ap)

    from ego_pipeline.pipeline import EgoPipeline
    from ego_pipeline.stages.vipe_stage import ViPEStage

    pipeline = EgoPipeline(stages=[ViPEStage(save_depth=True)], debug=False)
    pipeline.run(args.video, args.outdir)
    print(f"Worker done: {args.outdir}")

if __name__ == "__main__":
    main()
