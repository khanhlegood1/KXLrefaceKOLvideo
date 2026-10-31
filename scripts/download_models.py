#!/usr/bin/env python3
"""Download (once) all models the face-swap pipeline needs.

    python3 scripts/download_models.py [--models-dir models]

Files (total ~760 MB, cached on disk afterwards):
  - buffalo_l.zip  -> det_10g.onnx (SCRFD detector) + w600k_r50.onnx (recognition)
  - inswapper_128.onnx (fp32 swapper, ~554 MB)
  - emap_fp32.npy  (latent matrix, extracted from the swapper graph)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=None)
    a = ap.parse_args()
    d = pipeline.ensure_models(a.models_dir)
    print("models ready in:", d)


if __name__ == "__main__":
    main()
