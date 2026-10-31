"""
Build the source-identity embedding in a SEPARATE process.

Running this in the API server's own process would load ~500MB of model
weights (detector + recognition) permanently into the server, which on a 2GB
box leaves too little headroom for the part runners (they were OOM-killed in
testing when the server held the models). A short-lived subprocess returns all
memory to the OS when it exits.

Usage:
    python3 -m app.embed_runner --photos p1.jpg p2.jpg --models models --out out.npy
Prints {"photos": 4} on success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from app import pipeline  # heavy imports stay inside the subprocess

    n = pipeline.build_source_embedding(a.photos, a.models, a.out)
    print(json.dumps({"photos": n}), flush=True)


if __name__ == "__main__":
    main()
