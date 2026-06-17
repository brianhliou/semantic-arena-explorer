#!/usr/bin/env python3
"""Download the GloVe-6B slim embedding pack from the latest release.

The pack (~31 MB) is a release asset rather than a committed file. This pulls it
into data/embeddings/ so precompute.py can run. See DATA.md to rebuild from
source instead.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/brianhliou/semantic-arena-explorer/releases/download/data-v1"
ASSET = "glove-6B-300d-slim.npz"
DEST = Path(__file__).resolve().parent / "data" / "embeddings" / ASSET


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"already present: {DEST}")
        return 0
    url = f"{RELEASE}/{ASSET}"
    print(f"downloading {url}")
    try:
        urllib.request.urlretrieve(url, DEST)
    except Exception as exc:  # noqa: BLE001
        print(f"download failed: {exc}\nSee DATA.md to rebuild from source.", file=sys.stderr)
        return 1
    print(f"wrote {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
