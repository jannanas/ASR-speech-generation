#!/usr/bin/env python3
"""Trim *_concat.wav files under ./data to at most 3 minutes; shorter files unchanged."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import soundfile as sf

MAX_SECONDS = 180.0
SUFFIX = "_concat.wav"


def find_targets(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob(f"*{SUFFIX}") if p.is_file())


def trim_if_needed(path: Path, max_seconds: float, dry_run: bool) -> None:
    info = sf.info(path)
    if info.duration <= max_seconds + 1e-3:
        print(f"skip (<= {max_seconds:.0f}s): {path}")
        return

    print(f"trim {info.duration:.1f}s -> {max_seconds:.0f}s: {path}")
    if dry_run:
        return

    n_frames = int(max_seconds * info.samplerate)
    with sf.SoundFile(path, "r") as f:
        data = f.read(frames=n_frames)

    parent = path.parent
    fd, tmp_name = tempfile.mkstemp(suffix=".wav", dir=parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        sf.write(
            tmp_path,
            data,
            info.samplerate,
            subtype=info.subtype,
        )
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory to search (default: ./data)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=MAX_SECONDS,
        help=f"Maximum duration in seconds (default: {MAX_SECONDS:.0f})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List actions without writing files",
    )
    args = parser.parse_args()

    root = args.data_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    paths = find_targets(root)
    if not paths:
        print(f"No *{SUFFIX} files under {root}")
        return

    for p in paths:
        trim_if_needed(p, args.max_seconds, args.dry_run)


if __name__ == "__main__":
    main()
