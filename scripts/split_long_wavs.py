#!/usr/bin/env python3
"""Split .wav files longer than a max duration into fixed-length segments.

By default scans ./zwitserlood, skips filenames containing "concat" (case-insensitive),
and splits files longer than 3 minutes into chunks named like 01_T1_01.wav, 01_T1_02.wav, ...
The original file is kept unless you pass --delete-original.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import soundfile as sf

DEFAULT_MAX_SECONDS = 120


def should_skip(path: Path) -> bool:
    return "concat" in path.name.lower()


def list_wavs(root: Path, recursive: bool) -> list[Path]:
    it = root.rglob("*.wav") if recursive else root.glob("*.wav")
    paths = sorted(p for p in it if p.is_file() and not should_skip(p))
    return paths


def split_one(path: Path, max_seconds: float, dry_run: bool, delete_original: bool) -> None:
    info = sf.info(path)
    if info.duration <= max_seconds + 1e-3:
        print(f"skip (<= {max_seconds:.0f}s): {path}")
        return

    n_segments = math.ceil(info.duration / max_seconds)
    idx_width = max(2, len(str(n_segments)))
    segment_frames = int(max_seconds * info.samplerate)
    stem = path.stem
    parent = path.parent

    print(f"split {info.duration:.1f}s into {n_segments} part(s): {path}")
    if dry_run:
        return

    with sf.SoundFile(path, "r") as src:
        for i in range(n_segments):
            data = src.read(frames=segment_frames)
            if data.size == 0:
                break
            suffix = str(i + 1).zfill(idx_width)
            out_path = parent / f"{stem}_{suffix}.wav"
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
                os.replace(tmp_path, out_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

    if delete_original:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("zwitserlood"),
        help="Directory containing .wav files (default: ./zwitserlood)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan subdirectories for .wav files",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help=f"Maximum segment length in seconds (default: {DEFAULT_MAX_SECONDS:.0f})",
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Remove the source file after a successful split (default: keep original)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing or deleting files",
    )
    args = parser.parse_args()

    root = args.dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    paths = list_wavs(root, args.recursive)
    if not paths:
        print(f"No eligible .wav files under {root}")
        return

    for p in paths:
        split_one(p, args.max_seconds, args.dry_run, args.delete_original)


if __name__ == "__main__":
    main()
