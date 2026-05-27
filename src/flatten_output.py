"""Flatten output WAV hierarchies and build per-directory transcript.txt files."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Repo root (parent of ``src/``) so imports resolve when running ``python src/flatten_output.py``.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.append(str(_repo_root / "src"))

from config import DATA_DIR, OUTPUT_DIR  # noqa: E402

TRANSCRIPT_DIRS = {
    "678_wav": "678_transcript",
    "8910_wav": "8910_transcript",
}
# Explicit blacklist: only these characters/patterns are stripped from transcript text.
# Apostrophes and dashes are intentionally preserved.
BLACKLISTED_CHARS = "\n\r:.,()0123456789!?&=+_/\";"
TIMESTAMP_RE = re.compile(r"\b\d+ms\s*-\s*\d+ms:\s*")
BRACKETED_RE = re.compile(r"\[[^\]]*\]")
GENERATED_SOURCE_RE = re.compile(r"^Zwitserlood_(?P<dataset>678_wav|8910_wav)_(?P<speaker>\d+)$")
FLAT_SOURCE_RE = re.compile(r"^Zwitserlood_(?P<dataset>678_wav|8910_wav)_(?P<utterance>.+)$")


def flatten_generated_pairs(output_dir: Path, dry_run: bool = False, copy: bool = False) -> int:
    """Flatten generated pair output: source/target_wav/file.wav -> source-target_wav-file.wav."""
    count = 0
    for wav_path in sorted(output_dir.glob("Zwitserlood_*/*_wav/*.wav")):
        source_speaker = wav_path.parents[1].name
        target_speaker = wav_path.parent.name
        destination = output_dir / f"{source_speaker}-{target_speaker}-{wav_path.name}"
        if destination == wav_path:
            continue
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        print(f"{'COPY' if copy else 'MOVE'} {wav_path} -> {destination}")
        if not dry_run:
            if copy:
                shutil.copy2(wav_path, destination)
            else:
                shutil.move(str(wav_path), str(destination))
        count += 1
    return count


def flatten_split_sources(output_dir: Path, dry_run: bool = False, copy: bool = False) -> int:
    """Flatten test/validation source output: dataset/file.wav -> Zwitserlood_dataset_file.wav."""
    count = 0
    for dataset_name in TRANSCRIPT_DIRS:
        dataset_dir = output_dir / dataset_name
        if not dataset_dir.is_dir():
            continue
        for wav_path in sorted(dataset_dir.glob("*.wav")):
            destination = output_dir / f"Zwitserlood_{dataset_name}_{wav_path.name}"
            if destination.exists():
                raise FileExistsError(f"Destination already exists: {destination}")
            print(f"{'COPY' if copy else 'MOVE'} {wav_path} -> {destination}")
            if not dry_run:
                if copy:
                    shutil.copy2(wav_path, destination)
                else:
                    shutil.move(str(wav_path), str(destination))
            count += 1
    return count


def transcript_path_for_wav_name(wav_name: str) -> Path:
    stem = Path(wav_name).stem

    if "-" in stem:
        source_speaker, rest = stem.split("-", maxsplit=1)
        _target_speaker, source_utterance = rest.rsplit("-", maxsplit=1)
        match = GENERATED_SOURCE_RE.match(source_speaker)
        if match is None:
            raise ValueError(f"Cannot parse generated output filename: {wav_name}")
        dataset_name = match.group("dataset")
        transcript_stem = source_utterance
    else:
        match = FLAT_SOURCE_RE.match(stem)
        if match is None:
            raise ValueError(f"Cannot parse source output filename: {wav_name}")
        dataset_name = match.group("dataset")
        transcript_stem = match.group("utterance")

    transcript_dir = TRANSCRIPT_DIRS[dataset_name]
    return DATA_DIR / "Zwitserlood" / dataset_name / transcript_dir / f"{transcript_stem}_transcript.txt"


def transcript_words(transcript_path: Path) -> tuple[str, set[str]]:
    text = transcript_path.read_text(encoding="utf-8")
    text = TIMESTAMP_RE.sub(" ", text)
    text = BRACKETED_RE.sub(" ", text)
    text = text.translate(str.maketrans({char: " " for char in BLACKLISTED_CHARS}))

    residual_special_chars = {
        char for char in text if not (char.isalpha() or char.isspace() or char in {"'", "-"})
    }
    text = " ".join(text.split())
    return text, residual_special_chars


def transcript_wav_names(output_dir: Path) -> list[str]:
    """Return current or planned flat root-level WAV names for output_dir."""
    names = {wav_path.name for wav_path in output_dir.glob("*.wav")}

    for wav_path in output_dir.glob("Zwitserlood_*/*_wav/*.wav"):
        source_speaker = wav_path.parents[1].name
        target_speaker = wav_path.parent.name
        names.add(f"{source_speaker}-{target_speaker}-{wav_path.name}")

    for dataset_name in TRANSCRIPT_DIRS:
        dataset_dir = output_dir / dataset_name
        if not dataset_dir.is_dir():
            continue
        for wav_path in dataset_dir.glob("*.wav"):
            names.add(f"Zwitserlood_{dataset_name}_{wav_path.name}")

    return sorted(names)


def write_directory_transcript(output_dir: Path, dry_run: bool = False) -> int:
    """Write one transcript.txt row per flat WAV name in output_dir."""
    rows: list[str] = []
    special_chars_by_file: dict[Path, set[str]] = defaultdict(set)

    for wav_name in transcript_wav_names(output_dir):
        source_transcript_path = transcript_path_for_wav_name(wav_name)
        if not source_transcript_path.exists():
            raise FileNotFoundError(f"Missing transcript for {wav_name}: {source_transcript_path}")
        words, special_chars = transcript_words(source_transcript_path)
        if special_chars:
            special_chars_by_file[source_transcript_path].update(special_chars)
        rows.append(f"{Path(wav_name).stem} {words}")

    transcript_output_path = output_dir / "transcript.txt"
    print(f"WRITE {transcript_output_path} ({len(rows)} rows)")

    if special_chars_by_file:
        print("Residual non-letter characters found after timestamp/parenthesis/colon/period cleanup:")
        for path, chars in sorted(special_chars_by_file.items(), key=lambda item: str(item[0])):
            rendered_chars = " ".join(repr(char) for char in sorted(chars))
            print(f"  {path}: {rendered_chars}")

    if not dry_run:
        transcript_output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    return len(rows)


def default_output_dirs() -> list[Path]:
    return [path for path in sorted(OUTPUT_DIR.iterdir()) if path.is_dir()]


def main() -> None:
    global BLACKLISTED_CHARS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="*",
        type=Path,
        help="Output directories to process. Defaults to all direct directories under output/.",
    )
    parser.add_argument(
        "--blacklist",
        default=BLACKLISTED_CHARS,
        help="Exact characters to strip from transcripts after timestamp removal.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without writing files.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them when flattening.")
    parser.add_argument("--no-flatten", action="store_true", help="Only write transcript.txt files.")
    parser.add_argument("--no-transcript", action="store_true", help="Only flatten WAV files.")
    args = parser.parse_args()

    BLACKLISTED_CHARS = args.blacklist

    output_dirs = args.directories or default_output_dirs()
    for output_dir in output_dirs:
        if not output_dir.is_absolute():
            output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)

        print(f"\n== {output_dir} ==")
        if not args.no_flatten:
            generated_count = flatten_generated_pairs(output_dir, dry_run=args.dry_run, copy=args.copy)
            source_count = flatten_split_sources(output_dir, dry_run=args.dry_run, copy=args.copy)
            print(f"Flattened {generated_count + source_count} WAVs")
        if not args.no_transcript:
            row_count = write_directory_transcript(output_dir, dry_run=args.dry_run)
            print(f"Transcript rows: {row_count}")


if __name__ == "__main__":
    main()
