"""Chunk Zwitserlood WAVs on transcript line boundaries.

For each raw WAV such as ``01_T1.wav``, this script reads the matching
``01_T1_transcript.txt`` file, keeps only the transcribed audio spans, inserts a
short silence between spans, and writes chunks that are at most 30 seconds where
possible. Cuts are made only after transcript line ends, never inside a line.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Repo root (parent of ``src/``) so imports resolve when running ``python src/chunking.py``.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root / "src"))

from config import DATA_DIR  # noqa: E402

MAX_CHUNK_MS = 30_000
DEFAULT_SILENCE_MS = 300
TRANSCRIPT_LINE_RE = re.compile(r"^(?P<start>\d+)ms\s*-\s*(?P<end>\d+)ms:\s*(?P<text>.*)$")
NO_TIMESTAMP_LINE_RE = re.compile(r"^No Timestamp:\s*(?P<text>.*)$", re.IGNORECASE)
DATASETS = {
    "678_wav": "678_transcript",
    "8910_wav": "8910_transcript",
}


@dataclass(frozen=True)
class TranscriptLine:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class Chunk:
    start_ms: int
    end_ms: int
    lines: list[TranscriptLine]


def parse_transcript(path: Path) -> list[TranscriptLine]:
    parsed_lines: list[TranscriptLine | None] = []
    missing_timestamp_count = 0

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue

        timed_match = TRANSCRIPT_LINE_RE.match(raw_line)
        if timed_match is not None:
            start_ms = int(timed_match.group("start"))
            end_ms = int(timed_match.group("end"))
            if end_ms < start_ms:
                raise ValueError(f"Transcript line ends before it starts in {path}:{line_number}")
            parsed_lines.append(
                TranscriptLine(start_ms=start_ms, end_ms=end_ms, text=timed_match.group("text"))
            )
            continue

        no_timestamp_match = NO_TIMESTAMP_LINE_RE.match(raw_line)
        if no_timestamp_match is not None:
            parsed_lines.append(None)
            missing_timestamp_count += 1
            continue

        raise ValueError(f"Invalid transcript line in {path}:{line_number}: {raw_line!r}")

    if missing_timestamp_count:
        print(
            f"Warning: inferred timestamps for {missing_timestamp_count} 'No Timestamp' "
            f"line(s) in {path}",
            file=sys.stderr,
        )

    lines: list[TranscriptLine] = []
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_index = 0
    while raw_index < len(parsed_lines):
        parsed_line = parsed_lines[raw_index]
        if parsed_line is not None:
            lines.append(parsed_line)
            raw_index += 1
            continue

        missing_start_index = raw_index
        while raw_index < len(parsed_lines) and parsed_lines[raw_index] is None:
            raw_index += 1
        missing_end_index = raw_index
        missing_raw_lines = raw_lines[missing_start_index:missing_end_index]

        previous_end_ms = lines[-1].end_ms if lines else 0
        next_timed_line = parsed_lines[raw_index] if raw_index < len(parsed_lines) else None
        next_start_ms = next_timed_line.start_ms if next_timed_line is not None else previous_end_ms
        available_ms = max(0, next_start_ms - previous_end_ms)
        missing_count = missing_end_index - missing_start_index

        for offset, raw_missing_line in enumerate(missing_raw_lines):
            no_timestamp_match = NO_TIMESTAMP_LINE_RE.match(raw_missing_line)
            if no_timestamp_match is None:
                raise ValueError(f"Unexpected missing timestamp line in {path}: {raw_missing_line!r}")
            start_ms = previous_end_ms + round(available_ms * offset / missing_count)
            end_ms = previous_end_ms + round(available_ms * (offset + 1) / missing_count)
            lines.append(
                TranscriptLine(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=no_timestamp_match.group("text"),
                )
            )

    return lines


def line_duration_ms(line: TranscriptLine) -> int:
    return line.end_ms - line.start_ms


def compressed_duration_ms(lines: list[TranscriptLine], silence_ms: int) -> int:
    if not lines:
        return 0
    return sum(line_duration_ms(line) for line in lines) + silence_ms * (len(lines) - 1)


def choose_cut(
    lines: list[TranscriptLine],
    start_index: int,
    max_chunk_ms: int,
    silence_ms: int,
) -> int:
    """Return the last transcript-line index for the next chunk.

    Normal case: choose the line boundary that gets the compressed chunk closest
    to 30s without exceeding 30s.

    Tail case: if that cut would leave a final compressed tail shorter than 30s,
    choose a better midpoint cut between the current chunk and that final tail,
    still only cutting after line ends.
    """
    remaining_lines = lines[start_index:]
    if compressed_duration_ms(remaining_lines, silence_ms) <= max_chunk_ms:
        return len(lines) - 1

    valid_normal_cuts = [
        index
        for index in range(start_index, len(lines) - 1)
        if compressed_duration_ms(lines[start_index : index + 1], silence_ms) <= max_chunk_ms
    ]
    if not valid_normal_cuts:
        # A single transcript line is longer than the requested maximum. Keep it
        # intact because cutting inside a line is explicitly disallowed.
        return start_index

    normal_cut = min(
        valid_normal_cuts,
        key=lambda index: abs(
            max_chunk_ms - compressed_duration_ms(lines[start_index : index + 1], silence_ms)
        ),
    )
    tail_ms = compressed_duration_ms(lines[normal_cut + 1 :], silence_ms)

    if tail_ms >= max_chunk_ms:
        return normal_cut

    balanced_cuts = []
    for index in range(start_index, len(lines) - 1):
        current_ms = compressed_duration_ms(lines[start_index : index + 1], silence_ms)
        remaining_ms = compressed_duration_ms(lines[index + 1 :], silence_ms)
        if current_ms <= max_chunk_ms and remaining_ms <= max_chunk_ms:
            balanced_cuts.append((index, current_ms, remaining_ms))

    if not balanced_cuts:
        return normal_cut

    return min(
        balanced_cuts,
        key=lambda item: (abs(item[1] - item[2]), abs(max_chunk_ms - item[1])),
    )[0]


def plan_chunks(
    lines: list[TranscriptLine],
    max_chunk_ms: int = MAX_CHUNK_MS,
    silence_ms: int = DEFAULT_SILENCE_MS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    start_index = 0

    while start_index < len(lines):
        chunk_start_ms = lines[start_index].start_ms
        cut_index = choose_cut(lines, start_index, max_chunk_ms, silence_ms)
        chunk_lines = lines[start_index : cut_index + 1]
        chunks.append(
            Chunk(
                start_ms=chunk_start_ms,
                end_ms=chunk_lines[-1].end_ms,
                lines=chunk_lines,
            )
        )
        start_index = cut_index + 1

    return chunks


def write_chunk_wav(
    source_wav: Path,
    output_wav: Path,
    chunk: Chunk,
    silence_ms: int,
) -> None:
    info = sf.info(source_wav)
    audio_parts = []
    silence_frames = round(silence_ms * info.samplerate / 1000)
    silence = np.zeros((silence_frames, info.channels), dtype="float32")

    for line_index, line in enumerate(chunk.lines):
        start_frame = round(line.start_ms * info.samplerate / 1000)
        end_frame = round(line.end_ms * info.samplerate / 1000)
        audio, _sample_rate = sf.read(
            source_wav,
            start=start_frame,
            stop=end_frame,
            dtype="float32",
            always_2d=True,
        )
        audio_parts.append(audio)
        if line_index < len(chunk.lines) - 1 and silence_frames > 0:
            audio_parts.append(silence)

    chunk_audio = np.concatenate(audio_parts, axis=0) if audio_parts else silence[:0]
    sf.write(output_wav, chunk_audio, info.samplerate, subtype=info.subtype)


def write_chunk_transcript(output_transcript: Path, chunk: Chunk, silence_ms: int) -> None:
    transcript_lines = []
    cursor_ms = 0
    for line_index, line in enumerate(chunk.lines):
        duration_ms = line_duration_ms(line)
        transcript_lines.append(f"{cursor_ms}ms - {cursor_ms + duration_ms}ms: {line.text}")
        cursor_ms += duration_ms
        if line_index < len(chunk.lines) - 1:
            cursor_ms += silence_ms

    output_transcript.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")


def chunk_file(
    wav_path: Path,
    transcript_path: Path,
    transcript_dir: Path,
    max_chunk_ms: int,
    silence_ms: int,
    overwrite: bool,
    dry_run: bool,
) -> int:
    transcript_lines = parse_transcript(transcript_path)
    if not transcript_lines:
        print(f"Skip {wav_path}: empty transcript")
        return 0

    chunks = plan_chunks(transcript_lines, max_chunk_ms=max_chunk_ms, silence_ms=silence_ms)
    wav_stem = wav_path.stem

    for chunk_index, chunk in enumerate(chunks, start=1):
        output_wav = wav_path.with_name(f"{wav_stem}_chunk_{chunk_index}.wav")
        output_transcript = transcript_dir / f"{wav_stem}_chunk_{chunk_index}_transcript.txt"

        if dry_run:
            print(
                f"{wav_path.name} chunk {chunk_index}: "
                f"source {chunk.start_ms}ms-{chunk.end_ms}ms, "
                f"output {compressed_duration_ms(chunk.lines, silence_ms)}ms, "
                f"{len(chunk.lines)} transcript lines"
            )
            continue

        if not overwrite and (output_wav.exists() or output_transcript.exists()):
            raise FileExistsError(
                f"Refusing to overwrite existing chunk files for {wav_path}. Use --overwrite."
            )

        write_chunk_wav(wav_path, output_wav, chunk, silence_ms)
        write_chunk_transcript(output_transcript, chunk, silence_ms)

    return len(chunks)


def raw_wavs(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.glob("*.wav")
        if "_concat" not in path.name and "_chunk_" not in path.name
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        help="Only process one Zwitserlood dataset directory.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help="Maximum chunk duration in seconds where line boundaries allow it.",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N raw WAVs.")
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=DEFAULT_SILENCE_MS,
        help="Silence inserted between transcribed spans in each output chunk.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing chunk files.")
    parser.add_argument("--dry-run", action="store_true", help="Print chunk plan without writing files.")
    args = parser.parse_args()

    max_chunk_ms = round(args.max_seconds * 1000)
    selected_datasets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    total_files = 0
    total_chunks = 0
    for dataset_name, transcript_dir_name in selected_datasets.items():
        dataset_dir = DATA_DIR / "Zwitserlood" / dataset_name
        transcript_dir = dataset_dir / transcript_dir_name
        wav_paths = raw_wavs(dataset_dir)
        if args.limit is not None:
            wav_paths = wav_paths[: args.limit]

        for wav_path in tqdm(wav_paths, desc=f"chunk {dataset_name}", unit="wav"):
            transcript_path = transcript_dir / f"{wav_path.stem}_transcript.txt"
            if not transcript_path.exists():
                raise FileNotFoundError(f"Missing transcript for {wav_path}: {transcript_path}")
            total_chunks += chunk_file(
                wav_path=wav_path,
                transcript_path=transcript_path,
                transcript_dir=transcript_dir,
                max_chunk_ms=max_chunk_ms,
                silence_ms=args.silence_ms,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            total_files += 1

    action = "Planned" if args.dry_run else "Wrote"
    print(f"{action} {total_chunks} chunks from {total_files} WAVs")


if __name__ == "__main__":
    main()
