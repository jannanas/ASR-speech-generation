"""Create a speaker-level train/test/verification split for Zwitserlood.

The split is deterministic and approximately preserves both sex and age coverage in
all sets. It writes a CSV manifest to ./zwitserlood_speakers_split.csv by default.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from math import floor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from connectors import ZwitserloodCorpus  # noqa: E402
from models import Speaker  # noqa: E402

SPLIT_FRACTIONS = {
    "train": 0.70,
    "test": 0.20,
    "verification": 0.10,
}
SPLIT_ORDER = tuple(SPLIT_FRACTIONS)
DEFAULT_OUTPUT = REPO_ROOT / "zwitserlood_speakers_split.csv"


def allocate_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    """Allocate integer counts using largest remainders."""
    raw = {split: total * fraction for split, fraction in fractions.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remaining = total - sum(counts.values())

    remainders = sorted(
        fractions,
        key=lambda split: (raw[split] - counts[split], fractions[split]),
        reverse=True,
    )
    for split in remainders[:remaining]:
        counts[split] += 1
    return counts


def choose_split_for_age_rank(assigned: Counter[str], targets: dict[str, int], rank: int) -> str:
    """Choose the split most behind its target after seeing this age rank.

    Processing speakers sorted by age and repeatedly picking the most under-filled
    split spreads each split across the low/middle/high age range while still
    respecting the requested sex-specific quotas.
    """
    seen = rank + 1
    candidates = [split for split in SPLIT_ORDER if assigned[split] < targets[split]]
    return max(
        candidates,
        key=lambda split: (
            targets[split] * seen / sum(targets.values()) - assigned[split],
            targets[split],
            -SPLIT_ORDER.index(split),
        ),
    )


def allocate_counts_by_sex(by_sex: dict[str, list[Speaker]]) -> dict[str, dict[str, int]]:
    """Allocate sex-specific split counts while preserving global split totals."""
    total = sum(len(speakers) for speakers in by_sex.values())
    global_targets = allocate_counts(total, SPLIT_FRACTIONS)

    raw: dict[str, dict[str, float]] = {
        sex: {split: len(speakers) * fraction for split, fraction in SPLIT_FRACTIONS.items()}
        for sex, speakers in by_sex.items()
    }
    targets: dict[str, dict[str, int]] = {
        sex: {split: floor(value) for split, value in split_values.items()}
        for sex, split_values in raw.items()
    }
    remaining_by_sex = {
        sex: len(by_sex[sex]) - sum(split_values.values())
        for sex, split_values in targets.items()
    }
    deficits_by_split = {
        split: global_targets[split] - sum(targets[sex][split] for sex in by_sex)
        for split in SPLIT_ORDER
    }

    # Fill global split deficits from smallest split to largest. On equal remainders,
    # prefer the sex currently least represented in that split; this keeps the small
    # verification set from losing minority-sex coverage to a train-set tie.
    for split in sorted(SPLIT_ORDER, key=lambda item: global_targets[item]):
        while deficits_by_split[split] > 0:
            candidates = [sex for sex in by_sex if remaining_by_sex[sex] > 0]
            if not candidates:
                raise RuntimeError("Unable to allocate all split counts")
            sex = max(
                candidates,
                key=lambda candidate: (
                    raw[candidate][split] - targets[candidate][split],
                    -targets[candidate][split],
                    -len(by_sex[candidate]),
                    candidate,
                ),
            )
            targets[sex][split] += 1
            remaining_by_sex[sex] -= 1
            deficits_by_split[split] -= 1

    return targets


def split_by_sex_and_age(speakers: list[Speaker]) -> dict[str, list[Speaker]]:
    by_sex: dict[str, list[Speaker]] = defaultdict(list)
    for speaker in speakers:
        by_sex[speaker.sex or "unknown"].append(speaker)

    targets_by_sex = allocate_counts_by_sex(by_sex)
    split_speakers: dict[str, list[Speaker]] = {split: [] for split in SPLIT_ORDER}
    for sex in sorted(by_sex):
        sex_speakers = sorted(by_sex[sex], key=lambda speaker: (speaker.age, speaker.id))
        targets = targets_by_sex[sex]
        assigned: Counter[str] = Counter()

        for rank, speaker in enumerate(sex_speakers):
            split = choose_split_for_age_rank(assigned, targets, rank)
            split_speakers[split].append(speaker)
            assigned[split] += 1

    for split in SPLIT_ORDER:
        split_speakers[split].sort(key=lambda speaker: (speaker.sex or "", speaker.age, speaker.id))
    return split_speakers


def write_csv(output_path: Path, split_speakers: dict[str, list[Speaker]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["split", "speaker_id", "age", "sex", "dataset"],
        )
        writer.writeheader()
        for split in SPLIT_ORDER:
            for speaker in split_speakers[split]:
                writer.writerow(
                    {
                        "split": split,
                        "speaker_id": speaker.id,
                        "age": f"{speaker.age:.6f}",
                        "sex": speaker.sex or "",
                        "dataset": speaker.dataset,
                    }
                )


def print_report(split_speakers: dict[str, list[Speaker]]) -> None:
    all_ids = [speaker.id for speakers in split_speakers.values() for speaker in speakers]
    print(f"Speakers: {len(all_ids)}")
    print(f"Overlap check: {'ok' if len(all_ids) == len(set(all_ids)) else 'FAILED'}")
    print()

    for split in SPLIT_ORDER:
        speakers = split_speakers[split]
        ages = [speaker.age for speaker in speakers]
        sex_counts = Counter(speaker.sex or "unknown" for speaker in speakers)
        print(f"{split}: {len(speakers)} speakers")
        print(f"  sex: {dict(sorted(sex_counts.items()))}")
        print(
            "  age: "
            f"min={min(ages):.2f}, mean={sum(ages) / len(ages):.2f}, max={max(ages):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path; defaults to {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rescan Zwitserlood instead of loading the corpus cache.",
    )
    args = parser.parse_args()

    corpus = ZwitserloodCorpus(use_cache=not args.refresh_cache)
    split_speakers = split_by_sex_and_age(list(corpus.speakers.values()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, split_speakers)
    print_report(split_speakers)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
