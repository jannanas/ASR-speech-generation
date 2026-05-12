import logging
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from connectors import UltraSuiteCorpus
from utils import configure_logging

log = logging.getLogger(__name__)


def concat_wavs(
    paths: Sequence[Path],
    out_path: Path,
    *,
    sr: int = 16000,
    silence_ms: float = 300.0,
    desc: str | None = "load wav",
) -> Path:
    gap = int(sr * silence_ms / 1000.0)
    silence = np.zeros(gap, dtype=np.float32) if gap > 0 else np.zeros(0, dtype=np.float32)

    path_list = list(paths)
    pieces: list[np.ndarray] = []
    it = enumerate(path_list)
    if desc is not None and path_list:
        it = enumerate(tqdm(path_list, desc=desc, unit="file", leave=False))

    for i, p in it:
        y, _ = librosa.load(str(p), sr=sr, mono=True)
        pieces.append(np.asarray(y, dtype=np.float32))
        if gap > 0 and i < len(path_list) - 1:
            pieces.append(silence)

    combined = np.concatenate(pieces, axis=0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), combined, sr)
    return out_path


def main() -> None:
    configure_logging()
    log.info("Building UltraSuite corpus (scan or cache)")
    corpus = UltraSuiteCorpus()
    n_spk = len(corpus.speakers)
    log.info("Concatenating utterances | speakers=%d", n_spk)

    for speaker in tqdm(corpus.speakers.values(), desc="speakers", unit="spk"):
        merged_path = (
            corpus._dirpath
            / speaker.id.split("_")[1]
            / "core"
            / speaker.id.split("_")[2]
            / f"{speaker.id.split('_')[2]}_concat.wav"
        )
        utterance_paths = [u.filepath for u in corpus.utterances[speaker.id]]
        if not utterance_paths:
            log.warning("No utterances for speaker %s, skip", speaker.id)
            continue

        concat_wavs(
            utterance_paths,
            merged_path,
            desc=f"wav [{speaker.id[-12:]}]",
        )

    log.info("Done | processed %d speakers", n_spk)


if __name__ == "__main__":
    main()
