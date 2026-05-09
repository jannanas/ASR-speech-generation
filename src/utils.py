from __future__ import annotations

import logging
import pickle
import sys
import wave
from pathlib import Path
from typing import Any

import config

# --- WAV ---------------------------------------------------------------------------

def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()
    return n_frames / sample_rate


def wav_sample_rate(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
    return sample_rate


# --- Logging ------------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)


# --- Corpus cache (DATA_DIR / .cache / <PIPELINE_VERSION>) -------------------------

_log = logging.getLogger(__name__)


def cache_path(corpus_id: str) -> Path:
    return config.CACHE_DIR / f"{corpus_id}.pkl"


def _validate_payload(data: Any, corpus_id: str) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("pipeline_version") != config.PIPELINE_VERSION:
        return False
    if data.get("corpus_id") != corpus_id:
        return False
    if "speakers" not in data or "utterances" not in data:
        return False
    return True


def try_load(corpus_id: str, *, log_hit: bool = True) -> dict | None:
    path = cache_path(corpus_id)
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, OSError, EOFError) as e:
        _log.warning("Cache unreadable %s: %s", path, e)
        return None
    if not _validate_payload(data, corpus_id):
        _log.info("Cache miss or stale pipeline for corpus %r (%s)", corpus_id, path)
        return None
    if log_hit:
        _log.info("Loaded corpus cache %s", path)
    return data


def save(payload: dict) -> None:
    path = cache_path(payload["corpus_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    _log.info("Wrote corpus cache %s", path)


def save_after_scan(corpus_id: str, speakers: dict, utterances: dict) -> None:
    payload = {
        "pipeline_version": config.PIPELINE_VERSION,
        "corpus_id": corpus_id,
        "speakers": dict(speakers),
        "utterances": {k: list(v) for k, v in utterances.items()},
    }
    save(payload)


def merge_mfcc_from_corpus(corpus_id: str, corpus_speakers: dict) -> None:
    data = try_load(corpus_id, log_hit=False)
    if data is None:
        _log.warning(
            "No valid scan cache for %r; cannot persist MFCC. Run corpus with use_cache=True after scan.",
            corpus_id,
        )
        return
    for sid, sp in corpus_speakers.items():
        if sp.mfcc_vector is None:
            continue
        if sid not in data["speakers"]:
            continue
        data["speakers"][sid].mfcc_vector = (
            sp.mfcc_vector.copy() if sp.mfcc_vector is not None else None
        )
    save(data)
