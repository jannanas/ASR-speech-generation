from __future__ import annotations

from collections import defaultdict
import logging
import pickle
import sys
import warnings
import wave
from pathlib import Path
from typing import Any

import numpy as np
from sympy.core.expr import Float

from config import *

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

_speechbrain_log_filter: logging.Filter | None = None


class _SpeechBrainNoiseFilter(logging.Filter):
    """Drop speechbrain.* records below min_level (default: only ERROR+ reach stderr)."""

    __slots__ = ("min_level",)

    def __init__(self, min_level: int = logging.ERROR) -> None:
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "speechbrain" or record.name.startswith("speechbrain."):
            return record.levelno >= self.min_level
        return True


def _silence_speechbrain_logging(min_level: int = logging.ERROR) -> None:
    global _speechbrain_log_filter
    if _speechbrain_log_filter is None:
        _speechbrain_log_filter = _SpeechBrainNoiseFilter(min_level)
    else:
        _speechbrain_log_filter.min_level = min_level

    logging.getLogger("speechbrain").setLevel(min_level)
    for name in list(logging.root.manager.loggerDict):
        if isinstance(name, str) and name.startswith("speechbrain"):
            logging.getLogger(name).setLevel(min_level)

    for h in logging.getLogger().handlers:
        if _speechbrain_log_filter not in h.filters:
            h.addFilter(_speechbrain_log_filter)


def configure_logging(
    level: int = logging.INFO,
    *,
    speechbrain_min_level: int = logging.ERROR,
) -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
        root.setLevel(level)
    _silence_speechbrain_logging(speechbrain_min_level)


# --- Corpus cache (DATA_DIR / .cache / <PIPELINE_VERSION>) -------------------------

_log = logging.getLogger(__name__)


def _read_pickle(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, OSError, EOFError) as e:
        _log.warning("Cache unreadable %s: %s", path, e)
        return None


def _write_pickle_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    _log.info("Wrote cache %s", path)


def cache_path(corpus_id: str) -> Path:
    return CACHE_DIR / f"{corpus_id}.pkl"


def _validate_payload(data: Any, corpus_id: str) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("pipeline_version") != PIPELINE_VERSION:
        return False
    if data.get("corpus_id") != corpus_id:
        return False
    if "speakers" not in data or "utterances" not in data:
        return False
    return True


def try_load(corpus_id: str, *, log_hit: bool = True) -> dict | None:
    path = cache_path(corpus_id)
    data = _read_pickle(path)
    if data is None:
        return None
    if not _validate_payload(data, corpus_id):
        _log.info("Cache miss or stale pipeline for corpus %r (%s)", corpus_id, path)
        return None
    if log_hit:
        _log.info("Loaded corpus cache %s", path)
    return data


def save(payload: dict) -> None:
    _write_pickle_atomic(cache_path(payload["corpus_id"]), payload)


def save_after_scan(corpus_id: str, speakers: dict, utterances: dict) -> None:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
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


def merge_embeddings_from_corpus(
    corpus_id: str,
    corpus_speakers: dict,
    corpus_utterances: dict,
) -> None:
    """Persist speaker/utterance embeddings into the main corpus scan cache (same as __init__ load)."""
    data = try_load(corpus_id, log_hit=False)
    if data is None:
        _log.warning(
            "No valid scan cache for %r; cannot persist embeddings. Run corpus with use_cache=True after scan.",
            corpus_id,
        )
        return
    for sid, sp in corpus_speakers.items():
        if sid not in data["speakers"]:
            continue
        if sp.embedding is not None:
            data["speakers"][sid].embedding = np.asarray(sp.embedding, dtype=np.float32).copy()
    for sid, utts in corpus_utterances.items():
        if sid not in data["utterances"]:
            continue
        live_by_id = {u.id: u for u in utts}
        for u_cached in data["utterances"][sid]:
            live = live_by_id.get(u_cached.id)
            if live is not None and live.embedding is not None:
                u_cached.embedding = np.asarray(live.embedding, dtype=np.float32).copy()
    save(data)


# --- Speaker-embedding cache (pairing / SpeechBrain) --------------------------------

SPKREC_EMBEDDING_MODEL_SLUG = "speechbrain_spkrec-xvect-voxceleb"


def embedding_cache_path(corpus_id: str, model_slug: str = SPKREC_EMBEDDING_MODEL_SLUG) -> Path:
    return CACHE_DIR / "embeddings" / model_slug / f"{corpus_id}.pkl"


def try_load_embedding_cache(
    corpus_id: str,
    model_slug: str = SPKREC_EMBEDDING_MODEL_SLUG,
    *,
    log_hit: bool = True,
) -> tuple[dict, dict] | None:
    path = embedding_cache_path(corpus_id, model_slug)
    data = _read_pickle(path)
    if data is None:
        return None
    if not _validate_payload(data, corpus_id):
        _log.info("Embedding cache miss or stale for corpus %r (%s)", corpus_id, path)
        return None
    if log_hit:
        _log.info("Loaded embedding cache %s", path)
    return data["utterances"], data["speakers"]


def save_embedding_cache(
    corpus_id: str,
    utterances: dict,
    speakers: dict,
    model_slug: str = SPKREC_EMBEDDING_MODEL_SLUG,
) -> None:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "corpus_id": corpus_id,
        "utterances": utterances,
        "speakers": speakers,
    }
    _write_pickle_atomic(embedding_cache_path(corpus_id, model_slug), payload)


# --- Pairing -----------------------------------------------------------------

def invert(data: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    inverted = defaultdict(dict)
    for key, val in data.items():
        for subkey, subval in val.items():
            inverted[subkey][key] = subval
    return inverted


def _suppress_speechbrain_deprecation_warnings() -> None:
    """Inspect + lazy SpeechBrain shims emit UserWarning; hide before pairing imports speechbrain."""
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"^Module 'speechbrain\.[^']+' was deprecated, redirecting",
    )


_suppress_speechbrain_deprecation_warnings()
_silence_speechbrain_logging()