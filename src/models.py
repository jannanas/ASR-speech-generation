import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from torch import ScriptModule

# Repo root (parent of ``src/``) so ``vendor.*`` resolves when running scripts as ``python src/...py``.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))


class PairingStrategy(Enum):
    SIMILAR = 'similar'
    DISSIMILAR = 'dissimilar'
    STRATIFIED = 'stratified'

class Disorder(Enum):
    developmental_language_disorder = 'developmental language disorder'
    inconsistent_phonological_disorder = 'inconsistent phonological disorder'
    phonological_disorder  = 'phonological disorder'
    childhood_apraxia_of_speech = 'childhood apraxia of speech'
    phonological_delay = 'phonological delay'
    vowel_disorder = 'vowel disorder'
    articulation_disorder = 'articulation disorder'
    unknown = 'unknown'

@dataclass
class Utterance:
    id: str
    filepath: Path
    duration: int       # in seconds
    sample_rate: int
    speaker: str
    fragments: Optional[list["Utterance"]] = None
    mos: float | None = None 
    embedding: np.ndarray = None

@dataclass
class ConvertedUtterance(Utterance):
    source_speaker: str = None
    target_speaker: str = None

@dataclass
class Speaker:
    id: str
    dataset: str
    age: float
    sex: str | None = None
    disorder: Disorder = Disorder.unknown
    utterances_concat: Utterance | None = None
    embedding: np.ndarray = None
