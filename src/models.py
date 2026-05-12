from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np

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
    fragments: Optional[list[Path]] = None
    mos: float | None = None  # e.g. LDNet non-intrusive MOS, set by src.mos

@dataclass
class Speaker:
    id: str
    dataset: str
    age_range: tuple[int, int]      # more accurate info exists than currently implemented
    # sex: str                      # omitted
    disorder: Disorder = Disorder.unknown
    mfcc_vector: np.ndarray = None
    utterances_concat_filepath: Path | None = None

