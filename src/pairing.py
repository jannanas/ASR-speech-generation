from collections import defaultdict
import logging

from config import N_MFCC
from connectors import BaseCorpus, ZwitserloodCorpus, UltraSuiteCorpus
from utils import configure_logging, merge_mfcc_from_corpus
from models import *
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from pprint import pprint
import torch  # noqa: F401
import transformers  # noqa: F401
import flair  # noqa: F401
import soundfile as sf
from speechbrain.inference.classifiers import EncoderClassifier

log = logging.getLogger(__name__)


def extract_utterance_embedding(utterance: Utterance, classifier: EncoderClassifier) -> np.array:


def extract_speaker_embedding(speaker: Speaker, classifier: EncoderClassifier) -> None:
    
    # log.info(
    #     "MFCC extraction | corpus=%r | speakers=%d | utterances=%d | use_cache=%s",
    #     corpus.id,
    #     len(corpus.speakers),
    #     sum(len(corpus.utterances[sid]) for sid in corpus.speakers),
    #     use_cache,
    # )

    # speakers = list(corpus.speakers.values())
    # for speaker in tqdm(speakers, desc=f"MFCC {corpus.id}", unit="speaker"):
    #     if use_cache and speaker.mfcc_vector is not None:
    #         continue
    #     utterance_represantations = []
    #     for utterance in tqdm(
    #         corpus.utterances[speaker.id],
    #         desc="utterances",
    #         leave=False,
    #         unit="utt",
    #     ):
    #         utterance_mfcc = extract_mfcc(utterance.filepath)
    #         utterance_representation = np.concatenate([utterance_mfcc.mean(axis=1), utterance_mfcc.std(axis=1)])
    #         utterance_represantations.append(utterance_representation)
    #     speaker.mfcc_vector = np.mean(utterance_represantations, axis=0)
    # if use_cache:
    #     merge_mfcc_from_corpus(corpus.id, corpus.speakers)

def extract_all_speaker_embeddings(corpus: BaseCorpus, use_cache=True) -> None:
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")
    
    # signal, fs = load_wav_torch("C:/Users/Jannes/Repos/ASR-speech-generation/data/001E.wav")

def calculate_similarity(source: BaseCorpus, target: BaseCorpus) -> dict[str, dict[str, float]]:
    pairing_matrix: dict[str, dict[str, float]] = defaultdict(dict)

    source_speakers = list(source.speakers.values())
    target_speakers = list(target.speakers.values())
    log.info(
        "Similarity matrix | %r x %r | %d x %d speakers",
        source.id,
        target.id,
        len(source_speakers),
        len(target_speakers),
    )

    source_matrix = np.array([s.mfcc_vector for s in source_speakers])  # (N_src, N_MFCC)
    target_matrix = np.array([t.mfcc_vector for t in target_speakers])  # (N_tgt, N_MFCC)

    similarity_matrix = cosine_similarity(source_matrix, target_matrix)  # (N_src, N_tgt)

    for i, source_speaker in enumerate(source_speakers):
        for j, target_speaker in enumerate(target_speakers):
            pairing_matrix[source_speaker.id][target_speaker.id] = similarity_matrix[i, j]

    return pairing_matrix

def k_fold_match(
    pairing_matrix: dict[str, dict[str, float]],
    strategy: PairingStrategy,
    k: int,
) -> list[tuple[str, str]]:
    if strategy == PairingStrategy.STRATIFIED:
        raise NotImplementedError()
    reverse = strategy == PairingStrategy.SIMILAR

    pairs: list[tuple[str, str]] = []
    for source_id in sorted(pairing_matrix.keys()):
        targets = pairing_matrix[source_id]
        top_k_target = sorted(targets.items(), key=lambda x: x[1], reverse=reverse)[:k]
        for target_id, _ in top_k_target:
            pairs.append((source_id, target_id))
    return pairs

def pair_speakers(
    source: BaseCorpus,
    target: BaseCorpus,
    limit: int | None = None,
    strategy: PairingStrategy = PairingStrategy.SIMILAR,
    k: int = 2,
    use_mfcc_cache: bool = True,
) -> list[tuple[str, str]]:
    log.info(
        "Pair speakers | source=%r target=%r limit=%r k=%r strategy=%s use_mfcc_cache=%s",
        source.id,
        target.id,
        limit,
        k,
        strategy.name,
        use_mfcc_cache,
    )
    source.limit_speakers(limit)
    target.limit_speakers(limit)

    extract_mfcc_vectors(source, use_cache=use_mfcc_cache)
    extract_mfcc_vectors(target, use_cache=use_mfcc_cache)

    pairing_matrix = calculate_similarity(source, target)
    pairs = k_fold_match(pairing_matrix, strategy, k=k)
    log.info(
        "Pairing done | matrix %d x %d | %d (source, target) pairs",
        len(source.speakers),
        len(target.speakers),
        len(pairs),
    )

    return pairs
    

if __name__ == "__main__":
    configure_logging()
    pairs = pair_speakers(
        source=ZwitserloodCorpus(use_cache=True),
        target=UltraSuiteCorpus(use_cache=True),
        strategy=PairingStrategy.DISSIMILAR,
        use_mfcc_cache=True,
    )

    pprint(pairs)
