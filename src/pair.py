from collections import defaultdict
import logging
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from pprint import pprint

from config import N_MFCC
from connectors import BaseCorpus, UltraSuiteCorpus, ZwitserloodCorpus
from models import *
from utils import configure_logging, merge_mfcc_from_corpus, invert

log = logging.getLogger(__name__)


def extract_mfcc(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=16000)
    return librosa.feature.mfcc(y=y, sr=16000 , n_mfcc=N_MFCC)


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= eps:
        return np.zeros_like(vec)
    return vec / n

def extract_mfcc_vectors(corpus: BaseCorpus) -> None:
    '''
    We calculate the Mel-frequency cepstral coefficients for each utterance. 
    Then we L2 normalize them, average them per speaker, and renormalize.
    '''
    log.info(
        "MFCC extraction | corpus=%r | speakers=%d | utterances=%d | use_cache=%s",
        corpus.id,
        len(corpus.speakers),
        sum(len(corpus.utterances[sid]) for sid in corpus.speakers),
        corpus.use_cache,
    )

    speakers = list(corpus.speakers.values())
    for speaker in tqdm(speakers, desc=f"MFCC {corpus.id}", unit="speaker"):
        if corpus.use_cache and speaker.mfcc_vector is not None:
            continue
        utterance_represantations = []
        for utterance in tqdm(
            corpus.utterances[speaker.id],
            desc="utterances",
            leave=False,
            unit="utt",
        ):
            utterance_mfcc = extract_mfcc(utterance.filepath)
            utterance_representation = np.concatenate(
                [utterance_mfcc.mean(axis=1), utterance_mfcc.std(axis=1)]
            )
            utterance_represantations.append(_l2_normalize(utterance_representation))
        speaker.mfcc_vector = _l2_normalize(
            np.mean(utterance_represantations, axis=0)
        )
    if corpus.use_cache:
        merge_mfcc_from_corpus(corpus.id, corpus.speakers)

def calculate_similarity_scores(source: BaseCorpus, target: BaseCorpus) -> dict[str, dict[str, float]]:
    similarity_scores: dict[str, dict[str, float]] = defaultdict(dict)

    source_speakers = list(source.speakers.values())
    target_speakers = list(target.speakers.values())
    log.info(
        "Similarity scores | %r x %r | %d x %d speakers",
        source.id,
        target.id,
        len(source_speakers),
        len(target_speakers),
    )

    source_matrix = np.array([s.mfcc_vector for s in source_speakers])
    target_matrix = np.array([t.mfcc_vector for t in target_speakers])

    similarity_matrix = cosine_similarity(source_matrix, target_matrix)

    for i, source_speaker in enumerate(source_speakers):
        for j, target_speaker in enumerate(target_speakers):
            similarity_scores[source_speaker.id][target_speaker.id] = similarity_matrix[i, j]

    return similarity_scores

def calculate_rank_scores(
    silimarity_scores: dict[str, dict[str, float]],
    source_speakers: list[str],
    target_speakers: list[str],
    strategy: PairingStrategy,
) -> dict[str, dict[str, float]]:
    rank_scores: dict[str, dict[str, float]] = defaultdict(dict)

    if strategy == PairingStrategy.STRATIFIED:
        raise NotImplementedError()
    reverse = strategy == PairingStrategy.SIMILAR
    
    silimarity_scores_from_source = silimarity_scores
    silimarity_scores_from_target = invert(silimarity_scores_from_source)

    for source_speaker in source_speakers:
        ranked_targets = [
            target_id
            for target_id, _ in sorted(
                silimarity_scores_from_source[source_speaker].items(),
                key=lambda x: x[1],
                reverse=reverse
            )
        ]
   
        for target_speaker in target_speakers:
            ranked_sources = [
                target_id
                for target_id, _ in sorted(
                    silimarity_scores_from_target[target_speaker].items(),
                    key=lambda x: x[1],
                    reverse=reverse
                )
            ]
            
            target_rank = ranked_targets.index(target_speaker)
            source_rank = ranked_sources.index(source_speaker)

            rank_score = 1/(1 + source_rank) + 1/(1 + target_rank)
            rank_scores[source_speaker][target_speaker] = rank_score

    return rank_scores

def k_fold_match(
    scores: dict[str, dict[str, float]],
    k: int,
    max_target_use: int
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    target_use: dict[str, int] = defaultdict(int)

    for source_id in sorted(scores.keys()):
        ranked_targets = [
            target_id
            for target_id, _ in sorted(
                scores[source_id].items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        best_match = 0
        match_count = 0
        while match_count < k:
            target_id = ranked_targets[best_match]
            if target_use[target_id] < max_target_use:
                best_match += 1
                target_use[target_id] += 1
                match_count += 1
                pairs.append((source_id, target_id))
            else:
                best_match += 1

    return pairs

def pair_speakers(
    source: BaseCorpus,
    target: BaseCorpus,
    limit: int | None = None,
    strategy: PairingStrategy = PairingStrategy.SIMILAR,
    k: int = 2,
    max_target_use: int = 4
) -> list[tuple[str, str]]:
    log.info(
        "Pair speakers | source=%r target=%r limit=%r k=%r strategy=%s",
        source.id,
        target.id,
        limit,
        k,
        strategy.name,
    )
    source.limit_speakers(limit)
    target.limit_speakers(limit)

    # Extract mel-frequency cepstral coefficients for utterances, and combine to get one embeddings for each speaker
    extract_mfcc_vectors(source)
    extract_mfcc_vectors(target)

    # Calculate cosine similarity scores with MFCCs
    similarity_scores = calculate_similarity_scores(source, target)

    # Calculate rank normalized score to avoid all sources being matched with the same few targets
    rank_scores = calculate_rank_scores(similarity_scores, source.speakers.keys(), target.speakers.keys(), strategy)

    # Match pairs k-fold and limit reuse of targets
    pairs = k_fold_match(rank_scores, k=k, max_target_use=max_target_use)

    log.info(
        "Pairing done | matrix %d x %d | %d (source, target) pairs | %d unique targets used",
        len(source.speakers),
        len(target.speakers),
        len(pairs),
        len(set(target_id for _, target_id in pairs)),
    )

    return pairs
    
def main():
    configure_logging()
    
    pairs = pair_speakers(
        source=ZwitserloodCorpus(),
        target=UltraSuiteCorpus(),
        strategy=PairingStrategy.SIMILAR,
        k=2,
        max_target_use=3
    )

    pprint(pairs)


if __name__ == "__main__":
    main()