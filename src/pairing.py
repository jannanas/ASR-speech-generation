from collections import defaultdict
import logging

from connectors import BaseCorpus, ZwitserloodCorpus, UltraSuiteCorpus
from utils import configure_logging, merge_embeddings_from_corpus
from models import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
from pprint import pprint
import torch  # noqa: F401
import transformers  # noqa: F401
import flair  # noqa: F401
import soundfile as sf
from speechbrain.inference.classifiers import EncoderClassifier

log = logging.getLogger(__name__)


def extract_utterance_embedding(utterance: Utterance, encoder: EncoderClassifier) -> np.array:
    data, fs = sf.read(utterance.filepath, dtype="float32", always_2d=True)
    signal = torch.from_numpy(data.T)
    embedding = encoder.encode_batch(signal)[0][0].numpy()
    normalized_embedding = normalize(embedding.reshape(1, -1), norm="l2", axis=1)[0]
    
    utterance.embedding = normalized_embedding
    return normalized_embedding

def extract_speaker_embedding(corpus: BaseCorpus, speaker: Speaker, encoder: EncoderClassifier) -> np.array:
    # speaker_embedding = np.mean(utterance_embeddings, axis=0)
    # normalized_speaker_embedding = normalize(np.array(speaker_embedding).reshape(1, -1), norm="l2", axis=1)[0]
    # speaker.embedding = normalized_speaker_embedding
    # return normalized_speaker_embedding

    utterance_embeddings = []
    for utterance in corpus.utterances[speaker.id]:
        utterance_embedding = extract_utterance_embedding(utterance, encoder)
        utterance_embeddings.append(utterance_embedding)

    stacked_utts = np.stack(utterance_embeddings, axis=0)
    speaker_embedding = stacked_utts.mean(axis=0)
    normalized_speaker_embedding = normalize(
        np.asarray(speaker_embedding, dtype=np.float64).reshape(1, -1),
        norm="l2",
        axis=1,
    )[0]
    speaker.embedding = normalized_speaker_embedding.astype(np.float32)
    return normalized_speaker_embedding.astype(np.float32)


def _mean_center_and_renormalize(
    corpus: BaseCorpus, speaker_id_to_vec: dict[str, np.ndarray]
) -> None:
    """Subtract per-dimension corpus mean, then L2-normalize (fixes wrong scalar np.mean bug)."""
    if not speaker_id_to_vec:
        return
    mat = np.stack(
        [np.asarray(speaker_id_to_vec[sid], dtype=np.float64) for sid in corpus.speakers],
        axis=0,
    )
    mean_vec = mat.mean(axis=0)
    for sid in corpus.speakers:
        vec = np.asarray(speaker_id_to_vec[sid], dtype=np.float64)
        centered = vec - mean_vec
        corpus.speakers[sid].embedding = (
            normalize(centered.reshape(1, -1), norm="l2", axis=1)[0].astype(np.float32)
        )


def extract_all_speaker_embeddings(corpus: BaseCorpus) -> None:
    first_speaker = next(iter(corpus.speakers.values()))
    if corpus.use_cache and first_speaker.embedding is not None:
        log.info(
            "Speaker embeddings | corpus=%r | using cached values",
            corpus.id,
        )
        return

    log.info("Speaker embeddings | corpus=%r | computing with SpeechBrain", corpus.id)
    encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")

    speaker_embeddings: dict[str, np.ndarray] = {}
    for speaker in tqdm(
        corpus.speakers.values(),
        desc=f"spk-emb {corpus.id}",
        unit="speaker",
    ):
        speaker_embeddings[speaker.id] = extract_speaker_embedding(corpus, speaker, encoder)

    _mean_center_and_renormalize(corpus, speaker_embeddings)

    if corpus.use_cache:
        merge_embeddings_from_corpus(corpus.id, corpus.speakers, corpus.utterances)

 
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

    source_matrix = np.stack([s.embedding for s in source_speakers])
    target_matrix = np.stack([t.embedding for t in target_speakers])
    similarity_matrix = cosine_similarity(source_matrix, target_matrix)

    for i, source_speaker in enumerate(source_speakers):
        for j, target_speaker in enumerate(target_speakers):
            pairing_matrix[source_speaker.id][target_speaker.id] = similarity_matrix[i, j]

    return pairing_matrix

def k_fold_match(
    pairing_matrix: dict[str, dict[str, float]],
    strategy: PairingStrategy,
    k: int,
) -> list[tuple[str, str, float]]:
    if strategy == PairingStrategy.STRATIFIED:
        raise NotImplementedError()
    reverse = strategy == PairingStrategy.SIMILAR

    pairs: list[tuple[str, str, float]] = []
    for source_id in sorted(pairing_matrix.keys()):
        targets = pairing_matrix[source_id]
        top_k_target = sorted(targets.items(), key=lambda x: x[1], reverse=reverse)[:k]
        for target_id, score in top_k_target:
            pairs.append((source_id, target_id, score))
    return pairs

def pair_speakers(
    source: BaseCorpus,
    target: BaseCorpus,
    limit: int | None = None,
    strategy: PairingStrategy = PairingStrategy.SIMILAR,
    k: int = 2,
) -> list[tuple[str, str, float]]:
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

    extract_all_speaker_embeddings(source)
    extract_all_speaker_embeddings(target)

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
        strategy=PairingStrategy.SIMILAR,
        # limit=3
    )

    pprint(pairs)
