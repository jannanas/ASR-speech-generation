import csv
import random
from collections import defaultdict

from config import OUTPUT_DIR
from connectors import BaseCorpus, ZwitserloodCorpus, UltraSuiteCorpus
from utils import configure_logging, merge_embeddings_from_corpus, tlog
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

def extract_utterance_embedding(utterance: Utterance, encoder: EncoderClassifier) -> np.array:
    data, fs = sf.read(utterance.filepath, dtype="float32", always_2d=True)
    signal = torch.from_numpy(data.T)
    embedding = encoder.encode_batch(signal)[0][0].numpy()
    normalized_embedding = normalize(embedding.reshape(1, -1), norm="l2", axis=1)[0]
    
    utterance.embedding = normalized_embedding
    return normalized_embedding

    # data, fs = sf.read(utterance.filepath)
    # signal = torch.from_numpy(data).float().unsqueeze(0)
    # embedding = encoder.encode_batch(signal).squeeze()
    # return embedding

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
        tlog(
            __name__,
            "Speaker embeddings | corpus=%r | using cached values",
            corpus.id,
        )
        return

    tlog(__name__, "Speaker embeddings | corpus=%r | computing with SpeechBrain", corpus.id)
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
    tlog(
        __name__,
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
    random_seed: int | None = 42,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    if strategy == PairingStrategy.RANDOM:
        rng = random.Random(random_seed)
        for source_id in sorted(pairing_matrix.keys()):
            target_ids = sorted(pairing_matrix[source_id].keys())
            selected_targets = rng.sample(target_ids, k=min(k, len(target_ids)))
            for target_id in selected_targets:
                tlog(__name__, "Pair speaker | random: %s -> %s", source_id, target_id)
                pairs.append((source_id, target_id))
        return pairs

    reverse = strategy == PairingStrategy.SIMILAR
    for source_id in sorted(pairing_matrix.keys()):
        targets = pairing_matrix[source_id]
        top_k_target = sorted(targets.items(), key=lambda x: x[1], reverse=reverse)[:k]
        for target_id, score in top_k_target:
            tlog(__name__, "Pair speaker | %r: %s -> %s", score, source_id, target_id)
            pairs.append((source_id, target_id))
    return pairs

def load_split_speaker_ids(split_csv_path, split: str) -> set[str]:
    speaker_ids: set[str] = set()
    with open(split_csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        required_columns = {"split", "speaker_id"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Split CSV must contain columns {sorted(required_columns)}: {split_csv_path}"
            )
        for row in reader:
            if row["split"] == split:
                speaker_ids.add(row["speaker_id"])
    if not speaker_ids:
        raise ValueError(f"No speakers found for split {split!r} in {split_csv_path}")
    return speaker_ids


def keep_speakers(corpus: BaseCorpus, speaker_ids: set[str]) -> BaseCorpus:
    missing_speaker_ids = sorted(speaker_ids - set(corpus.speakers))
    if missing_speaker_ids:
        raise ValueError(
            f"Split contains {len(missing_speaker_ids)} speakers absent from {corpus.id}: "
            f"{missing_speaker_ids[:5]}"
        )

    corpus.speakers = {
        speaker_id: speaker
        for speaker_id, speaker in corpus.speakers.items()
        if speaker_id in speaker_ids
    }
    corpus.utterances = defaultdict(
        list,
        {
            speaker_id: list(utterances)
            for speaker_id, utterances in corpus.utterances.items()
            if speaker_id in corpus.speakers
        },
    )
    return corpus


def pair_speakers(
    source: BaseCorpus,
    target: BaseCorpus,
    limit: int | None = None,
    strategy: PairingStrategy = PairingStrategy.SIMILAR,
    k: int = 2,
    random_seed: int | None = 42,
) -> list[tuple[str, str]]:
    tlog(
        __name__,
        "Pair speakers | source=%r target=%r limit=%r k=%r strategy=%s",
        source.id,
        target.id,
        limit,
        k,
        strategy.name,
    )
    source.limit_speakers(limit)
    target.limit_speakers(limit)

    if strategy == PairingStrategy.RANDOM:
        pairing_matrix = {
            source_id: {target_id: 0.0 for target_id in target.speakers}
            for source_id in source.speakers
        }
        pairs = k_fold_match(pairing_matrix, strategy, k=k, random_seed=random_seed)
    else:
        extract_all_speaker_embeddings(source)
        extract_all_speaker_embeddings(target)

        pairing_matrix = calculate_similarity(source, target)
        pairs = k_fold_match(pairing_matrix, strategy, k=k, random_seed=random_seed)
    tlog(
        __name__,
        "Pairing done | matrix %d x %d | %d (source, target) pairs",
        len(source.speakers),
        len(target.speakers),
        len(pairs),
    )

    return pairs
    

if __name__ == "__main__":
    configure_logging()

    split_csv_path = OUTPUT_DIR / "zwitserlood_speakers_split.csv"
    train_speaker_ids = load_split_speaker_ids(split_csv_path, split="train")
    source = keep_speakers(ZwitserloodCorpus(use_cache=True), train_speaker_ids)
    tlog(
        __name__,
        "Loaded Zwitserlood train split | csv=%s | speakers=%d",
        split_csv_path,
        len(source.speakers),
    )

    pairs = pair_speakers(
        source=source,
        target=UltraSuiteCorpus(use_cache=True),
        strategy=PairingStrategy.RANDOM,
        # limit=3
    )

    pprint(pairs)
