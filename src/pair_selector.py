from collections import defaultdict
from config import N_MFCC
from connectors import BaseCorpus, ZwitserloodCorpus, UltraSuiteCorpus
from models import *
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint

def extract_mfcc(path: Path) -> np.ndarray:
    y, sr = librosa.load(path, sr=None)
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

def extract_mfcc_vectors(corpus: BaseCorpus) -> None:
    for speaker in corpus.speakers.values():

        utterance_represantations = []
        for utterance in corpus.utterances[speaker.id]:
            utterance_mfcc = extract_mfcc(utterance.filepath)
            utterance_representation = np.concatenate([utterance_mfcc.mean(axis=1), utterance_mfcc.std(axis=1)])
            utterance_represantations.append(utterance_representation)
        speaker.mfcc_vector = np.mean(utterance_represantations, axis=0)

def calculate_similarity(source: BaseCorpus, target: BaseCorpus) -> dict[str, dict[str, float]]:
    pairing_matrix: dict[str, dict[str, float]] = defaultdict(dict)

    source_speakers = list(source.speakers.values())
    target_speakers = list(target.speakers.values())

    src_matrix = np.array([s.mfcc_vector for s in source_speakers])  # (N_src, N_MFCC)
    tgt_matrix = np.array([t.mfcc_vector for t in target_speakers])  # (N_tgt, N_MFCC)

    sim_matrix = cosine_similarity(src_matrix, tgt_matrix)  # (N_src, N_tgt)

    for i, src in enumerate(source_speakers):
        for j, tgt in enumerate(target_speakers):
            pairing_matrix[src.id][tgt.id] = sim_matrix[i, j]

    return pairing_matrix

def pair_speakers(source: BaseCorpus, target: BaseCorpus, limit: int = 3) -> dict[str, dict[str, float]]:
    # Keep first `limit` speakers
    source.limit_speakers(limit)
    target.limit_speakers(limit)

    extract_mfcc_vectors(source)
    extract_mfcc_vectors(target)

    pairing_matrix = calculate_similarity(source, target)

    return pairing_matrix
    

if __name__ == "__main__":    
    pairing_matrix = pair_speakers(
        source = ZwitserloodCorpus(),
        target = UltraSuiteCorpus(),
        limit = 5
    )

    pprint(pairing_matrix)
