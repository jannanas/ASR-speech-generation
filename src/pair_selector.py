from collections import defaultdict
from speechbrain.inference.speaker import SpeakerRecognition
from connectors import BaseCorpus, ZwitserloodCorpus, UltraSuiteCorpus
from models import *
import torch

def pair_speakers(source: BaseCorpus, target: BaseCorpus, keep: int = 2, limit: int = -1) -> dict[str, dict[str, float]]:
    # Make dict of speaker.id to utterances
    source_data: dict[Speaker, list[Utterance]] = {}
    for dataset in source.datasets:
        items = [(data[0].id, data[1]) for _, data in dataset.data.items()][:limit]
        source_data.update(dict(items))
 
    target_data: dict[Speaker, list[Utterance]] = {}
    for dataset in target.datasets:
        items = [(data[0].id, data[1]) for _, data in dataset.data.items()][:limit]
        target_data.update(dict(items))

    # Setup cosine similiarity model
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        # run_opts={"device":"cuda"}
    )

    # Calculate pairwise similarity scores
    pairing_matrix: dict[str, dict[str, float]] = {}
    for source_speaker, source_utterances in source_data.items():
        for target_speaker, target_utterances in target_data.items():

            source_utterance_filepath = str(source_utterances[0].filepath.resolve().absolute())
            target_utterance_filepath = str(target_utterances[0].filepath.resolve().absolute())
            
            score, _ = verification.verify_files(source_utterance_filepath, target_utterance_filepath)
            pairing_matrix[source_speaker.id][target_speaker.id] = score

    return pairing_matrix
    

if __name__ == "__main__":    
    pairing_matrix = pair_speakers(
        source = ZwitserloodCorpus(),
        target = UltraSuiteCorpus(),
        limit = 5
    )

    print()
