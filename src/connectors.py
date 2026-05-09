from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Self
import pandas as pd
from pathlib import Path

import config
from models import *
from utils import wav_duration, wav_sample_rate


class BaseCorpus(ABC):
    def __init__(self) -> None:
        self.id: str = None
        self.dirpath: Path = None
        self.datasets: list[BaseDataset] = []
        self.speakers: dict[str, Speaker] = {}
        self.utterances: dict[str, list[Utterance]] = defaultdict(list)
        self._setup_corpus_info()
        self._scan()
        self._consolidate_datasets()

    @abstractmethod
    def _setup_corpus_info(self) -> None:
        pass

    @abstractmethod
    def _scan(self) -> None:
        pass

    def _consolidate_datasets(self) -> None:
        for dataset in self.datasets:
            for speaker in dataset.speakers.values():
                self.speakers[speaker.id] = speaker
            for speaker_id, utterances in dataset.utterances.items():
                self.utterances[speaker_id].extend(utterances)

    def limit_speakers(self, limit: int) -> Self:
        self.speakers = dict(list(self.speakers.items())[:limit])
        self.utterances = {speaker_id: utterance for speaker_id, utterance in self.utterances.items() if speaker_id in self.speakers}
        return self
   
class ZwitserloodCorpus(BaseCorpus):
    def _setup_corpus_info(self) -> None:
        self.id = "Zwitserlood"
        self.dirpath = config.DATA_DIR / self.id

    def _scan(self) -> None:
        self.datasets = [
            SixToEightDataset(self),
            EightToTenDataset(self),
        ]

class UltraSuiteCorpus(BaseCorpus):
    def _setup_corpus_info(self) -> None:
        self.id = "UltraSuite"
        self.dirpath = config.DATA_DIR / self.id

    def _scan(self) -> None:
        upxDataset = UPXDataSet(self)
        upxDataset._scan()

        ux2020Dataset = UX2020DataSet(self)
        ux2020Dataset._scan()

        uxssdDataset = UXSSDDataSet(self)
        uxssdDataset._scan()

        self.datasets = [
            upxDataset,
            ux2020Dataset,
            uxssdDataset
        ]

class BaseDataset(ABC):
    def __init__(self, corpus: BaseCorpus) -> None:
        self.id: str = None
        self.dirpath: Path = None
        self.speakers: dict[str, Speaker] = {}
        self.utterances: dict[str, list[Utterance]] = defaultdict(list)
        self._setup_dataset_info(corpus)
        self._scan()

    @abstractmethod
    def _setup_dataset_info(self, corpus: BaseCorpus) -> None:
        pass

    @abstractmethod
    def _scan(self) -> None:
        pass

class ZwitserloodDataset(BaseDataset):
    def _scan(self) -> None:
        for utterance_path in self.dirpath.iterdir():
            if utterance_path.is_dir():
                continue
            if utterance_path.suffix != ".wav":
                continue

            speaker_id, utterance_id = utterance_path.name.split("_")

            # Add speaker (override if already exists)
            speaker = Speaker(
                id=f"{self.id}_{speaker_id}",
                age_range=(6, 8),   # more accurate info exists in the headers of .cha files
                disorder=Disorder.developmental_language_disorder
            )
            self.speakers[speaker.id] = speaker
            
            # Add utterance to that speaker
            utterance = Utterance(
                id=utterance_id[:-4],
                filepath=utterance_path,
                duration=wav_duration(utterance_path),
                sample_rate=wav_sample_rate(utterance_path),
                speaker=speaker_id,
                fragments=None
            )
            self.utterances[speaker.id].append(utterance)

class SixToEightDataset(ZwitserloodDataset):
    def _setup_dataset_info(self, corpus: BaseCorpus) -> None:
        dirname = '678_wav'
        self.id = f"{corpus.id}_{dirname}"
        self.dirpath = corpus.dirpath / dirname

class EightToTenDataset(ZwitserloodDataset):
    def _setup_dataset_info(self, corpus: BaseCorpus) -> None:
        dirname = '8910_wav'
        self.id = f"{corpus.id}_{dirname}"
        self.dirpath = corpus.dirpath / dirname

class UltraSuiteDataset(BaseDataset):
    def get_speaker(self, speaker_id: str) -> Speaker:
        speaker_info_filepath = self.docpath / "speakers"
        speaker_info_df = pd.read_csv(speaker_info_filepath, delimiter='\t')

        if 'speaker_id' in speaker_info_df.columns: 
            id_col = 'speaker_id'
        else: 
            id_col = "id"
        speaker_info = speaker_info_df[speaker_info_df[id_col] == speaker_id].iloc[0]

        disorder = Disorder.unknown
        if 'ssd_subtype' in speaker_info_df.columns: 
            disorder = Disorder(speaker_info['ssd_subtype'])
        
        speaker = Speaker(
            id=f"{self.id}_{speaker_id}",
            age_range=(int(speaker_info['age']), int(speaker_info['age'])),
            disorder=disorder
        )

        return speaker

    def _scan(self) -> None:
        for speaker_path in self.dirpath.iterdir():
            if not speaker_path.is_dir(): continue

            speaker = self.get_speaker(speaker_path.name)
            self.speakers[speaker.id] = speaker

            utterances = []
            for utterance_path in speaker_path.rglob('*'):
                if utterance_path.is_dir(): continue
                if utterance_path.suffix != ".wav": continue

                utterance = Utterance(
                    id=f"{utterance_path.parent.name}_{utterance_path.name[:-4]}" if utterance_path.parent.name != speaker_path.name else utterance_path.name[:-4],
                    filepath=utterance_path,
                    duration=wav_duration(utterance_path),
                    sample_rate=wav_sample_rate(utterance_path),
                    speaker=speaker.id,
                    fragments=None
                )
                self.utterances[speaker.id].append(utterance)

class UPXDataSet(UltraSuiteDataset):
    def _setup_dataset_info(self, corpus: BaseCorpus) -> None:
        dirname = 'core-upx'
        self.id = f"{corpus.id}_{dirname}"
        self.dirpath = corpus.dirpath / dirname / 'core'
        self.docpath = corpus.dirpath / dirname / 'doc'

class UX2020DataSet(UltraSuiteDataset):
    def _setup_dataset_info(self, corpus: BaseCorpus) -> None:
        dirname = 'core-ux2020'
        self.id = f"{corpus.id}_{dirname}"
        self.dirpath = corpus.dirpath / dirname / 'core'
        self.docpath = corpus.dirpath / dirname / 'doc'

class UXSSDDataSet(UltraSuiteDataset):
    def _setup_dataset_info(self, corpus: BaseCorpus) -> None:
        dirname = 'core-uxssd'
        self.id = f"{corpus.id}_{dirname}"
        self.dirpath = corpus.dirpath / dirname / 'core'
        self.docpath = corpus.dirpath / dirname / 'doc'
