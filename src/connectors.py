from abc import ABC, abstractmethod
from collections import Counter, defaultdict
import re
from typing import Self
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import DATA_DIR
from models import *
from utils import *

class BaseCorpus(ABC):
    def __init__(self, use_cache: bool = True) -> None:
        self.use_cache = use_cache
        self.id: str = None
        self._dirpath: Path = None
        self._datasets: list[BaseDataset] = []
        self.speakers: dict[str, Speaker] = {}
        self.utterances: dict[str, list[Utterance]] = defaultdict(list)
        self._setup_corpus_info()

        if use_cache:
            cached = try_load(self.id)
            if cached is not None:
                self._datasets = []
                self.speakers = cached["speakers"]
                self.utterances = defaultdict(list)
                for sid, utts in cached["utterances"].items():
                    self.utterances[sid] = list(utts)
                n_utts = sum(len(u) for u in self.utterances.values())
                tlog(
                    __name__,
                    "Corpus %r (from cache): %d speakers, %d utterances",
                    self.id,
                    len(self.speakers),
                    n_utts,
                )
                return

        self._scan()
        self._consolidate_datasets()
        n_utts = sum(len(u) for u in self.utterances.values())
        tlog(
            __name__,
            "Corpus %r: %d speakers, %d utterances",
            self.id,
            len(self.speakers),
            n_utts,
        )
        if use_cache:
            save_after_scan(self.id, self.speakers, self.utterances)

    @abstractmethod
    def _setup_corpus_info(self) -> None:
        pass

    @abstractmethod
    def _scan(self) -> None:
        pass

    def _consolidate_datasets(self) -> None:
        for dataset in self._datasets:
            for speaker in dataset.speakers.values():
                self.speakers[speaker.id] = speaker
            for speaker_id, utterances in dataset.utterances.items():
                self.utterances[speaker_id].extend(utterances)

    def limit_speakers(self, limit: int | None) -> Self:
        if limit is None or limit < 0:
            return self
        self.speakers = dict(list(self.speakers.items())[:limit])
        self.utterances = {
            speaker_id: utts
            for speaker_id, utts in self.utterances.items()
            if speaker_id in self.speakers
        }
        return self
   
class ZwitserloodCorpus(BaseCorpus):
    def _setup_corpus_info(self) -> None:
        self.id = "Zwitserlood"
        self._dirpath = DATA_DIR / self.id

    def _scan(self) -> None:
        self._datasets = [
            SixToEightDataset(self),
            EightToTenDataset(self),
        ]

class UltraSuiteCorpus(BaseCorpus):
    def _setup_corpus_info(self) -> None:
        self.id = "UltraSuite"
        self._dirpath = DATA_DIR / self.id

    def _scan(self) -> None:
        upxDataset = UPXDataSet(self)
        upxDataset._scan()

        ux2020Dataset = UX2020DataSet(self)
        ux2020Dataset._scan()

        uxssdDataset = UXSSDDataSet(self)
        uxssdDataset._scan()

        self._datasets = [
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
    _CHAT_AGE_RE = re.compile(r"^(?P<years>\d+);(?P<months>\d+)\.")

    def _parse_chat_age_years(self, age: str) -> float | None:
        match = self._CHAT_AGE_RE.match(age.strip())
        if match is None:
            return None
        return int(match.group("years")) + int(match.group("months")) / 12

    def _load_speaker_metadata(self) -> dict[str, dict[str, object]]:
        metadata: dict[str, dict[str, object]] = {}
        raw: dict[str, list[tuple[Path, float | None, str | None]]] = defaultdict(list)

        for cha_path in sorted(self.chapath.glob("*.cha")):
            speaker_id = cha_path.stem.split("_", maxsplit=1)[0]
            for line in cha_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.startswith("@ID:") or "|CHI|" not in line:
                    continue
                fields = line.split("\t", maxsplit=1)[-1].split("|")
                age = self._parse_chat_age_years(fields[3]) if len(fields) > 3 else None
                sex = fields[4].strip().lower() if len(fields) > 4 and fields[4].strip() else None
                raw[speaker_id].append((cha_path, age, sex))
                break

        for speaker_id, rows in raw.items():
            ages = [age for _, age, _ in rows if age is not None]
            sexes = [sex for _, _, sex in rows if sex]
            sex = None
            if sexes:
                sex_counts = Counter(sexes)
                sex = sorted(sex_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
                if len(sex_counts) > 1:
                    twarning(
                        __name__,
                        "Inconsistent Zwitserlood sex metadata for %s_%s: %s; using %s",
                        self.id,
                        speaker_id,
                        dict(sex_counts),
                        sex,
                    )
            if not ages:
                twarning(__name__, "No Zwitserlood age metadata for %s_%s", self.id, speaker_id)
                continue
            metadata[speaker_id] = {
                "age": sum(ages) / len(ages),
                "sex": sex,
            }

        return metadata

    def _scan(self) -> None:
        speaker_metadata = self._load_speaker_metadata()
        paths = list(self.dirpath.iterdir())
        for utterance_path in tqdm(paths, desc=f"Scan {self.id}", unit="file"):
            if utterance_path.is_dir() or utterance_path.suffix != ".wav" or "_concat" in utterance_path.name:
                continue

            speaker_id, utterance_id = utterance_path.name.split("_", maxsplit=1)
            metadata = speaker_metadata.get(speaker_id)
            if metadata is None:
                twarning(__name__, "No Zwitserlood metadata for %s_%s", self.id, speaker_id)
                age = 0.0
                sex = None
            else:
                age = metadata["age"]
                sex = metadata["sex"]

            # utterances_concat_filepath = self.dirpath / f"{speaker_id}_concat.wav"
            speaker = Speaker(
                id=f"{self.id}_{speaker_id}",
                dataset=self.id,
                age=age,
                sex=sex,
                disorder=Disorder.developmental_language_disorder,
                # utterances_concat=Utterance(
                #     id=f"{speaker_id}_concat.wav",
                #     filepath=utterances_concat_filepath,
                #     duration=wav_duration(utterances_concat_filepath),
                #     sample_rate=wav_sample_rate(utterances_concat_filepath),
                #     speaker=f"{self.id}_{speaker_id}"
                # )
            )
            self.speakers[speaker.id] = speaker

            # # Get fragments
            # frag_paths = (self.dirpath / f"{self.id[12:-4]}_fragments").iterdir()
            # fragments = []
            # for frag_path in frag_paths:
            #     if f"{speaker_id}_{utterance_id[:-4]}" not in frag_path.name:
            #         continue
            #     fragments.append(Utterance(
            #         id=frag_path.name[:-4],
            #         filepath=frag_path,
            #         duration=wav_duration(frag_path),
            #         sample_rate=wav_sample_rate(frag_path),
            #         speaker=speaker_id
            #     ))
       
            
            # Add utterance to that speaker
            utterance = Utterance(
                id=utterance_id[:-4],
                filepath=utterance_path,
                duration=wav_duration(utterance_path),
                sample_rate=wav_sample_rate(utterance_path),
                speaker=speaker_id,
                # fragments=fragments
            )
            self.utterances[speaker.id].append(utterance)

class SixToEightDataset(ZwitserloodDataset):
    def _setup_dataset_info(self, corpus: BaseCorpus) -> None:
        dirname = '678_wav'
        self.id = f"{corpus.id}_{dirname}"
        self.dirpath = corpus._dirpath / dirname
        self.chapath = self.dirpath / '678_cha'

class EightToTenDataset(ZwitserloodDataset):
    def _setup_dataset_info(self, corpus: BaseCorpus) -> None:
        dirname = '8910_wav'
        self.id = f"{corpus.id}_{dirname}"
        self.dirpath = corpus._dirpath / dirname
        self.chapath = self.dirpath / '8910_cha'

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
        
        utterances_concat_filepath = self.dirpath / speaker_id / f"{speaker_id}_concat.wav"
        speaker = Speaker(
            id=f"{self.id}_{speaker_id}",
            dataset=self.id,
            age=float(speaker_info['age']),
            sex=str(speaker_info['sex']).lower() if 'sex' in speaker_info_df.columns else None,
            disorder=disorder,
            utterances_concat=Utterance(
                id=f"{speaker_id}_concat.wav",
                filepath=utterances_concat_filepath,
                duration=wav_duration(utterances_concat_filepath),
                sample_rate=wav_sample_rate(utterances_concat_filepath),
                speaker=f"{self.id}_{speaker_id}"
            )
        )

        return speaker

    def _scan(self) -> None:
        speaker_dirs = [p for p in self.dirpath.iterdir() if p.is_dir()]
        for speaker_path in tqdm(speaker_dirs, desc=f"Scan {self.id}", unit="spk"):
            speaker = self.get_speaker(speaker_path.name)
            self.speakers[speaker.id] = speaker

            wav_files = [
                p
                for p in speaker_path.rglob("*")
                if not p.is_dir() and p.suffix == ".wav" and p.name and "_concat" not in p.name
           
            ]
            for utterance_path in tqdm(
                wav_files,
                leave=False,
                desc="wav",
                unit="file",
            ):

                utterance = Utterance(
                    id=f"{utterance_path.parent.name}_{utterance_path.name[:-4]}" if utterance_path.parent.name != speaker_path.name else utterance_path.name[:-4],
                    filepath=utterance_path,
                    duration=wav_duration(utterance_path),
                    sample_rate=wav_sample_rate(utterance_path),
                    speaker=speaker.id,
                )
                self.utterances[speaker.id].append(utterance)

class UPXDataSet(UltraSuiteDataset):
    def _setup_dataset_info(self, corpus: BaseCorpus) -> None:
        dirname = 'core-upx'
        self.id = f"{corpus.id}_{dirname}"
        self.dirpath = corpus._dirpath / dirname / 'core'
        self.docpath = corpus._dirpath / dirname / 'doc'

class UX2020DataSet(UltraSuiteDataset):
    def _setup_dataset_info(self, corpus: BaseCorpus) -> None:
        dirname = 'core-ux2020'
        self.id = f"{corpus.id}_{dirname}"
        self.dirpath = corpus._dirpath / dirname / 'core'
        self.docpath = corpus._dirpath / dirname / 'doc'

class UXSSDDataSet(UltraSuiteDataset):
    def _setup_dataset_info(self, corpus: BaseCorpus) -> None:
        dirname = 'core-uxssd'
        self.id = f"{corpus.id}_{dirname}"
        self.dirpath = corpus._dirpath / dirname / 'core'
        self.docpath = corpus._dirpath / dirname / 'doc'
