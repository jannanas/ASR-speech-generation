import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# Repo root (parent of ``src/``) so ``vendor.*`` resolves when running ``python .../src/vc.py``.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from torch import ScriptModule
from config import OUTPUT_DIR, ROOT_DIR, SAMPLE_RATE
from connectors import BaseCorpus, UltraSuiteCorpus, ZwitserloodCorpus
from models import ConvertedUtterance, Utterance
from utils import configure_logging, tlog, wav_duration, wav_sample_rate
from vendor.MeanVC.src.infer.infer_ref import *
from vendor.MeanVC.src.infer.dit_kvcache import DiT
from vendor.MeanVC.src.runtime.speaker_verification.ecapa_tdnn import ECAPA_TDNN

pairs = [
 ('Zwitserlood_678_wav_01', 'UltraSuite_core-ux2020_02F'),
 ('Zwitserlood_678_wav_01', 'UltraSuite_core-ux2020_09M'),
 ('Zwitserlood_678_wav_02', 'UltraSuite_core-ux2020_17M'),
 ('Zwitserlood_678_wav_02', 'UltraSuite_core-upx_18F'),
 ('Zwitserlood_678_wav_03', 'UltraSuite_core-upx_10M'),
 ('Zwitserlood_678_wav_03', 'UltraSuite_core-upx_02F'),
 ('Zwitserlood_678_wav_04', 'UltraSuite_core-ux2020_02F'),
 ('Zwitserlood_678_wav_04', 'UltraSuite_core-ux2020_06F'),
 ('Zwitserlood_678_wav_05', 'UltraSuite_core-ux2020_17M'),
 ('Zwitserlood_678_wav_05', 'UltraSuite_core-ux2020_19M'),
 ('Zwitserlood_678_wav_06', 'UltraSuite_core-ux2020_03M'),
 ('Zwitserlood_678_wav_06', 'UltraSuite_core-ux2020_02F'),
 ('Zwitserlood_678_wav_07', 'UltraSuite_core-upx_17M'),
 ('Zwitserlood_678_wav_07', 'UltraSuite_core-ux2020_11M'),
 ('Zwitserlood_678_wav_08', 'UltraSuite_core-ux2020_15M'),
 ('Zwitserlood_678_wav_08', 'UltraSuite_core-ux2020_11M'),
 ('Zwitserlood_678_wav_09', 'UltraSuite_core-upx_08M'),
 ('Zwitserlood_678_wav_09', 'UltraSuite_core-ux2020_17M'),
 ('Zwitserlood_678_wav_10', 'UltraSuite_core-ux2020_29M'),
 ('Zwitserlood_678_wav_10', 'UltraSuite_core-upx_11M'),
 ('Zwitserlood_678_wav_11', 'UltraSuite_core-ux2020_17M'),
 ('Zwitserlood_678_wav_11', 'UltraSuite_core-upx_08M'),
 ('Zwitserlood_678_wav_12', 'UltraSuite_core-ux2020_15M'),
 ('Zwitserlood_678_wav_12', 'UltraSuite_core-ux2020_25M'),
 ('Zwitserlood_678_wav_13', 'UltraSuite_core-upx_04M'),
 ('Zwitserlood_678_wav_13', 'UltraSuite_core-upx_09M'),
 ('Zwitserlood_678_wav_14', 'UltraSuite_core-ux2020_32M'),
 ('Zwitserlood_678_wav_14', 'UltraSuite_core-ux2020_22M'),
 ('Zwitserlood_678_wav_15', 'UltraSuite_core-ux2020_07M'),
 ('Zwitserlood_678_wav_15', 'UltraSuite_core-ux2020_37F'),
 ('Zwitserlood_678_wav_16', 'UltraSuite_core-ux2020_37F'),
 ('Zwitserlood_678_wav_16', 'UltraSuite_core-ux2020_02F'),
 ('Zwitserlood_678_wav_17', 'UltraSuite_core-ux2020_10M'),
 ('Zwitserlood_678_wav_17', 'UltraSuite_core-uxssd_01M'),
 ('Zwitserlood_678_wav_18', 'UltraSuite_core-upx_04M'),
 ('Zwitserlood_678_wav_18', 'UltraSuite_core-upx_08M'),
 ('Zwitserlood_678_wav_19', 'UltraSuite_core-upx_01F'),
 ('Zwitserlood_678_wav_19', 'UltraSuite_core-ux2020_13F'),
 ('Zwitserlood_678_wav_20', 'UltraSuite_core-upx_12M'),
 ('Zwitserlood_678_wav_20', 'UltraSuite_core-upx_17M'),
 ('Zwitserlood_678_wav_21', 'UltraSuite_core-ux2020_19M'),
 ('Zwitserlood_678_wav_21', 'UltraSuite_core-ux2020_23M'),
 ('Zwitserlood_678_wav_22', 'UltraSuite_core-ux2020_29M'),
 ('Zwitserlood_678_wav_22', 'UltraSuite_core-ux2020_07M'),
 ('Zwitserlood_678_wav_23', 'UltraSuite_core-ux2020_18M'),
 ('Zwitserlood_678_wav_23', 'UltraSuite_core-upx_08M'),
 ('Zwitserlood_678_wav_24', 'UltraSuite_core-upx_17M'),
 ('Zwitserlood_678_wav_24', 'UltraSuite_core-uxssd_01M'),
 ('Zwitserlood_678_wav_25', 'UltraSuite_core-ux2020_02F'),
 ('Zwitserlood_678_wav_25', 'UltraSuite_core-ux2020_05M'),
 ('Zwitserlood_678_wav_26', 'UltraSuite_core-upx_17M'),
 ('Zwitserlood_678_wav_26', 'UltraSuite_core-ux2020_33M'),
 ('Zwitserlood_678_wav_27', 'UltraSuite_core-ux2020_15M'),
 ('Zwitserlood_678_wav_27', 'UltraSuite_core-ux2020_29M'),
 ('Zwitserlood_678_wav_28', 'UltraSuite_core-ux2020_07M'),
 ('Zwitserlood_678_wav_28', 'UltraSuite_core-ux2020_27M'),
 ('Zwitserlood_678_wav_29', 'UltraSuite_core-upx_04M'),
 ('Zwitserlood_678_wav_29', 'UltraSuite_core-upx_02F'),
 ('Zwitserlood_678_wav_30', 'UltraSuite_core-upx_15M'),
 ('Zwitserlood_678_wav_30', 'UltraSuite_core-upx_01F'),
 ('Zwitserlood_8910_wav_01', 'UltraSuite_core-upx_04M'),
 ('Zwitserlood_8910_wav_01', 'UltraSuite_core-ux2020_02F'),
 ('Zwitserlood_8910_wav_02', 'UltraSuite_core-upx_04M'),
 ('Zwitserlood_8910_wav_02', 'UltraSuite_core-upx_09M'),
 ('Zwitserlood_8910_wav_03', 'UltraSuite_core-upx_04M'),
 ('Zwitserlood_8910_wav_03', 'UltraSuite_core-ux2020_36M'),
 ('Zwitserlood_8910_wav_04', 'UltraSuite_core-upx_09M'),
 ('Zwitserlood_8910_wav_04', 'UltraSuite_core-upx_04M'),
 ('Zwitserlood_8910_wav_05', 'UltraSuite_core-ux2020_18M'),
 ('Zwitserlood_8910_wav_05', 'UltraSuite_core-upx_08M'),
 ('Zwitserlood_8910_wav_06', 'UltraSuite_core-ux2020_33M'),
 ('Zwitserlood_8910_wav_06', 'UltraSuite_core-ux2020_13F'),
 ('Zwitserlood_8910_wav_07', 'UltraSuite_core-ux2020_02F'),
 ('Zwitserlood_8910_wav_07', 'UltraSuite_core-ux2020_03M'),
 ('Zwitserlood_8910_wav_08', 'UltraSuite_core-ux2020_29M'),
 ('Zwitserlood_8910_wav_08', 'UltraSuite_core-ux2020_15M'),
 ('Zwitserlood_8910_wav_09', 'UltraSuite_core-ux2020_31M'),
 ('Zwitserlood_8910_wav_09', 'UltraSuite_core-ux2020_07M'),
 ('Zwitserlood_8910_wav_10', 'UltraSuite_core-upx_08M'),
 ('Zwitserlood_8910_wav_10', 'UltraSuite_core-ux2020_17M'),
 ('Zwitserlood_8910_wav_11', 'UltraSuite_core-ux2020_12M'),
 ('Zwitserlood_8910_wav_11', 'UltraSuite_core-ux2020_02F'),
 ('Zwitserlood_8910_wav_12', 'UltraSuite_core-ux2020_28F'),
 ('Zwitserlood_8910_wav_12', 'UltraSuite_core-ux2020_24F'),
 ('Zwitserlood_8910_wav_13', 'UltraSuite_core-ux2020_20M'),
 ('Zwitserlood_8910_wav_13', 'UltraSuite_core-ux2020_19M'),
 ('Zwitserlood_8910_wav_14', 'UltraSuite_core-ux2020_34F'),
 ('Zwitserlood_8910_wav_14', 'UltraSuite_core-ux2020_15M'),
 ('Zwitserlood_8910_wav_15', 'UltraSuite_core-ux2020_17M'),
 ('Zwitserlood_8910_wav_15', 'UltraSuite_core-upx_13M'),
 ('Zwitserlood_8910_wav_16', 'UltraSuite_core-ux2020_09M'),
 ('Zwitserlood_8910_wav_16', 'UltraSuite_core-ux2020_15M'),
 ('Zwitserlood_8910_wav_17', 'UltraSuite_core-ux2020_33M'),
 ('Zwitserlood_8910_wav_17', 'UltraSuite_core-ux2020_03M'),
 ('Zwitserlood_8910_wav_18', 'UltraSuite_core-ux2020_02F'),
 ('Zwitserlood_8910_wav_18', 'UltraSuite_core-ux2020_03M'),
 ('Zwitserlood_8910_wav_19', 'UltraSuite_core-ux2020_17M'),
 ('Zwitserlood_8910_wav_19', 'UltraSuite_core-upx_05M'),
 ('Zwitserlood_8910_wav_20', 'UltraSuite_core-upx_09M'),
 ('Zwitserlood_8910_wav_20', 'UltraSuite_core-upx_04M'),
 ('Zwitserlood_8910_wav_21', 'UltraSuite_core-ux2020_27M'),
 ('Zwitserlood_8910_wav_21', 'UltraSuite_core-ux2020_15M'),
 ('Zwitserlood_8910_wav_22', 'UltraSuite_core-upx_10M'),
 ('Zwitserlood_8910_wav_22', 'UltraSuite_core-ux2020_32M'),
 ('Zwitserlood_8910_wav_23', 'UltraSuite_core-ux2020_02F'),
 ('Zwitserlood_8910_wav_23', 'UltraSuite_core-ux2020_12M'),
 ('Zwitserlood_8910_wav_24', 'UltraSuite_core-ux2020_21M'),
 ('Zwitserlood_8910_wav_24', 'UltraSuite_core-upx_13M'),
 ('Zwitserlood_8910_wav_25', 'UltraSuite_core-ux2020_02F'),
 ('Zwitserlood_8910_wav_25', 'UltraSuite_core-ux2020_03M'),
 ('Zwitserlood_8910_wav_26', 'UltraSuite_core-ux2020_27M'),
 ('Zwitserlood_8910_wav_26', 'UltraSuite_core-ux2020_07M'),
 ('Zwitserlood_8910_wav_27', 'UltraSuite_core-ux2020_23M'),
 ('Zwitserlood_8910_wav_27', 'UltraSuite_core-ux2020_25M'),
 ('Zwitserlood_8910_wav_28', 'UltraSuite_core-upx_08M'),
 ('Zwitserlood_8910_wav_28', 'UltraSuite_core-ux2020_21M'),
 ('Zwitserlood_8910_wav_29', 'UltraSuite_core-ux2020_18M'),
 ('Zwitserlood_8910_wav_29', 'UltraSuite_core-ux2020_17M'),
 ('Zwitserlood_8910_wav_30', 'UltraSuite_core-ux2020_29M'),
 ('Zwitserlood_8910_wav_30', 'UltraSuite_core-ux2020_18M')
]


def select_longest_target_reference(target_corpus: BaseCorpus, target_speaker: str) -> Utterance:
    """Select the longest individual target utterance for MeanVC reference audio.

    Ties are resolved lexically by utterance id/path so selection stays deterministic.
    """
    if target_speaker not in target_corpus.speakers:
        raise KeyError(f"Unknown target speaker: {target_speaker}")

    utterances = list(target_corpus.utterances.get(target_speaker, []))
    if not utterances:
        raise ValueError(
            f"Target speaker {target_speaker!r} has no individual utterances to use as reference"
        )

    return sorted(
        utterances,
        key=lambda utterance: (-utterance.duration, utterance.id, str(utterance.filepath)),
    )[0]


class VoiceConverter:
    dit_model: DiT = None
    voco_model: ScriptModule = None
    asr_model: ScriptModule = None
    sv_model: ECAPA_TDNN = None
    mel_extractor: MelSpectrogramFeatures = None
    output_dir = OUTPUT_DIR
    chunk_size = 20
    steps = 2
    device = 'cpu'

    def __init__(self):
        self._load_vc()
        
    def _load_vc(self):
        tlog(
            __name__,
            "Loading voice converter (MeanVC) | device=%s",
            self.device,
        )
        meanvc_path = ROOT_DIR / "vendor" / "MeanVC"
        model_config = str(meanvc_path / "src" / "config" / "config_200ms.json")
        ckpt_path = str(meanvc_path / "src" / "ckpt" / "model_200ms.safetensors")
        voco_ckpt_path = str(meanvc_path / "src" / "ckpt" / "vocos.pt")
        asr_ckpt_path = str(meanvc_path / "src" / "ckpt" / "fastu2++.pt")
        sv_ckpt_path = str(meanvc_path / "src" / "runtime" / "speaker_verification" / "ckpt" / "wavlm_large_finetune.pth")
        
        with open(model_config) as f:
            model_config = json.load(f)

        model_cls = DiT
        dit_model = model_cls(**model_config["model"])
        total_params = sum(p.numel() for p in dit_model.parameters())
        # print(f"Total parameters: {total_params}")
        dit_model = dit_model.to(self.device)
        dit_model = load_checkpoint(dit_model, ckpt_path, device=self.device, use_ema=False)
        dit_model = dit_model.float()
        dit_model.eval()

        voco_model = torch.jit.load(voco_ckpt_path).to(self.device)

        asr_model = torch.jit.load(asr_ckpt_path).to(self.device)

        # First run can download ~1.2GB; after that, mmap/torch. 
        # Load of that file may take several minutes."
        sv_model = init_sv_model("wavlm_large", sv_ckpt_path)
        sv_model = sv_model.to(self.device)
        sv_model.eval()
        
        mel_extractor = MelSpectrogramFeatures(
            sample_rate=SAMPLE_RATE, n_fft=1024, win_size=640, hop_length=160, 
            n_mels=80, fmin=0, fmax=8000, center=True
        ).to(self.device)

        self.dit_model = dit_model
        self.voco_model = voco_model
        self.asr_model = asr_model
        self.sv_model = sv_model
        self.mel_extractor = mel_extractor

    def convert(self, source_corpus: BaseCorpus, source_speaker: str, target_corpus: BaseCorpus, target_speaker: str) -> list[ConvertedUtterance]:
        output_dir = OUTPUT_DIR / source_speaker / target_speaker
        os.makedirs(output_dir, exist_ok=True)

        wav_subdir = OUTPUT_DIR / source_speaker / f"{target_speaker}_wav"
        sources: list[str] = []
        skipped = 0
        for utterance in source_corpus.utterances[source_speaker]:
            converted_wav = wav_subdir / f"{source_speaker[-2:]}_{utterance.id}.wav"
            if converted_wav.is_file():
                tlog(__name__, "Skipping %s already converted", converted_wav.name)
                skipped += 1
                continue
            tlog(__name__, "Adding %s to batch", converted_wav.name)
            sources.append(str(utterance.filepath))

            # for fragment in utterance.fragments:
            #     converted_wav = wav_subdir / f"{fragment.id}.wav"
            #     if converted_wav.is_file():
            #         tlog(__name__, "Skip %s, already converted", converted_wav.name)
            #         skipped += 1
            #         continue
            #     sources.append(str(utterance.filepath))
            # if skipped == 0:
            #     tlog(__name__, "Added %s to batch", utterance.id)
            # elif skipped == len(utterance.fragments):
            #     tlog(__name__, "Skipped %s, all %r fragments converted", utterance.id, skipped)
            # else:
            #     tlog(__name__, "Added %s to batch, skipped %r fragments", utterance.id, skipped)

        target_reference = select_longest_target_reference(target_corpus, target_speaker)
        tlog(
            __name__,
            "Selected reference | target=%s utterance=%s duration=%.2fs path=%s",
            target_speaker,
            target_reference.id,
            target_reference.duration,
            target_reference.filepath,
        )

        if sources:
            inference_list(
                model=self.dit_model,
                vocos=self.voco_model,
                asr_model=self.asr_model,
                sv_model=self.sv_model,
                mel_extractor=self.mel_extractor,
                sources=sources,
                reference_path=str(target_reference.filepath),
                chunk_size=self.chunk_size,
                steps=self.steps,
                output_dir=str(output_dir),
                device=self.device,
            )
        
        converted_utterances = []
        for utterance in source_corpus.utterances[source_speaker]:
            filepath = wav_subdir / f"{source_speaker[-2:]}_{utterance.id}.wav"
            converted_utterances.append(ConvertedUtterance(
                id = utterance.id,
                filepath = filepath,
                duration = wav_duration(filepath),
                sample_rate = wav_sample_rate(filepath),
                source_speaker = source_speaker,
                target_speaker = target_speaker,
                speaker=None
            ))

        return converted_utterances

def convert_pairs(source_corpus: BaseCorpus, target_corpus: BaseCorpus, pairs: tuple[str, str]) -> dict[str, dict[str, list[ConvertedUtterance]]]:
    vc = VoiceConverter()

    conversions: dict[str, dict[str, list[ConvertedUtterance]]] = defaultdict(dict)
    pair_iter = tqdm(
        pairs,
        desc="convert_pairs",
        unit="pair",
        dynamic_ncols=True,
        mininterval=0.25,
        file=sys.stderr,
    )
    for source_speaker, target_speaker in pair_iter:
        tlog(__name__, "Converting %s -> %s", source_speaker, target_speaker)
        converted_utterances = vc.convert(
            source_corpus,
            source_speaker,
            target_corpus,
            target_speaker
        )
        conversions[source_speaker][target_speaker] = converted_utterances
    
    return conversions

def main():
    configure_logging()
    source = ZwitserloodCorpus(use_cache=True)
    target = UltraSuiteCorpus(use_cache=True)
    convert_pairs(source, target, pairs)

if __name__ == "__main__":
    main()
