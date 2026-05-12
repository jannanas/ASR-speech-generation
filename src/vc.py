"""
Voice conversion (MeanVC): source speaker audio -> target speaker timbre/prompt.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from connectors import *
from models import Speaker, Utterance

log = logging.getLogger(__name__)

MEANVC_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "MeanVC"


def convert_voice(
    source: Speaker,
    target: Speaker,
    output_wav: Path,
    *,
    source_wav: Path | None = None,
    reference_wav: Path | None = None,
    model_config: Path | None = None,
    ckpt_path: Path | None = None,
    vocoder_ckpt: Path | None = None,
    asr_ckpt: Path | None = None,
    sv_ckpt: Path | None = None,
    chunk_size: int = 20,
    steps: int = 2,
    seed: int = 42,
) -> Path:
    """
    Run MeanVC: **content** from ``source`` wav, **timbre / prompt** from ``target`` wav.

    By default uses ``source.utterances_concat_filepath`` and ``target.utterances_concat_filepath``
    (e.g. after ``concat_ultrasuite_speakers`` / Zwitserlood concat). Pass ``source_wav`` / ``reference_wav`` to override.

    Returns the path to ``output_wav`` (written on disk).
    """
    src_path = Path(source_wav) if source_wav is not None else source.utterances_concat.filepath
    ref_path = Path(reference_wav) if reference_wav is not None else target.utterances_concat.filepath

    if not src_path or not src_path.is_file():
        raise ValueError(
            f"Source wav missing for speaker {source.id!r}: set speaker.utterances_concat_filepath "
            "or pass source_wav="
        )
    if not ref_path or not ref_path.is_file():
        raise ValueError(
            f"Target reference wav missing for speaker {target.id!r}: set target.utterances_concat_filepath "
            "or pass reference_wav="
        )

    model_config = model_config or MEANVC_ROOT / "src" / "config" / "config_200ms.json"
    ckpt_path = ckpt_path or MEANVC_ROOT / "src" / "ckpt" / "model_200ms.safetensors"
    vocoder_ckpt = vocoder_ckpt or MEANVC_ROOT / "src" / "ckpt" / "vocos.pt"
    asr_ckpt = asr_ckpt or MEANVC_ROOT / "src" / "ckpt" / "fastu2++.pt"
    sv_ckpt = sv_ckpt or (
        MEANVC_ROOT
        / "src"
        / "runtime"
        / "speaker_verification"
        / "ckpt"
        / "wavlm_large_finetune.pth"
    )

    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="meanvc_vc_") as tmp:
        tmp = Path(tmp)
        mel_dir = tmp / "mels"
        mel_dir.mkdir()

        env = {**os.environ, "PYTHONPATH": str(MEANVC_ROOT)}
        cmd = [
            sys.executable,
            str(MEANVC_ROOT / "src" / "infer" / "infer_ref.py"),
            "--model-config",
            str(model_config),
            "--ckpt-path",
            str(ckpt_path),
            "--asr-ckpt-path",
            str(asr_ckpt),
            "--sv-ckpt-path",
            str(sv_ckpt),
            "--vocoder-ckpt-path",
            str(vocoder_ckpt),
            "--output-dir",
            str(mel_dir),
            "--source-path",
            str(src_path),
            "--reference-path",
            str(ref_path),
            "--chunk-size",
            str(chunk_size),
            "--steps",
            str(steps),
            "--seed",
            str(seed),
        ]
        log.info(
            "MeanVC | source=%s | target=%s | out=%s",
            src_path,
            ref_path,
            output_wav,
        )
        subprocess.run(cmd, cwd=str(MEANVC_ROOT), env=env, check=True)

        produced = mel_dir.parent / f"{mel_dir.name}_wav" / f"{src_path.stem}.wav"
        if not produced.is_file():
            raise FileNotFoundError(f"MeanVC did not write expected wav: {produced}")

        shutil.copyfile(produced, output_wav)

    log.info("Wrote %s", output_wav)
    return output_wav

def main():
    sourceCorpus = ZwitserloodCorpus()
    targetCorpus = UltraSuiteCorpus()

    convert_voice(
        sourceCorpus.speakers['Zwitserlood_678_wav_02'],
        targetCorpus.speakers['UltraSuite_core-ux2020_17M'],
        output_wav='./out/test.wav',
        source_wav='./data/Zwitserlood/678_wav/02_T1.wav'
    )


if __name__ == "__main__":
    main()