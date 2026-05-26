"""
LDNet non-intrusive MOS (1–5) for utterances / corpora.

Uses the bundled VoiceMOS pretrained LDNet (BVCC-style STFT). Typical use::

    from connectors import ZwitserloodCorpus
    from mos import LDNetMOSPredictor, assign_ldnet_mos_to_corpus

    corpus = ZwitserloodCorpus(use_cache=True)
    predictor = LDNetMOSPredictor()
    assign_ldnet_mos_to_corpus(corpus, predictor)
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import librosa
import numpy as np
import scipy
import torch
import yaml
from tqdm import tqdm

from connectors import BaseCorpus
from models import Utterance
from utils import tlog
from vendor.LDNet.models.LDNet import LDNet

_ROOT = Path(__file__).resolve().parents[1]
_LDNET_ROOT = _ROOT / "vendor" / "LDNet"


def _hamming_window(n: int) -> np.ndarray:
    if hasattr(scipy.signal, "hamming"):
        return scipy.signal.hamming(n)
    return scipy.signal.windows.hamming(n)


def default_ldnet_pretrained_paths() -> tuple[Path, Path]:
    base = _LDNET_ROOT / "exp" / "Pretrained-LDNet-ML-2337"
    return base / "config.yml", base / "model-27000.pt"


def wav_to_mag_sgram(path: Path) -> np.ndarray:
    """Same STFT as BVCC path in vendor/LDNet/dataset.py (BCVCCDataset)."""
    wav, _ = librosa.load(str(path), sr=16000)
    mag = np.abs(
        librosa.stft(
            wav,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=_hamming_window,
        )
    ).astype(np.float32)
    return mag.T  # (time, 257)


MAX_FRAMES = 1250


class LDNetMOSPredictor:
    """Loads LDNet once; predicts one MOS score per waveform (mean-listener head)."""

    def __init__(
        self,
        config_path: Path | None = None,
        ckpt_path: Path | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        cfg_path, weights_path = default_ldnet_pretrained_paths()
        self.config_path = Path(config_path or cfg_path)
        self.ckpt_path = Path(ckpt_path or weights_path)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not self.config_path.is_file():
            raise FileNotFoundError(f"LDNet config not found: {self.config_path}")
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"LDNet checkpoint not found: {self.ckpt_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        if self._config.get("model") != "LDNet":
            raise ValueError(f"Expected LDNet in config, got {self._config.get('model')!r}")

        self._model = LDNet(self._config).to(self.device)
        state = torch.load(str(self.ckpt_path), map_location=self.device)
        self._model.load_state_dict(state, strict=False)
        self._model.eval()
        tlog(
            __name__,
            "LDNet MOS predictor ready | device=%s | ckpt=%s",
            self.device,
            self.ckpt_path,
        )

    def predict_path(self, path: Path) -> float:
        mag = wav_to_mag_sgram(Path(path))
        t = min(mag.shape[0], MAX_FRAMES)
        x = torch.from_numpy(mag[:t]).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self._model.mean_listener_inference(x)
        return float(pred.squeeze().cpu().numpy())

    def predict_utterance(self, utterance: Utterance) -> float:
        return self.predict_path(utterance.filepath)


def assign_ldnet_mos_to_corpus(
    corpus: BaseCorpus,
    predictor: LDNetMOSPredictor | None = None,
    *,
    show_progress: bool = True,
) -> None:
    """
    Sets ``utterance.mos`` for every utterance in ``corpus.utterances``.
    Pass a shared :class:`LDNetMOSPredictor` to avoid reloading weights.
    """
    pred = predictor or LDNetMOSPredictor()
    flat: list[Utterance] = []
    for utts in corpus.utterances.values():
        flat.extend(utts)
    it: list[Utterance] | tqdm = flat
    if show_progress:
        it = tqdm(flat, desc="LDNet MOS", unit="utt")
    for utt in it:
        utt.mos = pred.predict_utterance(utt)


def assign_ldnet_mos_to_utterances(
    utterances: Iterable[Utterance],
    predictor: LDNetMOSPredictor | None = None,
    *,
    show_progress: bool = True,
) -> None:
    """Sets ``mos`` on each utterance (any iterable, e.g. one speaker's list)."""
    pred = predictor or LDNetMOSPredictor()
    flat = list(utterances)
    it: list[Utterance] | tqdm = flat
    if show_progress:
        it = tqdm(flat, desc="LDNet MOS", unit="utt")
    for utt in it:
        utt.mos = pred.predict_utterance(utt)
