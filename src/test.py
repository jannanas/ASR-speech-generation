# SpeechBrain 1.x + Flair + Transformers + Torch 2.11 can circular-import when SB's
# lazy loaders pull in `flair` while HuggingFace submodules are still loading.
# Eagerly initializing torch, transformers, and flair before any SpeechBrain import
# avoids that re-entrancy for pretrained x-vector loading.
import torch  # noqa: F401
import transformers  # noqa: F401
import flair  # noqa: F401
import soundfile as sf
from speechbrain.inference.classifiers import EncoderClassifier
from pprint import pprint


def load_wav_torch(path: str) -> tuple[torch.Tensor, int]:
    """Load WAV as [channels, time] float32 — same layout as torchaudio.load (mono → [1, T])."""
    data, fs = sf.read(path, dtype="float32", always_2d=True)
    signal = torch.from_numpy(data.T)
    return signal, int(fs)


classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")
signal, fs = load_wav_torch("C:/Users/Jannes/Repos/ASR-speech-generation/data/001E.wav")

embeddings = classifier.encode_batch(signal)[0][0].numpy()