import wave
from pathlib import Path

def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()
    return n_frames / sample_rate

def wav_sample_rate(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
    return sample_rate