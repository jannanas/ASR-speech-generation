# ASR speech generation

## Pairing pipeline

All of this is in **`src/pairing.py`**.

1. **Speaker vectors** — **`extract_mfcc_vectors`** loads each utterance with librosa, builds MFCCs, reduces each utterance to a fixed vector (per-coefficient mean and std over time, concatenated), then averages those vectors over the speaker’s utterances and stores the result on **`Speaker.mfcc_vector`**. With **`use_mfcc_cache=True`**, speakers that already have a vector are skipped; new vectors are merged back into the corpus checkpoint on disk via **`merge_mfcc_from_corpus`** in `src/utils.py`.

2. **Cross-corpus similarity** — **`calculate_similarity`** stacks source and target **`mfcc_vector`** rows and runs **cosine similarity** (scikit-learn), producing a dense map: each source speaker id → each target speaker id → score.

3. **Top‑k pairs** — **`k_fold_match`** sorts targets per source (by score). **`PairingStrategy.SIMILAR`** keeps the highest scores; **`DISSIMILAR`** the lowest. **`STRATIFIED`** is not implemented yet. Returns a flat list of **`(source_speaker_id, target_speaker_id)`**, up to **`k`** targets per source.

The public entry point is **`pair_speakers(source, target, limit=None, strategy=..., k=2, use_mfcc_cache=True)`**.

## Getting started

1. Put data under **`data/`** as expected by the corpus classes (see `connectors.py` and `config.DATA_DIR`).

2. Install dependencies (to be cleaned up):

   - **Conda:** `conda env create -f environment.yml` then `conda activate asr-speech-generation`

   - **Pip only:** `pip install -r requirements.txt`

3. Run pairing from **`src/`** so imports resolve:

   ```bash
   cd src
   python pairing.py
   ```
