# ASR speech generation

## Pairing pipeline

All of this is in **`src/pairing.py`**.

1. **Speaker vectors** — **`extract_mfcc_vectors`** loads each utterance with librosa, builds MFCCs, reduces each utterance to a fixed vector (per-coefficient mean and std over time, concatenated), then averages those vectors over the speaker’s utterances and stores the result on **`Speaker.mfcc_vector`**. For each corpus, MFCC disk merge and “skip if vector already present” follow that corpus’s constructor flag **`use_cache`** (same as scan cache): see **`merge_mfcc_from_corpus`** in `src/utils.py`.

2. **Cross-corpus similarity** — **`calculate_similarity`** stacks source and target **`mfcc_vector`** rows and runs **cosine similarity** (scikit-learn), producing a dense map: each source speaker id → each target speaker id → score.

3. **Top‑k pairs** — **`k_fold_match`** sorts targets per source (by score). **`PairingStrategy.SIMILAR`** keeps the highest scores; **`DISSIMILAR`** the lowest. **`STRATIFIED`** is not implemented yet. Returns a flat list of **`(source_speaker_id, target_speaker_id)`**, up to **`k`** targets per source.

The public entry point is **`pair_speakers(source, target, limit=None, strategy=..., k=2)`**; MFCC caching uses **`source.use_cache`** and **`target.use_cache`** from how each corpus was constructed.

## Getting started

1. Put data under **`data/`** as expected by the corpus classes (see `connectors.py` and `config.DATA_DIR`).

2. Install dependencies:

   - **Core / pairing (env name `DataAugmentation`):** from the repo root, `conda env create -f ./envs/rp_core_env.yml` (first time only). To refresh packages later: `conda env update -n DataAugmentation -f ./envs/rp_core_env.yml`. Then `conda activate DataAugmentation`.

   - **MeanVC stack (env name `MeanVC`):** `conda env create -f ./envs/rp_meanvc_env.yml`, then `conda activate MeanVC`.

3. Run pairing from **`src/`** so imports resolve:

   ```bash
   cd src
   python pairing.py
   ```
   
(All *_concat.wav files are limited to 3 mins)

4. Credit

https://huggingface.co/datasets/kgrosero14/ultrasuite-benchmark
https://ultrasuite.github.io/

https://talkbank.org/childes/access/Clinical-Other/Zwitserlood.html











## Pairs Top 2
```
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
```