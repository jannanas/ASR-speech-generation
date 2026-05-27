## General
This is the final dataset used by Jannes Kelso in the 2026 Q4 Research Project "Improving speech recognition for children with developmental language disorders using synthetic data".

## Data
The data is split into the `train`, `test`, and `validation` sets at 70/20/10 ratios.
```
train: 42 speakers
    sex: {'female': 10, 'male': 32}
    age: min=7.11, mean=8.38, max=9.61

test: 12 speakers
    sex: {'female': 3, 'male': 9}
    age: min=7.22, mean=8.36, max=9.50

verification: 6 speakers
    sex: {'female': 2, 'male': 4}
    age: min=7.28, mean=8.41, max=9.47
```

Each audio file is at most 30 seconds long.

## Directory and file structure

```
output/
├── train/
│   ├── original/                      # real Zwitserlood train speech
│   │   ├── transcript.txt
│   │   └── {source_speaker_id}-{source-utterance-id}.wav
│   │
│   ├── top_2/                         # MeanVC conversions to top-2 most similar targets
│   │   ├── top_2_pairs.csv
│   │   ├── transcript.txt
│   │   └── {source_speaker_id}-{target-speaker-id}-{source-utterance-id}.wav
│   │
│   ├── bottom_2/                      # MeanVC conversions to bottom-2 least similar targets
│   │   ├── bottom_2_pairs.csv
│   │   ├── transcript.txt
│   │   └── {source_speaker_id}-{target-speaker-id}-{source-utterance-id}.wav
│   │
│   └── random_2/                      # MeanVC conversions to 2 random targets
│       ├── random_2_pairs.csv
│       ├── transcript.txt
│       └── {source_speaker_id}-{target-speaker-id}-{source-utterance-id}.wav
│
├── test/                              # held-out real Zwitserlood speech
│   ├── transcript.txt
│   └── {source_speaker_id}-{source-utterance-id}.wav
│
└── validation/                        # validation real Zwitserlood speech
    ├── transcript.txt
    └── {source_speaker_id}-{source-utterance-id}.wav
```

The transcripts follow use the file names, exluding extensions, to identify files. They follow this format:
```
filename the transcription follows directly after the filename and a blank space
```

## Experiments

I would like to see what the WER is inferring speech from our test set for each of the following models. In each case we start with a pretrained ASR model. We use the validation set during finetuning as needed.

| Model ID  | Name                      | Description                                   |
|-----------|---------------------------|-----------------------------------------------|
| Model0    | Out of the box            | We do not finetune the ASR model at all.      |
| Model1    | Original                  | We finetune on only the original data.        |
| Model2    | Top 2                     | We finetune on only the top_2 data.           |
| Model3    | Original + top 2          | We finetune on the original and top_2 data    |
| Model4    | Bottom 2                  | We finetune on only the bottom_2 data.        |
| Model5    | Original + bottom 2       | We finetune on the original and bottom_2 data |
| Model6    | Random 2                  | We finetune on only the random_2 data.        |
| Model7    | Original + random 2       | We finetune on the original and random_2 data |
