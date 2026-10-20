# Delta2Gamma: Self-Supervised Multi-Band EEG Representation Learning for Dementia Classification

PyTorch implementation of a multi-head SimCLR framework that learns separate representations for the
delta, theta, alpha, beta and gamma bands with adaptive temperature regularization, evaluated on
Alzheimer's disease (AD) vs. cognitively normal (CN) classification from resting-state EEG.

## Project Structure

```
dementia/
├── config.py          # Settings and hyperparameters (all fixed a priori)
├── data.py            # Local / external EEG datasets, SimCLR view dataset
├── augmentation.py    # Augmentations and adaptive multi-head NT-Xent loss
├── models.py          # Band extractor, encoder, projection heads, classifiers
├── training.py        # Pre-training, nested validation, linear evaluation, fine-tuning
├── evaluation.py      # LOSO, subject-level metrics, bootstrap CIs, external validation
├── main.py            # Full pipeline (CLI)
└── requirements.txt
```

## Data

The local cohort is OpenNeuro **ds004504** (19-channel, 500 Hz, eyes-closed resting state), exported
as one CSV per subject (`sub-XXX_*.csv`, one column per channel plus an optional `time` column).

| Subjects | Group | Use |
|---|---|---|
| 1–36  | AD  | LOSO evaluation (label 1, **positive class**) |
| 37–65 | CN  | LOSO evaluation (label 0) |
| 66–88 | FTD | Self-supervised pre-training only; labels never used |

Recordings are split into non-overlapping 30 s segments (15,000 samples).

## Usage

```bash
pip install -r requirements.txt

# Default: pre-train on subjects 66-88, LOSO on 1-65, both protocols, 5 seeds
python main.py --root-dir /path/to/csv
```

## Requirements

See `requirements.txt` (torch, scipy, scikit-learn, pandas, numpy, pytorch-metric-learning, tqdm).
