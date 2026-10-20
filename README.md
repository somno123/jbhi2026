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

## Usage

```bash
pip install -r requirements.txt

python main.py --root-dir /path/to/csv
```

## Requirements

See `requirements.txt` (torch, scipy, scikit-learn, pandas, numpy, pytorch-metric-learning, tqdm).
