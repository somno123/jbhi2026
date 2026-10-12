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
├── profile_cost.py    # Computational cost: multi-head vs. single-head
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

# Choose protocols / seeds
python main.py --modes linear finetune --seeds 2025 2026 2027 2028 2029

# Pre-train on a large external unlabeled corpus (e.g. TUH EEG), optionally a fraction of it
python main.py --pretrain-dir /path/to/tuh_npy --pretrain-fs 250 --pretrain-fraction 0.25

# Independent external validation cohort (cross-dataset transfer, no domain adaptation)
python main.py --external-dir /path/to/cohort --external-labels cohort_labels.csv --external-fs 500

# Computational cost comparison (random input, no data needed)
python profile_cost.py --batch-size 64
```

All outputs are written to `results/` (`--output-dir`). Pre-trained encoders are cached per
seed as `pretrained_<source>_frac<f>_seed<s>.pt`. Delete them after you change pre-training settings.

## Evaluation Protocol

### Downstream protocols (`--modes`)

| Protocol | Encoder | Classifier | Learning rate |
|---|---|---|---|
| `linear`   | **Frozen**: weights *and* BatchNorm running statistics | single `nn.Linear` on standardized 512-d features | `LINEAR_LR` = 1e-3 |
| `finetune` | All parameters updated | 3-layer MLP | `FINETUNE_LR` = 1e-4 |

"Linear evaluation" here always means a frozen encoder. Results that update the encoder are always
labelled "fine-tuning". Because the frozen encoder is deterministic, linear-evaluation features are
extracted once per pre-trained model and reused across folds.

### Leave-one-subject-out with nested validation

For each of the 65 AD/CN subjects:

1. The test subject is held out. It is **never** used for training, early stopping or model selection.
2. The remaining 64 subjects are split **subject-wise and label-stratified** into inner-train (~85%)
   and inner-validation (~15%, `INNER_VAL_FRACTION`).
3. The model is trained on inner-train. Early stopping (`PATIENCE`) keeps the epoch with the lowest
   **inner-validation** cross-entropy.
4. That model predicts every 30 s epoch of the test subject once.

The inner-validation subjects and selected epoch for every fold are saved in `loso_folds_*.csv`.

### Subject-level metrics (primary)

Epochs from one person are correlated and share a single diagnosis, so the subject is the unit of
analysis:

- **Subject score** = mean of the epoch-level P(AD) over all epochs of the held-out subject.
- **Subject prediction** = AD if the score is at least 0.5. The threshold is fixed a priori, not tuned.
- Accuracy, balanced accuracy, sensitivity (AD recall), specificity, precision, F1 and **AUC** are
  computed across the 65 subject scores, one per LOSO fold.
- **AUC** uses the 65 subject-level scores. It never pools epochs and is never computed within a
  single subject.
- **95% CIs**: percentile bootstrap that resamples subjects (`N_BOOTSTRAP` = 2000).

Epoch-level pooled metrics are also reported, but they are labelled as secondary because the epochs
are not independent.

### Repetitions and mean ± std

Each seed in `SEEDS` (default 5: 2025–2029) repeats the **entire** pipeline, both pre-training and
all 65 LOSO folds. The final `mean ± std` is the sample standard deviation (ddof = 1) of the
subject-level metrics **across seeds**. The number of seeds is printed and stored in
`final_summary.json`.

### Leakage audit: what is fitted where

| Operation | Scope |
|---|---|
| Band-pass filters (Butterworth, fixed cut-offs) | Not data-dependent |
| Artifact removal (ICA/ASR) | Per recording, done before export. Never fitted across subjects |
| Segmentation / resampling | Per recording, deterministic |
| Amplitude normalization | None across subjects |
| SimCLR pre-training | Subjects 66–88 (or an external corpus) only. Disjoint from all LOSO subjects; labels unused |
| Linear-probe feature standardization | Fitted on inner-train features of each fold |
| Early stopping / model selection | Inner-validation subjects of each fold |
| Augmentation parameters, hyperparameters, decision threshold | Fixed in `config.py`; not tuned on LOSO test subjects |

## Model

- **Band extractor**: 6th-order Butterworth band-pass (SciPy SOS) for delta (0.5–4 Hz), theta (4–8),
  alpha (8–13), beta (13–30) and gamma (30–45), followed by a depthwise conv per band.
- **Encoder**: one 3-layer 1-D CNN per band. The pooled band vectors are fused into a 512-d representation.
- **Multi-head projection**: an independent projection head per band (128-d, L2-normalized).
- **Adaptive temperature**: per band, τ = τ_min + σ(g([φ(z_i); φ(z_j)])) · (τ_max − τ_min), with τ ∈ [0.05, 0.5].

### Loss (Algorithm 1)

```
L = Σ_b NTXent_b(z_i^b, z_j^b; τ̄_b) + λ Σ_b Ω(τ_b),      λ = 0.01

Ω(τ_b) = mean_n [ (d/2) · log τ_{b,n} + 1/τ_{b,n} ],     d = 128
```

- `τ̄_b` is the batch mean of the adaptive temperature. It is used as a constant inside NT-Xent, so the
  temperature networks receive gradients only through Ω.
- `1/τ` penalizes overly small temperatures and `(d/2) log τ` penalizes overly large ones. The
  unconstrained minimizer is τ* = 2/d.

See `OptimizedAdaptiveMultiHead_NTXentLoss` in `augmentation.py`.

## External Data

### Unlabeled pre-training corpus (`--pretrain-dir`)

The directory is searched recursively for `*.npy` (channels × time, float32) or `*.csv` files.
Recordings must contain the 19 channels of `CHANNEL_NAMES`. Named CSV columns are reordered
automatically, and T7/T8/P7/P8 are mapped to T3/T4/T5/T6. Files that already match the 500 Hz target
rate are memory-mapped, so large corpora such as TUH do not need to fit in RAM. Files at other rates
are resampled on load. Use `--pretrain-fraction` to run data-scaling experiments.

### Labeled external cohort (`--external-dir`, `--external-labels`)

The labels file is a CSV with columns `file` (relative path), `label` (`AD`/`CN` or `1`/`0`) and an
optional `subject_id`. The downstream model is trained on all 65 local subjects, using the same
nested inner validation, and then applied to the external cohort **without domain adaptation**. No
target-side fine-tuning, normalization or threshold tuning is done. Subject-level metrics are reported
the same way as for LOSO.

## Computational Cost

`profile_cost.py` compares the multi-head model with a single-head baseline that uses the same encoder,
one projection head on the fused 512-d feature and a fixed temperature. It reports parameter counts,
pre-training step time, inference latency per 30 s epoch and peak GPU memory. The extra heads and
temperature networks are used only during pre-training. At inference time both models run the same
encoder, so downstream latency is the same.

## Outputs

| File | Content |
|---|---|
| `loso_epochs_<mode>_seed<s>.csv`   | Epoch-level P(AD) for every held-out epoch |
| `loso_subjects_<mode>_seed<s>.csv` | Subject score, prediction and n_epochs per subject |
| `loso_folds_<mode>_seed<s>.csv`    | Inner-validation subjects, selected epoch and inner-val loss per fold |
| `loso_summary_<mode>_seed<s>.json` | Subject-level metrics + 95% CI, epoch-level pooled metrics |
| `external_*`                       | Same for the external cohort |
| `final_summary.json`               | Mean ± std across seeds and the run configuration |

## Requirements

See `requirements.txt` (torch, scipy, scikit-learn, pandas, numpy, pytorch-metric-learning, tqdm).
