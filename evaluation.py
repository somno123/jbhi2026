"""
LOSO cross-validation, subject-level metrics and external (cross-dataset) evaluation

Unit of analysis: the subject. For every held-out subject the epoch-level P(AD) of all its 30 s
segments are averaged into one subject score; accuracy, sensitivity, specificity, precision, F1
and AUC are then computed across the 65 subject scores (one per LOSO fold). Epoch-level pooled
metrics are reported only as a secondary, clearly labelled number, since epochs of one subject are
not independent.
"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, f1_score, precision_score)

from config import (DOWNSTREAM_EPOCHS, LOSO_SUBJECTS, SUBJECT_THRESHOLD, N_BOOTSTRAP,
                    LABEL_AD, LABEL_CN)
from data import subject_label
from training import (split_inner_validation, train_finetune, predict_proba, build_frozen_encoder,
                      extract_features, train_linear_probe, predict_proba_features)

METRIC_NAMES = ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'auc']


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def binary_metrics(y_true, y_prob, threshold=SUBJECT_THRESHOLD):
    """Metrics with AD (label 1) as the positive class"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[LABEL_CN, LABEL_AD]).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def aggregate_subjects(epoch_df, threshold=SUBJECT_THRESHOLD):
    """One row per subject: mean epoch P(AD) -> subject score and prediction"""
    subject_df = epoch_df.groupby('subject_id').agg(
        label=('label', 'first'),
        n_epochs=('prob', 'size'),
        prob=('prob', 'mean'),
        epoch_accuracy=('correct', 'mean'),
    ).reset_index()
    subject_df['pred'] = (subject_df['prob'] >= threshold).astype(int)
    return subject_df


def bootstrap_ci(subject_df, n_boot=N_BOOTSTRAP, seed=0, alpha=0.05):
    """Percentile CI from resampling subjects (not epochs) with replacement"""
    rng = np.random.RandomState(seed)
    y, p = subject_df['label'].values, subject_df['prob'].values
    samples = {m: [] for m in METRIC_NAMES}
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        for m, v in binary_metrics(y[idx], p[idx]).items():
            if m in samples:
                samples[m].append(v)
    return {m: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
            for m, v in samples.items() if v}


def summarize_predictions(epoch_df):
    """Subject-level (primary) and epoch-level pooled (secondary) metrics for one run"""
    epoch_df = epoch_df.copy()
    epoch_df['correct'] = ((epoch_df['prob'] >= SUBJECT_THRESHOLD).astype(int) == epoch_df['label']).astype(float)
    subject_df = aggregate_subjects(epoch_df)
    subject_metrics = binary_metrics(subject_df['label'], subject_df['prob'])
    subject_metrics['n_subjects'] = len(subject_df)
    epoch_metrics = binary_metrics(epoch_df['label'], epoch_df['prob'])
    epoch_metrics['n_epochs'] = len(epoch_df)
    return {
        'subject_level': subject_metrics,
        'subject_level_ci95': bootstrap_ci(subject_df),
        'epoch_level_pooled': epoch_metrics,
    }, subject_df


# ----------------------------------------------------------------------------------------------
# Training / prediction dispatch for the two downstream protocols
# ----------------------------------------------------------------------------------------------
class DownstreamContext:
    """Holds the pre-trained encoder and, for linear evaluation, cached frozen features"""
    def __init__(self, mode, pretrained_encoder_state, encoder_config, device):
        self.mode = mode
        self.state = pretrained_encoder_state
        self.config = encoder_config
        self.device = device
        self._cache = {}
        self.encoder = (build_frozen_encoder(pretrained_encoder_state, encoder_config, device)
                        if mode == 'linear' else None)

    def features(self, dataset, indices):
        key = id(dataset)
        if key not in self._cache:
            self._cache[key] = {}
        cache = self._cache[key]
        missing = [i for i in indices if i not in cache]
        if missing:
            feats = extract_features(self.encoder, dataset, missing, self.device)
            cache.update(zip(missing, feats))
        return torch.stack([cache[i] for i in indices])

    def fit(self, dataset, train_indices, val_indices, seed):
        if self.mode == 'linear':
            y = lambda idx: torch.tensor([dataset.labels[i] for i in idx], dtype=torch.long)
            return train_linear_probe(self.features(dataset, train_indices), y(train_indices),
                                      self.features(dataset, val_indices), y(val_indices),
                                      self.device, seed, num_epochs=DOWNSTREAM_EPOCHS)
        return train_finetune(self.state, self.config, dataset, train_indices, val_indices,
                              self.device, seed, num_epochs=DOWNSTREAM_EPOCHS)

    def predict(self, model, dataset, indices):
        if self.mode == 'linear':
            return predict_proba_features(model, self.features(dataset, indices), self.device)
        return predict_proba(model, dataset, indices, self.device)


def _indices_for(dataset, subjects):
    subjects = set(subjects)
    return [i for i, s in enumerate(dataset.subject_ids) if s in subjects]


# ----------------------------------------------------------------------------------------------
# LOSO with nested (inner) validation
# ----------------------------------------------------------------------------------------------
def loso_evaluation(dataset, pretrained_encoder_state, encoder_config, device, mode, seed):
    """
    For each of the 65 AD/CN subjects:
      1. test subject = held out, never used for training, early stopping or model selection
      2. remaining 64 subjects -> label-stratified subject-wise split into inner-train / inner-val
      3. train on inner-train, early stop on inner-val loss
      4. predict every 30 s epoch of the test subject once
    """
    target_subjects = sorted(s for s in set(dataset.subject_ids) if s in LOSO_SUBJECTS)
    labels_by_subject = {s: subject_label(s) for s in target_subjects}
    ctx = DownstreamContext(mode, pretrained_encoder_state, encoder_config, device)

    print(f"\n{'='*60}")
    print(f"LOSO [{mode}] seed={seed}: {len(target_subjects)} subjects")
    print(f"{'='*60}")

    epoch_rows, fold_rows = [], []
    for fold_idx, test_subject in enumerate(tqdm(target_subjects, desc=f"LOSO {mode} seed {seed}")):
        fold_seed = seed * 1000 + fold_idx
        train_subjects = [s for s in target_subjects if s != test_subject]
        inner_train, inner_val = split_inner_validation(train_subjects, labels_by_subject, fold_seed)
        assert test_subject not in inner_train and test_subject not in inner_val

        test_indices = _indices_for(dataset, [test_subject])
        model, info = ctx.fit(dataset, _indices_for(dataset, inner_train), _indices_for(dataset, inner_val), fold_seed)
        probs = ctx.predict(model, dataset, test_indices)

        label = labels_by_subject[test_subject]
        epoch_rows.extend({'fold': fold_idx, 'subject_id': test_subject, 'label': label, 'prob': float(p)}
                          for p in probs)
        fold_rows.append({'fold': fold_idx, 'test_subject': test_subject, 'label': label,
                          'n_test_epochs': len(test_indices), 'subject_prob': float(np.mean(probs)),
                          'inner_val_subjects': ' '.join(map(str, inner_val)), **info})
        print(f"  Fold {fold_idx+1:2d} | subject {test_subject:2d} (label {label}) | "
              f"P(AD)={np.mean(probs):.3f} | best epoch {info['best_epoch']}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(epoch_rows), pd.DataFrame(fold_rows)


# ----------------------------------------------------------------------------------------------
# External cohort: train on all local AD/CN subjects, test on an unseen dataset, no adaptation
# ----------------------------------------------------------------------------------------------
def external_evaluation(dataset, external_dataset, pretrained_encoder_state, encoder_config,
                        device, mode, seed):
    """No domain adaptation: no target-side fine-tuning, re-normalization or threshold tuning"""
    train_subjects = sorted(s for s in set(dataset.subject_ids) if s in LOSO_SUBJECTS)
    labels_by_subject = {s: subject_label(s) for s in train_subjects}
    inner_train, inner_val = split_inner_validation(train_subjects, labels_by_subject, seed)

    ctx = DownstreamContext(mode, pretrained_encoder_state, encoder_config, device)
    model, info = ctx.fit(dataset, _indices_for(dataset, inner_train), _indices_for(dataset, inner_val), seed)

    ext_indices = list(range(len(external_dataset)))
    probs = ctx.predict(model, external_dataset, ext_indices)
    epoch_df = pd.DataFrame({'subject_id': [external_dataset.subject_ids[i] for i in ext_indices],
                             'label': [external_dataset.labels[i] for i in ext_indices],
                             'prob': probs})
    return epoch_df, info


# ----------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------
def print_run_summary(title, summary):
    s, ci, e = summary['subject_level'], summary['subject_level_ci95'], summary['epoch_level_pooled']
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print(f"SUBJECT-LEVEL (primary; n={s['n_subjects']} subjects, mean epoch P(AD) >= {SUBJECT_THRESHOLD})")
    for m in METRIC_NAMES:
        lo, hi = ci.get(m, (np.nan, np.nan))
        print(f"  {m:18s}: {s[m]:.4f}  (95% bootstrap CI over subjects {lo:.4f}-{hi:.4f})")
    print(f"  confusion (AD positive): TP={s['tp']} FN={s['fn']} TN={s['tn']} FP={s['fp']}")
    print(f"EPOCH-LEVEL POOLED (secondary; n={e['n_epochs']} correlated 30 s epochs)")
    for m in METRIC_NAMES:
        print(f"  {m:18s}: {e[m]:.4f}")


def summarize_across_seeds(summaries):
    """Mean ± sample std (ddof=1) of the subject-level metrics across independent seeds"""
    rows = [s['subject_level'] for s in summaries]
    result = {'n_seeds': len(rows)}
    for m in METRIC_NAMES:
        vals = np.array([r[m] for r in rows], dtype=float)
        result[m] = {'mean': float(np.nanmean(vals)),
                     'std': float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                     'per_seed': vals.tolist()}
    return result


def print_seed_summary(title, agg):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print(f"Subject-level metrics, mean ± std (ddof=1) across {agg['n_seeds']} independent seeds")
    print("(each seed repeats pre-training and all downstream training)")
    for m in METRIC_NAMES:
        print(f"  {m:18s}: {agg[m]['mean']:.4f} ± {agg[m]['std']:.4f}")
