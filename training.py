"""
Pre-training and downstream training (linear evaluation / fine-tuning) for the SimCLR-based model

Model selection (early stopping) is always done on an inner validation split of the *training*
subjects. The LOSO test subject is only passed to `predict_proba` after training has finished.
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
import warnings

from config import (
    PRETRAIN_EPOCHS, DOWNSTREAM_EPOCHS, BATCH_SIZE, DOWNSTREAM_BATCH_SIZE, LEARNING_RATE,
    FINETUNE_LR, LINEAR_LR, WEIGHT_DECAY, EPS, SCHEDULER_T0, SCHEDULER_T_MULT, ETA_MIN,
    PATIENCE, NUM_WORKERS, PIN_MEMORY, DROP_LAST, PREFETCH_FACTOR, SAMPLING_RATE,
    PRETRAIN_SUBJECTS, INNER_VAL_FRACTION, BASE_TEMPERATURE, REGULARIZATION_WEIGHT
)
from data import OptimizedSimCLR_EEG_Dataset
from augmentation import SimCLR_EEG_Augmentations, OptimizedAdaptiveMultiHead_NTXentLoss
from models import (create_scipy_multihead_simclr_model, ScipyFineTuningClassifier,
                    ScipyMultiFrequency_SimCLR_Encoder, build_linear_head)

warnings.filterwarnings('ignore')


def seed_everything(seed):
    """결과 재현을 위한 시드 고정 함수"""
    import os
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, indices, batch_size, shuffle, seed, drop_last=False, num_workers=NUM_WORKERS):
    """Seeded DataLoader over a subset of indices"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    subset = dataset if indices is None else Subset(dataset, indices)
    return DataLoader(
        subset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=num_workers > 0,
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn, generator=generator
    )


# ----------------------------------------------------------------------------------------------
# Self-supervised pre-training
# ----------------------------------------------------------------------------------------------
def pretrain_simclr(dataset, pretrain_indices, device, seed, num_epochs=PRETRAIN_EPOCHS,
                    batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, desc=""):
    """Multi-head SimCLR pre-training on the given (unlabeled) segment indices"""
    print(f"\n{'='*60}")
    print(f"PRE-TRAINING {desc}")
    print(f"{'='*60}")
    print(f"Total pretraining samples: {len(pretrain_indices)}")

    if len(pretrain_indices) == 0:
        print("ERROR: No pretraining data found!")
        return None

    seed_everything(seed)
    augmentations = SimCLR_EEG_Augmentations(sampling_rate=SAMPLING_RATE)
    simclr_dataset = OptimizedSimCLR_EEG_Dataset(dataset, pretrain_indices, augmentations)
    simclr_loader = make_loader(simclr_dataset, None, batch_size, shuffle=True, seed=seed,
                                drop_last=DROP_LAST)

    simclr_model = create_scipy_multihead_simclr_model(
        n_channels=np.asarray(dataset.get_raw(pretrain_indices[0])).shape[0],
        sampling_rate=SAMPLING_RATE
    ).to(device)

    optimizer = torch.optim.AdamW(simclr_model.parameters(), lr=learning_rate,
                                  weight_decay=WEIGHT_DECAY, eps=EPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_T_MULT, eta_min=ETA_MIN)
    criterion = OptimizedAdaptiveMultiHead_NTXentLoss(
        base_temperature=BASE_TEMPERATURE,
        regularization_weight=REGULARIZATION_WEIGHT
    ).to(device)

    # Early stopping on the self-supervised training loss only (no labels, no LOSO subjects)
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    simclr_model.train()
    for epoch in range(num_epochs):
        total_loss = total_contrastive_loss = total_reg_loss = 0
        num_batches = 0

        pbar = tqdm(simclr_loader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch_idx, (view1, view2) in enumerate(pbar):
            view1, view2 = view1.to(device, non_blocking=True), view2.to(device, non_blocking=True)

            optimizer.zero_grad()
            try:
                _, projections_1 = simclr_model(view1)
                _, projections_2, adaptive_temps = simclr_model(view2, compute_temperature=True, other_view=view1)

                loss, contrastive_loss, reg_loss = criterion(projections_1, projections_2, adaptive_temps)

                if torch.isfinite(loss) and loss.item() > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(simclr_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    total_contrastive_loss += contrastive_loss.item()
                    total_reg_loss += reg_loss.item()
                    num_batches += 1
                    pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                      'cont': f'{contrastive_loss.item():.4f}',
                                      'reg': f'{reg_loss.item():.4f}'})
            except RuntimeError as e:
                print(f"Error during training batch {batch_idx}: {e}")
                continue

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, "
                      f"Cont={total_contrastive_loss / num_batches:.4f}, Reg={total_reg_loss / num_batches:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = {k: v.cpu() for k, v in simclr_model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        simclr_model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    print(f"Pre-training completed with best loss: {best_loss:.4f}")
    return simclr_model.encoder.eval()


def scipy_pretrain_on_subjects_66_88(dataset, device, seed, num_epochs=PRETRAIN_EPOCHS,
                                      batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    """66~88번 subject(FTD, 레이블 미사용)로만 사전학습"""
    pretrain_indices = [i for i, subj in enumerate(dataset.subject_ids) if subj in PRETRAIN_SUBJECTS]
    return pretrain_simclr(dataset, pretrain_indices, device, seed, num_epochs, batch_size,
                           learning_rate, desc="ON SUBJECTS 66-88")


# ----------------------------------------------------------------------------------------------
# Nested validation split
# ----------------------------------------------------------------------------------------------
def split_inner_validation(subjects, subject_labels, seed, fraction=INNER_VAL_FRACTION):
    """Label-stratified, subject-wise split of the training subjects into inner train / inner val"""
    rng = np.random.RandomState(seed)
    inner_train, inner_val = [], []
    for label in sorted(set(subject_labels[s] for s in subjects)):
        group = sorted(s for s in subjects if subject_labels[s] == label)
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * fraction)))
        inner_val.extend(group[:n_val])
        inner_train.extend(group[n_val:])
    return sorted(inner_train), sorted(inner_val)


def _evaluate_loss(model, loader, device, criterion, feature_input=False):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if feature_input:
                x, y = batch
            else:
                x, y = batch['raw_data'], batch['label']
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            total += criterion(model(x), y).item() * y.size(0)
            n += y.size(0)
    return total / max(n, 1)


def _train_with_early_stopping(model, params, lr, train_loader, val_loader, device,
                               num_epochs, feature_input=False):
    """Train, keep the epoch with the lowest inner-validation loss (patience = PATIENCE)"""
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_T_MULT, eta_min=ETA_MIN)

    best_val_loss, best_epoch = float('inf'), -1
    best_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            if feature_input:
                x, y = batch
            else:
                x, y = batch['raw_data'], batch['label']
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_loss = _evaluate_loss(model, val_loader, device, criterion, feature_input)
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model, {'best_epoch': best_epoch, 'best_inner_val_loss': best_val_loss}


# ----------------------------------------------------------------------------------------------
# Fine-tuning protocol: encoder + MLP head, all parameters updated
# ----------------------------------------------------------------------------------------------
def train_finetune(pretrained_encoder_state, encoder_config, dataset, train_indices, val_indices,
                   device, seed, num_epochs=DOWNSTREAM_EPOCHS):
    seed_everything(seed)
    classifier = ScipyFineTuningClassifier(
        pretrained_encoder_state=pretrained_encoder_state,
        encoder_config=encoder_config, n_classes=2, freeze_encoder=False
    ).to(device)
    workers = min(NUM_WORKERS, 4)
    train_loader = make_loader(dataset, train_indices, DOWNSTREAM_BATCH_SIZE, True, seed,
                               drop_last=True, num_workers=workers)
    val_loader = make_loader(dataset, val_indices, DOWNSTREAM_BATCH_SIZE, False, seed, num_workers=workers)
    return _train_with_early_stopping(classifier, classifier.parameters(), FINETUNE_LR,
                                      train_loader, val_loader, device, num_epochs)


def predict_proba(model, dataset, indices, device):
    """Epoch-level P(AD) for raw segments"""
    loader = make_loader(dataset, indices, DOWNSTREAM_BATCH_SIZE, False, 0, num_workers=min(NUM_WORKERS, 4))
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch['raw_data'].to(device, non_blocking=True))
            probs.append(F.softmax(outputs, dim=1)[:, 1].cpu())
    return torch.cat(probs).numpy()


# ----------------------------------------------------------------------------------------------
# Linear-evaluation protocol: frozen encoder (weights + BN statistics) + single linear layer
# ----------------------------------------------------------------------------------------------
def build_frozen_encoder(pretrained_encoder_state, encoder_config, device):
    encoder = ScipyMultiFrequency_SimCLR_Encoder(**encoder_config)
    encoder.load_state_dict(pretrained_encoder_state)
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder.to(device).eval()


def extract_features(encoder, dataset, indices, device):
    """Frozen-encoder features. The encoder is in eval mode and deterministic, so features can be
    computed once per pre-trained model and shared by all LOSO folds without any leakage."""
    loader = make_loader(dataset, indices, DOWNSTREAM_BATCH_SIZE, False, 0, num_workers=min(NUM_WORKERS, 4))
    feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting frozen features", leave=False):
            features, _, _ = encoder(batch['raw_data'].to(device, non_blocking=True))
            feats.append(features.cpu())
    return torch.cat(feats)


class LinearProbe(nn.Module):
    """Linear layer with feature standardization fitted on the inner-train features of the fold"""
    def __init__(self, mean, std, n_classes=2):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.linear = build_linear_head(mean.shape[0], n_classes)

    def forward(self, x):
        return self.linear((x - self.mean) / self.std)


def train_linear_probe(train_x, train_y, val_x, val_y, device, seed, num_epochs=DOWNSTREAM_EPOCHS):
    seed_everything(seed)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)
    probe = LinearProbe(mean, std).to(device)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=DOWNSTREAM_BATCH_SIZE,
                              shuffle=True, drop_last=True, generator=generator)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=DOWNSTREAM_BATCH_SIZE)
    return _train_with_early_stopping(probe, probe.linear.parameters(), LINEAR_LR,
                                      train_loader, val_loader, device, num_epochs, feature_input=True)


def predict_proba_features(probe, x, device):
    probe.eval()
    with torch.no_grad():
        return F.softmax(probe(x.to(device)), dim=1)[:, 1].cpu().numpy()
