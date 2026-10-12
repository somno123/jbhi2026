"""
Data loading and dataset classes for EEG dementia analysis

Preprocessing note (leakage audit): nothing in this module is fitted on data.
Segmentation and (optional) resampling are deterministic per-recording operations, and no
normalization statistics are estimated across subjects. Any artifact removal (e.g. ICA) must be
done per recording before export, so it cannot couple training and test subjects.
"""
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from math import gcd
from scipy.signal import resample_poly
from tqdm import tqdm
from config import (SEGMENT_LENGTH, SAMPLING_RATE, N_CHANNELS, CHANNEL_NAMES, CHANNEL_ALIASES,
                    AD_SUBJECTS, CN_SUBJECTS, LABEL_AD, LABEL_CN)

UNLABELED = -1


def _reorder_channels(df):
    """Return a (C, T) array in CHANNEL_NAMES order if the columns are named, else the raw order"""
    columns = {CHANNEL_ALIASES.get(c.strip(), c.strip()).lower(): c for c in df.columns}
    if all(name.lower() in columns for name in CHANNEL_NAMES):
        df = df[[columns[name.lower()] for name in CHANNEL_NAMES]]
    return df.values.T.astype(np.float32)


def load_recording(file_path, source_fs=SAMPLING_RATE, target_fs=SAMPLING_RATE, mmap=False):
    """Load one recording as a (C, T) float32 array (.csv with optional 'time' column, or .npy)"""
    if file_path.endswith('.npy'):
        eeg_np = np.load(file_path, mmap_mode='r' if mmap and source_fs == target_fs else None)
    else:
        df = pd.read_csv(file_path)
        df = df.drop(columns=[c for c in df.columns if c.lower() == 'time'])
        eeg_np = _reorder_channels(df)

    if eeg_np.shape[0] != N_CHANNELS:
        raise ValueError(f"{file_path}: expected {N_CHANNELS} channels, got {eeg_np.shape[0]}")

    if source_fs != target_fs:
        g = gcd(int(source_fs), int(target_fs))
        eeg_np = resample_poly(eeg_np, int(target_fs) // g, int(source_fs) // g, axis=1).astype(np.float32)
    return eeg_np


def subject_label(sub_id):
    """ds004504 diagnosis: AD -> 1, CN -> 0, FTD (pre-training only) -> -1"""
    if sub_id in AD_SUBJECTS:
        return LABEL_AD
    if sub_id in CN_SUBJECTS:
        return LABEL_CN
    return UNLABELED


class EEGDataset(Dataset):
    """Local cohort (ds004504 CSV export) split into non-overlapping 30 s segments"""
    def __init__(self, root_dir, segment_length=SEGMENT_LENGTH, sampling_rate=SAMPLING_RATE):
        self.segment_length = segment_length
        self.data, self.labels, self.subject_ids = [], [], []
        self.sampling_rate = sampling_rate

        target_files = sorted(glob(os.path.join(root_dir, "sub-*.csv")))
        for file_path in tqdm(target_files, desc="Loading Raw EEG Data"):
            filename = os.path.basename(file_path)
            eeg_np = load_recording(file_path)

            sub_id = int(filename.split('_')[0].split('-')[1])
            label = subject_label(sub_id)

            n_segments = eeg_np.shape[1] // self.segment_length
            for i in range(n_segments):
                start, end = i * self.segment_length, (i + 1) * self.segment_length
                self.data.append(eeg_np[:, start:end])
                self.labels.append(label)
                self.subject_ids.append(sub_id)

    def get_raw(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'raw_data': torch.from_numpy(np.ascontiguousarray(self.get_raw(idx))).float(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'subject_id': self.subject_ids[idx]
        }


class ExternalEEGDataset(Dataset):
    """
    External EEG directory used either as
      * an unlabeled pre-training corpus (e.g. TUH EEG converted to 19-ch .npy), or
      * an independent labeled validation cohort (labels_csv given).

    Files: recursive *.npy (C x T) or *.csv (one column per channel, optional 'time').
    .npy files at the target rate are memory-mapped, so large corpora are not loaded into RAM.

    labels_csv columns: 'file' (path relative to root_dir), 'label' (AD/CN or 1/0),
    optional 'subject_id' (defaults to the file stem; several files may share a subject).
    """
    def __init__(self, root_dir, labels_csv=None, source_fs=SAMPLING_RATE,
                 segment_length=SEGMENT_LENGTH, max_files=None):
        self.segment_length = segment_length
        self.recordings = []
        self.index, self.labels, self.subject_ids = [], [], []

        if labels_csv is not None:
            table = pd.read_csv(labels_csv)
            if 'subject_id' not in table.columns:
                table['subject_id'] = table['file'].map(lambda f: os.path.splitext(os.path.basename(f))[0])
            label_map = {'AD': LABEL_AD, 'CN': LABEL_CN, '1': LABEL_AD, '0': LABEL_CN}
            entries = [(os.path.join(root_dir, row.file), label_map[str(row.label).strip().upper()],
                        str(row.subject_id)) for row in table.itertuples()]
        else:
            files = sorted(glob(os.path.join(root_dir, '**', '*.npy'), recursive=True) +
                           glob(os.path.join(root_dir, '**', '*.csv'), recursive=True))
            entries = [(f, UNLABELED, os.path.relpath(f, root_dir)) for f in files]

        if max_files is not None:
            entries = entries[:max_files]

        for file_path, label, sub_id in tqdm(entries, desc=f"Indexing {root_dir}"):
            try:
                eeg_np = load_recording(file_path, source_fs=source_fs, mmap=True)
            except ValueError as e:
                print(f"Skipping {e}")
                continue
            rec_idx = len(self.recordings)
            self.recordings.append(eeg_np)
            for i in range(eeg_np.shape[1] // segment_length):
                self.index.append((rec_idx, i * segment_length))
                self.labels.append(label)
                self.subject_ids.append(sub_id)

    def get_raw(self, idx):
        rec_idx, start = self.index[idx]
        return self.recordings[rec_idx][:, start:start + self.segment_length]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return {
            'raw_data': torch.from_numpy(np.array(self.get_raw(idx), dtype=np.float32)),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'subject_id': self.subject_ids[idx]
        }


class OptimizedSimCLR_EEG_Dataset(Dataset):
    """SimCLR dataset: returns two augmented views of one segment (labels are never used)"""
    def __init__(self, original_dataset, indices, augmentations):
        self.original_dataset = original_dataset
        self.indices = indices
        self.augmentations = augmentations

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        raw_data = torch.from_numpy(np.array(self.original_dataset.get_raw(original_idx), dtype=np.float32))
        view1, view2 = self.augmentations(raw_data)
        return view1, view2
