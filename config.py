"""
Configuration and constants for dementia EEG analysis
"""
import os
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_DEVICE = '0'

# Dataset configuration
ROOT_DIR = "/home/pcw/data/eeg/demen/csv"
SEGMENT_LENGTH = 15000          # 30 s at 500 Hz
SAMPLING_RATE = 500
N_CHANNELS = 19
N_CLASSES = 2

# Standard 10-20 montage order expected for external datasets (old/new temporal names are aliased)
CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
CHANNEL_ALIASES = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}

# Subject configuration (OpenNeuro ds004504: 1-36 AD, 37-65 CN, 66-88 FTD)
AD_SUBJECTS = list(range(1, 37))
CN_SUBJECTS = list(range(37, 66))
PRETRAIN_SUBJECTS = list(range(66, 89))  # 66~88 (FTD, used without labels)
LOSO_SUBJECTS = list(range(1, 66))       # 1~65

# Labels: AD is the positive class (sensitivity = AD recall)
LABEL_CN = 0
LABEL_AD = 1
CLASS_NAMES = ['CN', 'AD']

# Model hyperparameters (fixed a priori; never tuned on LOSO test subjects)
PRETRAIN_EPOCHS = 100
DOWNSTREAM_EPOCHS = 20
BATCH_SIZE = 64
DOWNSTREAM_BATCH_SIZE = 32
LEARNING_RATE = 1e-3
FINETUNE_LR = 1e-4
LINEAR_LR = 1e-3
WEIGHT_DECAY = 1e-5
EPS = 1e-8

# Downstream evaluation protocol
#   'linear'   : encoder frozen (weights and BatchNorm statistics), single linear layer trained
#   'finetune' : all encoder and classifier parameters updated
EVAL_MODES = ['linear', 'finetune']

# Nested validation: fraction of the LOSO *training* subjects held out (subject-wise,
# label-stratified) as an inner validation set for early stopping / model selection.
# The LOSO test subject is never seen until the final prediction.
INNER_VAL_FRACTION = 0.15

# Subject-level decision rule: mean of epoch-level P(AD) over a subject, threshold fixed a priori
SUBJECT_THRESHOLD = 0.5
N_BOOTSTRAP = 2000

# SimCLR configuration
SIMCLR_HIDDEN_DIM = 256
SIMCLR_OUTPUT_DIM = 128
TEMP_MIN = 0.05
TEMP_MAX = 0.5
BASE_TEMPERATURE = 0.1
REGULARIZATION_WEIGHT = 0.01

# Optimizer configuration
SCHEDULER_T0 = 10
SCHEDULER_T_MULT = 2
ETA_MIN = 1e-6

# Training configuration
PATIENCE = 10
NUM_WORKERS = min((os.cpu_count() or 2) // 2, 8)
PIN_MEMORY = True
DROP_LAST = True
PREFETCH_FACTOR = 2

# Frequency bands (Hz)
FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Augmentation parameters
GAUSSIAN_NOISE_FACTOR = 0.1
CHANNEL_DROPOUT_PROB = 0.1
TIME_SHIFT_RANGE = 0.1
AMPLITUDE_SCALE_RANGE = (0.8, 1.2)
FREQUENCY_MASK_PROB = 0.3
FREQUENCY_MASK_RATIO = 0.1
TIME_MASK_PROB = 0.3
TIME_MASK_RATIO = 0.1

# Seeds: every seed repeats the whole pipeline (pre-training + all LOSO folds).
# Reported mean ± std is across these independent repetitions.
SEED = 2025
SEEDS = [2025, 2026, 2027, 2028, 2029]

# Output
OUTPUT_DIR = "results"
