"""
Main execution script for dementia EEG analysis using SimCLR

Pipeline per seed:
  1. Self-supervised pre-training (local FTD subjects 66-88, or an external unlabeled corpus)
  2. LOSO on the 65 AD/CN subjects with nested inner validation, for each downstream protocol
     ('linear' = frozen encoder + linear layer, 'finetune' = all parameters updated)
  3. Optional: independent external cohort, trained on all 65 local subjects, no domain adaptation
"""
import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
import warnings

from config import (
    DEVICE, SEEDS, ROOT_DIR, CUDA_DEVICE, PRETRAIN_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    SAMPLING_RATE, EVAL_MODES, OUTPUT_DIR, PRETRAIN_SUBJECTS
)
from data import EEGDataset, ExternalEEGDataset
from training import pretrain_simclr, seed_everything
from evaluation import (loso_evaluation, external_evaluation, summarize_predictions,
                        print_run_summary, summarize_across_seeds, print_seed_summary)

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE


def parse_args():
    p = argparse.ArgumentParser(description="Delta2Gamma multi-band SimCLR for AD vs CN EEG classification")
    p.add_argument('--root-dir', default=ROOT_DIR, help="local ds004504 CSV directory")
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    p.add_argument('--modes', nargs='+', default=EVAL_MODES, choices=EVAL_MODES)
    p.add_argument('--output-dir', default=OUTPUT_DIR)
    # Pre-training source (scalability / cross-dataset transfer)
    p.add_argument('--pretrain-dir', default=None,
                   help="external unlabeled corpus (e.g. TUH EEG as 19-ch .npy); default: local subjects 66-88")
    p.add_argument('--pretrain-fs', type=float, default=SAMPLING_RATE, help="sampling rate of --pretrain-dir")
    p.add_argument('--pretrain-max-files', type=int, default=None)
    p.add_argument('--pretrain-fraction', type=float, default=1.0,
                   help="fraction of pre-training segments used (data-scaling experiments)")
    p.add_argument('--pretrain-epochs', type=int, default=PRETRAIN_EPOCHS)
    # Independent external validation cohort
    p.add_argument('--external-dir', default=None)
    p.add_argument('--external-labels', default=None, help="CSV with columns file,label[,subject_id]")
    p.add_argument('--external-fs', type=float, default=SAMPLING_RATE)
    return p.parse_args()


def get_pretrained_encoder(args, dataset, pretrain_dataset, device, seed):
    """Pre-train (or load the cached checkpoint) for one seed"""
    tag = 'local' if args.pretrain_dir is None else 'external'
    ckpt = os.path.join(args.output_dir, f"pretrained_{tag}_frac{args.pretrain_fraction}_seed{seed}.pt")
    encoder_config = {'n_channels': np.asarray(dataset.get_raw(0)).shape[0], 'sampling_rate': SAMPLING_RATE}
    if os.path.exists(ckpt):
        print(f"Loading cached pre-trained encoder: {ckpt}")
        return torch.load(ckpt, map_location='cpu'), encoder_config

    if pretrain_dataset is dataset:
        indices = [i for i, s in enumerate(dataset.subject_ids) if s in PRETRAIN_SUBJECTS]
    else:
        indices = list(range(len(pretrain_dataset)))
    if args.pretrain_fraction < 1.0:
        rng = np.random.RandomState(seed)
        indices = sorted(rng.choice(indices, int(len(indices) * args.pretrain_fraction), replace=False).tolist())

    encoder = pretrain_simclr(pretrain_dataset, indices, device, seed, num_epochs=args.pretrain_epochs,
                              batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                              desc=f"[{tag}, {len(indices)} segments, seed {seed}]")
    if encoder is None:
        raise RuntimeError("Pretraining failed!")
    state = {k: v.cpu() for k, v in encoder.state_dict().items()}
    torch.save(state, ckpt)
    return state, encoder_config


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)


def scipy_main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = DEVICE
    print(f"Using device: {device}")
    print(f"Seeds: {args.seeds} (n={len(args.seeds)}) | protocols: {args.modes}")

    dataset = EEGDataset(root_dir=args.root_dir)
    print(f"Local dataset: {len(dataset)} segments, {len(set(dataset.subject_ids))} subjects")

    pretrain_dataset = dataset
    if args.pretrain_dir is not None:
        pretrain_dataset = ExternalEEGDataset(args.pretrain_dir, source_fs=args.pretrain_fs,
                                              max_files=args.pretrain_max_files)
        print(f"External pre-training corpus: {len(pretrain_dataset)} segments")

    external_dataset = None
    if args.external_dir is not None:
        external_dataset = ExternalEEGDataset(args.external_dir, labels_csv=args.external_labels,
                                              source_fs=args.external_fs)
        print(f"External validation cohort: {len(set(external_dataset.subject_ids))} subjects")

    loso_summaries = {m: [] for m in args.modes}
    external_summaries = {m: [] for m in args.modes}

    for seed in args.seeds:
        seed_everything(seed)
        encoder_state, encoder_config = get_pretrained_encoder(args, dataset, pretrain_dataset, device, seed)

        for mode in args.modes:
            run = f"{mode}_seed{seed}"
            epoch_df, fold_df = loso_evaluation(dataset, encoder_state, encoder_config, device, mode, seed)
            summary, subject_df = summarize_predictions(epoch_df)
            epoch_df.to_csv(os.path.join(args.output_dir, f"loso_epochs_{run}.csv"), index=False)
            subject_df.to_csv(os.path.join(args.output_dir, f"loso_subjects_{run}.csv"), index=False)
            fold_df.to_csv(os.path.join(args.output_dir, f"loso_folds_{run}.csv"), index=False)
            save_json(summary, os.path.join(args.output_dir, f"loso_summary_{run}.json"))
            print_run_summary(f"LOSO RESULT [{run}]", summary)
            loso_summaries[mode].append(summary)

            if external_dataset is not None:
                ext_epoch_df, info = external_evaluation(dataset, external_dataset, encoder_state,
                                                         encoder_config, device, mode, seed)
                ext_summary, ext_subject_df = summarize_predictions(ext_epoch_df)
                ext_summary['training'] = info
                ext_subject_df.to_csv(os.path.join(args.output_dir, f"external_subjects_{run}.csv"), index=False)
                save_json(ext_summary, os.path.join(args.output_dir, f"external_summary_{run}.json"))
                print_run_summary(f"EXTERNAL COHORT RESULT [{run}] (no domain adaptation)", ext_summary)
                external_summaries[mode].append(ext_summary)

    final = {}
    for mode in args.modes:
        agg = summarize_across_seeds(loso_summaries[mode])
        print_seed_summary(f"LOSO FINAL [{mode}]", agg)
        final[f"loso_{mode}"] = agg
        if external_summaries[mode]:
            agg = summarize_across_seeds(external_summaries[mode])
            print_seed_summary(f"EXTERNAL FINAL [{mode}]", agg)
            final[f"external_{mode}"] = agg
    final['config'] = vars(args)
    save_json(final, os.path.join(args.output_dir, "final_summary.json"))

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    scipy_main()
