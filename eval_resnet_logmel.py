# eval_resnet_logmel_eval.py

import numpy as np
import pandas as pd
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, roc_curve, classification_report

# Import dataset/model utilities from your training script
from train_resnet_logmel import (
    load_protocol,
    add_audio_path,
    ASVspoofLADataset,
    ResNetSmall,
    SR,
    N_MELS,
    AUDIO_EXT,
)

# ---------- CONFIG: EVAL paths ----------

# Eval dataset
PATH_EVAL_PROTO = "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
PATH_EVAL_AUDIO_DIR = "data/LA/ASVspoof2019_LA_eval/flac"

# Dev dataset
#PATH_DEV_PROTO = "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
#PATH_DEV_AUDIO_DIR = "data/LA/ASVspoof2019_LA_dev/flac"

CHECKPOINT_PATH = "checkpoints/resnet_logmel_pos1_gain_best.pth"
BATCH_SIZE = 32


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER) given true binary labels and scores.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer


def main():
    # ----- Device -----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # ----- Load EVAL protocol -----
    print("Loading EVAL protocol from:", PATH_EVAL_PROTO)
    df_eval = load_protocol(PATH_EVAL_PROTO)
    df_eval = add_audio_path(df_eval, PATH_EVAL_AUDIO_DIR, AUDIO_EXT)
    print("EVAL size:", len(df_eval))

    # Sanity check: at least one file exists
    example_path = df_eval["path"].iloc[0]
    if not os.path.exists(example_path):
        raise FileNotFoundError(f"Example EVAL audio path does not exist: {example_path}")

    # ----- Dataset & DataLoader -----
    eval_dataset = ASVspoofLADataset(df_eval, train=False)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    # ----- Load trained ResNet -----
    print("Loading checkpoint:", CHECKPOINT_PATH)
    model = ResNetSmall(n_mels=N_MELS).to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ----- Inference on EVAL -----
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in eval_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels).astype(int)

    # ----- Metrics -----
    auc = roc_auc_score(all_labels, all_probs)
    eer = compute_eer(all_labels, all_probs)
    y_pred = (all_probs >= 0.5).astype(int)

    print("\n=== EVAL RESULTS (ResNet-small, log-Mel) ===")
    print(f"[EVAL] ROC AUC: {auc:.4f}")
    print(f"[EVAL] EER: {eer * 100:.2f}%")
    print("[EVAL] Classification report (threshold=0.5):")
    print(classification_report(all_labels, y_pred, target_names=["bonafide", "spoof"]))


if __name__ == "__main__":
    main()
