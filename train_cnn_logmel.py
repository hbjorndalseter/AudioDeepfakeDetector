import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, roc_curve, classification_report


# ================== CONFIG ==================

# Protocol and audio paths – adjust to your structure
PATH_TRAIN_PROTO = "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
PATH_TRAIN_AUDIO_DIR = "data/LA/ASVspoof2019_LA_train/flac"

PATH_DEV_PROTO = "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
PATH_DEV_AUDIO_DIR = "data/LA/ASVspoof2019_LA_dev/flac"

AUDIO_EXT = ".flac"

SR = 16000
DURATION = 3.0          # seconds of audio per clip (cropped/padded)
N_MELS = 64

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
PATIENCE = 5          # for early stopping

# optionally limit number for faster testing (None = use all)
MAX_TRAIN_UTTS = None
MAX_DEV_UTTS = None


# ================== UTIL: PROTOKOLL ==================

def load_protocol(protocol_path: str) -> pd.DataFrame:
    rows = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            speaker_id = parts[0]
            utt_id = parts[1]
            system_id = parts[2]
            label = parts[-1]  # 'bonafide' eller 'spoof'
            rows.append((speaker_id, utt_id, system_id, label))
    df = pd.DataFrame(rows, columns=["speaker_id", "utt_id", "system_id", "label"])
    return df


def add_audio_path(df: pd.DataFrame, audio_dir: str | Path, ext: str) -> pd.DataFrame:
    audio_dir = Path(audio_dir)
    df = df.copy()
    df["path"] = df["utt_id"].apply(lambda u: str(audio_dir / f"{u}{ext}"))
    return df


# ================== DATASET ==================

class ASVspoofLADataset(Dataset):
    def __init__(self, df: pd.DataFrame, sr: int = SR, duration: float = DURATION,
                 n_mels: int = N_MELS, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.sr = sr
        self.n_samples = int(sr * duration)
        self.n_mels = n_mels
        self.train = train

    def __len__(self):
        return len(self.df)

    def _load_audio_fixed(self, path: str) -> np.ndarray:
        y, sr = librosa.load(path, sr=self.sr)
        if len(y) < self.n_samples:
            pad = self.n_samples - len(y)
            y = np.pad(y, (0, pad))
        else:
            if self.train:
                start = np.random.randint(0, len(y) - self.n_samples + 1)
            else:
                start = (len(y) - self.n_samples) // 2
            y = y[start:start + self.n_samples]
        return y

    def _spec_augment(self, log_mel: np.ndarray) -> np.ndarray:
        """
        Enkel SpecAugment: masker noen frekvensbånd og tidsvinduer.
        log_mel: (F, T)
        """
        F, T = log_mel.shape

        # Frequency masking
        num_freq_masks = 2
        max_width_freq = F // 8
        for _ in range(num_freq_masks):
            f = np.random.randint(0, max_width_freq + 1)
            f0 = np.random.randint(0, max(1, F - f))
            log_mel[f0:f0 + f, :] = log_mel.mean()

        # Time masking
        num_time_masks = 2
        max_width_time = T // 8
        for _ in range(num_time_masks):
            t = np.random.randint(0, max_width_time + 1)
            t0 = np.random.randint(0, max(1, T - t))
            log_mel[:, t0:t0 + t] = log_mel.mean()

        return log_mel

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = self._load_audio_fixed(row["path"])

        # melspectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=self.n_mels,
            power=2.0
        )
        log_mel = np.log(mel + 1e-6)  # (F, T)

        # Normalize per example (zero-mean, unit-var)
        mean = log_mel.mean()
        std = log_mel.std() + 1e-6
        log_mel = (log_mel - mean) / std

        # SpecAugment only on train
        if self.train:
            log_mel = self._spec_augment(log_mel)

        x = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)  # (1, F, T)

        label = 0 if row["label"] == "bonafide" else 1
        label = torch.tensor(label, dtype=torch.float32)  # for BCEWithLogitsLoss

        return x, label


# ================== MODELL ==================

class SimpleCNNLA(nn.Module):
    def __init__(self, n_mels: int = N_MELS):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # global average pooling over freq and time
        x = x.mean(dim=[2, 3])  # (B, 128)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc_out(x).squeeze(1)  # (B,)
        return x


# ================== EER-CALCULATION ==================

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer


# ================== MAIN ==================

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # --- load protocols and add audio paths ---
    print("Loading TRAIN protocol...")
    df_train = load_protocol(PATH_TRAIN_PROTO)
    df_train = add_audio_path(df_train, PATH_TRAIN_AUDIO_DIR, AUDIO_EXT)
    if MAX_TRAIN_UTTS is not None:
        df_train = df_train.iloc[:MAX_TRAIN_UTTS].reset_index(drop=True)
    print("TRAIN size:", len(df_train))

    print("Loading DEV protocol...")
    df_dev = load_protocol(PATH_DEV_PROTO)
    df_dev = add_audio_path(df_dev, PATH_DEV_AUDIO_DIR, AUDIO_EXT)
    if MAX_DEV_UTTS is not None:
        df_dev = df_dev.iloc[:MAX_DEV_UTTS].reset_index(drop=True)
    print("DEV size:", len(df_dev))

    # sanity check
    for p in [df_train["path"].iloc[0], df_dev["path"].iloc[0]]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Sample path does not exist: {p}")

    # class counts for train (to pos_weight)
    y_train_labels = (df_train["label"] == "spoof").astype(int)
    num_pos = int(y_train_labels.sum())
    num_neg = len(y_train_labels) - num_pos
    pos_weight_value = num_neg / max(num_pos, 1)
    print(f"[TRAIN] class counts: neg={num_neg}, pos={num_pos}, pos_weight={pos_weight_value:.2f}")

    # --- datasets and dataloaders ---
    train_dataset = ASVspoofLADataset(df_train, train=True)
    dev_dataset = ASVspoofLADataset(df_dev, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    # --- model, loss, optimizer ---
    model = SimpleCNNLA(n_mels=N_MELS).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Early stopping state
    best_auc = -np.inf
    best_state_dict = None
    epochs_no_improve = 0

    # --- training loop ---
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        avg_train_loss = running_loss / len(train_dataset)

        # --- eval on dev ---
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for x_batch, y_batch in tqdm(dev_loader, desc=f"Epoch {epoch} [dev]"):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels).astype(int)

        # ROC AUC, EER, classification report ved threshold 0.5
        auc = roc_auc_score(all_labels, all_probs)
        eer = compute_eer(all_labels, all_probs)

        y_pred = (all_probs >= 0.5).astype(int)
        print(f"\nEpoch {epoch}:")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  [DEV] ROC AUC: {auc:.4f}")
        print(f"  [DEV] EER: {eer * 100:.2f}%")
        print("  [DEV] Classification report (threshold=0.5):")
        print(classification_report(all_labels, y_pred, target_names=["bonafide", "spoof"]))

        if auc > best_auc + 1e-4:  # small tolerance to avoid tiny float noise
            best_auc = auc
            epochs_no_improve = 0
            # save best weights in RAM
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [Early stopping] New best AUC: {best_auc:.4f} (saving model state)")
        else:
            epochs_no_improve += 1
            print(f"  [Early stopping] No improvement in AUC for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs (patience={PATIENCE}).")
            break
    
    # Load best model state before exiting
    if best_state_dict is not None:
        torch.save(best_state_dict, "checkpoints/cnn_logmel_best.pth")
        print("Saved best model weights to checkpoints/cnn_logmel_best.pth")
        model.load_state_dict(best_state_dict)
        print(f"\nLoaded best model weights (best dev AUC = {best_auc:.4f}).")
    else:
        print("\nNo best_state_dict saved; training probably ended before any eval improvement.")



if __name__ == "__main__":
    main()
