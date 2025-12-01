import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from xgboost import XGBClassifier


# ========= CONFIG =========

# Path to your ASVspoof2019 LA train protocol file
PATH_TRAIN_PROTO = "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

# Path to the directory containing the *audio* files for the train split
PATH_TRAIN_AUDIO_DIR = "data/LA/ASVspoof2019_LA_train/flac"

# Audio file extension in the train folder (likely '.flac' or '.wav')
AUDIO_EXT = ".flac"

# DEV
PATH_DEV_PROTO = "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
PATH_DEV_AUDIO_DIR = "data/LA/ASVspoof2019_LA_dev/flac"

# Optional: for quick tests, you can limit utterances; set to None for full
MAX_TRAIN_UTTERANCES = 5000      # e.g. 5000 for quick test, then None
MAX_DEV_UTTERANCES = None        # same idea

# Caches
FEATURE_CACHE_TRAIN = "data/cache_mfcc_train.npz"
FEATURE_CACHE_DEV = "data/cache_mfcc_dev.npz"


# ========= STEP 1: LOAD PROTOCOL =========

def load_protocol(protocol_path: str) -> pd.DataFrame:
    """
    Parse ASVspoof 2019 LA protocol file into a DataFrame with columns:
    ['speaker_id', 'utt_id', 'system_id', 'label'].
    """
    rows = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            speaker_id = parts[0]
            utt_id = parts[1]
            system_id = parts[2]
            label = parts[-1]  # 'bonafide' or 'spoof'
            rows.append((speaker_id, utt_id, system_id, label))
    df = pd.DataFrame(rows, columns=["speaker_id", "utt_id", "system_id", "label"])
    return df


def add_audio_path(df: pd.DataFrame, audio_dir: str | Path, ext: str = ".flac") -> pd.DataFrame:
    """
    Add a 'path' column pointing to the audio file for each utt_id.
    """
    audio_dir = Path(audio_dir)
    df = df.copy()
    df["path"] = df["utt_id"].apply(lambda u: str(audio_dir / f"{u}{ext}"))
    return df


# ========= STEP 2: MFCC FEATURE EXTRACTION =========

def extract_mfcc_features(
    audio_path: str,
    sr: int = 16000,
    n_mfcc: int = 20,
    win_length: float = 0.025,
    hop_length: float = 0.010,
) -> np.ndarray:
    """
    Load audio, compute MFCCs + Δ + ΔΔ per frame, then aggregate stats over time.
    Returns a 1D feature vector of shape (4 * 3 * n_mfcc,).
    """
    y, sr_loaded = librosa.load(audio_path, sr=sr)
    # librosa may return a float sample rate in some environments; ensure an int for downstream computations
    sr = int(sr_loaded)
    
    n_fft = int(sr * win_length)
    hop = int(sr * hop_length)
    
    # MFCCs: shape (n_mfcc, T)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # concatenate along feature axis: (3*n_mfcc, T)
    feats = np.concatenate([mfcc, delta, delta2], axis=0)
    
    # aggregate over time with simple stats
    stats = []
    for stat in (np.mean, np.std, np.min, np.max):
        stats.append(stat(feats, axis=1))
    stats = np.concatenate(stats, axis=0)  # shape (4 * 3*n_mfcc,)
    return stats.astype(np.float32)


def build_feature_matrix(df: pd.DataFrame, cache_path: str | None = None, max_utts: int | None = None):
    """
    Given a DataFrame with columns ['path', 'label'], extract MFCC features for each utterance.
    Optionally caches to / loads from an .npz file.
    Returns:
        X: np.ndarray of shape (N, D)
        y: np.ndarray of shape (N,)
    """
    label_map = {"bonafide": 0, "spoof": 1}

    # If cache exists and we want to load it
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        data = np.load(cache_path)
        return data["X"], data["y"]

    if max_utts is not None:
        df = df.iloc[:max_utts].reset_index(drop=True)

    X_list = []
    y_list = []

    print(f"Extracting MFCC features for {len(df)} utterances...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row["path"]
        label = row["label"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        feat = extract_mfcc_features(path)
        X_list.append(feat)
        y_list.append(label_map[label])

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"Saved features to cache: {cache_path}")

    return X, y


# ========= STEP 3: TRAIN XGBOOST BASELINE =========

def train_xgboost_baseline(X: np.ndarray, y: np.ndarray):
    """
    Train an XGBoost binary classifier and evaluate on a held-out validation set.
    """
    # Stratified split within the training set
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardise features
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    # Handle class imbalance via scale_pos_weight
    num_pos = (y_tr == 1).sum()
    num_neg = (y_tr == 0).sum()
    scale_pos_weight = num_neg / max(num_pos, 1)
    print(f"Class counts in train split: neg={num_neg}, pos={num_pos}, scale_pos_weight={scale_pos_weight:.2f}")

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
    )

    clf.fit(X_tr, y_tr)

    # Predictions
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_val, y_val_proba)
    print(f"Validation ROC AUC: {auc:.4f}")
    print("Classification report:")
    print(classification_report(y_val, y_val_pred, target_names=["bonafide", "spoof"]))

    # Compute EER (simple implementation)
    fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
    fnr = 1 - tpr
    # Find threshold where FNR ≈ FPR
    eer_threshold_index = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
    print(f"Approximate EER: {eer * 100:.2f}%")

    return clf, scaler

def train_xgboost_on_train_eval_on_dev(X_train: np.ndarray, y_train: np.ndarray,
                                       X_dev: np.ndarray, y_dev: np.ndarray):
    """
    Train XGBoost on the official train set, evaluate on the official dev set.
    """
    # Standardise based on TRAIN only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    # Handle class imbalance via scale_pos_weight (on TRAIN)
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    scale_pos_weight = num_neg / max(num_pos, 1)
    print(f"[TRAIN] Class counts: neg={num_neg}, pos={num_pos}, scale_pos_weight={scale_pos_weight:.2f}")

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
    )

    clf.fit(X_train_scaled, y_train)

    # Predictions on DEV
    y_dev_proba = clf.predict_proba(X_dev_scaled)[:, 1]
    y_dev_pred = (y_dev_proba >= 0.5).astype(int)

    from sklearn.metrics import roc_auc_score, roc_curve, classification_report

    auc = roc_auc_score(y_dev, y_dev_proba)
    print(f"[DEV] ROC AUC: {auc:.4f}")
    print("[DEV] Classification report:")
    print(classification_report(y_dev, y_dev_pred, target_names=["bonafide", "spoof"]))

    # EER
    fpr, tpr, thresholds = roc_curve(y_dev, y_dev_proba)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    print(f"[DEV] Approximate EER: {eer * 100:.2f}%")

    return clf, scaler


def main():
    # 1. Load TRAIN protocol
    print("Loading TRAIN protocol...")
    df_train = load_protocol(PATH_TRAIN_PROTO)
    df_train = add_audio_path(df_train, PATH_TRAIN_AUDIO_DIR, ext=AUDIO_EXT)

    missing_train = df_train[~df_train["path"].apply(os.path.exists)]
    if not missing_train.empty:
        print("Warning: some TRAIN audio paths do not exist. First few missing:")
        print(missing_train.head())
        raise SystemExit("Fix PATH_TRAIN_AUDIO_DIR/AUDIO_EXT or protocol paths.")

    print("TRAIN head:")
    print(df_train.head())

    # 2. Load DEV protocol
    print("\nLoading DEV protocol...")
    df_dev = load_protocol(PATH_DEV_PROTO)
    df_dev = add_audio_path(df_dev, PATH_DEV_AUDIO_DIR, ext=AUDIO_EXT)

    missing_dev = df_dev[~df_dev["path"].apply(os.path.exists)]
    if not missing_dev.empty:
        print("Warning: some DEV audio paths do not exist. First few missing:")
        print(missing_dev.head())
        raise SystemExit("Fix PATH_DEV_AUDIO_DIR/AUDIO_EXT or protocol paths.")

    print("DEV head:")
    print(df_dev.head())

    # 3. Extract (or load) MFCC features
    X_train, y_train = build_feature_matrix(
        df_train,
        cache_path=FEATURE_CACHE_TRAIN,
        max_utts=MAX_TRAIN_UTTERANCES,
    )
    X_dev, y_dev = build_feature_matrix(
        df_dev,
        cache_path=FEATURE_CACHE_DEV,
        max_utts=MAX_DEV_UTTERANCES,
    )

    print("Train feature matrix shape:", X_train.shape)
    print("Dev   feature matrix shape:", X_dev.shape)

    # 4. Train and evaluate XGBoost baseline (train on train, eval on dev)
    train_xgboost_on_train_eval_on_dev(X_train, y_train, X_dev, y_dev)


if __name__ == "__main__":
    main()
