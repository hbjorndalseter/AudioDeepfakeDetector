import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

# ==== CONFIG: adapt to your paths ====

PATH_DEV_PROTO = "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
PATH_DEV_AUDIO_DIR = "data/LA/ASVspoof2019_LA_dev/flac"

AUDIO_EXT = ".flac"

SR = 16000
N_MELS = 64
N_MFCC = 20
WIN_LENGTH = 0.025  # seconds
HOP_LENGTH = 0.010  # seconds


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
            label = parts[-1]  # 'bonafide' or 'spoof'
            rows.append((speaker_id, utt_id, system_id, label))
    df = pd.DataFrame(rows, columns=["speaker_id", "utt_id", "system_id", "label"])
    return df


def add_audio_path(df: pd.DataFrame, audio_dir: str | Path, ext: str) -> pd.DataFrame:
    audio_dir = Path(audio_dir)
    df = df.copy()
    df["path"] = df["utt_id"].apply(lambda u: str(audio_dir / f"{u}{ext}"))
    return df


def pick_examples(df: pd.DataFrame):
    """Pick one bonafide and one spoof example from the dev set."""
    df_bona = df[df["label"] == "bonafide"]
    df_spoof = df[df["label"] == "spoof"]

    if df_bona.empty or df_spoof.empty:
        raise ValueError("Dev set does not contain both bonafide and spoof examples!")

    ex_bona = df_bona.iloc[0]
    ex_spoof = df_spoof.iloc[0]

    return ex_bona, ex_spoof


def compute_features(y, sr):
    """Compute MFCCs and log-Mel spectrogram for a waveform."""
    n_fft = int(sr * WIN_LENGTH)
    hop = int(sr * HOP_LENGTH)

    # MFCCs (+ delta + delta-delta)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=N_MFCC,
        n_fft=n_fft,
        hop_length=hop
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # log-Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = np.log(mel + 1e-6)

    return mfcc, delta, delta2, log_mel


def plot_example(y, sr, mfcc, log_mel, title):
    """Plot waveform, MFCCs, and log-Mel spectrogram for a single utterance."""
    # time axis for waveform
    t = np.linspace(0, len(y) / sr, num=len(y))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)

    # 1) Waveform
    axes[0].plot(t, y)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")

    # 2) MFCCs
    img1 = librosa.display.specshow(
        mfcc,
        x_axis="time",
        sr=sr,
        hop_length=int(sr * HOP_LENGTH),
        ax=axes[1]
    )
    axes[1].set_title("MFCCs ({} coefficients)".format(N_MFCC))
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

    # 3) Log-Mel spectrogram
    img2 = librosa.display.specshow(
        log_mel,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=int(sr * HOP_LENGTH),
        ax=axes[2]
    )
    axes[2].set_title("Log-Mel spectrogram ({} Mel bands)".format(N_MELS))
    fig.colorbar(img2, ax=axes[2], format="%+2.0f")

    plt.tight_layout()
    plt.show()


def main():
    # Load dev protocol and paths
    df_dev = load_protocol(PATH_DEV_PROTO)
    df_dev = add_audio_path(df_dev, PATH_DEV_AUDIO_DIR, AUDIO_EXT)

    ex_bona, ex_spoof = pick_examples(df_dev)

    print("Bonafide example:")
    print(ex_bona)
    print("\nSpoof example:")
    print(ex_spoof)

    # Load waveforms
    for ex, name in [(ex_bona, "Bonafide"), (ex_spoof, "Spoof")]:
        path = ex["path"]
        print(f"\nLoading {name} from {path}")
        y, sr = librosa.load(path, sr=SR)
        print(f"{name} duration: {len(y) / sr:.2f} seconds")

        mfcc, delta, delta2, log_mel = compute_features(y, sr)

        print(f"{name}: MFCC shape = {mfcc.shape}, log-Mel shape = {log_mel.shape}")

        plot_example(y, sr, mfcc, log_mel, title=f"{name} example: {ex['utt_id']}")

if __name__ == "__main__":
    main()
