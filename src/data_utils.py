import os
from pathlib import Path
import pandas as pd

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

def add_audio_path(df: pd.DataFrame, audio_dir: str, ext: str = ".flac") -> pd.DataFrame:
    """
    Add a 'path' column pointing to the audio file for each utt_id.
    """
    audio_dir = Path(audio_dir)
    df["path"] = df["utt_id"].apply(lambda u: str(audio_dir / f"{u}{ext}"))
    return df
