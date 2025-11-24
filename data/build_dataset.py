from torch.utils.data import Dataset
import torch
import librosa
import numpy as np

class ASVspoofLADataset(Dataset):
    def __init__(self, df, sr=16000, duration=3.0, n_mels=64, train=True):
        self.df = df.reset_index(drop=True)
        self.sr = sr
        self.n_samples = int(sr * duration)
        self.n_mels = n_mels
        self.train = train

    def __len__(self):
        return len(self.df)

    def _load_audio_fixed(self, path):
        y, sr = librosa.load(path, sr=self.sr)
        if len(y) < self.n_samples:
            # pad with zeros
            pad = self.n_samples - len(y)
            y = np.pad(y, (0, pad))
        else:
            # random crop during training, center crop otherwise
            if self.train:
                start = np.random.randint(0, len(y) - self.n_samples + 1)
            else:
                start = (len(y) - self.n_samples) // 2
            y = y[start:start + self.n_samples]
        return y

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self._load_audio_fixed(row["path"])
        # compute log-mel
        mel = librosa.feature.melspectrogram(
            y, sr=self.sr,
            n_fft=512, hop_length=160, win_length=400,
            n_mels=self.n_mels, power=2.0
        )
        log_mel = np.log(mel + 1e-6)
        # (F, T) -> (1, F, T)
        x = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        label = 0 if row["label"] == "bonafide" else 1
        label = torch.tensor(label, dtype=torch.long)
        return x, label
