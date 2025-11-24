import librosa
import numpy as np

def extract_mfcc_features(
    audio_path: str,
    sr: int = 16000,
    n_mfcc: int = 20,
    win_length: float = 0.025,
    hop_length: float = 0.010,
) -> np.ndarray:
    """
    Load audio, compute MFCCs + Δ + ΔΔ per frame, then aggregate stats over time.
    Returns a 1D feature vector.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    
    n_fft = int(sr * win_length)
    hop = int(sr * hop_length)
    
    # MFCCs: shape (n_mfcc, T)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # concatenate along "feature" axis: (3*n_mfcc, T)
    feats = np.concatenate([mfcc, delta, delta2], axis=0)
    
    # aggregate over time with simple stats
    # (mean, std, min, max) for each of the 3*n_mfcc features
    stats = []
    for stat in (np.mean, np.std, np.min, np.max):
        stats.append(stat(feats, axis=1))
    # shape (4, 3*n_mfcc) -> flatten to (4*3*n_mfcc,)
    stats = np.concatenate(stats, axis=0)
    return stats.astype(np.float32)
