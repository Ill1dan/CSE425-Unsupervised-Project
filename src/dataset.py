import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa


# -----------------------------
# Feature helpers
# -----------------------------

def _fix_frames(feat_2d, target_frames):
    n_feats, frames = feat_2d.shape

    if frames == target_frames:
        return feat_2d

    if frames > target_frames:
        return feat_2d[:, :target_frames]

    pad = target_frames - frames
    return np.pad(feat_2d, ((0, 0), (0, pad)), mode="constant")


def extract_mfcc(y, sr, n_mfcc=40, n_fft=2048, hop_length=512, target_frames=256):
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    mfcc = _fix_frames(mfcc, target_frames)
    return mfcc


def extract_mel(y, sr, n_mels=128, n_fft=2048, hop_length=512, target_frames=256, to_db=True):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    mel = _fix_frames(mel, target_frames)
    return mel


# -----------------------------
# Normalization (scaler)
# -----------------------------

def fit_standard_scaler(features, eps=1e-8):
    # MFCC flat: (N, D)
    if features.ndim == 2:
        mean = features.mean(axis=0)
        std = features.std(axis=0) + eps
        return {"mode": "standard", "mean": mean, "std": std}

    # Mel / 2D: (N, H, W) or (N, 1, H, W)
    if features.ndim == 4:
        features_ = features[:, 0, :, :]
    else:
        features_ = features

    mean = features_.mean(axis=(0, 2))  # (H,)
    std = features_.std(axis=(0, 2)) + eps
    return {"mode": "standard_mel", "mean": mean, "std": std}


def apply_scaler(features, scaler):
    if scaler is None:
        return features

    mode = scaler.get("mode")

    if mode == "standard":
        return (features - scaler["mean"]) / scaler["std"]

    if mode == "standard_mel":
        mean = scaler["mean"]  # (H,)
        std = scaler["std"]    # (H,)

        if features.ndim == 4:
            out = features.copy()
            out[:, 0, :, :] = (out[:, 0, :, :] - mean[:, None]) / std[:, None]
            return out

        return (features - mean[None, :, None]) / std[None, :, None]

    raise ValueError("Unknown scaler mode: %s" % str(mode))


def save_scaler(path, scaler):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mode=scaler["mode"], mean=scaler["mean"], std=scaler["std"])


def load_scaler(path):
    obj = np.load(path, allow_pickle=True)
    return {"mode": str(obj["mode"]), "mean": obj["mean"], "std": obj["std"]}


# -----------------------------
# Metadata reading
# -----------------------------

def read_metadata_csv(metadata_path):
    """
    Expects columns:
      id, filepath, language, title, lyrics
    """
    rows = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _normalize_audio_path(p):
    """
    Handles mixed slashes like:
      data/processed\\english\\en_001.wav
    Works on Windows + Linux.
    """
    if p is None:
        return None
    p = p.strip()

    # Convert all backslashes to os.sep
    p = p.replace("\\", os.sep).replace("/", os.sep)

    # Normalize (collapse .., duplicate separators)
    return os.path.normpath(p)


# -----------------------------
# Dataset
# -----------------------------

class AudioFeatureDataset(Dataset):
    """
    Reads metadata.csv and loads audio from `filepath`.

    feature_type:
      - "mfcc_flat"  -> (D,)
      - "mfcc_2d"    -> (1, n_mfcc, frames)
      - "mel"        -> (1, n_mels, frames)

    Returns: (x_tensor, id, language, title, lyrics)
    """
    def __init__(
        self,
        metadata_csv,
        root_dir=None,
        feature_type="mel",
        sample_rate=22050,
        mono=True,
        n_fft=2048,
        hop_length=512,
        n_mfcc=40,
        n_mels=128,
        target_frames=256,
        scaler=None
    ):
        self.rows = read_metadata_csv(metadata_csv)
        self.root_dir = root_dir
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.mono = mono

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.target_frames = target_frames

        self.scaler = scaler

    def __len__(self):
        return len(self.rows)

    def _resolve_path(self, rel_path):
        rel_path = _normalize_audio_path(rel_path)
        if self.root_dir is None:
            return rel_path
        return os.path.normpath(os.path.join(self.root_dir, rel_path))

    def __getitem__(self, idx):
        r = self.rows[idx]

        _id = r.get("id", str(idx))
        rel_path = r.get("filepath")
        lang = r.get("language", "")
        title = r.get("title", "")
        lyrics = r.get("lyrics", "")

        wav_path = self._resolve_path(rel_path)

        y, sr = librosa.load(wav_path, sr=self.sample_rate, mono=self.mono)

        # ---- MFCC flat ----
        if self.feature_type == "mfcc_flat":
            mfcc = extract_mfcc(
                y, sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                target_frames=self.target_frames
            )
            x = mfcc.reshape(-1).astype(np.float32)

            if self.scaler is not None:
                x = apply_scaler(x[None, :], self.scaler)[0]

            if torch is None:
                return x, _id, lang, title, lyrics
            return torch.from_numpy(x), _id, lang, title, lyrics

        # ---- MFCC 2D ----
        if self.feature_type == "mfcc_2d":
            mfcc = extract_mfcc(
                y, sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                target_frames=self.target_frames
            ).astype(np.float32)

            x = mfcc[None, :, :]  # (1, n_mfcc, frames)

            if self.scaler is not None:
                x = apply_scaler(x[None, :, :, :], self.scaler)[0]

            if torch is None:
                return x, _id, lang, title, lyrics
            return torch.from_numpy(x), _id, lang, title, lyrics

        # ---- Mel ----
        if self.feature_type == "mel":
            mel = extract_mel(
                y, sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                target_frames=self.target_frames,
                to_db=True
            ).astype(np.float32)

            x = mel[None, :, :]  # (1, n_mels, frames)

            if self.scaler is not None:
                x = apply_scaler(x[None, :, :, :], self.scaler)[0]

            if torch is None:
                return x, _id, lang, title, lyrics
            return torch.from_numpy(x), _id, lang, title, lyrics

        raise ValueError("Unknown feature_type: %s" % self.feature_type)


# -----------------------------
# Offline feature export (Deliverable)
# -----------------------------

def export_features(
    metadata_csv,
    out_dir,
    root_dir=None,
    feature_type="mel",
    sample_rate=22050,
    n_fft=2048,
    hop_length=512,
    n_mfcc=40,
    n_mels=128,
    target_frames=256,
    fit_scaler=True
):
    ds = AudioFeatureDataset(
        metadata_csv=metadata_csv,
        root_dir=root_dir,
        feature_type=feature_type,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        target_frames=target_frames,
        scaler=None
    )

    feats = []
    ids = []
    langs = []

    for i in range(len(ds)):
        x, _id, lang, title, lyrics = ds[i]

        if torch is not None and hasattr(x, "detach"):
            x = x.detach().cpu().numpy()

        feats.append(x)
        ids.append(_id)
        langs.append(lang)

    feats = np.stack(feats, axis=0)

    scaler = None
    if fit_scaler:
        scaler = fit_standard_scaler(feats)
        feats = apply_scaler(feats, scaler)
        save_scaler(os.path.join(out_dir, "scaler.npz"), scaler)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "features.npy"), feats)
    np.save(os.path.join(out_dir, "ids.npy"), np.array(ids, dtype=object))
    np.save(os.path.join(out_dir, "languages.npy"), np.array(langs, dtype=object))

    return feats.shape


def main():
    # You can run:
    #   python -m src.dataset
    metadata_csv = "data/metadata.csv"

    # Change this to:
    #   "mfcc_flat" for easy baseline
    #   "mel" for Conv-VAE
    feature_type = "mfcc_flat"

    out_dir = "data/features_%s" % feature_type

    shape = export_features(
        metadata_csv=metadata_csv,
        out_dir=out_dir,
        root_dir=None,
        feature_type=feature_type,
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mfcc=40,
        n_mels=128,
        target_frames=256,
        fit_scaler=True
    )

    print("Saved:", out_dir)
    print("Feature array shape:", shape)


if __name__ == "__main__":
    main()
