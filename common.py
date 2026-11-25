"""Eğitim / değerlendirme / tahmin betiklerinin ortak yardımcıları."""
import json
import os
import random

import numpy as np
import torch

from model import GenreCNN


def get_device():
    """NVIDIA/AMD(ROCm) GPU -> 'cuda', Apple Silicon Mac GPU -> 'mps', yoksa 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def parse_classes(text, all_classes):
    """'hip-hop, Rock' gibi bir metni veri setindeki gerçek sınıf isimlerine çevirir."""
    lookup = {c.lower(): c for c in all_classes}
    out = []
    for name in (s.strip() for s in text.split(",")):
        if not name:
            continue
        if name.lower() not in lookup:
            raise SystemExit(f"Bilinmeyen sınıf: '{name}'. Mevcut sınıflar: {all_classes}")
        out.append(lookup[name.lower()])
    if len(out) < 2:
        raise SystemExit("En az 2 sınıf seçmelisiniz.")
    return out


def select_classes(X, y, track, all_classes, keep):
    """Sadece 'keep' sınıflarının segmentlerini alır ve etiketleri 0..len(keep)-1 yapar."""
    remap = np.full(len(all_classes), -1, dtype=np.int64)
    for new_i, name in enumerate(keep):
        remap[all_classes.index(name)] = new_i
    sel = np.nonzero(remap[y] >= 0)[0]
    return np.asarray(X[sel]), remap[y[sel]], track[sel]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_meta(feat_dir):
    with open(os.path.join(feat_dir, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def load_split(feat_dir, split, meta, in_memory=True):
    """(X, y, track) döndürür. X float16; in_memory=False ise diskten okunur (memmap)."""
    n = meta["counts"][split]
    X = np.load(os.path.join(feat_dir, f"{split}_X.npy"), mmap_mode="r")[:n]
    if in_memory:
        X = np.ascontiguousarray(X)
    y = np.load(os.path.join(feat_dir, f"{split}_y.npy"))
    track = np.load(os.path.join(feat_dir, f"{split}_track.npy"))
    return X, y, track


class Normalizer:
    """Her mel bandı için (x - ortalama) / std. İstatistikler eğitim setinden gelir."""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 1, -1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 1, -1, 1)

    def __call__(self, x_np):
        x = torch.from_numpy(np.asarray(x_np, dtype=np.float32)).unsqueeze(1)  # (B,1,M,T)
        return (x - self.mean) / self.std


def iterate_batches(X, y, batch_size, shuffle):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        if shuffle:
            b = np.sort(b)  # memmap'ten okurken sıralı erişim daha hızlı
        yield X[b], y[b]


def spec_augment(x, n_freq_masks=2, max_f=15, n_time_masks=2, max_t=20):
    """SpecAugment: rastgele frekans ve zaman bantlarını sıfırlar (veri artırma)."""
    B, _, H, W = x.shape
    f_idx = torch.arange(H).view(1, 1, H, 1)
    t_idx = torch.arange(W).view(1, 1, 1, W)
    for _ in range(n_freq_masks):
        f = torch.randint(0, max_f + 1, (B, 1, 1, 1))
        f0 = (torch.rand(B, 1, 1, 1) * (H - f)).long()
        x = x.masked_fill((f_idx >= f0) & (f_idx < f0 + f), 0.0)
    for _ in range(n_time_masks):
        t = torch.randint(0, max_t + 1, (B, 1, 1, 1))
        t0 = (torch.rand(B, 1, 1, 1) * (W - t)).long()
        x = x.masked_fill((t_idx >= t0) & (t_idx < t0 + t), 0.0)
    return x


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GenreCNN(n_classes=len(ckpt["classes"]), base_channels=ckpt["width"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_proba(model, X, normalizer, batch_size=128, device="cpu"):
    """Segmentler için softmax olasılıklarını döndürür: (N, n_classes)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = normalizer(X[i:i + batch_size]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def track_level(proba, y, track):
    """Aynı parçanın segment olasılıklarının ortalamasını alarak parça bazında tahmin yapar."""
    tracks = np.unique(track)
    t_proba = np.stack([proba[track == t].mean(axis=0) for t in tracks])
    t_true = np.array([y[track == t][0] for t in tracks])
    return t_proba, t_true, tracks
