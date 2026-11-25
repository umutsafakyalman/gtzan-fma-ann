"""GTZAN ve FMA veri setlerindeki parçaları listeler ve train/val/test olarak böler.

Her fonksiyon şu formatta bir liste döndürür:
    [(ses_dosyasi_yolu, etiket_indeksi, split, parca_id), ...], sınıf_isimleri
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import GTZAN_GENRES, SEED

AUDIO_EXTS = {".wav", ".au", ".mp3"}


def list_gtzan(root: Path, val_ratio=0.1, test_ratio=0.1):
    """GTZAN: klasör yapısından bağımsız olarak 'blues.00000.wav' gibi dosyaları bulur.

    Hem Kaggle sürümü (genres_original/*.wav) hem orijinal sürüm (genres/*.au) çalışır.
    Bölme PARÇA bazında yapılır: aynı şarkının segmentleri hem train hem test'e düşmez
    (aksi halde veri sızıntısı olur ve doğruluk yapay olarak yüksek çıkar).
    """
    root = Path(root)
    paths, labels = [], []
    for idx, genre in enumerate(GTZAN_GENRES):
        files = sorted(p for p in root.rglob(f"{genre}.*") if p.suffix.lower() in AUDIO_EXTS)
        paths += files
        labels += [idx] * len(files)
    if not paths:
        raise FileNotFoundError(
            f"{root} altında GTZAN ses dosyası bulunamadı. "
            "Beklenen örnek: data/gtzan/genres/blues/blues.00000.au"
        )

    idx_all = np.arange(len(paths))
    labels = np.array(labels)
    train_idx, rest_idx = train_test_split(
        idx_all, test_size=val_ratio + test_ratio, stratify=labels, random_state=SEED)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=test_ratio / (val_ratio + test_ratio),
        stratify=labels[rest_idx], random_state=SEED)

    split_of = {}
    for name, ids in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for i in ids:
            split_of[i] = name

    items = [(str(paths[i]), int(labels[i]), split_of[i], i) for i in idx_all]
    return items, list(GTZAN_GENRES)


def list_fma(root: Path, subset="small"):
    """FMA: fma_metadata/tracks.csv'den tür (genre_top) ve resmi split bilgisini okur.

    Ses dosyaları: fma_<subset>/<ilk 3 hane>/<6 haneli id>.mp3
    FMA'nın kendi train/validation/test bölmesi kullanılır (sanatçı bazında ayrılmıştır).
    """
    root = Path(root)
    csvs = list(root.rglob("tracks.csv"))
    if not csvs:
        raise FileNotFoundError(f"{root} altında fma_metadata/tracks.csv bulunamadı.")
    audio_dirs = [p for p in root.rglob(f"fma_{subset}") if p.is_dir()]
    if not audio_dirs:
        raise FileNotFoundError(f"{root} altında fma_{subset} klasörü bulunamadı.")
    audio_dir = audio_dirs[0]

    tracks = pd.read_csv(csvs[0], index_col=0, header=[0, 1])
    # Alt kümeler iç içedir: small ⊂ medium ⊂ large (fma_medium.zip small'u da içerir)
    order = {"small": 0, "medium": 1, "large": 2}
    sub = tracks[tracks[("set", "subset")].map(order) <= order[subset]]
    sub = sub[sub[("track", "genre_top")].notna()]

    classes = sorted(sub[("track", "genre_top")].unique())
    cls_idx = {c: i for i, c in enumerate(classes)}
    split_map = {"training": "train", "validation": "val", "test": "test"}

    items, missing = [], 0
    for tid, row in sub.iterrows():
        tid_str = f"{tid:06d}"
        path = audio_dir / tid_str[:3] / f"{tid_str}.mp3"
        if not path.exists():
            missing += 1
            continue
        items.append((str(path), cls_idx[row[("track", "genre_top")]],
                      split_map[row[("set", "split")]], int(tid)))
    if missing:
        print(f"[uyarı] {missing} FMA dosyası diskte bulunamadı, atlandı.")
    return items, classes


def list_dataset(name: str, root: Path):
    if name == "gtzan":
        return list_gtzan(root)
    if name.startswith("fma"):
        subset = name.split("_", 1)[1] if "_" in name else "small"
        return list_fma(root, subset)
    raise ValueError(f"Bilinmeyen veri seti: {name}")
