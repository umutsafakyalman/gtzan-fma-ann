"""1. ADIM: Ses dosyalarından mel-spektrogram çıkarıp diske kaydeder.

Kullanım:
    python extract_features.py --dataset gtzan
    python extract_features.py --dataset fma_small

Çıktı: features/<dataset>/{train,val,test}_{X,y,track}.npy + meta.json
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

import config
from audio_features import segment_melspecs
from datasets import list_dataset

SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="gtzan | fma_small | fma_medium")
    ap.add_argument("--data-root", default=None, help="varsayılan: data/<gtzan|fma>")
    ap.add_argument("--out", default=None, help="varsayılan: features/<dataset>")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    default_root = config.DATA_DIR / ("gtzan" if args.dataset == "gtzan" else "fma")
    data_root = args.data_root or default_root
    out_dir = config.FEATURE_DIR / args.dataset if args.out is None else args.out
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    items, classes = list_dataset(args.dataset, data_root)
    print(f"{len(items)} parça, {len(classes)} sınıf: {classes}")

    # Her split için üst sınıra göre disk üzerinde (memmap) dizi aç -> RAM'e sığmayan FMA için.
    arrays, counts = {}, {s: 0 for s in SPLITS}
    for s in SPLITS:
        n_max = sum(1 for it in items if it[2] == s) * config.SEGMENTS_PER_TRACK
        arrays[s] = {
            "X": np.lib.format.open_memmap(os.path.join(out_dir, f"{s}_X.npy"), mode="w+",
                                           dtype=np.float16,
                                           shape=(max(n_max, 1), config.N_MELS, config.N_FRAMES)),
            "y": np.zeros(max(n_max, 1), dtype=np.int64),
            "track": np.zeros(max(n_max, 1), dtype=np.int64),
        }

    failed = 0
    paths = [it[0] for it in items]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = ex.map(segment_melspecs, paths, chunksize=4)
        for (path, label, split, tid), mels in tqdm(zip(items, results), total=len(items),
                                                     desc="öznitelik çıkarılıyor"):
            if mels is None:
                failed += 1
                continue
            a, c = arrays[split], counts[split]
            n = len(mels)
            a["X"][c:c + n] = mels
            a["y"][c:c + n] = label
            a["track"][c:c + n] = tid
            counts[split] = c + n

    for s in SPLITS:
        arrays[s]["X"].flush()
        np.save(os.path.join(out_dir, f"{s}_y.npy"), arrays[s]["y"][:counts[s]])
        np.save(os.path.join(out_dir, f"{s}_track.npy"), arrays[s]["track"][:counts[s]])

    # Normalizasyon istatistikleri: SADECE eğitim verisinden, her mel bandı için ortalama/std
    X_tr = arrays["train"]["X"]
    n_tr = counts["train"]
    s1 = np.zeros(config.N_MELS, dtype=np.float64)
    s2 = np.zeros(config.N_MELS, dtype=np.float64)
    for i in range(0, n_tr, 2048):
        chunk = X_tr[i:min(i + 2048, n_tr)].astype(np.float64)
        s1 += chunk.sum(axis=(0, 2))
        s2 += (chunk ** 2).sum(axis=(0, 2))
    denom = max(n_tr * config.N_FRAMES, 1)
    mean = s1 / denom
    std = np.sqrt(np.maximum(s2 / denom - mean ** 2, 1e-8))

    meta = {
        "dataset": args.dataset,
        "classes": classes,
        "counts": counts,
        "failed_tracks": failed,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "sample_rate": config.SAMPLE_RATE,
        "segment_seconds": config.SEGMENT_SECONDS,
        "n_mels": config.N_MELS,
        "n_frames": config.N_FRAMES,
        "hop_length": config.HOP_LENGTH,
        "n_fft": config.N_FFT,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Bitti. Segment sayıları: {counts} | okunamayan parça: {failed}")
    print(f"Kaydedildi: {out_dir}")


if __name__ == "__main__":
    main()
