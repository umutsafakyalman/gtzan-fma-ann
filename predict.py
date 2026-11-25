"""Herhangi bir ses dosyasının (wav/mp3/au...) müzik türünü tahmin eder.

Kullanım:
    python predict.py sarki.mp3 --dataset gtzan
    python predict.py sarki.mp3 --model outputs/fma_small/best_model.pt
"""
import argparse
import os

import numpy as np

import config
from audio_features import segment_melspecs
from common import Normalizer, get_device, load_checkpoint, predict_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="ses dosyası yolu")
    ap.add_argument("--dataset", "--run", dest="dataset", default="gtzan",
                    help="hangi model: outputs/<bu isim>/best_model.pt")
    ap.add_argument("--model", default=None, help="model dosyasının tam yolu")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    model_path = args.model or os.path.join(config.OUTPUT_DIR, args.dataset, "best_model.pt")
    device = get_device()
    model, ckpt = load_checkpoint(model_path, device)
    classes = ckpt["classes"]

    # Şarkının tamamı 3 sn'lik parçalara bölünür, her parça sınıflandırılır, olasılıklar ortalanır
    mels = segment_melspecs(args.audio, max_segments=None, max_duration=None)
    if mels is None:
        raise SystemExit("Dosya okunamadı veya 3 saniyeden kısa.")
    proba = predict_proba(model, mels, Normalizer(ckpt["mean"], ckpt["std"]),
                          device=device).mean(axis=0)

    print(f"\n{os.path.basename(args.audio)}  ({len(mels)} segment x {config.SEGMENT_SECONDS:.0f} sn)")
    for i in np.argsort(proba)[::-1][:args.top]:
        print(f"  {classes[i]:<15s} %{100 * proba[i]:5.1f}  {'#' * int(40 * proba[i])}")
    print(f"\nTahmin: {classes[int(proba.argmax())].upper()}")


if __name__ == "__main__":
    main()
