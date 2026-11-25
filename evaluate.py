"""3. ADIM: Eğitilmiş modeli TEST seti üzerinde değerlendirir.

Kullanım:
    python evaluate.py --dataset gtzan

Çıktılar (outputs/<dataset>/):
    test_report.txt                 sınıf bazında precision / recall / F1
    confusion_matrix_segment.png    3 sn'lik segment bazında karmaşıklık matrisi
    confusion_matrix_track.png      30 sn'lik parça bazında (segment oylaması)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

import config
from common import (Normalizer, get_device, load_checkpoint, load_meta, load_split,
                    predict_proba, select_classes, track_level)


def plot_cm(cm, classes, title, path):
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 * len(classes) + 3, 0.9 * len(classes) + 2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Tahmin edilen"); ax.set_ylabel("Gerçek"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--features", default=None)
    ap.add_argument("--run", default=None, help="sonuç klasörü adı: outputs/<run>")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"],
                    help="best: en iyi doğrulama epoch'u, last: son epoch")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count())
    feat_dir = args.features or str(config.FEATURE_DIR / args.dataset)
    out_dir = args.out or str(config.OUTPUT_DIR / (args.run or args.dataset))
    device = get_device()

    meta = load_meta(feat_dir)
    model, ckpt = load_checkpoint(os.path.join(out_dir, f"{args.checkpoint}_model.pt"), device)
    tag = args.split if args.checkpoint == "best" else f"{args.split}_last"
    classes = ckpt["classes"]
    normalizer = Normalizer(ckpt["mean"], ckpt["std"])

    X, y, track = load_split(feat_dir, args.split, meta, in_memory=False)
    if classes != meta["classes"]:  # model sınıf alt kümesiyle eğitildiyse testi de filtrele
        X, y, track = select_classes(X, y, track, meta["classes"], classes)
    proba = predict_proba(model, X, normalizer, device=device)
    seg_pred = proba.argmax(1)
    t_proba, t_true, _ = track_level(proba, y, track)
    t_pred = t_proba.argmax(1)

    seg_acc = (seg_pred == y).mean()
    trk_acc = (t_pred == t_true).mean()
    labels = list(range(len(classes)))
    rep_seg = classification_report(y, seg_pred, labels=labels, target_names=classes,
                                     digits=3, zero_division=0)
    rep_trk = classification_report(t_true, t_pred, labels=labels, target_names=classes,
                                    digits=3, zero_division=0)

    text = (
        f"Veri seti: {args.dataset} | split: {args.split} | model: {args.checkpoint} (epoch {ckpt['epoch']})\n"
        f"Segment sayısı: {len(y)} | Parça sayısı: {len(t_true)}\n\n"
        f"SEGMENT (3 sn) doğruluğu : {seg_acc:.4f}\n"
        f"PARÇA (30 sn) doğruluğu  : {trk_acc:.4f}\n\n"
        f"--- Segment bazında rapor ---\n{rep_seg}\n"
        f"--- Parça bazında rapor ---\n{rep_trk}\n"
    )
    print(text)
    with open(os.path.join(out_dir, f"{tag}_report.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    plot_cm(confusion_matrix(y, seg_pred, labels=labels), classes,
            f"{args.dataset} [{args.checkpoint}, ep {ckpt['epoch']}] – segment bazında (acc={seg_acc:.3f})",
            os.path.join(out_dir, f"confusion_matrix_segment_{tag}.png"))
    plot_cm(confusion_matrix(t_true, t_pred, labels=labels), classes,
            f"{args.dataset} [{args.checkpoint}, ep {ckpt['epoch']}] – parça bazında (acc={trk_acc:.3f})",
            os.path.join(out_dir, f"confusion_matrix_track_{tag}.png"))
    print(f"Rapor ve grafikler kaydedildi: {out_dir}")


if __name__ == "__main__":
    main()
