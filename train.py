"""2. ADIM: Kendi CNN modelimizi sıfırdan eğitir.

Kullanım:
    python train.py --dataset gtzan
    python train.py --dataset fma_small --width 16 --epochs 30

Çıktılar (outputs/<dataset>/):
    best_model.pt        en iyi doğrulama doğruluğuna sahip model
    history.csv          epoch başına kayıp / doğruluk
    training_curves.png  öğrenme eğrileri
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import config
from common import (Normalizer, get_device, iterate_batches, load_meta, load_split,
                    parse_classes, predict_proba, select_classes, set_seed,
                    spec_augment, track_level)
from model import GenreCNN, count_parameters


def evaluate(model, X, y, track, normalizer, criterion, device):
    proba = predict_proba(model, X, normalizer, device=device)
    loss = criterion(torch.log(torch.from_numpy(proba).clamp_min(1e-8)),
                     torch.from_numpy(y)).item() if len(y) else float("nan")
    seg_acc = float((proba.argmax(1) == y).mean())
    t_proba, t_true, _ = track_level(proba, y, track)
    trk_acc = float((t_proba.argmax(1) == t_true).mean())
    return loss, seg_acc, trk_acc


def plot_history(hist, path):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="eğitim")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="doğrulama")
    ax[0].set_title("Kayıp (loss)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="eğitim (segment)")
    ax[1].plot(ep, [h["val_acc"] for h in hist], label="doğrulama (segment)")
    ax[1].plot(ep, [h["val_track_acc"] for h in hist], label="doğrulama (parça)")
    ax[1].set_title("Doğruluk"); ax[1].set_xlabel("epoch"); ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="gtzan | fma_small | fma_medium")
    ap.add_argument("--features", default=None, help="varsayılan: features/<dataset>")
    ap.add_argument("--run", default=None,
                    help="sonuç klasörü adı: outputs/<run> (varsayılan: veri seti adı)")
    ap.add_argument("--out", default=None, help="sonuç klasörünün tam yolu (--run yerine)")
    ap.add_argument("--classes", default=None,
                    help='sadece bu sınıflarla eğit, ör. "Hip-Hop,Rock,Electronic,Folk"')
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=32, help="ilk katmandaki kanal sayısı")
    ap.add_argument("--patience", type=int, default=10, help="erken durdurma sabrı")
    ap.add_argument("--no-augment", action="store_true", help="SpecAugment kapat")
    ap.add_argument("--disk", action="store_true", help="veriyi RAM'e yükleme (büyük FMA için)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--resume", action="store_true",
                    help="last_model.pt'den kaldığı yerden devam et")
    args = ap.parse_args()

    set_seed(config.SEED)
    torch.set_num_threads(args.threads)
    device = get_device()

    feat_dir = args.features or str(config.FEATURE_DIR / args.dataset)
    out_dir = args.out or str(config.OUTPUT_DIR / (args.run or args.dataset))
    os.makedirs(out_dir, exist_ok=True)

    meta = load_meta(feat_dir)
    classes = meta["classes"]
    in_mem = not args.disk and not args.classes  # sınıf seçiliyse sadece seçilenler RAM'e alınır
    X_tr, y_tr, t_tr = load_split(feat_dir, "train", meta, in_memory=in_mem)
    X_va, y_va, t_va = load_split(feat_dir, "val", meta, in_memory=in_mem)
    if args.classes:
        keep = parse_classes(args.classes, classes)
        X_tr, y_tr, t_tr = select_classes(X_tr, y_tr, t_tr, classes, keep)
        X_va, y_va, t_va = select_classes(X_va, y_va, t_va, classes, keep)
        classes = keep
    normalizer = Normalizer(meta["mean"], meta["std"])
    print(f"Sınıflar ({len(classes)}): {classes}")
    print(f"Eğitim segmenti: {len(X_tr)} | Doğrulama segmenti: {len(X_va)} | cihaz: {device}")

    model = GenreCNN(n_classes=len(classes), base_channels=args.width).to(device)
    print(f"Model parametre sayısı: {count_parameters(model):,}")

    # Sınıf dengesizliği varsa (ör. FMA medium) kayıp fonksiyonunu ağırlıklandır
    counts = np.bincount(y_tr, minlength=len(classes)).astype(np.float64)
    class_w = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.1)
    eval_criterion = nn.NLLLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=3)

    best_acc, bad_epochs, history, start_epoch = -1.0, 0, [], 1
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    last_path = os.path.join(out_dir, "last_model.pt")

    def make_ckpt(epoch, val_acc):
        return {
            "model": model.state_dict(),
            "classes": classes,
            "width": args.width,
            "mean": meta["mean"],
            "std": meta["std"],
            "epoch": epoch,
            "val_acc": val_acc,
            "dataset": args.dataset,
        }

    if args.resume:
        if not os.path.exists(last_path):
            raise SystemExit(f"Devam edilecek kayıt yok: {last_path}")
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        best_acc, bad_epochs, history = st["best_acc"], st["bad_epochs"], st["history"]
        start_epoch = st["epoch"] + 1
        torch.set_rng_state(st["torch_rng"])
        np.random.set_state(st["numpy_rng"])
        print(f"Devam ediliyor: epoch {start_epoch} (şimdiye kadarki en iyi val acc {best_acc:.3f})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        for xb_np, yb_np in iterate_batches(X_tr, y_tr, args.batch_size, shuffle=True):
            xb = normalizer(xb_np)
            if not args.no_augment:
                xb = spec_augment(xb)
            xb, yb = xb.to(device), torch.from_numpy(yb_np).to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item() * len(yb)
            tot_correct += (logits.argmax(1) == yb).sum().item()
            tot_n += len(yb)

        val_loss, val_acc, val_trk = evaluate(model, X_va, y_va, t_va, normalizer,
                                              eval_criterion, device)
        scheduler.step(val_acc)
        rec = {
            "epoch": epoch,
            "train_loss": tot_loss / tot_n,
            "train_acc": tot_correct / tot_n,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_track_acc": val_trk,
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        history.append(rec)
        print(f"Epoch {epoch:3d} | train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} (parça {val_trk:.3f}) | "
              f"lr {rec['lr']:.1e} | {rec['sec']:.0f}s")

        if val_acc > best_acc:
            best_acc, bad_epochs = val_acc, 0
            torch.save(make_ckpt(epoch, val_acc), ckpt_path)
            print(f"   -> en iyi model kaydedildi (val acc {val_acc:.3f})")
        else:
            bad_epochs += 1

        # Son epoch her zaman kaydedilir (+ devam ettirmek için optimizer durumu)
        last = make_ckpt(epoch, val_acc)
        last.update({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "bad_epochs": bad_epochs,
            "history": history,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        })
        torch.save(last, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)  # yarım kalmış dosya oluşmasın

        with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
        plot_history(history, os.path.join(out_dir, "training_curves.png"))

        if bad_epochs >= args.patience:
            print(f"Erken durdurma: {args.patience} epoch boyunca iyileşme yok.")
            break

    print(f"Eğitim bitti. En iyi doğrulama doğruluğu: {best_acc:.3f}")
    print(f"Model: {ckpt_path}")
    run_arg = f" --run {args.run}" if args.run else ""
    print(f"Test sonuçları için: python evaluate.py --dataset {args.dataset}{run_arg}")


if __name__ == "__main__":
    main()
