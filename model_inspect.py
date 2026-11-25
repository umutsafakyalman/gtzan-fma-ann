"""Modelin içini gözlemleme: katman aktivasyonları, nöronlar, Grad-CAM, öğrenilmiş filtreler.

Tek bir 3 sn'lik segment modele verilir; ileri geçiş sırasında her ReLU katmanının çıktısı
kancalarla (forward hook) yakalanır. Bunlardan:
  - katman tablosu  : çıktı boyutu, nöron sayısı, aktif (>0) nöron oranı, parametre sayısı
  - feature map'ler : her evrişim bloğunda en çok aktifleşen filtrelerin çıktıları
  - Grad-CAM        : tahmin edilen sınıf için spektrogramın hangi bölgesi belirleyici oldu
  - gizli katman    : 256 nöronun aktivasyonu ve tahmine en çok katkı veren nöronlar
  - ilk katman      : öğrenilmiş 3x3 evrişim filtreleri
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import config


def _fmt(n):
    return f"{int(n):,}".replace(",", ".")


def _layers(model):
    """(isim, ReLU modülü, parametre taşıyan modüller) listesi."""
    out = []
    for b, block in enumerate(model.features, 1):
        net = block.net
        out.append((f"Blok {b} · Conv {b}a", net[2], [net[0], net[1]]))
        out.append((f"Blok {b} · Conv {b}b", net[5], [net[3], net[4]]))
    out.append(("Tam bağlı gizli katman", model.classifier[2], [model.classifier[1]]))
    return out


def _n_params(mods):
    return sum(p.numel() for m in mods for p in m.parameters())


def inspect_segment(model, normalizer, mel, classes, device):
    """Bir segment (N_MELS x N_FRAMES) için modelin iç durumunu çıkarır."""
    layers = _layers(model)
    acts, hooks = {}, []
    for name, mod, _ in layers:
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, n=name: acts.__setitem__(n, o)))

    x = normalizer(mel[None]).to(device)          # (1, 1, 128, 130)
    model.eval()
    try:
        with torch.enable_grad():
            logits = model(x)
            proba = torch.softmax(logits, dim=1)[0]
            pred = int(proba.argmax())
            last = acts["Blok 4 · Conv 4b"]          # (1, C, 16, 16) son evrişim çıktısı
            grad = torch.autograd.grad(logits[0, pred], last)[0]
    finally:
        for h in hooks:
            h.remove()

    # Grad-CAM: kanal ağırlığı = gradyanın uzamsal ortalaması
    w = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * last).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(config.N_MELS, config.N_FRAMES), mode="bilinear",
                        align_corners=False)[0, 0]
    cam = (cam / cam.max().clamp_min(1e-8)).detach().cpu().numpy()

    acts = {k: v.detach().cpu().numpy()[0] for k, v in acts.items()}
    return {
        "layers": layers,
        "acts": acts,
        "proba": proba.detach().cpu().numpy(),
        "pred": pred,
        "classes": classes,
        "cam": cam,
        "mel": np.asarray(mel, dtype=np.float32),
    }


def layer_table(model, res):
    """Katman bazında nöron / aktif nöron / parametre tablosu (Markdown)."""
    rows = ["| Katman | Çıktı boyutu | Nöron | Aktif nöron (>0) | Aktif % | Ort. aktivasyon | Parametre |",
            "|---|---|---:|---:|---:|---:|---:|"]
    tot_n = tot_a = tot_p = 0
    rows.append(f"| Girdi (log-mel) | 1×{config.N_MELS}×{config.N_FRAMES} | "
                f"{_fmt(config.N_MELS * config.N_FRAMES)} | – | – | – | 0 |")
    for name, _, mods in res["layers"]:
        a = res["acts"][name]
        n, act = a.size, int((a > 0).sum())
        p = _n_params(mods)
        shape = "×".join(str(s) for s in a.shape)
        rows.append(f"| {name} | {shape} | {_fmt(n)} | {_fmt(act)} | %{100 * act / n:.1f} | "
                    f"{a.mean():.3f} | {_fmt(p)} |")
        tot_n, tot_a, tot_p = tot_n + n, tot_a + act, tot_p + p
    k = len(res["classes"])
    p_out = _n_params([model.classifier[4]])
    rows.append(f"| Çıktı (softmax) | {k} | {k} | – | – | – | {_fmt(p_out)} |")
    tot_n, tot_p = tot_n + k, tot_p + p_out
    rows.append(f"| **Toplam** | | **{_fmt(tot_n)}** | **{_fmt(tot_a)}** | "
                f"**%{100 * tot_a / tot_n:.1f}** | | **{_fmt(tot_p)}** |")
    return "\n".join(rows)


def featuremap_plot(res, top_k=6):
    """Her bloğun son evrişim çıktısı: katman ortalaması + en aktif top_k filtre."""
    names = [f"Blok {b} · Conv {b}b" for b in range(1, 5)]
    fig, axes = plt.subplots(4, top_k + 1, figsize=(2.0 * (top_k + 1), 8.2))
    for r, name in enumerate(names):
        a = res["acts"][name]                       # (C, H, W)
        order = np.argsort(a.mean(axis=(1, 2)))[::-1][:top_k]
        panels = [("katman ortalaması", a.mean(axis=0))] + \
                 [(f"filtre #{c}", a[c]) for c in order]
        for col, (title, img) in enumerate(panels):
            ax = axes[r, col]
            ax.imshow(img, origin="lower", aspect="auto", cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=8)
            if col == 0:
                ax.set_ylabel(f"Blok {r + 1}\n{a.shape[0]}×{a.shape[1]}×{a.shape[2]}", fontsize=9)
    fig.suptitle("Evrişim bloklarının çıktıları (feature map) — her satırda en çok aktifleşen filtreler\n"
                 "dikey eksen: frekans (mel), yatay eksen: zaman; derin bloklarda çözünürlük azalır",
                 fontsize=10)
    fig.tight_layout()
    return fig


def gradcam_plot(res):
    """Spektrogram ve Grad-CAM ısı haritası."""
    mel, cam = res["mel"], res["cam"]
    cls = res["classes"][res["pred"]]
    ext = [0, config.SEGMENT_SECONDS, 0, config.N_MELS]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].imshow(mel, origin="lower", aspect="auto", cmap="magma", extent=ext)
    ax[0].set_title("Girdi: log-mel spektrogram")
    ax[1].imshow(mel, origin="lower", aspect="auto", cmap="gray", extent=ext)
    im = ax[1].imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.5, extent=ext,
                      vmin=0, vmax=1)
    ax[1].set_title(f"Grad-CAM: '{cls}' kararında belirleyici bölgeler")
    for a in ax:
        a.set_xlabel("zaman (sn)"); a.set_ylabel("mel bandı (düşük → yüksek frekans)")
    fig.colorbar(im, ax=ax[1], fraction=0.04, label="önem")
    fig.tight_layout()
    return fig


def dense_plot(model, res, top_k=12):
    """256 gizli nöronun aktivasyonu, tahmine katkı veren nöronlar ve çıktı katmanı."""
    h = res["acts"]["Tam bağlı gizli katman"]       # (256,)
    W = model.classifier[4].weight.detach().cpu().numpy()   # (k, 256)
    pred, classes = res["pred"], res["classes"]
    contrib = h * W[pred]
    order = np.argsort(np.abs(contrib))[::-1][:top_k][::-1]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1.3, 1]})
    side = int(np.ceil(np.sqrt(len(h))))
    grid = np.full(side * side, np.nan); grid[:len(h)] = h
    im = ax[0].imshow(grid.reshape(side, side), cmap="viridis")
    ax[0].set_title(f"Gizli katman: {len(h)} nöron\n(aktif: {(h > 0).sum()}, "
                    f"sönük: {(h <= 0).sum()})", fontsize=10)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    colors = ["tab:green" if c > 0 else "tab:red" for c in contrib[order]]
    ax[1].barh([f"nöron #{i}" for i in order], contrib[order], color=colors)
    ax[1].axvline(0, color="gray", lw=0.8)
    ax[1].set_title(f"'{classes[pred]}' kararına en çok katkı veren {top_k} nöron\n"
                    "(aktivasyon × çıktı ağırlığı; yeşil: destekliyor, kırmızı: karşı)",
                    fontsize=10)
    ax[1].tick_params(axis="y", labelsize=8)

    p = res["proba"]
    ax[2].bar(classes, p, color=["tab:orange" if i == pred else "tab:blue"
                                 for i in range(len(classes))])
    ax[2].set_ylim(0, 1); ax[2].set_title("Çıktı katmanı (softmax olasılıkları)", fontsize=10)
    ax[2].tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    return fig


def filters_plot(model):
    """İlk evrişim katmanının öğrenilmiş 3x3 filtreleri (girdi: tek kanal spektrogram)."""
    w = model.features[0].net[0].weight.detach().cpu().numpy()[:, 0]   # (C, 3, 3)
    n = len(w)
    cols = 8
    rows = int(np.ceil(n / cols))
    lim = np.abs(w).max()
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.2))
    for i, ax in enumerate(np.ravel(axes)):
        ax.set_xticks([]); ax.set_yticks([])
        if i < n:
            ax.imshow(w[i], cmap="coolwarm", vmin=-lim, vmax=lim, origin="lower")
            ax.set_title(f"#{i}", fontsize=7)
        else:
            ax.axis("off")
    fig.suptitle(f"İlk katmanın öğrenilmiş {n} filtresi (3×3; kırmızı +, mavi −)", fontsize=10)
    fig.tight_layout()
    return fig
