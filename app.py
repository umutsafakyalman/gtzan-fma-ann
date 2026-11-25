"""Web arayüzü: şarkı yükle / mikrofondan kaydet -> müzik türü tahmini.

Kullanım:
    python app.py
Tarayıcıda http://127.0.0.1:7860 adresi açılır.
"""
import os
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import model_inspect as mi
from audio_features import segment_melspecs
from common import Normalizer, get_device, load_checkpoint, predict_proba

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
DEVICE = get_device()

# outputs/<dataset>/best_model.pt dosyalarını bul
MODELS = {p.parent.name: p for p in sorted(config.OUTPUT_DIR.glob("*/best_model.pt"))}
_cache = {}


def get_model(name):
    if name not in _cache:
        model, ckpt = load_checkpoint(MODELS[name], DEVICE)
        _cache[name] = (model, ckpt, Normalizer(ckpt["mean"], ckpt["std"]))
    return _cache[name]


def timeline_plot(seg_proba, classes):
    """Şarkı boyunca her 3 sn'lik segment için tür olasılıkları (yığılmış alan grafiği)."""
    n = len(seg_proba)
    t = np.arange(n) * config.SEGMENT_SECONDS + config.SEGMENT_SECONDS / 2
    fig, ax = plt.subplots(figsize=(9, 3.2))
    cmap = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")
    if n == 1:
        ax.bar(classes, seg_proba[0], color=[cmap(i) for i in range(len(classes))])
        ax.set_ylabel("olasılık")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.stackplot(t, seg_proba.T, labels=classes, colors=[cmap(i) for i in range(len(classes))])
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, 1)
        ax.set_xlabel("zaman (sn)"); ax.set_ylabel("olasılık")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    ax.set_title("Zamana göre tür olasılıkları (her 3 sn'lik segment)")
    fig.tight_layout()
    return fig


def spectrogram_plot(mels):
    """İlk 30 saniyenin log-mel spektrogramı (modelin gördüğü girdi)."""
    spec = np.concatenate(list(mels[:10]), axis=1).astype(np.float32)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    dur = spec.shape[1] * config.HOP_LENGTH / config.SAMPLE_RATE
    im = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, dur, 0, config.N_MELS])
    for k in range(1, min(len(mels), 10)):
        ax.axvline(k * config.SEGMENT_SECONDS, color="white", lw=0.6, alpha=0.6)
    ax.set_xlabel("zaman (sn)"); ax.set_ylabel("mel bandı")
    ax.set_title("Log-mel spektrogram (beyaz çizgiler: 3 sn'lik segment sınırları)")
    fig.colorbar(im, ax=ax, label="dB", fraction=0.03)
    fig.tight_layout()
    return fig


def classify(audio_path, model_name):
    if not audio_path:
        raise gr.Error("Önce bir ses dosyası yükleyin ya da kayıt yapın.")
    if model_name not in MODELS:
        raise gr.Error("Model bulunamadı. Önce train.py ile model eğitin.")
    model, ckpt, norm = get_model(model_name)
    classes = ckpt["classes"]

    mels = segment_melspecs(audio_path, max_segments=None, max_duration=300.0)
    if mels is None:
        raise gr.Error("Ses okunamadı ya da 3 saniyeden kısa.")

    seg_proba = predict_proba(model, mels, norm, device=DEVICE)
    proba = seg_proba.mean(axis=0)
    best = int(proba.argmax())
    votes = np.bincount(seg_proba.argmax(1), minlength=len(classes))

    summary = (
        f"## Tahmin: **{classes[best].upper()}**  (%{100 * proba[best]:.1f})\n"
        f"- İncelenen süre: {len(mels) * config.SEGMENT_SECONDS:.0f} sn "
        f"({len(mels)} segment × {config.SEGMENT_SECONDS:.0f} sn)\n"
        f"- Segment oyları: {votes[best]}/{len(mels)} segment **{classes[best]}** dedi\n"
        f"- Model: `{model_name}` (epoch {ckpt['epoch']}, doğrulama doğruluğu "
        f"%{100 * ckpt['val_acc']:.1f})"
    )
    label = {c: float(p) for c, p in zip(classes, proba)}
    return summary, label, timeline_plot(seg_proba, classes), spectrogram_plot(mels)


def inspect(audio_path, model_name, seg_no):
    """'Modelin içi' sekmesi: seçilen 3 sn'lik segment için katmanlar ve nöronlar."""
    if not audio_path:
        raise gr.Error("Önce bir ses dosyası yükleyin ya da kayıt yapın.")
    if model_name not in MODELS:
        raise gr.Error("Model bulunamadı. Önce train.py ile model eğitin.")
    model, ckpt, norm = get_model(model_name)
    mels = segment_melspecs(audio_path, max_segments=None, max_duration=300.0)
    if mels is None:
        raise gr.Error("Ses okunamadı ya da 3 saniyeden kısa.")
    i = min(max(int(seg_no), 1), len(mels)) - 1

    res = mi.inspect_segment(model, norm, mels[i], ckpt["classes"], DEVICE)
    cls = ckpt["classes"][res["pred"]]
    s0 = i * config.SEGMENT_SECONDS
    note = "" if int(seg_no) - 1 == i else f" (şarkıda {len(mels)} segment var, sonuncusu gösteriliyor)"
    header = (
        f"### Segment {i + 1}/{len(mels)} ({s0:.0f}–{s0 + config.SEGMENT_SECONDS:.0f}. sn) → "
        f"**{cls}** (%{100 * res['proba'][res['pred']]:.1f}){note}\n"
        "Aşağıdaki tablo bu segment modelden geçerken her katmanda kaç nöronun aktifleştiğini "
        "(ReLU çıktısı > 0) gösterir. *Nöron* = katman çıktısındaki her bir değer.\n\n"
        + mi.layer_table(model, res)
    )
    return (header, mi.gradcam_plot(res), mi.featuremap_plot(res),
            mi.dense_plot(model, res), mi.filters_plot(model))


def example_files():
    """Örnek olarak GTZAN TEST setinden (modelin hiç görmediği) her türden bir parça."""
    try:
        from datasets import list_gtzan
        items, classes = list_gtzan(config.DATA_DIR / "gtzan")
    except Exception:
        return []
    seen, out = set(), []
    for path, label, split, _ in items:
        if split == "test" and label not in seen and Path(path).exists():
            seen.add(label)
            out.append([path])
    return out


with gr.Blocks(title="Müzik Türü Sınıflandırma") as demo:
    gr.Markdown(
        "# 🎵 Derin Öğrenme ile Müzik Türü Sınıflandırma\n"
        "Bir şarkı yükleyin veya mikrofondan kaydedin. Ses 3 saniyelik parçalara bölünür, "
        "her parçanın log-mel spektrogramı sıfırdan eğitilmiş CNN'e verilir ve "
        "olasılıkların ortalaması alınarak şarkının türü tahmin edilir."
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                             label="Ses dosyası (wav, mp3, au, ...)")
            model_dd = gr.Dropdown(choices=list(MODELS), value=next(iter(MODELS), None),
                                   label="Model (eğitildiği veri seti)")
        with gr.Column(scale=1):
            ex = example_files()
            if ex:
                gr.Examples(ex, inputs=[audio],
                            label="Örnekler (GTZAN test seti, modelin görmediği parçalar)")

    with gr.Tabs():
        with gr.Tab("Tahmin"):
            btn = gr.Button("Türü tahmin et", variant="primary")
            with gr.Row():
                summary = gr.Markdown()
                label = gr.Label(num_top_classes=5, label="Tür olasılıkları")
            timeline = gr.Plot(label="Zamana göre tahmin")
            spec = gr.Plot(label="Spektrogram")

        with gr.Tab("Modelin içi (katmanlar ve nöronlar)"):
            gr.Markdown(
                "Şarkının seçilen 3 saniyelik segmenti modelden geçirilir ve her katmanda olanlar "
                "gösterilir: aktif nöron sayıları, Grad-CAM (model nereye bakıyor), evrişim "
                "filtrelerinin çıktıları, gizli katmandaki 256 nöron ve ilk katmanın öğrenilmiş filtreleri."
            )
            with gr.Row():
                seg_no = gr.Slider(1, 100, value=1, step=1, label="Segment no (1 = ilk 3 sn)")
                ibtn = gr.Button("Modelin içini göster", variant="primary")
            itable = gr.Markdown()
            icam = gr.Plot(label="Grad-CAM — model nereye bakıyor?")
            ifmap = gr.Plot(label="Evrişim katmanlarının çıktıları (feature map)")
            idense = gr.Plot(label="Gizli katman nöronları ve çıktı")
            ifilt = gr.Plot(label="İlk katmanın öğrenilmiş filtreleri")

    btn.click(classify, inputs=[audio, model_dd], outputs=[summary, label, timeline, spec])
    ibtn.click(inspect, inputs=[audio, model_dd, seg_no],
               outputs=[itable, icam, ifmap, idense, ifilt])

if __name__ == "__main__":
    if not MODELS:
        print("Uyarı: outputs/*/best_model.pt bulunamadı. Önce train.py ile model eğitin.")
    import sys
    demo.launch(inbrowser="--no-browser" not in sys.argv, allowed_paths=[str(config.DATA_DIR)])
