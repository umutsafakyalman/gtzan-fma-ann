# Derin Öğrenme ile Müzik Türü Sınıflandırma

Ses özelliklerinden (log-mel spektrogram) müzik türünü tespit eden, **sıfırdan tasarlanmış bir
evrişimli sinir ağı (CNN)**. Hazır ya da önceden eğitilmiş bir model kullanılmaz; ağın bütün
katmanları `model.py` içinde tanımlanır ve rastgele ağırlıklarla başlayıp eğitilir.

Veri setleri: **GTZAN** (10 tür, 1000 parça) ve **FMA** (Free Music Archive, `fma_small`: 8 tür, 8000 parça).

## Yöntem

```
ses dosyası (30 sn)
   └─► 22050 Hz mono'ya dönüştür
        └─► 10 adet 3 sn'lik segmente böl
             └─► her segmentten 128 x 130 log-mel spektrogram (dB)
                  └─► mel bandı bazında normalizasyon (eğitim setinin ortalama/std'si)
                       └─► GenreCNN ──► softmax ──► segment olasılıkları
                                                     └─► ortalama ──► parça tahmini
```

### Model mimarisi (`model.py` → `GenreCNN`)

| Katman | Çıktı boyutu |
|---|---|
| Girdi (log-mel spektrogram) | 1 × 128 × 130 |
| ConvBlock 1: [Conv3×3 → BN → ReLU] ×2 → MaxPool 2×2 → Dropout | 32 × 64 × 65 |
| ConvBlock 2 | 64 × 32 × 32 |
| ConvBlock 3 | 128 × 16 × 16 |
| ConvBlock 4 | 256 × 8 × 8 |
| Global ortalama + global maksimum havuzlama (birleştirme) | 512 |
| Dropout 0.5 → Dense 256 → ReLU → Dropout 0.5 | 256 |
| Dense (softmax) | sınıf sayısı |

Yaklaşık 1,3 milyon eğitilebilir parametre (`--width 32`). `--width 16` ile ~360 bin parametre, 2 kat hız.

### Eğitim ayrıntıları
- **Kayıp:** Cross-entropy, label smoothing 0.1, sınıf dengesizliğine karşı sınıf ağırlıkları
- **Optimizasyon:** AdamW (lr 1e-3, weight decay 1e-4), doğrulama doğruluğu düşmezse lr yarıya iner
- **Düzenlileştirme:** BatchNorm, Dropout, **SpecAugment** (rastgele frekans/zaman maskeleme)
- **Erken durdurma:** 10 epoch iyileşme olmazsa durur, en iyi model saklanır
- **Bölme parça bazındadır:** Aynı şarkının segmentleri hem eğitim hem test setine düşmez
  (veri sızıntısı önlenir). GTZAN için %80/%10/%10 tabakalı bölme, FMA için veri setinin kendi
  resmi `training/validation/test` bölmesi kullanılır.
- **İki seviyede değerlendirme:** 3 sn'lik segment doğruluğu ve segment olasılıklarının ortalaması
  alınarak 30 sn'lik parça doğruluğu.

## Kurulum

```powershell
pip install -r requirements.txt
```

GPU varsa PyTorch otomatik olarak CUDA kullanır; yoksa CPU'da çalışır.

## Mac'te çalıştırma (Apple Silicon GPU)

Kod cihazı kendisi seçer: NVIDIA/AMD GPU → `cuda`, **Apple Silicon (M1/M2/M3/M4) → `mps`**, yoksa `cpu`.
Eğitim çıktısındaki `cihaz:` satırında `mps` görmelisiniz. Intel Mac'lerde `cpu` kullanılır.

Repoda yalnızca kod ve `outputs/` (eğitilmiş GTZAN modeli) bulunur. `data/` ve `features/`
büyük olduğu için `.gitignore` ile dışarıda bırakılmıştır; Mac'te yeniden indirilip üretilir:

```bash
python download_data.py gtzan && python extract_features.py --dataset gtzan
python download_data.py fma_small && python extract_features.py --dataset fma_small
```

Kurulum (Terminal):

```bash
cd ~/Desktop/"erdal proje"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"   # True olmalı
```

### Sınıf seçerek eğitim (`--classes`)
Sadece belirli türlerle eğitmek için `--classes` verin, sonuçlar `outputs/<run>/` klasörüne gider.
Test ve tahmin aynı sınıflarla otomatik yapılır.

```bash
# FMA small, sınırları belirgin 4 tür (öznitelikler hazır, indirme yok)
python train.py --dataset fma_small --classes "Hip-Hop,Rock,Electronic,Folk" --run fma_4sinif
python evaluate.py --dataset fma_small --run fma_4sinif

# FMA small, 8 türün hepsi (standart kurulum)
python train.py --dataset fma_small
python evaluate.py --dataset fma_small

# FMA medium, 6 belirgin tür (önce ~22 GB indirme + öznitelik çıkarma gerekir)
python download_data.py fma_medium
python extract_features.py --dataset fma_medium
python train.py --dataset fma_medium --classes "Classical,Electronic,Folk,Hip-Hop,Jazz,Rock" --run fma_medium_6sinif
python evaluate.py --dataset fma_medium --run fma_medium_6sinif
```

Eğitim yarıda kalırsa aynı komutun sonuna `--resume` ekleyin. Arayüz (`python app.py`)
`outputs/` altındaki bütün modelleri listeler.

## Veri setleri

İki yol var: indirme betiği ya da elle indirip klasöre koymak.

```powershell
python download_data.py gtzan        # ~1,2 GB
python download_data.py fma_small    # ~7,7 GB + 360 MB metadata
```

Elle indirirseniz klasör yapısı şöyle olmalı (GTZAN'ın Kaggle sürümü `genres_original/*.wav` de olur):

```
data/
├── gtzan/
│   └── genres/blues/blues.00000.au ... rock/rock.00099.au
└── fma/
    ├── fma_metadata/tracks.csv
    └── fma_small/000/000002.mp3 ...
```

## Çalıştırma

```powershell
# 1) Öznitelik çıkarma (bir kere yapılır)
python extract_features.py --dataset gtzan
python extract_features.py --dataset fma_small

# 2) Eğitim
python train.py --dataset gtzan
python train.py --dataset fma_small --width 16 --epochs 30

#    yarıda kalırsa kaldığı epoch'tan devam:
python train.py --dataset gtzan --resume

# 3) Test seti değerlendirmesi (en iyi doğrulama epoch'u / son epoch)
python evaluate.py --dataset gtzan
python evaluate.py --dataset gtzan --checkpoint last
python evaluate.py --dataset fma_small

# 4) Kendi şarkınızı sınıflandırma (komut satırı)
python predict.py "sarki.mp3" --dataset gtzan

# 5) Web arayüzü
python app.py
```

### Web arayüzü (`app.py`)
`python app.py` çalıştırınca tarayıcıda **http://127.0.0.1:7860** açılır (kapatmak için terminalde Ctrl+C).
- Şarkı yükleyin (wav, mp3, au...) ya da mikrofon simgesiyle kayıt yapın, **Türü tahmin et**'e basın.
- Sağda tahmin edilen tür ve ilk 5 türün olasılıkları, altta şarkı boyunca her 3 sn'lik segmentin
  tür olasılıkları ve modelin gördüğü log-mel spektrogram gösterilir.
- **Örnekler** listesindeki dosyalar GTZAN test setindendir; model bunları eğitimde hiç görmedi.
- Birden fazla model eğitildiyse (ör. `gtzan` ve `fma_small`) açılır listeden seçilebilir.
- En fazla ilk 5 dakika incelenir.

Sonuçlar `outputs/<dataset>/` klasörüne yazılır:

| Dosya | İçerik |
|---|---|
| `best_model.pt` | doğrulama doğruluğu en yüksek epoch'un ağırlıkları |
| `last_model.pt` | son epoch'un ağırlıkları + devam ettirmek için optimizer durumu |
| `history.csv`, `training_curves.png` | epoch bazında kayıp / doğruluk eğrileri |
| `test_report.txt` | precision, recall, F1 (segment ve parça bazında) |
| `confusion_matrix_*_test.png` | karmaşıklık matrisleri |

### Süre (bu bilgisayarda, 12 çekirdekli CPU, GPU yok)
| | GTZAN (~8000 eğitim segmenti) | FMA small (~64000 eğitim segmenti) |
|---|---|---|
| `--width 32` | ~2,2 dk / epoch | ~18 dk / epoch |
| `--width 16` | ~1,1 dk / epoch | ~9 dk / epoch |

Öznitelik dosyaları: GTZAN ≈ 330 MB, FMA small ≈ 2,7 GB disk. FMA için RAM yetmezse `train.py`'ye
`--disk` ekleyin.

## Dosyalar

| Dosya | Görev |
|---|---|
| `config.py` | ortak ayarlar (örnekleme hızı, segment süresi, mel sayısı...) |
| `datasets.py` | GTZAN/FMA parçalarını listeleme ve train/val/test bölme |
| `audio_features.py` | segmentleme + log-mel spektrogram çıkarma |
| `extract_features.py` | tüm veri setinin özniteliklerini diske yazma |
| `model.py` | **sıfırdan CNN mimarisi** |
| `common.py` | normalizasyon, SpecAugment, parça bazında oylama vb. |
| `train.py` / `evaluate.py` / `predict.py` | eğitim / test / tek dosya tahmini |
| `app.py` | web arayüzü (Gradio) |
| `download_data.py` | veri setlerini indirme |

## Notlar
- GTZAN'daki `jazz.00054.wav` dosyası bozuktur; otomatik olarak atlanır.
- FMA small'da birkaç bozuk/kısa mp3 vardır; bunlar da atlanır.
- GTZAN'da aynı sanatçının parçaları farklı setlere düşebildiği için literatürde rastgele bölmeyle
  elde edilen sonuçların biraz iyimser olduğu bilinir; FMA'nın resmi bölmesi sanatçı bazındadır.
  Bu yüzden FMA doğrulukları genellikle GTZAN'dan belirgin şekilde düşük çıkar ve bu beklenen bir durumdur.
