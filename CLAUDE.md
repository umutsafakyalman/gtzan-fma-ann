# Proje bağlamı (Windows'taki oturumdan devir)

Kullanıcıyla **Türkçe** konuş. Kısa ve net cevapları tercih ediyor. Üniversite projesi.

## Konu
"Derin Öğrenme ile Müzik Türü Sınıflandırma: ses özelliklerinden müzik türü tespiti".
Veri setleri: **GTZAN ve FMA**. Şart: **kendi sinir ağını sıfırdan kurmak** (hazır/önceden
eğitilmiş model yok). Kod ve kullanım ayrıntıları `README.md` içinde.

## Mimari özeti
- 30 sn parça → 10 × 3 sn segment → 128×130 log-mel spektrogram (`audio_features.py`, `config.py`)
- `model.py` → `GenreCNN`: 4 × [Conv-BN-ReLU ×2, MaxPool, Dropout] + global avg+max pool + FC.
  `--width 32` ≈ 1,3M parametre, `--width 16` ≈ 360k.
- Eğitim: AdamW, label smoothing 0.1, sınıf ağırlıkları, SpecAugment, ReduceLROnPlateau, erken durdurma.
  Her epoch `last_model.pt` (devam için `--resume`), en iyisi `best_model.pt`.
- Bölme PARÇA bazında (veri sızıntısı yok, kontrol edildi). GTZAN: 80/10/10 tabakalı, seed 42.
  FMA: resmi training/validation/test bölmesi.
- Değerlendirme hem segment hem parça (segment olasılıkları ortalaması) bazında.
- `get_device()` cuda → mps → cpu seçer. Mac'te `mps` beklenir (MPS'te henüz test edilmedi).
- `--classes "A,B,C"` sınıf alt kümesiyle eğitir; `--run isim` sonuçları `outputs/isim/`'e yazar.
- `app.py`: Gradio web arayüzü (dosya yükle / mikrofon, olasılıklar, zaman grafiği, spektrogram).

## Sonuçlar
- **GTZAN** (`outputs/gtzan/`, width 32, 40 epoch, CPU'da eğitildi, en iyi epoch 38):
  test segment **%79,3**, parça **%83,0**. En iyi: classical, hiphop, blues (10/10).
  En zayıf: disco 5/10, rock 6/10, pop 7/10 (birbirine karışıyor).
  Bu çalıştırma eski kodla yapıldığı için `last_model.pt` yok.
- **FMA small, 4 tür** (`outputs/fma_4sinif/`, `Hip-Hop,Rock,Electronic,Folk`, width 32,
  batch 128, Mac M1 MPS, 37 epoch'ta erken durdu, en iyi epoch 27, val segment %81,4):
  test segment **%76,0**, parça **%80,0** (400 parça). Parça recall: Hip-Hop 92, Electronic 85,
  Rock 81, Folk 62. Ana hata: Folk→Rock (31/100). Epoch ~5 dk (MPS, şarjda), bataryada ~8 dk.
- Hendrix "Who Knows" (canlı, funk-rock) denendi → model reggae/hiphop dedi. Açıklama: GTZAN'da
  funk sınıfı yok, 83 BPM funk groove'u reggae/hiphop ritmine benziyor, domain shift.
  Raporda "Sınırlılıklar" için iyi örnek.

## Mac'te ilk kurulum: veri setlerini SEN indir (kullanıcı onayladı)
Proje Mac'e GitHub üzerinden **sadece kod + `outputs/`** olarak geldi; `data/` ve `features/`
repoda yok (`.gitignore`). Kullanıcı veri setlerinin Mac'te Claude tarafından indirilmesini ve
özniteliklerin yeniden çıkarılmasını istedi. GTZAN (~1,2 GB) ve FMA small (~7,7 GB + 360 MB
metadata) için **önceden onay verildi, tekrar sorma**. FMA medium (~23 GB) seçilirse yine sor.
Sıra (venv kurulup MPS doğrulandıktan sonra, uzun işleri arka planda çalıştır):
```bash
python download_data.py gtzan
python extract_features.py --dataset gtzan        # birkaç dakika
python download_data.py fma_small
python extract_features.py --dataset fma_small    # uzun sürer (Windows'ta ~1 saat civarı)
```
- GTZAN aynı kaynaktan (HF marsyas/gtzan) indiği için seed 42 ile bölme Windows'takiyle aynı olur;
  `outputs/gtzan/best_model.pt` normalizasyon istatistiklerini kendi içinde taşır, yeniden eğitmek gerekmez.
- `extract_features.py` bittiğinde `features/fma_small/meta.json`'daki sayıları kontrol et
  (Windows'ta: train 60188, val 7544, test 7564 segment; 6 bozuk mp3 atlandı).
- python.org Python'unda `SSL: CERTIFICATE_VERIFY_FAILED` çıkarsa
  `/Applications/Python 3.x/Install Certificates.command` çalıştır.
- Sınıf seçimi B (FMA medium) olursa fma_small indirmeye gerek yok, medium small'u da içerir.

## Mac kurulumu yapıldı (2026-11-25)
- Klasör: `~/Desktop/gtzan-fma`, venv `.venv` (python.org 3.13 arm64, torch 2.15, MPS True).
- SSL hatası `SSL_CERT_FILE=$(.venv/bin/python -c "import certifi;print(certifi.where())")` ile aşıldı.
- `data/gtzan`, `data/fma` (fma_small) indirildi; `features/gtzan` ve `features/fma_small` üretildi
  (sayılar Windows'takiyle aynı). Eski GTZAN modeli Mac özniteliklerinde yine %79,3 / %83,0 verdi.
- Mac: Apple M1, **8 GB RAM**. batch 256 belleğe sığmayıp swap'a düşüyor, fp16 autocast hız
  kazandırmıyor → **batch 128, fp32** kullan. FMA medium diske sığmıyor (~44 GB tepe, ~30 GB boş).
- Kullanıcı sınıf seçimi olarak **A (4 tür)** seçti; eğitim + test tamamlandı (yukarıda).
- Sonuç beğenilmezse önerilen: AdamW + OneCycle/cosine, weight decay 1e-2, EMA (koda henüz eklenmedi).

## Durum / yarım kalanlar (Windows'tan kalan notlar)
- `features/fma_small/` Windows'ta hazırdı ama Mac'e taşınmadı; yukarıdaki adımlarla yeniden üret.
  FMA small sınıfları: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock.
- **FMA ile henüz eğitim yapılmadı.** Kullanıcı Windows'ta GPU kullanamadı (AMD RX 7700 XT,
  ROCm/MIOpen BatchNorm çekirdeği derlenemiyor) ve eğitime Mac'te devam etmek istedi.
- **Sınıf seçimi bekliyor.** Kullanıcı "görece kolay tahmin edilecek sınıflar" istedi. Seçenekler:
  - A: FMA small, `Hip-Hop,Rock,Electronic,Folk` (indirme yok)
  - B (önerilen): FMA medium (~23,8 GB indirme), `Classical,Electronic,Folk,Hip-Hop,Jazz,Rock`
  - C: FMA small, 8 türün hepsi (standart, beklenen doğruluk ~%50–65)
  Komutlar README'nin "Mac'te çalıştırma" bölümünde.
- Sonraki adımlar: Mac'te `pip install -r requirements.txt`, MPS'in görüldüğünü doğrula,
  seçilen sınıflarla eğit, `evaluate.py` ile test et, GTZAN sonuçlarıyla karşılaştır.

## Mac'te eğitim: EN HIZLI yöntem zorunlu (kullanıcı isteği)
Kullanıcı eğitimin Mac'te mümkün olan en hızlı şekilde, **GPU (Apple Silicon → MPS)** ile
yapılmasını istiyor. Eğitime başlamadan önce:
1. `python -c "import platform, torch; print(platform.machine(), torch.__version__, torch.backends.mps.is_available())"`
   → `arm64 ... True` görülmeli. Eğitim çıktısında `cihaz: mps` yazmalı.
2. **Görmüyorsa kendin kur/düzelt**, CPU'ya düşüp öyle eğitme:
   - `platform.machine()` `x86_64` ise Python Rosetta altında (Intel sürümü). arm64 Python kur
     (Homebrew `/opt/homebrew/bin/python3` veya python.org universal installer), yeni `.venv` oluştur.
   - `pip install --upgrade torch` (MPS için macOS 12.3+ ve güncel torch gerekli).
   - Desteklenmeyen işlem hatası çıkarsa `PYTORCH_ENABLE_MPS_FALLBACK=1` ile çalıştır.
   - Mac Intel ise (MPS yok) bunu kullanıcıya söyle, CPU süresini tahmin et.
3. Hız ayarları: öznitelikleri RAM'de tut (`--disk` kullanma, RAM yetmezse kullan), batch'i
   büyütmeyi dene (`--batch-size 128` veya 256) ve ilk epoch süresini ölçüp kullanıcıya bildir.
   Windows'ta CPU'da FMA small width 32 ≈ 18 dk/epoch idi; MPS bundan belirgin hızlı olmalı.
4. Eğitim uzun sürer: arka planda çalıştır, `--resume` ile devam ettirilebildiğini hatırlat.

## Notlar
- İndirme yapmadan önce (dosya, kaynak, boyut) kullanıcıya sor.
- Windows'a özel: GPU ortamı `C:\Users\PC\.venvs\genre-gpu` işe yaramadı, Mac'te gereksiz.
