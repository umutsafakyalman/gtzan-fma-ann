"""Projenin tüm ortak ayarları tek yerde."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # ham ses dosyaları (GTZAN, FMA)
FEATURE_DIR = ROOT / "features"   # çıkarılmış mel-spektrogramlar
OUTPUT_DIR = ROOT / "outputs"     # eğitilmiş modeller, grafikler, raporlar

# --- Ses / öznitelik ayarları ---
SAMPLE_RATE = 22050               # Hz
SEGMENT_SECONDS = 3.0             # her parça 3 saniye
SEGMENTS_PER_TRACK = 10           # 30 sn'lik parça -> 10 adet 3 sn'lik segment
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128                      # mel frekans bandı sayısı (spektrogram yüksekliği)
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)
N_FRAMES = 1 + SEGMENT_SAMPLES // HOP_LENGTH   # = 130 (spektrogram genişliği)

GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

SEED = 42
