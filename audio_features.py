"""Ses dosyasını 3 saniyelik segmentlere bölüp log-mel spektrogram çıkarır."""
import warnings

import librosa
import numpy as np

from config import (HOP_LENGTH, N_FFT, N_FRAMES, N_MELS, SAMPLE_RATE,
                    SEGMENT_SAMPLES, SEGMENTS_PER_TRACK)


def segment_melspecs(path, max_segments=SEGMENTS_PER_TRACK, max_duration=30.0):
    """Bir ses dosyasından (n_segment, N_MELS, N_FRAMES) boyutlu float16 dizi döndürür.

    Dosya okunamazsa veya 3 saniyeden kısaysa None döner.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=max_duration)
    except Exception as e:  # bozuk dosyalar (GTZAN jazz.00054, bazı FMA mp3'leri)
        print(f"[atlandı] {path}: {type(e).__name__}")
        return None

    n_seg = len(y) // SEGMENT_SAMPLES
    if max_segments:
        n_seg = min(n_seg, max_segments)
    if n_seg == 0:
        return None

    out = np.empty((n_seg, N_MELS, N_FRAMES), dtype=np.float16)
    for i in range(n_seg):
        seg = y[i * SEGMENT_SAMPLES:(i + 1) * SEGMENT_SAMPLES]
        mel = librosa.feature.melspectrogram(
            y=seg, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        # dB ölçeği; ref=max ile her segment [-80, 0] aralığına gelir (ses seviyesinden bağımsız)
        out[i] = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return out
