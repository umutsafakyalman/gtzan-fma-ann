"""Veri setlerini indirir ve data/ altına açar.

Kullanım:
    python download_data.py gtzan        # ~1.2 GB  (Hugging Face: marsyas/gtzan)
    python download_data.py fma_small    # ~7.7 GB ses + ~360 MB metadata (FMA resmi sunucusu)
    python download_data.py fma_medium   # ~22 GB, 25.000 parça, 16 tür

Not: Dosyaları elle de indirebilirsiniz; README'deki klasör yapısına koymanız yeterli.
"""
import argparse
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

import config

SOURCES = {
    "gtzan": [("https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz",
               config.DATA_DIR / "gtzan")],
    "fma_metadata": [("https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
                      config.DATA_DIR / "fma")],
    "fma_small": [("https://os.unil.cloud.switch.ch/fma/fma_metadata.zip", config.DATA_DIR / "fma"),
                  ("https://os.unil.cloud.switch.ch/fma/fma_small.zip", config.DATA_DIR / "fma")],
    "fma_medium": [("https://os.unil.cloud.switch.ch/fma/fma_metadata.zip", config.DATA_DIR / "fma"),
                   ("https://os.unil.cloud.switch.ch/fma/fma_medium.zip", config.DATA_DIR / "fma")],
}


def download(url, dest: Path):
    if dest.exists():
        print(f"Zaten var: {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                        desc=dest.name) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def extract(archive: Path, target: Path):
    print(f"Açılıyor: {archive.name} -> {target}")
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target, filter="data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(SOURCES))
    ap.add_argument("--keep-archive", action="store_true", help="arşiv dosyasını silme")
    args = ap.parse_args()

    for url, target in SOURCES[args.dataset]:
        target.mkdir(parents=True, exist_ok=True)
        archive = target / url.rsplit("/", 1)[1]
        download(url, archive)
        extract(archive, target)
        if not args.keep_archive:
            archive.unlink()
    print("Tamam.")


if __name__ == "__main__":
    main()
