"""Sıfırdan tasarlanmış evrişimli sinir ağı (CNN) — hazır/önceden eğitilmiş model KULLANILMAZ.

Girdi : (batch, 1, 128 mel bandı, 130 zaman adımı) log-mel spektrogram
Çıktı : (batch, sınıf_sayısı) logit

Mimari (GenreCNN):
    4 x ConvBlock  [Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> MaxPool 2x2 -> Dropout]
        kanal: 1 -> 32 -> 64 -> 128 -> 256,   boyut: 128x130 -> 64x65 -> 32x32 -> 16x16 -> 8x8
    Global Ortalama + Global Maksimum havuzlama (zaman/frekans üzerinde) -> 512
    Tam bağlı: 512 -> 256 -> sınıf_sayısı  (arada Dropout 0.5)
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GenreCNN(nn.Module):
    def __init__(self, n_classes, base_channels=32, dropout=0.5):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            ConvBlock(1, c, 0.05),
            ConvBlock(c, 2 * c, 0.1),
            ConvBlock(2 * c, 4 * c, 0.15),
            ConvBlock(4 * c, 8 * c, 0.2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(16 * c, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)                     # (B, 8c, 8, 8)
        avg = x.mean(dim=(2, 3))                 # global ortalama havuzlama
        mx = x.amax(dim=(2, 3))                  # global maksimum havuzlama
        x = torch.cat([avg, mx], dim=1)          # (B, 16c)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = GenreCNN(n_classes=10)
    print(m)
    print(f"Eğitilebilir parametre sayısı: {count_parameters(m):,}")
    print("Çıktı boyutu:", m(torch.randn(2, 1, 128, 130)).shape)
