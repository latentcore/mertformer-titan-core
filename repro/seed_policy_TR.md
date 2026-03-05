# Seed Politikası

- Python, NumPy ve PyTorch için global seed ayarla.
- Her koşu metadata dosyasında seed değerini kaydet.
- Not: GPU çekirdekleri arasında tam determinism garanti edilmez.

## Örnek Log Kaydı
```
2026-02-05 07:12:44,021 - [INFO] - seed=1337
2026-02-05 07:12:44,021 - [INFO] - torch.deterministic=False
2026-02-05 07:12:44,022 - [INFO] - cudnn.benchmark=True
```
