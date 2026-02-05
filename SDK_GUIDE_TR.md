# MertFormer SDK Kilavuzu (Hizli)

## SDK Nedir?
MertFormer Titan icin model yukleme, uretim, export ve benchmark adimlarini saran hafif bir Python paketidir.
**Opt-in** calisir ve egitim pipeline'ina **dokunmaz**.

## Kurulum (repo kokunden)
```bash
python -m pip install -e .
```
Istege bagli ekler:
```bash
python -m pip install -e ".[cuda,onnx,cli]"
```

## Hizli Baslangic (Python API)
```python
from mertformer_sdk.api import load_model, generate, enable_lowbit_kernels

# Opsiyonel: low-bit kernel (CUDA + Triton gerekli)
enable_lowbit_kernels(True)

model, tokenizer, device = load_model(ckpt="latest", device=None)
text = generate(model, tokenizer, "Merhaba MertFormer!", max_new_tokens=64)
print(text)
```

## CLI Hizli Baslangic
```bash
mertformer info
mertformer run --prompt "Merhaba MertFormer" --ckpt latest
mertformer export --ckpt latest --bitpack
mertformer benchmark --ckpt latest --samples 0
```

## Low-bit Kernel (Opt-in)
- Python: `enable_lowbit_kernels(True)`
- CLI: `mertformer run --lowbit ...`
- Env: `MERTFORMER_LOWBIT_KERNEL=1`
- Tensor Core (deneysel): `MERTFORMER_TENSORCORE=1`

**Not:** Low-bit kernel **deneysel**dir. CUDA/Triton yoksa otomatik olarak float yola geri doner.

## Export (ONNX + Bitpack)
- `mertformer export --bitpack` uretir:
  - `titan_s25_bitpack.bin`
  - `titan_s25_bitpack.json`
- ONNX dosyalarina metadata eklenir:
  - `mertformer.bitpack=ternary5in8`

## CPU Fallback
CPU-only sistemlerde low-bit kernel otomatik kapatilir ve standart float yol kullanilir.

## SSS
**S: QAT nedir?**
C: Quantization-Aware Training. Egitim sirasinda quantization simulasyonu yaparak low-bit inference kalitesini artirir.
Genelde **stabil bir temel model** olustuktan sonra uygulanir.

**S: Turkce tokenizer var mi?**
C: Evet, **opt-in**. Varsayilan kapali.
- `scripts/download_tr_tokenizer.py` ile indir
- `config/config.py` icinde `use_tr_tokenizer=true`
- Distillation stabilitesi icin risk kontrollu POC onerilir.

**S: Kernel production-ready mi?**
C: **Deneysel referans kernel** (dogruluk oncelikli). Performans iddialari icin gercek profil gerekir.
