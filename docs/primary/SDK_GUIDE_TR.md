# MertFormer SDK Kilavuzu (Hizli)

## SDK Nedir?
MertFormer Titan için model yukleme, üretim, export ve benchmark adimlarini saran hafif bir Python paketidir.
**Opt-in** çalışir ve eğitim pipeline'ina **dokunmaz**.

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

`load_model(..., strict_checkpoint=True)` varsayilan davranistir. Checkpoint yoksa `FileNotFoundError` verir.
Sadece random-weight demo/smoke amaciyla `load_model(..., strict_checkpoint=False)` kullanin.

## CLI Hizli Baslangic
```bash
mertformer info
mertformer run --prompt "Merhaba MertFormer" --ckpt latest
mertformer export --ckpt latest --bitpack
mertformer benchmark --ckpt latest --samples 0
mertformer verify
mertformer pilot-report --out reports/pilot_report.json
```

CLI tarafinda `run` ve `benchmark` komutlari varsayilan olarak checkpoint ister.
Sadece claim disi demo için `--allow-random` kullanin.

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
- ONNX dosyalarına metadata eklenir:
  - `mertformer.bitpack=ternary5in8`

## CPU Fallback
CPU-only sistemlerde low-bit kernel otomatik kapatilir ve standart float yol kullanilir.

## SSS
**S: QAT nedir?**
C: Quantization-Aware Training. Eğitim sirasinda quantization simülasyonu yaparak low-bit inference kalitesini artirir.
Genelde **stabil bir temel model** olustuktan sonra uygulanir.

**S: Türkçe tokenizer var mi?**
C: Evet, **opt-in**. Varsayilan kapali.
- `scripts/download_tr_tokenizer.py` ile indir
- `config/config.py` icinde `use_tr_tokenizer=true`
- Distillation stabilitesi için risk kontrollu POC onerilir.

**S: Kernel production-ready mi?**
C: **Deneysel referans kernel** (dogruluk oncelikli). Performans iddialari için gerçek profil gerekir.
