# MertFormer SDK Kılavuzu (Hızlı)

## SDK Nedir?
MertFormer Titan için model yükleme, üretim, dışa aktarma ve benchmark adımlarını sağlayan hafif bir Python paketidir.  
SDK **opt-in** çalışır ve eğitim hattını değiştirmez.

## Kurulum (repo kökünden)
```bash
python -m pip install -e .
```
İsteğe bağlı ekler:
```bash
python -m pip install -e ".[cuda,onnx,cli]"
```

## Hızlı Başlangıç (Python API)
```python
from mertformer_sdk.api import load_model, generate, enable_lowbit_kernels

# Opsiyonel: low-bit kernel (CUDA + Triton gerekir)
enable_lowbit_kernels(True)

model, tokenizer, device = load_model(ckpt="latest", device=None)
text = generate(model, tokenizer, "Merhaba MertFormer!", max_new_tokens=64)
print(text)
```

`load_model(..., strict_checkpoint=True)` varsayılandır. Checkpoint yoksa `FileNotFoundError` verir.  
Sadece random-weight smoke/demo için `load_model(..., strict_checkpoint=False)` kullanın.

## CLI Hızlı Başlangıç
```bash
mertformer info
mertformer run --prompt "Merhaba MertFormer" --ckpt latest
mertformer export --ckpt latest --bitpack
mertformer benchmark --ckpt latest --samples 0
mertformer verify
mertformer pilot-report --out reports/pilot_report.json
mertformer kpi-report --out reports/kpi_report_v1.json
mertformer 57-report --out reports/closure_57_matrix.json
```

`run` ve `benchmark` komutları varsayılan olarak checkpoint ister.  
Sadece iddia dışı demo için `--allow-random` kullanın.

## KPI Raporu (Build 30)
- Üretim: `mertformer kpi-report --out reports/kpi_report_v1.json`
- Opsiyonel ONNX KPI: `mertformer kpi-report --out reports/kpi_report_v1.json --onnx-check`
- Şema: `interfaces/kpi_report_v1.schema.json`

## Closure 57 Kapısı (Build 30)
- Makine doğrulamalı kapanış raporu üretimi:
  - `mertformer 57-report --out reports/closure_57_matrix.json`
- Opsiyonel markdown çıktıları:
  - `mertformer 57-report --out reports/closure_57_matrix.json --md-out reports/closure_57_matrix.md --md-tr-out reports/closure_57_matrix_TR.md`
- Eşdeğer script:
  - `python3 scripts/check_57_matrix.py`

## Low-bit Kernel (Opt-in)
- Python: `enable_lowbit_kernels(True)`
- CLI: `mertformer run --lowbit ...`
- Env: `MERTFORMER_LOWBIT_KERNEL=1`
- Tensor Core (deneysel): `MERTFORMER_TENSORCORE=1`

**Not:** Low-bit kernel deneysel referans yoldur. CUDA/Triton yoksa otomatik olarak float fallback kullanılır.

## Export (ONNX + Bitpack)
- `mertformer export --bitpack` çıktıları:
  - `titan_s25_bitpack.bin`
  - `titan_s25_bitpack.json`
- ONNX metadata etiketi:
  - `mertformer.bitpack=ternary5in8`

## CPU Fallback
CPU-only ortamlarda low-bit kernel otomatik kapanır ve standart float yol çalışır.

## Yasal Güvenlik Kapsamı
- SDK yalnızca yasal, denetlenebilir ve insan onaylı operasyonlar için tasarlanmıştır.
- İzinsiz gözetim/takip ve yetkisiz müdahale kapsam dışıdır.
- Ağır eğitim kanıtları kapanış raporlarında `Evidence Pending` olarak açıkça işaretlenir.

## SSS
**S: QAT nedir?**  
C: Quantization-Aware Training. Eğitim sırasında quantization simülasyonu yaparak low-bit inference kalitesini artırır. Genelde stabil bir temel modelden sonra uygulanır.

**S: Türkçe tokenizer var mı?**  
C: Evet, opt-in. Varsayılan olarak kapalıdır.
- `scripts/download_tr_tokenizer.py` ile indir
- `config/config.py` içinde `use_tr_tokenizer=true` ayarla

**S: Kernel production-ready mi?**  
C: Doğruluk öncelikli deneysel referans kernel vardır. Performans iddiaları için gerçek profil ölçümü gerekir.
