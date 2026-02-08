# MertFormer SDK Guide (Quick)

## What is the SDK?
A lightweight Python package + CLI that wraps model loading, generation, export, and benchmarking for MertFormer Titan.
It is **opt-in** and does **not** change the training pipeline.

## Install (from repo root)
```bash
python -m pip install -e .
```
Optional extras:
```bash
python -m pip install -e ".[cuda,onnx,cli]"
```

## Quick Start (Python API)
```python
from mertformer_sdk.api import load_model, generate, enable_lowbit_kernels

# Optional: enable low-bit kernels (CUDA + Triton required)
enable_lowbit_kernels(True)

model, tokenizer, device = load_model(ckpt="latest", device=None)
text = generate(model, tokenizer, "Hello from MertFormer!", max_new_tokens=64)
print(text)
```

`load_model(..., strict_checkpoint=True)` is the default. If the checkpoint is missing, it raises `FileNotFoundError`.
For random-weight smoke/demo usage only, call `load_model(..., strict_checkpoint=False)`.

## CLI Quick Start
```bash
mertformer info
mertformer run --prompt "Hello MertFormer" --ckpt latest
mertformer export --ckpt latest --bitpack
mertformer benchmark --ckpt latest --samples 0
mertformer verify
mertformer pilot-report --out reports/pilot_report.json
```

CLI commands now enforce checkpoint presence by default (`run` and `benchmark`).
Use `--allow-random` only for non-claim demos.

## Low-bit Kernel Toggle (Opt-in)
- Python: `enable_lowbit_kernels(True)`
- CLI: `mertformer run --lowbit ...`
- Env: `MERTFORMER_LOWBIT_KERNEL=1`
- Tensor Core (experimental): `MERTFORMER_TENSORCORE=1`

**Note:** Low-bit kernels are **experimental**. If CUDA/Triton is unavailable, the SDK falls back to the float path.

## Export (ONNX + Bitpack)
- `mertformer export --bitpack` produces:
  - `titan_s25_bitpack.bin`
  - `titan_s25_bitpack.json`
- ONNX files are tagged with metadata:
  - `mertformer.bitpack=ternary5in8`

## CPU Fallback
On CPU-only systems, low-bit kernels are disabled automatically and the SDK uses the standard float path.

## FAQ
**Q: What is QAT?**
A: Quantization-Aware Training. It simulates quantization during training to improve low-bit inference quality.
It is **optional** and usually applied **after** a stable baseline model exists.

**Q: Is there a Turkish tokenizer?**
A: Yes, **opt-in**. Default is off.
- Download with `scripts/download_tr_tokenizer.py`
- Enable via `use_tr_tokenizer=true` in `config/config.py`
- Recommended as a risk-controlled POC to protect distillation stability.

**Q: Is the kernel production-ready?**
A: It is an **experimental reference kernel** (correctness-first). Performance claims require real profiling.
