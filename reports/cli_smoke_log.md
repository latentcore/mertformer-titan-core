# CLI Smoke Test (CPU) — 2026-02-05

## Command
```
python3 -m mertformer_sdk.cli info
```

## Output
```
⛔ CRITICAL: NO GPU DETECTED OR VRAM UNREADABLE.
   -> Switching to CPU/MPS Safe Mode (Very Slow)
✅ Layer configuration validated: No Liquid/MoE conflicts
⚠️  TR: Flash Attention 2 mevcut değil. / EN: Flash Attention 2 not available.
{
  "cuda": false,
  "mps": false,
  "device_count": 0
}
```

## Notes
- `run` and `export` commands require a checkpoint and/or external tokenizer downloads.
- These are intentionally **not executed** in the pre-training CPU smoke test to avoid large downloads.
