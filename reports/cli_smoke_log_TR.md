# CLI Smoke Testi (CPU) — 2026-02-05

## Komut
```
python3 -m mertformer_sdk.cli info
```

## Çıktı
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

## Notlar
- `run` ve `export` komutları checkpoint ve/veya tokenizer indirmesi gerektirir.
- Bu adım, büyük indirmelerden kaçınmak için **bilerek** çalıştirilmadi.
