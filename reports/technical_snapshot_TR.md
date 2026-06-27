# Teknik Özet (TR)

**Version:** Build 30 V2

## Mimari
- Model boyutu: 2.64B parametre (tasarım hedefi); mevcut ölçülen runtime toplamı: ~3.67B.
- Kuantizasyon: BitNet 1.58‑bit.
- Yönlendirme: LiquidRouter MoE (zamansal dinamik).
- Dikkat: GQA attention (grouped-query, mevcut implementasyon) ile uzun bağlam hazırlığı.
- QINN: deneysel bir yol olarak mevcut, mevcut baseline yapılandırmasında varsayılan olarak devre dışı.

## Aktif Runtime Bileşenleri
- Çekirdek tokenizer yolu: mevcut LM baseline'ı için Llama ailesi tokenizer kurulumu.
- Çekirdek token embedding'leri: eğitim sırasında model içinde öğrenilir.
- Orchestrator metin embedding'leri: `sentence-transformers/all-MiniLM-L6-v2`.
- Orchestrator görüntü embedding'leri: `openai/clip-vit-base-patch32`.

## Aktivasyonlar
- MoE uzmanları: `BitSwiGLU`.
- Router akışkan yolu: `SiLU`.
- Liquid / CfC dinamikleri: `softplus` ve `tanh`.

## Eğitim Pipeline
- Offline logits destekli distillation.
- Curriculum aşamaları ve güvenlik kontrolleri.
- Akıllı checkpoint saklama.

## Safety & Gate’ler
- NaN kill‑switch testi: `scripts/nan_kill_test.py`.
- Failure budget izleme: `orchestrator/failure_budget.py`.
- Checkpoint restore drill: `scripts/checkpoint_restore_drill.py`.
- 1MB overfit gate: `scripts/overfit_gate.py`.

## Observability
- Grad norm loglama.
- Router entropy aracı: `orchestrator/telemetry.py`.
- VRAM ve sistem snapshot: `orchestrator/telemetry.py`.

## Reproducibility
- Git hash + config + seed + dataset manifest loglama: `utils/logger.py` ve `scripts/operator_mode_gate.py`.

## Değerlendirme
- Golden sample seti (50 prompt): `datasets/golden_samples.jsonl`.
- HumanEval/MBPP çıktı üretimi: `scripts/benchmarks_internal.py`.

## Dağıtım
- ONNX export ve mobil pipeline `scripts/` altında.


V2 refactor: dedup pipeline, parallel MoE dispatch, CfC fast path, stricter train gates.
