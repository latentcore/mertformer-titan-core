# Teknik Özet (TR)

## Mimari
- Model boyutu: 2.64B parametre (tasarım hedefi); mevcut ölçülen runtime toplamı: ~3.70B.
- Kuantizasyon: BitNet 1.58‑bit.
- Yönlendirme: LiquidRouter MoE (zamansal dinamik).
- Dikkat: MLA ile uzun bağlam hazırlığı.

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
