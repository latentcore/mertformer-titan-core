# Görev: Specialized Edge Coding Launch (Operator Mode - Pre-Training)

## Phase -1: Safety & Failure Budget
- [x] Auto-Kill NaN Injection (kill switch testi): `scripts/nan_kill_test.py`
- [x] Failure Budget tanımı ve Pivot tetikleyici: `orchestrator/failure_budget.py`
- [x] Checkpoint Restore Drill (kill/resume bütünlük testi): `scripts/checkpoint_restore_drill.py`

## Phase 0: Infrastructure & Reality Contact Gates
- [x] Reproducibility Stamp (git, config, seed, dataset): `scripts/operator_mode_gate.py` + `utils/logger.py`
- [x] Overfit Gate (1MB kod): `scripts/overfit_gate.py`
- [x] Observability Layer (grad norm, router entropy, VRAM): `orchestrator/telemetry.py`
- [x] Golden Sample Eval (50 prompt): `datasets/golden_samples.jsonl` + `scripts/golden_eval.py`

## Phase 1: Telemetry-Driven Execution
- [x] Expected vs Actual tracking iskeleti: `orchestrator/telemetry.py`
- [x] Master Training run yolu hazır (eğitim donanımında koşulacak)
- [x] Internal Truth Benchmarking runner (HumanEval/MBPP): `scripts/benchmarks_internal.py`

## Phase 2: Minimal Prototype Stack (Asset’ler)
- [x] Investor Deck / One-Pager / Technical Snapshot: `TECHNICAL_REPORT.md` (tracked). Ticari öğeler (`one_pager.md`, `technical_snapshot.md`, `PITCH.md`) `private/commercial/`'a taşındı — untracked.
- [x] Microsoft Founders Hub Başvuru Taslağı: `private/commercial/founders_hub_application.md` (`reports/`'tan taşındı; untracked). 2026-05-31'de yapıldı.

## Phase 3: Future Horizons (Backlog)
- [x] White Paper & Defense Licensing (post-validation): `WHITE_PAPER_LIQUIDROUTER.md`

## Verification Plan
- [x] Sanity Drill: Kill/Resume bütünlüğü ve Failure Budget uyarıları
- [x] Tek girişli gate runner: `scripts/operator_mode_gate.py`

## Teslim Notları
- Tek giriş test: `python scripts/operator_mode_gate.py`
- Varsayılan güvenli mod; tam gate için `--full` kullanılır.
