# MertFormer Titan (Build 30 V2)

An on-device-oriented LLM research stack combining **BitNet b1.58 ternary weights**, a
**sparse Mixture-of-Experts (MoE)**, a **Liquid/CfC mixer**, and **GQA attention** (grouped-query). Evidence-first: this repo is a **pre-training,
closure-complete engineering PoC** — the canonical model has **not been trained yet**, and
no capability/benchmark claim is made until a real run produces checkpoints.

Türkçe: [README_TR.md](README_TR.md).

## Status (canonical: [STATUS.md](STATUS.md))
- **Build:** `728 passed, 9 skipped` (offline-first `pytest`).
- **Readiness:** `decision_reason_code = READY_REMOTE_BOOTSTRAP` · `recommended_path = remote_bootstrap` · `train_allowed = true`.
- **Run blockers:** `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`, `online_teacher:MISSING_HF_TOKEN`.
- **The one real gap:** a real 45K GPU run (H100/H200). Crash-class bugs: none.

## Architecture (measured, not benchmarked)
- 18 layers · hidden 2048 · 16 heads / 8 KV (**GQA attention (grouped-query, current implementation)**) · 8 experts top-2, MoE every 3rd layer · Liquid/CfC mixer at layers [4, 10, 16] · BitNet b1.58 ternary.
- **Measured runtime params:** `3,672,982,022` (~3.67B). **Design target:** 2.64B. Both labels are deliberate — see [reports/param_accounting_report.md](reports/param_accounting_report.md) and [DECISIONS.md](DECISIONS.md).
- Full architecture: [ARCHITECTURE.md](ARCHITECTURE.md).

## What is measured vs. not (honest boundary)
- **Measured:** repo self-tests, an offline smoke harness, and a 12-seed Liquid ablation.
- **Liquid ablation verdict ([ABLATION.md](ABLATION.md)):** OFF 96.32% / ON 94.69% ID exact-accuracy (Δ−1.63 pp, p=0.305, Cohen's d=−0.43). **No measured accuracy benefit; ~30% slower; inconclusive at toy scale.** **No Liquid speed/latency claim** is made until a verified 45K run.
- **Not measured (the gap):** the canonical 3.67B model has never been trained, so convergence/generalization are **unverified**. This is hardware-bound, not a code edit.

## Claim boundary
This repo is a **pre-training**, **proof-of-system** PoC; it is explicitly intended **not to claim a production-ready or certified platform** and is **NOT ELIGIBLE FOR CLAIM** of trained capability until a real run produces checkpoints.
- **Routing policy: token-choice top-k.** Attention is GQA (grouped-query); see Architecture above.
- Closure-matrix scope: agi/asi rows are out-of-scope pending — `out_of_scope_pending_ids=[8, 9, 11, 12, 51, 52, 53, 54, 55, 56, 57]` (see [reports/closure_57_matrix.md](reports/closure_57_matrix.md)).
- Output discipline: **Default mode is `verified`.** No claim survives final docs without evidence.

## Quickstart
```bash
bash scripts/bootstrap_venv.sh        # .titan-venv with pinned deps (Python 3.11)
bash scripts/verify_all.sh            # offline: 728 passed, 9 skipped + gates (no GPU, no network)
bash zero_touch_start.sh --check-only # readiness verdict + blockers (no training)
python scripts/train_smoke.py --cleanup  # tiny CUDA/MPS/CPU sanity loop -- proves it trains, not a claim
```
Full verify/launch flow: [REPRODUCE.md](REPRODUCE.md).

### Config sample (current defaults)
A few representative `config/config.py` defaults (documentation only — do not edit here):
```python
use_torch_compile = False
moe_intermediate = 8192     # holds the measured 3.67B total
liquid_layers_idx = [4, 10, 16]
```

## Canonical surfaces (read these first)
- [AGENTS.md](AGENTS.md) — project-closure constitution; #1 in this repo's own source-of-truth order.
- [STATUS.md](STATUS.md) — stage, readiness, the one gap.
- [TRUTH_MATRIX.md](TRUTH_MATRIX.md) — every claim → evidence class (measured/target/vision).
- [BACKLOG.md](BACKLOG.md) — 45K gate + deferred post-run findings.
- [GOVERNANCE.md](GOVERNANCE.md) — policies/contracts index + privacy posture.
- [REPRODUCE.md](REPRODUCE.md) — verify & launch commands.
- [DECISIONS.md](DECISIONS.md) — deliberate decisions (incl. documented-not-changed findings).
- [ARCHITECTURE.md](ARCHITECTURE.md) · [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) · [MODEL_CARD.md](MODEL_CARD.md).
- Master-truth docs: [docs/PROJECT_MASTER_TRUTH.md](docs/PROJECT_MASTER_TRUTH.md) · [docs/CHESS_ONEFILE_MASTER_TRUTH.md](docs/CHESS_ONEFILE_MASTER_TRUTH.md).
- External-reviewer onboarding path (separate from the list above): [START_HERE.md](START_HERE.md) · [README_SUMMARY.md](README_SUMMARY.md).

## 📂 Project Structure
### Canonical Layout (Build 30 V2)
```text
Full tracked-file tree: docs/PROJECT_STRUCTURE.md
```

## 🚀 Hiring & Commercial Opportunities

I am open to roles and collaborations in LLM pre-training, efficient architectures
(ternary/low-bit, MoE, recurrent mixers) and training infrastructure.

What this repository is meant to show: an end-to-end pre-training stack built and audited
in the open — architecture, data pipeline, distillation, evaluation harness, closure gates
and claim discipline. Read [STATUS.md](STATUS.md) for what is *measured* today versus what
is *target* or *vision*; the separation is deliberate and enforced by
[TRUTH_MATRIX.md](TRUTH_MATRIX.md).

**Compute sponsorship:** the one real gap between this repo and a trained checkpoint is
GPU time for the canonical 45K run (H100/H200). If your organization can offer compute
credits or a short rental window, reach out — [BACKLOG.md](BACKLOG.md) has the exact
pre-run checklist this would unblock.

Separately, a small side project mirroring this architecture family (BitNet/MoE/Liquid/GQA)
completed a real, checkpoint-bound training run on consumer hardware — see
[evidence/2026-08-02-chess-searchless-5070/](evidence/2026-08-02-chess-searchless-5070/)
(not the same model, not the same checkpoint as the flagship above). Checkpoint weights:
[Hugging Face](https://huggingface.co/Mert21779033/mertformer-chess-searchless).

**Contact:** [mert.yunlu08@gmail.com](mailto:mert.yunlu08@gmail.com) — or open an issue
on this repository.

## 🤝 Contributions & PR Rules

Contributions are welcome under the Apache 2.0 license.

Before opening a pull request:

1. **`bash scripts/verify_all.sh` must pass with zero regressions.** This runs the full
   pytest suite plus the closure gates (manifest sync, doc-claim consistency, secret scan,
   fact drift). A PR that reduces the passing test count will not be merged.
2. **Add a test with behaviour changes.** The suite is the contract; several bugs closed in
   this repository survived for months precisely because a test mirrored the code instead
   of importing it.
3. **Do not weaken claim boundaries.** The measured/target/vision separation, the reason
   codes and `TRUTH_MATRIX.md` are load-bearing. Do not relabel an unmeasured number as
   measured, and do not add `trained`, `benchmark-verified` or `production-ready` to any
   surface that is not backed by a real run.
4. **Keep the EN/TR twins in sync.** `README.md`/`README_TR.md`, `BACKLOG.md`/`BACKLOG_TR.md`,
   `DECISIONS.md`/`DECISIONS_TR.md`, `STATUS.md`/`STATUS_TR.md` and their siblings change
   together.
5. **Use `feature/` or `fix/` branch prefixes**, and disclose AI-assistant involvement in the
   commit trailer — see [CONTRIBUTING.md](CONTRIBUTING.md).

All participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

Report security issues privately via [SECURITY.md](SECURITY.md), never in a public issue.

## License
Apache License 2.0 — see [LICENSE](LICENSE). Copyright 2026 Mert Yünlü. Built with Llama — see [NOTICE](NOTICE) and [MODEL_LICENSE.md](MODEL_LICENSE.md).
