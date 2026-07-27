# V2 Backlog Seed (Post V1 Closure)

## Completed in Build30 V2 Refactor
- Build30 V2 version sync across core docs + metadata
- .env.example + TROUBLESHOOTING + MODEL_LICENSE added
- CfC/MoE tolerance check integrated into SOP
- Kaggle/Colab wall-time caps (11.5h/23.5h) + resume guard
- dataset dedup (rolling hash, configurable scope)
- stage 4/5 curriculum expansion + ratio rebalance
- MoE dispatch parallel gather/scatter path (sequential fallback retained)
- Liquid CfC training fast path (torch.compile guarded)
- FlashAttention inference opt-in flag
- SOP plot integration + artifacts zip refresh

## Track A — Core Runtime
- compile policy auto-profiler
- cudagraph static-shape validator hardening
- distributed mode contract (DDP/FSDP/ZeRO)
- optimizer plugin matrix (AdamW/Lion/Adafactor)
- liquid_spike_counter cleanup (v1.0.1 housekeeping: wire to safeguard path or remove dead placeholder, then add regression test)

## Track B — Data and Evaluation
- unseen-range curriculum sweeps
- compositional arithmetic benchmark pack
- error taxonomy expansions
- seed significance report automation

## Track C — Interpretability
- expert specialization stability panel
- attention entropy trend snapshots
- layer update/weight ratio dashboard

## Track D — Artifact and Reporting
- safetensors dual-save rollout
- export benchmark matrix (TorchScript/ONNX/quant)
- html/pdf report renderer hardening
- claim-evidence auto-map v2

## Track E — Productization
- constrained decode runtime policy
- REST surface and inference batch API
- demo packaging and canned prompts set
- KPI scorecard contract v2

## Track F — Post-45K verified-finding fixes (logged 2026-06-16; see DECISIONS.md)
> **2026-07-27 sync note:** 3 of the 7 items below were resolved by later passes and this file was never updated to say so (a `BACKLOG.md`/`DECISIONS.md` cross-check caught the drift). Marked `RESOLVED` inline rather than deleted, per the same history-preservation discipline as `BACKLOG.md`/`DECISIONS.md`'s own `⚠ DÜZELTME`/correction pattern. The remaining 4 are still genuinely open.
- **RESOLVED (2026-07-12).** z-loss double-multiply: pick intended effective weight (currently `1e-4 * 0.02 = 2e-6`), apply once, re-validate with `scripts/cfc_moe_tolerance_check.py`. → `z_loss_coef` changed `1e-4 → 0.05` (`config/config.py:329`), landing the effective weight at the ~1e-3 literature convention; candidate fix, applied and tested on real 36M/171M hardware (still diverges for other reasons — see `BACKLOG.md`), `cfc_moe_tolerance_check.py` re-validation itself remains a real-GPU task. See `BACKLOG.md`/`DECISIONS.md` "Second real-hardware confirmation... z-loss double-multiply fix" (2026-07-12).
- Liquid variable-`dt`: wire input/time-dependent `dt` (currently fixed `1.0` → gated RNN) and ablate vs the gated-RNN baseline; otherwise relabel. **Still open** — no change since 2026-06-16.
- Sequential MoE dispatch: pass `capacity_mask` to skip dropped-token FLOPs (GPU-perf; output already correct). **Still open** — GPU-perf only, deferred to the training box.
- Train-loop `.item()` hot-path: remove per-micro-step host-device syncs (GPU-perf). **Still open** — GPU-perf only, deferred to the training box.
- **RESOLVED, "confirm" question closed (2026-07-12), not called.** `LiquidCell.mark_weights_updated()`: confirm whether the JIT/quant eval cache needs this invalidation hook on weight update (do not remove blindly). → Investigated: `_ensure_qcache()` already auto-invalidates every forward via each param's PyTorch `._version` counter, so ordinary in-place optimizer steps never need this manual hook; the one edge case that might still matter (a full `.data = new_tensor` rebind) is undetermined without a GaLore/checkpoint-load-path test. Left uncalled and undeleted, with the full reasoning now documented at the call site in `layers/liquid.py`. See `BACKLOG.md`/`DECISIONS.md` "Uçtan uca pre/post-45K kalan-madde denetimi" (2026-07-12).
- ONNX export opset: prod is `14` (`scripts/mobile_export.py`); revisit bump to 17/18 for S25 NPU QDQ once device profiling exists (legacy smoke test deliberately stays at 12). **Still open** — no change since 2026-06-16; tracked alongside the other pre-45K-frozen items in `DECISIONS.md` (search "opset").
- **RESOLVED (2026-07-12).** `liquid_warmup_steps`: add an env override for parity with the other tunables (currently hardcoded `10000`). → `config/config.py:361` now reads `TITAN_LIQUID_WARMUP_STEPS` (mirrors the `TITAN_ROUTER_LR_MULT`/`TITAN_WARMUP_RATIO` pattern); default unchanged at `10000`. Verified live via `grep` 2026-07-27.

## Engineering Rules
- no direct edits on v1 closure branches
- one feature family per PR
- evidence-first payload updates
- fail-fast schema enforcement retained
