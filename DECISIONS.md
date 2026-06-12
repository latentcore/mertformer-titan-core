# Architecture Decisions

- **Config discipline**: Keep `config.py` as runtime default; allow optional YAML overlays.
- **Swarm architecture**: Documented as target; not enforced in core training.
- **Multimodal**: Deferred until text core is proven.
- **Canonical KD lane**: Keep the external lane name `offline_clean`, but treat it as the strict precomputed-KD path for the 45K closure run.
- **Teacher surface**: Pin the teacher to `meta-llama/Llama-3.3-70B-Instruct`; do not silently swap teacher IDs on the canonical path.
- **Fallback boundary**: Remove teacherless fallback from the canonical `offline_clean` training path. If logits are missing, fail with the exact blocker instead of silently downgrading the run.
- **Prompt surface**: Keep `prompts/system_v1.txt` as the only canonical system prompt surface until post-training behavioral evidence justifies expansion.
- **Artifact strategy**: Preserve the existing release/repo zips and add a separate training outputs bundle zip for real run retrieval (`artifacts/mertformer_training_outputs_bundle.zip` + SHA256 + manifests).

## Audited decisions (2026-06-12 zero-contradiction closure) — document, do NOT "re-fix"

A multi-agent audit verified these against current code. They are deliberate, NOT bugs;
a future reviewer/AI should not "fix" them:

- **P5/P8 — README performance table**: throughput/latency figures (~1.2s/step, ~107 tok/s,
  11,000 tok/s aggregate) are explicitly labeled "Projected / Est. / Not Measured /
  architecture simulation" — claim-discipline-compliant projections, not measured claims.
- **P13 — utils/safety.py `kill_if_non_finite`**: exported public utility (`utils/__init__.py`
  `__all__`), not orphaned. Absence of an internal caller is not a defect.
- **P18 — `config.auto_configure_batch_size` nvidia-smi**: no naive LOCAL_RANK guard is added —
  a rank-0-only guard would desync per-rank micro-batch; Method-A (`torch.cuda.get_device_properties`)
  already avoids nvidia-smi on real GPU nodes, and prints are env-gated (`TITAN_CONFIG_VERBOSE`).
- **P11 — `trust_remote_code=True`** on the gated Meta `Llama-3.3-70B-Instruct` teacher load is
  deliberate (trusted first-party repo). Production dataset loads do not enable it.
- **Measured param count `3,672,982,022` (~3.67B)** is the canonical measured runtime total;
  explicit `moe_intermediate=8192` (config/config.py) holds it. `economics/flops_estimator.py`
  `DEFAULT_PARAMS=2.64e9` is the separate design-target and is preserved. See reports/FACTS.json.
- **D3 — Stage-1 code source**: switched `bigcode/the-stack-v2` → `bigcode/the-stack-dedup`. v2 rows
  carry `blob_id`/`src_encoding` (the `content` field is empty and needs Software-Heritage S3
  resolution), which silently starved Stage-1's ~28% code corpus; the-stack-dedup exposes a real
  `content` field. A preflight source smoke-probe now fails loudly (`SOURCE_FIELD_EMPTY`) on any
  empty-field source. `datasets/hashes.json` revision/sha256 for the-stack-dedup must be re-pinned on
  the training machine (network required). Deliberate dataset-policy change.
- **Llama naming**: the Llama 3.3 Community License requests derivative model NAMES include "Llama".
  Current posture: name "MertFormer Titan" + the "Built with Llama" NOTICE; full naming compliance is
  EXTERNAL-PENDING legal (see NOTICE and reports/teacher_output_license_assessment.md).

## Mac-doable backlog dispositions (2026-06-13, Pass 7)

Closed as won't-change-now WITH reasons, so the backlog has no silently-"open" rows:
- **33 topk in bf16**: rejected — bf16 lowers top-k routing precision and would break the
  `scripts/cfc_moe_tolerance_check.py` gate (the fp32 cast at layers/moe.py is deliberate).
- **14 Liquid impl default**: already selectable via `TITAN_LIQUID_TRAIN_IMPL`
  (baseline / precompute_input / packed_pair / packed_pair_compile). Flipping the DEFAULT is the
  #1 MFU lever and needs pilot timing data → bound to the pilot, not changed blind.
- **17 MoE capacity-loop vectorization**: GPU-perf only (benefit shows on CUDA); do it on the
  training box where the speedup is measurable and the tolerance gate validates the numerics.
- **20 teacher pad-mask**: saves compute on one sub-max-seq sequence per stage (already masked
  downstream at train time) — GPU-perf only; do it on the teacher-precompute machine.
- **36 legacy logit-realign path**: kept as a deliberate env-gated escape hatch
  (`TITAN_ALLOW_LEGACY_LOGIT_REALIGN`, off by default); the canonical packing path is the default.
- **48 sharded checkpoint (accelerator.save_state)**: needs real multi-GPU to implement+validate;
  belongs on the training box. Single-GPU/Mac uses the standard checkpoint save.
- **23 MLA→GQA rename / 34 train.py split / 68 whole-repo mypy**: permanently left — high blast
  radius (onefile sha256 manifests / AST singleton tests / red CI gate) for ~zero functional gain.
  The MLA naming gap is documented honestly in README and ARCHITECTURE.md.
- **40 drift telemetry**: `lifelong_safety.safety_metrics()` already exposes `last_drift`; wiring it
  into the run logger is purely additive and the layer is off by default → deferred (no behavior risk).
- **24 / 25 / 31 / 38 test backlog**: additive pure-CPU coverage (RoPE position, block-order schema,
  SIGTERM handler, packing resume-seam). No defect — `consumed_through` monotonicity and the freeze/RNG/
  qinn/MoE-tolerance paths are already tested; these extend coverage and are deferred without risk.
