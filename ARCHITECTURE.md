# MertFormer Titan — Architecture

> Status: **PRE-TRAINING (UNVERIFIED)**. This document explains *what* each component is and
> *why* it exists. Capability/benchmark numbers are target/vision until a measured run exists.
> Canonical numbers live in [`reports/FACTS.json`](reports/FACTS.json).

## At a glance

| Property | Value | Source |
|---|---|---|
| Decoder layers | 18 | `config/config.py` |
| Hidden size | 2048 | `config/config.py` |
| Attention | GQA, 16 query / 8 KV heads, head_dim 128, RoPE θ=100000 | `layers/mla.py` |
| FFN | BitNet SwiGLU (dense layers) | `layers/mertformer_block.py`, `model/transformers.py` (`MertFormerFFN`) |
| MoE | Sparse, 8 experts top-2 + shared expert, `moe_intermediate=8192`, every 3rd layer | `layers/moe.py` |
| Dynamics | Liquid / CfC mixer on a subset of layers | `layers/liquid.py` |
| Weights | BitNet b1.58 (ternary) via `BitLinear` (fake-quant in training) | `layers/bitlinear.py` |
| Vocab | 128256 (Llama-3 tokenizer) | `utils/tokenizer_resolver.py` |
| Measured params | **3,672,982,022 (~3.67B)** | `reports/param_accounting_report.md` |
| Design target | **2.64B** | `economics/flops_estimator.py` (`DEFAULT_PARAMS`) |
| Active params / token | ~1.86B (MoE top-2 + shared) | `economics/flops_estimator.py` |
| Training | Offline Top-K knowledge distillation from `meta-llama/Llama-3.3-70B-Instruct` | `scripts/precompute_logits_topk.py`, `train/train.py` |

## Components and the *why*

- **BitLinear (BitNet b1.58)** — ternary {-1,0,+1} weight simulation via straight-through
  estimation; the routing/`tau`/gate/norm layers are kept FP16/BF16 (see `layers/bitnet_patch.py`
  whitelist) because quantizing routing collapses expert selection. *Why:* edge/NPU memory budget.
- **GQA attention (`layers/mla.py`, class `GQA`)** — grouped-query KV (8 KV heads) keeps
  the KV cache small for long context on-device. *Why:* mobile KV-cache footprint. (Naming: a true
  latent-MLA low-rank KV bottleneck is intentionally NOT implemented; the class was renamed
  `MLA` -> `GQA` to match the implementation — see DECISIONS / README truth-boundary.)
- **Sparse MoE (`layers/moe.py`)** — 8 experts, top-2 routing + a shared expert, Switch-style
  capacity control with overflow drop + renormalize, plus router-collapse detection/recovery.
  *Why:* more capacity at ~constant active FLOPs.
- **Liquid / CfC mixer (`layers/liquid.py`)** — continuous-time recurrent dynamics on a few layers,
  with a warmup freeze + spike-cooldown safeguard. *Why:* a research bet on long-range dynamics. A
  12-seed toy-scale ablation ([ABLATION.md](ABLATION.md)) found **no measured accuracy benefit**
  (OFF 96.32% / ON 94.69%, Δ−1.63pp, p=0.305) at **~30% slower** wall-clock; value at real scale is
  inconclusive (cost certain, benefit unmeasured). The earlier single-seed +0.50 pilot is superseded.
- **DDP safety (`train/train.py`)** — `find_unused_parameters=True` because the liquid/tau warmup
  freeze and MoE top-2 routing both leave params without grads in a step.

## Knowledge-distillation lane

Canonical = **offline sparse Top-K**: `scripts/precompute_logits_topk.py` precomputes the teacher's
Top-K logits (int32 indices + bf16 values) into shards; `train/packing.py` byte-aligns teacher/student
streams with a hard identity assert (silent realignment is structurally impossible). A dense lane exists
in `orchestrator/distillation_manager.py` but is **debug-only** (hard-fails unless
`TITAN_ALLOW_DENSE_PRECOMPUTE=1`). Disk scales ~linearly with `--top-k` (≈36 TB @ 256 … ≈4.5 TB @ 32 for
the ~23.6B-token budget); a pre-flight disk gate enforces this.

## Open ablations (gate the research claims)

The `ablations/` scaffold (`no_liquid/`, `no_moe/`, `bitlinear_off/`, `dense_only/`) is the place where
the architecture's bets get *measured*. The `no_liquid` bet has now been **measured at toy scale**: a
12-seed ablation ([ABLATION.md](ABLATION.md)) shows **no accuracy benefit** from the CfC mixer
(OFF 96.32% / ON 94.69%, Δ−1.63pp, p=0.305, Cohen's d=−0.43) at ~30% slower wall-clock — superseding the
earlier single-seed +0.50 pilot (one lucky seed). The reading is **inconclusive at scale** (the toy task
is ceiling-saturated and the test underpowered): the cost is certain, the benefit unmeasured. The
`no_moe` / `bitlinear_off` / `dense_only` bets remain hypotheses until a measured run.

## Feature flags (off by default) & out-of-scope surfaces

`use_qinn` (orthogonal Cayley transform, `layers/qinn.py`), `use_lifelong_safety_layer`
(`layers/lifelong_safety.py`), `use_latent_ode_state_channel` / `use_neuromodulatory_gain`
(`layers/cognitive_extensions.py`), and the world-model head (`layers/world_model_head.py`,
diagnostic-only) are **flag-OFF on the canonical path**: not instantiated, zero measured params,
inert during the 45K run.

The `orchestrator/` cognitive runtime (`cognitive_loop.py`, `reasoning_engine.py`, `planner.py`,
`self_audit.py`, etc.) is an **inference/agentic sandbox that is NOT part of pre-training**. The only
`orchestrator/` modules the 45K training path imports are `distillation_manager.py` and `telemetry.py`.
Everything else there, plus the flags above, is **explicitly out of scope for the trained model** —
kept for research continuity, not a capability claim, and not exercised by the canonical run.

## Projections (claim boundary)
All capability / throughput / device numbers in README and reports are **projections or targets**,
explicitly labeled (Est. / Projected / Not Measured / architecture simulation) — not measured results.
The only measured facts today are structural (param count, test suite, gate status — see
`reports/FACTS.json`); capability numbers become claims only after a trained checkpoint + an
lm-eval run. See the README "Truth Boundary" / "Parameter Disclosure" sections.

## Curriculum & the precomputed lane (stage-3 note)
The 5-stage curriculum is honored fully in the ONLINE data path. In the canonical OFFLINE
precomputed-KD lane, each stage maps to its own precomputed shard set; the stage-3 source *mixture*
(the small TR/synthetic blend) is an online-path feature — the offline lane consumes the pre-baked
stage-3 shards as produced, so the mixing ratio is a precompute-time decision, not a train-time one.
