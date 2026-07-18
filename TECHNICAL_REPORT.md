# Technical Analysis of the MertFormer Titan Architecture

> **External review note:** For compute sponsorship, start with
> `private/commercial/outreach_compute_sponsorship_messages.md` and
> `reports/ocean_pre45k_h200_20260514_partial_evidence.md`. This report is
> technical background and strategic framing; it is not a benchmark, production,
> deployment, AGI, or model-superiority claim.

**Date:** 2026-06-18
**Version:** v1.0 (Build 30 V2)
**Author:** Mert Yünlü

---

## 1. Executive Summary
The AI ecosystem is shifting from large cloud-based models toward on-device, energy-efficient Small Language Models (SLMs). The **MertFormer Titan v1.0 (Build 30 V2)** project combines four paradigms from recent deep-learning literature:

1.  **BitNet 1.58-bit Quantization** (Efficiency)
2.  **GQA Attention (grouped-query, current implementation)** (Memory)
3.  **Sparse Mixture of Experts (MoE)** (Capacity)
4.  **Liquid Neural Networks (LNN)** (Dynamism)

This architecture is an engineering prototype targeting next-generation on-device hardware platforms such as the Samsung Galaxy S25 and Snapdragon 8 Elite.

---

## 1.1 V2 Refactor Highlights
- Cross-dataset deduplication enabled in the data pipeline.
- MoE dispatch supports parallel gather/scatter mode.
- LiquidMixer fast path available behind `liquid_fast_path`.
- Training gates now default to fixed-step token budgeting.

## 2. Deep Technical Architecture Analysis

The cornerstone of the MertFormer Titan project is its hardware-aware structure that goes beyond standard transformer blocks. The model has a **measured runtime parameter total of ~3.67B (3,672,982,022)** — the figure used for all factual claims — yet its inference cost is significantly lower than classical models. (Note: 2.64B is the architecture *design target*, not the measured total; per `DECISIONS.md`, factual claims use the 3.67B measured figure. README and `ARCHITECTURE.md` state this distinction the same way.)

### 2.1 BitNet b1.58 and the Ternary Computing Revolution
While traditional models use 16-bit (BF16), MertFormer Titan reduces weights to values of $\{-1, 0, 1\}$ based on **BitNet b1.58** technology.

*   **Memory Savings (Theoretical, fp-simulation only — relabeled 2026-07-19, see `DECISIONS.md`):** 93.75% is the *arithmetic* reduction of packing weights to {-1,0,1} instead of BF16 — it is **not delivered by the current runtime**, which forward-passes through a plain fp simulation (`layers/bitlinear.py`'s BitNet forward, straight-through estimator), not a real ternary-math kernel. No such kernel exists yet in this repo (see the Energy Efficiency note below and `bitnet_cpu.cpp`'s own comments). The one *measured* low-bit number that IS real is **disk packing**: 17.99MB→1.06MB (~17.0x) for a saved checkpoint, verified with a matching CE difference of 6.19e-05 — a storage-format result, not a training/inference-speed result.
*   **VRAM Requirement (Estimate):** ~0.65 GB (computed for the **2.64B design-target** parameter count, not the 3.67B measured total; requires a real low-bit inference kernel, which does not exist yet — see above).
*   **Energy Efficiency (Unmeasured Target):** Up to 70x energy savings on NPUs *if* ternary math runs on optimized kernels. No such optimized kernel exists yet — the current Metal/Vulkan/NPU code paths fall back to a generic `F.linear` (`torch`) passthrough, so this figure is a projected target, not a measurement.

**Mathematical Quantization Formula (`bitlinear.py`):**
$$w_q = \text{clamp}(\text{round}(\frac{w}{\gamma + \epsilon}), -1, 1)$$

*   **Residual Scaling Effect (Target):** Signal stability is maintained across 18 layers using the 1/√2 (1/sqrt(2)) factor; empirical validation required.

### 2.2 GQA Attention (grouped-query, current implementation)
The `mla.py` module implements a GQA attention core in the `GQA` class. KV heads are reduced and shared (`num_kv_heads`) and then broadcast to query heads at runtime. (The class was formerly named `MLA`; renamed to match the implementation — a true latent-MLA bottleneck is not implemented.)

*   **Current mechanism:** GQA projection + KV head replication (not latent down/up bottleneck).
*   **Cache efficiency path:** Optional hierarchical short/long KV cache mode can reduce decode-time memory pressure (target behavior; profile-dependent).
*   **RoPE:** Long-context support with $\theta = 100,000$.
*   **Truth boundary:** Full latent-MLA bottleneck remains a roadmap item.

### 2.3 Liquid Neural Networks (CfC)
A continuous-time recurrent mixer on a few layers. Inspired by biological neurons (C. elegans), Closed-Form Continuous-time (CfC) cells use input-dependent differential equations. (Value boundary: a 12-seed toy ablation showed no measured accuracy benefit at ~30% slower wall-clock; see [ABLATION.md](ABLATION.md).)

*   **Time Perception:** The `tau` (time constant) parameter is dynamic.
*   **Continuity:** It tracks momentum between tokens.
*   **Application:** Runs at NPU speed with JIT-compiled loops (`liquid.py`).

**Formula:**
$$h(t) = A + (h_{prev} - A) \odot \exp(-\tau \Delta t)$$

### 2.4 LiquidRouter & MoE
`LiquidRouter` is implemented as a temporal Conv router (`causal depthwise Conv1d` + rolling state buffer) that informs MoE token routing.
Routing policy is token-choice top-k and should be read separately from the CfC path used by `LiquidMixer/LiquidCell`.

| Parameter | Value |
| :--- | :--- |
| Number of Experts | 8 |
| Active Experts (Top-k) | 2 |
| Router | `LiquidRouter` (Conv1d + state buffer) |
| MoE expert intermediate (`moe_intermediate`) | 8192 (SwiGLU) |
| Dense-FFN intermediate (`intermediate_size`) | 5632 (SwiGLU) |

**Strategic Edge of LiquidRouter (Claim-Safe):**
*   **Temporal routing:** It analyzes data-arrival speed and short history (`Fluid Path`) without claiming formal superiority.
*   **Causal Conv1d Integration:** Displays strategic intelligence by considering the past 4-token window (`history_window`) during expert selection.
*   **Hardware efficiency (Target):** Aims for lower routing instability and better NPU behavior; requires device profiling.

### 2.5 Layer Taxonomy (Layer-by-Layer)
MertFormer Titan's 18-layer stack, by role:
*   **L0-L2 (Foundation):** Base representation with RMSNorm stabilization and BitNet ternary linears.
*   **L3-L9 (Mid):** MoE expert distribution; first Liquid/CfC mixer at **L4**.
*   **L10-L15 (Deep):** Second Liquid/CfC mixer at **L10**; deeper feature composition.
*   **L16-L17 (Output):** Third Liquid/CfC mixer at **L16**; logits via the LM head.

---

## 3. Training Strategy: Knowledge Distillation

To bring the 2.64B model closer to the intelligence of a 70B model, a "Teacher-Student" structure was established.

### 3.1 Offline Distillation
With `distillation_manager.py`, the outputs (logits) of the Llama-3.3-70B model are pre-recorded to disk.
*   **Speed (Target / estimate):** ~12x training speedup vs. an online 70B teacher (offline precompute removes the per-step teacher forward); not yet measured at 45K.
*   **Memory:** No need to load the teacher model into VRAM during training.

### 3.2 5-Stage Curriculum
1.  **Pure Logic & Code (45%):** Structural thinking.
2.  **World Knowledge:** FineWeb-Edu.
3.  **Identity & Language (TR):** Cultural adaptation to Turkish.
4.  **Soul:** Character and instruction following.
5.  **Tool Use:** API and function calling.

### 3.3 Build30 Profile Contract (Stable vs Max-Arch)
Build30 closes with an explicit runtime profile contract:

| Profile | Contract | Activation |
| :--- | :--- | :--- |
| `stable` (default) | Regression-safe baseline for repeatable training starts | `bash run.sh` |
| `max_arch` | Enables advanced architecture flags through overlay (`mertformer_max_arch.yaml`) | `TITAN_PROFILE=max_arch bash run.sh` |

Readiness-only validation remains identical under both profiles:
```bash
bash run.sh --train-ready
```

QINN remains intentionally disabled by default (`use_qinn=false`) to preserve throughput and convergence stability in the primary training path.

### 3.4 Evidence-First Claim Policy
- Verified items are reported from gate outputs (pytest/verify/preflight/policy checks).
- Performance projections remain explicitly labeled as simulation targets until trained-checkpoint benchmarks exist.
- Dataset scope is locked in this convergence pass (manifest preserved; no major expansion).

---

## 4. Hardware Target: Samsung S25 & Snapdragon 8 Elite

MertFormer Titan is designed as an **"NPU-Native"** engine. Note: the optimized NPU/Metal/Vulkan kernels are not yet implemented — these backends currently run a generic `torch` (`F.linear`) fallback rather than dedicated low-bit shaders. The following data points are therefore **Architectural Simulation Targets** calculated from the model's operational complexity (OPs) and memory footprint, not measured throughput.

| Platform | Estimated Speed (Target) | Memory | Optimization |
| :--- | :---: | :---: | :--- |
| **Samsung S25 (NPU)** | **45 - 107 t/s** | < 2.0 GB | Full (JIT + BitNet) |
| iPhone 17 Pro | 40 - 80 t/s | < 2.5 GB | High (CoreML) |
| MacBook Pro (M4) | 110+ t/s | ~3.0 GB | Maximum (Metal, target — no optimized Metal kernel yet) |

> [!NOTE]
> Real-world performance metrics will be validated through physical device testing following the completion of the training phase.

---

## 5. Strategic Context (claim-boundary)

Commercial valuation, fundraising, and career framing are kept **out** of this technical report and
live in the private dealroom (`mertformer-titan-dealroom-private`). In scope here:

*   **Direction:** an on-device, low-bit (BitNet b1.58) + sparse-MoE + GQA architecture targeting
    NPU-class hardware (e.g. Samsung S25). Device-utilization, latency, energy, and cost figures are
    **target/estimate** until physical target-device measurement + checkpoint-bound runs exist.
*   **Market context (neutral):** the industry is adding on-device NPUs; this architecture targets
    that hardware class. No market-size, valuation, or "moat" claim is made here.
*   **Honest posture:** no benchmark, production, deployment, or model-superiority claim. The value to
    evaluate is the engineering discipline (low-bit runtime, training reliability, evidence/claim
    discipline), not a finished frontier model.

---

## 6. Forensic Verification & Security

The model's reliability is supported by explicit verification and logging mechanisms:
*   **SHA256 Chaining (designed):** training is *designed* to seal each step with the hash of the previous step (`TITAN_POC_PROOF.jsonl`); the chain is emitted by a real run — there is no completed 45K chain yet.
*   **Proof-of-Life:** Benchmark results are designed to be tied to cryptographic hashes and proof-of-life artifacts after benchmark runs.
*   **Dynamic Balance:** `z_loss` and `switch_loss` mechanisms prevent the model from collapsing into a single expert.

---

## 7. Conclusion

MertFormer Titan is an on-device-oriented architecture (BitNet + MoE + Liquid/CfC + GQA) built with a disciplined evidence boundary. The architecture is internally consistent and the hardware target is concrete, but component value and model quality remain hypotheses: there is no trained checkpoint yet. What to evaluate is the engineering discipline (low-bit runtime, training reliability, claim discipline) — not a finished model. Success now depends on operational execution and checkpoint-bound evidence.

---

## 8. Validation Roadmap (claim-boundary)

Steps required before any production or capability claim — none of the below is complete yet:
1. **Whitepaper**: A technical paper documenting the `LiquidRouter + GQA + BitNet` design and its measured ablation results (incl. the inconclusive Liquid ablation).
2. **Open Benchmarks**: Independent verification of MMLU, GSM8K, and HumanEval scores.
3. **Live Demo**: A video demonstrating 100% on-device code generation on a physical Samsung S25.

---

## 9. Speculative Research Horizons (out of scope; not implemented)

The following are **long-range research directions only**. None is implemented on the canonical training path, none is part of the 45K run or the trained model, and none is claimed as a capability — they are listed for transparency, not as features:
*   **Persistent contextual memory** — a vector-based episodic cache for user/project context across sessions (research idea; not built).
*   **On-inference plasticity** — Hebbian-style updates confined to isolated Liquid layers for real-time adaptation (research idea; not built).
*   **Adaptive gain regulation** — dynamic per-layer gating for signal stability (research idea; not built).

---

## 10. Lawful Safety Constraints

- Deployment is policy-bound and auditable.
- Human authorization is required for operational actions.
- Covert surveillance/tracking and unauthorized intervention are explicitly excluded.
- Build closure uses `Code+Test Green` criteria; heavy training evidence is reported as `Evidence Pending`.

## 11. Closure-57 Gate

```bash
python3 scripts/check_57_matrix.py
```

Generated artifacts:
- `reports/closure_57_matrix.json`
- `reports/closure_57_matrix.md`
- `reports/closure_57_matrix_TR.md`
