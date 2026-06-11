# MertFormer Titan - Executive Technical Brief

**Version:** Build 30 V2
**Date:** May 2026
**Status:** Pre-training baseline; repo-side remote-bootstrap start allowed; final checkpoint/evaluation pending
**Author:** Mert / MertFormer AI Team
**Origin:** Turkiye
**Public summary:** https://gist.github.com/latentcore/dac0aa0c56b12177e4a0e8e8f684bccf

---

## 1. Glossary of Technical Terms

| Term | Plain-English Meaning |
|---|---|
| **AI (Artificial Intelligence)** | Software designed to perform tasks that normally require reasoning, language understanding, planning, or pattern recognition. |
| **LLM (Large Language Model)** | A language model trained on large text/code datasets to understand and generate language, code, or structured outputs. |
| **Parameter** | A learned numerical value inside a model. Parameter count is a capacity indicator, not a quality guarantee. |
| **Pre-training** | The main training phase where a model learns broad patterns before specialized tuning or deployment. |
| **Checkpoint** | A saved model state. Reliable capability claims require a checkpoint tied to reproducible evaluation artifacts. |
| **Benchmark / Evaluation** | A standardized test of model performance. In this project, benchmark claims remain pending until a trained checkpoint exists. |
| **Token** | A unit of text processed by a model, such as a word, word fragment, number, or symbol. |
| **Low-bit architecture** | A model design that uses fewer bits for weights or operations to reduce memory and compute cost. |
| **BitNet-style / 1.58-bit direction** | A low-bit modeling approach where many weights are represented with ternary-like values such as -1, 0, and +1. |
| **MoE (Mixture of Experts)** | An architecture where selected expert sub-networks are activated per token instead of activating the full model every time. |
| **Top-2 routing** | A MoE policy where two experts are selected for each token. |
| **LiquidRouter** | This project's temporal routing component for MoE. It uses short-history convolutional context to help choose experts. |
| **CfC / Liquid Neural Networks** | Continuous-time neural layers intended to model temporal flow or changing state over a sequence. |
| **GQA (Grouped-Query Attention)** | An efficient attention mechanism that shares key/value heads across groups of query heads to reduce memory cost. |
| **MLA-labeled GQA** | The code uses the class name MLA, but the current inspected attention implementation is best described as GQA-style, not a proven full latent-MLA bottleneck. |
| **Knowledge Distillation** | Training a smaller student model with guidance from a larger teacher model. |
| **Top-K logits / sparse shards** | A storage strategy that keeps only the highest-scoring teacher outputs instead of storing the full output distribution. |
| **DDP (Distributed Data Parallel)** | A multi-GPU training method where GPUs process different batches and synchronize gradients. |
| **H100 / H200 / B300** | High-end data center GPU families used or targeted for large AI training and validation. |
| **NCCL** | NVIDIA's library for multi-GPU communication. NCCL errors can indicate distributed-training communication failures. |
| **OOM (Out Of Memory)** | A failure where available memory is insufficient. |
| **Artifact** | Evidence produced by a run, such as logs, checkpoints, manifests, reports, examples, or archives. |
| **Manifest** | A structured file listing run configuration, artifacts, hashes, and metadata. |
| **SHA256** | A cryptographic fingerprint used to verify file integrity. |
| **ONNX** | A model export format for running neural networks across different runtimes and hardware. |
| **NPU (Neural Processing Unit)** | A specialized AI chip used in modern phones or edge devices for efficient inference. |
| **Measured / Target / Vision** | Claim labels: measured is artifact-backed, target is planned but unverified, and vision is a long-range direction. |

---

## 2. Executive Summary

MertFormer Titan is an evidence-first AI systems and model-architecture project developed in Turkiye. It combines a low-bit BitNet-style direction, Sparse Mixture of Experts, Liquid/CfC layers, temporal routing, distributed training, and strict claim-boundary documentation.

The project should not be read as a completed model-quality claim. The strongest current evidence is engineering readiness and partial operational training evidence:

- repository-side training readiness is allowed through the `remote_bootstrap` path
- a previous 2x H200 run captured startup, DDP worker boot, training progress, curriculum transition, and stable partial throughput
- the cleaned captured window did not show traceback, CUDA OOM, or NCCL failure markers
- final checkpoint, final evaluation, and final archive artifacts were not recovered

Correct short positioning:

> MertFormer Titan is a pre-training architecture validation project with repo-side remote-bootstrap readiness and partial 2x H200 operational evidence; the next step is reliable compute to produce final checkpoint-bound evaluation artifacts.

This document intentionally summarizes the full project at executive level. It carries the relevant README/Gist context without copying raw logs, full file inventories, source code, or unverified marketing claims.

---

## 3. Project Overview

MertFormer Titan is designed as an offline-first, edge-oriented AI infrastructure and language-model architecture. Its long-term direction is efficient private inference on controlled local hardware rather than always-on cloud dependency.

The project has three layers of value:

1. **Architecture:** low-bit model design, sparse expert routing, liquid temporal dynamics, and efficient attention.
2. **Systems engineering:** training gates, distributed execution, logging, checkpoint planning, artifact packaging, and failure analysis.
3. **Evidence discipline:** every strong claim is separated into measured, target, or vision categories.

Current status:

- The repository has implemented architecture and training surfaces.
- The repository reports `TRAIN_ALLOWED` through the `remote_bootstrap` lane.
- The full 45K training run is still pending.
- Final model quality is not yet proven because final checkpoint-bound evaluation artifacts are missing.

What the project is not claiming:

- not a trained model yet
- not benchmark-verified
- not production-ready
- not mobile-ready
- not security-certified
- not an AGI or ASI claim
- not a claim that it beats Claude, Llama, Gemma, Phi, or any other model

---

## 4. Architecture Summary

MertFormer Titan's main architecture direction combines four components:

### 4.1 BitNet-Style Low-Bit Layers

The architecture uses BitNet-style low-bit linear layers in core model paths. The intended benefit is lower memory and inference cost compared with standard 16-bit or 32-bit weights.

Important boundary: theoretical compression is not the same as measured deployment speed. Mobile or production performance requires trained checkpoints and device profiling.

### 4.2 MLA-Labeled GQA Attention

The attention module is labeled MLA in the codebase, but the current implementation is best described as GQA-style attention:

- 16 query heads
- 8 key/value heads
- KV sharing to reduce memory pressure
- RoPE positional encoding with long-context-oriented settings

Important boundary: true latent-MLA bottleneck behavior is not claimed as current measured implementation.

### 4.3 LiquidRouter + Sparse MoE

The model uses Sparse Mixture of Experts with 8 experts and top-2 routing. LiquidRouter is a temporal routing component that uses a short rolling context window to inform routing decisions.

The intended benefit is more context-aware expert selection and better efficiency. This remains a target until full training and evaluation evidence exists.

### 4.4 Liquid / CfC Layers

Liquid/CfC-style layers appear at selected layers to provide continuous-time dynamics and temporal state behavior. These layers are intended to help sequence processing beyond a purely static feed-forward path.

Important boundary: the presence of the mechanism is code-level evidence; its final model-quality effect requires trained checkpoint evaluation.

### 4.5 Model Specification Snapshot

| Item | Current Claim-Safe Statement |
|---|---|
| Design target parameter count | 2.64B design/positioning target |
| Measured runtime parameter count | approximately 3.70B in current repo artifacts |
| Main architecture | BitNet-style + MoE + Liquid/CfC + MLA-labeled GQA |
| Default model depth | 18 layers in main configuration |
| Experts | 8 experts, top-2 routing |
| Tokenizer direction | Llama-3-family tokenizer surface in the main path |
| Sequence length | 4096 target/default in main documentation |
| Status | pre-training / claim-unverified for model quality |

---

## 5. Training Strategy

The broader training plan uses teacher-student knowledge distillation. A larger teacher model can guide the smaller student model by providing target distributions or top-k logits.

The practical training plan separates two paths:

- **Strict offline lane:** requires precomputed teacher logits. This lane is currently blocked until required logits exist or Phase-0 precompute becomes actionable.
- **Remote-bootstrap lane:** allows a rented machine to inject credentials and fetch/generate required artifacts at runtime. This is the current recommended lane.

Current readiness report:

- `final_status: TRAIN_ALLOWED`
- `decision_reason_code: READY_REMOTE_BOOTSTRAP`
- `recommended_path: remote_bootstrap`

Remaining non-winning blockers:

- `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`
- `online_teacher:MISSING_HF_TOKEN`

Training reliability surfaces include:

- start gates
- run locks
- checkpoint planning
- OOM and NaN/Inf handling paths
- distributed training support
- post-train artifact bundle planning
- evaluation and benchmark harnesses prepared for after checkpoint availability

---

## 6. Current Evidence Boundary

### 6.1 Latest Recorded Repository Evidence

The project documentation records:

- `pytest`: 354 passed, 4 skipped in the latest recorded closure context
- code quality checks passed in the latest recorded closure context
- `bash scripts/verify_all.sh`: reported OK in the latest recorded closure context
- training readiness decision: `TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP`

These are repository and systems-readiness signals. They are not model-quality benchmark results.

### 6.2 2026-05-14 Partial 2x H200 Evidence

A captured 2x H200 run produced partial operational evidence:

| Field | Captured Value |
|---|---:|
| GPUs | 2x H200/H100-class captured DDP startup |
| World size | 2 |
| Batch per GPU | 96 |
| Global batch | 192 |
| Sequence length | 512 |
| Last captured step | 1880 |
| Captured duration | approximately 89.29 minutes |
| Captured samples | 360,960 |
| Captured supervised tokens | approximately 45.56M |
| Tail throughput | approximately 8,505 supervised tokens/sec |
| Last captured loss | 0.041666 |
| Last weighted CE loss | 0.001362 |
| Last grad norm | 0.088019 |

Captured operational positives:

- clean repository/package materialization
- two GPU DDP worker boot
- fixed held-out evaluation set creation
- deterministic case-bank creation
- live child heartbeats
- stable captured training throughput
- curriculum transition into `multi_op` at the expected progress boundary
- no captured traceback, CUDA OOM, or NCCL marker in the cleaned evidence window

### 6.3 What This Evidence Does Not Prove

The same captured run does not include:

- final evaluation
- saved final checkpoint
- terminal recovery payload
- final archive completion
- benchmark results
- public checkpoint release

Therefore, the run is partial operational evidence only. It does not prove final arithmetic capability, general LLM capability, mobile deployment, production readiness, or benchmark superiority.

Training loss alone is not treated as proof of capability. The success condition remains checkpoint-bound held-out exact accuracy plus reproducible artifacts.

---

## 7. Compute Requirement

### 7.1 Immediate Proof Window

Preferred immediate proof window:

- 8-10 hours of 2x H100/H200 or equivalent
- persistent disk for logs, checkpoints, reports, evaluation outputs, and archives
- reliable artifact retrieval after the job exits

Minimum useful window:

- 2-4 hours of 2x H100/H200
- useful for launch and early-curriculum validation
- may not reach final proof completion

### 7.2 Larger 45K / Scale-Up Path

For the broader 45K architecture validation path, larger hardware such as 8x H200/B300-class compute has been discussed as a target planning surface. Time and cost estimates are planning estimates only until measured on target hardware.

The next infrastructure partner must provide more than raw GPU time:

- SSH or equivalent terminal access
- `tmux` or `screen`
- persistent output volume
- visible job identity and status
- reliable log access
- checkpoint retention after process exit
- explicit artifact sync/download command

The key lesson from the partial H200 run is that the captured model/DDP path did not show a low-level failure marker in the cleaned window; the weak point was provider-side artifact retrieval and job-status reliability.

---

## 8. Expected Successful-Run Artifacts

A successful proof run should produce:

- final held-out evaluation summary
- exact accuracy
- parser / format validity
- representative generated examples
- final checkpoint path
- checkpoint manifest
- SHA256 hashes
- report JSON
- report Markdown
- JSONL training logs
- evaluation history CSV
- fixed held-out cases
- final examples
- zip archive
- tar.gz archive

Public claim after a clean proof should be limited to:

> The documented proof run completed on the stated GPU setup, produced checkpoint-bound held-out evaluation artifacts, and generated reproducible logs, manifests, and archives under the stated configuration.

Even that would not automatically prove broad LLM capability, production readiness, mobile readiness, or superiority over other models.

---

## 9. Strategic Relevance

### 9.1 Edge AI and Data Sovereignty

The long-term project direction is private, local, controlled inference with reduced dependence on always-on cloud services. This is relevant to environments where data control, connectivity, latency, and cost matter.

### 9.2 Turkish Digital Sovereignty

The project has Turkish-first documentation and a stated goal of supporting local AI capability development. This makes the project strategically relevant beyond a narrow model benchmark.

### 9.3 AI Systems Engineering Signal

The strongest current signal is not a model-performance claim. It is systems discipline:

- explicit readiness gates
- blocker reason codes
- distributed training evidence
- source-of-truth hierarchy
- artifact manifests and hash planning
- refusal to treat missing checkpoint/eval evidence as completed work

### 9.4 Research Direction

The combination of low-bit layers, sparse expert routing, temporal routing, and liquid dynamics is technically interesting. However, it should be described as an architecture validation direction until trained-checkpoint evaluations establish measurable gains.

---

## 10. Governance, Safety, and Claim Discipline

This project uses an explicit evidence boundary:

- `measured`: backed by artifact, manifest, benchmark, or log
- `target`: planned or estimated behavior, not verified
- `vision`: long-range direction outside current proof scope

Claims that should not be made from current evidence:

- trained model
- benchmark-verified model
- production-ready model
- mobile-ready deployment
- security-certified system
- broad LLM capability
- arithmetic capability without final held-out evaluation
- model superiority over Claude, Llama, Gemma, Phi, or other models
- AGI or ASI

The current project should be reviewed as a serious pre-training architecture and systems-evidence project, not as a finished AI product.

---

## 11. Repository Status Summary

| Surface | Status |
|---|---|
| Core architecture code paths | implemented in repository |
| BitNet + MoE + Liquid direction | implemented surfaces exist |
| Attention truth | MLA-labeled GQA-style attention |
| Unit/integration tests | latest recorded closure context reports 354 passed, 4 skipped |
| Offline verification | latest recorded closure context reports verify gate OK |
| Training readiness | `TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP` |
| Partial H200 evidence | captured through step 1880 and approximately 89 minutes |
| Full 45K run | pending |
| Final checkpoint | pending |
| Benchmark evaluation | not eligible for claim without checkpoint |
| Mobile/device profiling | target / roadmap, not measured |
| Production readiness | not claimed |

---

## 12. Bottom Line

MertFormer Titan is technically serious as an AI systems and architecture-validation project. It has a coherent low-bit / MoE / Liquid direction, repository-side training readiness, and partial 2x H200 operational evidence. It does not yet have final checkpoint-bound evaluation artifacts.

The next rational step is not more marketing language. The next rational step is reliable compute with persistent artifact retrieval, followed by checkpoint-bound held-out evaluation and reproducible manifests.

If that proof succeeds, the project can move from partial operational evidence to a much stronger evidence milestone. Until then, the correct public posture is disciplined, measured, and explicitly claim-limited.
