# Turk Telekom Call FAQ

This note is for early discovery or compute-collaboration conversations. It is intentionally short and claim-safe.

## 1. What is the project right now?
- MertFormer Titan is a pilot-ready pre-training baseline for an offline-first, auditable, edge-oriented AI stack.
- The current architecture centers on a 2.64B design target with a latest measured runtime total of ~3.70B.
- The repository is ready for controlled training and verification flows, but full benchmark claims remain blocked until a trained checkpoint exists.

## 2. What is still missing?
- The main missing asset is an owned trained checkpoint plus measured benchmark evidence.
- Without that checkpoint, model-quality claims must stay in `NOT ELIGIBLE FOR CLAIM` territory.
- The next milestone is not feature expansion; it is a real training run with clean evidence capture.

## 3. Why should Turk Telekom care?
- The strongest fit is not “train a frontier chatbot from scratch.”
- The stronger fit is privacy-preserving, offline-capable, auditable AI for regulated and infrastructure-sensitive workflows.
- The architecture direction aligns with data sovereignty, edge deployment, and controlled enterprise pilots.

## 4. Why not just use open-source models?
- Open-source models may already be enough for many generic use cases.
- The value here is not “LLM exists”; the value is controlled architecture, offline-first design, auditability, and the option to shape the stack around Turkish enterprise constraints.
- The best commercial entry is a narrow PoC where these constraints matter.

## 5. What should the ask be?
- Ask for a scoped PoC or compute-backed validation lane, not a vague AGI-scale research commitment.
- Good framing: “The repo is pilot-ready at the systems level; we want a controlled training/evaluation path for a concrete private-edge use case.”
- If the PoC works, larger model or deployment discussions can follow with evidence instead of speculation.

## 6. Which technical details should be memorized first?
- Distillation and teacher-student setup.
- MoE routing basics and what makes LiquidRouter different.
- MLA-labeled GQA as the current attention truth.
- BitNet / low-bit efficiency tradeoffs.
- ONNX / mobile / edge deployment path.
- Why checkpoint + benchmark evidence is the real gate.
