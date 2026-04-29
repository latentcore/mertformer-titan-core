# Internal AGI Gap Map (Build 30 V2)

This document is an internal unresolved-problems register. It is not a public AGI claim and it does not upgrade repo readiness into capability proof.

## Current Truth Boundary
- Current repo-side readiness verdict: `TRAIN_ALLOWED`
- Current repo-side reason code: `READY_REMOTE_BOOTSTRAP`
- Current recommended active lane: `remote_bootstrap`
- Strict local lane: `offline_clean`
- Remaining non-winning blockers: `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`, `online_teacher:MISSING_HF_TOKEN`
- Most important missing evidence class: real owned training run, trained checkpoints, checkpoint-bound benchmarks, trained demo bundle, and trained export or device measurements

## Status Legend
- `implemented_scaffold`: code surface exists, but not claim-grade proof
- `partial_research`: some ingredients exist, but the core problem is still open
- `open_problem`: no convincing closure today

## Unresolved Math + Systems Register

| Problem Class | Why It Is Still Open | Current Repo Scaffold | Status | What Would Count as Real Progress |
| --- | --- | --- | --- | --- |
| Quadratic or subquadratic context scaling | Long-context cost and memory growth still dominate practical scaling. | `layers/mla.py`, long-context positioning, runtime notes in docs | `partial_research` | Measured long-context training and inference evidence with sustained quality and bounded cost |
| Memory wall and host-device bandwidth | Low-bit weights do not remove activation, cache, and transfer bottlenecks by themselves. | BitNet-style layers, export/runtime notes, benchmark scaffolds | `partial_research` | Measured bandwidth profile, cache policy evidence, and end-to-end throughput gains on trained checkpoints |
| Low-bit and sparse training stability | Sparse routing and low-bit math can destabilize gradients, routing, or convergence. | `layers/bitlinear.py`, `layers/moe.py`, `scripts/titan_preflight.py`, safety guards | `partial_research` | Long-run convergence evidence plus ablations against denser or higher-precision baselines |
| Router collapse and expert load balance | MoE usefulness depends on healthy expert utilization over time. | `layers/moe.py`, router health signals, tolerance checks | `partial_research` | Real run telemetry showing stable expert usage, no collapse, and justified quality gain |
| Continual learning and catastrophic forgetting | Training a new skill without erasing old capability is still unresolved. | `train/continual_adapter.py`, feature flags, roadmap docs | `implemented_scaffold` | Sequential-task evidence showing retained capability after adaptation |
| Calibrated uncertainty and abstention | Knowing when the model should say “I do not know” remains unsolved. | Governance and verification surfaces, but no calibrated uncertainty layer | `open_problem` | Confidence calibration benchmarks, abstention policy, and measured truthfulness gains |
| Long-horizon credit assignment | Useful planning over long chains remains much harder than local next-token prediction. | Orchestrator planner, verifier, and runtime scaffolds | `implemented_scaffold` | Benchmarks showing stable multi-step planning with verified task completion |
| Causal abstraction and counterfactual reasoning | Pattern completion is not the same as robust cause-and-effect reasoning. | World-model and cognitive-extension scaffolds | `implemented_scaffold` | Controlled causal tasks with intervention or counterfactual evaluation |
| World modeling and partial observability | A reliable latent world state under uncertainty is not solved here. | `layers/world_model_head.py`, orchestrator sensing modules | `implemented_scaffold` | Measured prediction quality in interactive or simulated environments |
| Tool-grounded planning reliability | Tool use is only useful if the agent can verify tool outputs and recover safely. | `orchestrator/tool_executor.py`, governance, verifier, swarm runtime | `implemented_scaffold` | Tool-use benchmarks with safety checks, verification, and low failure rates |
| Mechanistic interpretability to intervention | Reading internals is not enough; steering or intervention proof is missing. | Audit and verifier surfaces, reporting discipline | `open_problem` | Interventions that predictably change model behavior without breaking quality |
| Adversarial robustness and auditability | Powerful systems need robust measurement under prompt attacks and misuse. | Policies, governance docs, failure-budget logic, tool-abuse docs | `partial_research` | Independent red-team results, jailbreak resistance evidence, and audit-grade traces |

## Capability Gaps That Remain Open Even After Repo Closure
- Human-level novel problem solving
- Transferable planning across domains without narrow prompting tricks
- Memory reliability under long-running agent workloads
- Grounded multimodal understanding tied to real tasks
- Robust truthfulness under pressure, ambiguity, and adversarial prompting
- Safe tool-grounded execution with bounded failure modes
- Auditable deployment standards that survive external review

## What Exists Today Versus What Does Not

### Implemented scaffolds that matter
- Memory, planner, verifier, governance, self-audit, and swarm runtime surfaces exist in code.
- Zero-touch training and post-train orchestration surfaces exist.
- Readiness, freeze, manifest, and claim-boundary governance are unusually explicit.

### What still does not exist as evidence
- A real long training run
- A trained checkpoint story
- Checkpoint-bound benchmark proof
- Measured device/runtime proof on trained artifacts
- Any independent basis for AGI language

## Internal Summary
- AGI proximity: far
- Repo-side engineering closure: strong
- Current blocker to stronger claims: evidence, not folder count
- Most important next step: owned run plus checkpoint-bound measurement

## Policy Note
This file is internal. It is a registry of unresolved mathematical and systems problems, not a capability claim.
