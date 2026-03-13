# V2 Backlog Seed (Post V1 Closure)

## Completed in Build30 V2 Refactor
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

## Engineering Rules
- no direct edits on v1 closure branches
- one feature family per PR
- evidence-first payload updates
- fail-fast schema enforcement retained
