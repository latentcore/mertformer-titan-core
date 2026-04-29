# Science of Scaling / Tokens Interview Prep

## Core Question Bank
1. What is the strongest systems or scaling problem you have solved?
2. How do you distinguish measured evidence from projection in a live technical discussion?
3. If throughput drops during training, what metrics do you inspect first?
4. How do you reason about latency versus throughput versus utilization?
5. What makes a training or evaluation result trustworthy enough to narrate externally?
6. What is the most defensible low-level or runtime story in your repo today?
7. Why does your repo keep fallback and maturity labels explicit?
8. What would still need to happen before you made stronger checkpoint-level claims?

## Strong Answer Skeletons
### Strongest Systems or Scaling Problem
My strongest answer is the repo-wide systems/evidence story: I built low-bit runtime and backend-routing surfaces, but the real value was making readiness, fallback semantics, manifests, and claim boundaries explicit so the repo can say exactly what it has measured and what still depends on a real training run.

### Measured vs Projected
I explicitly separate what is measured now, what is a target, and what is a vision surface. If a claim is not tied to a real checkpoint or a real report, I will say that directly rather than smooth it over.

### Five Metrics When Throughput Drops
1. effective tokens or examples per second
2. backend or path actually selected
3. step or kernel latency distribution
4. device memory pressure and allocation churn
5. non-compute stall time such as data, orchestration, or transfer overhead

### Latency vs Throughput vs Utilization
Latency is the time one unit of work takes; throughput is sustained work rate; utilization only tells me whether a resource is busy, not whether it is the active bottleneck. I want all three before I make optimization decisions.

### Trustworthy External Narration
I want a real artifact chain: run identity, checkpoint identity, benchmark or raw outputs, and docs that match those artifacts. If that chain does not exist, I keep the claim in target or future-work territory.

## Gaps To Say Clearly
- I do not yet have a real owned long-run checkpoint story for this repo.
- The offline-clean dataset lane closes the repo contract, but it is not a claim of large-scale ETL ownership.
- Some eval surfaces are deterministic mini harnesses for discipline and contracts, not frontier-grade capability benchmarks.
- My fit is strongest on systems rigor, evidence hygiene, and research engineering instincts rather than on claiming that I have already operated Anthropic-scale training infrastructure alone.

## Evidence To Keep Primary
- `applications/anthropic/application_strategy.md`
- `applications/anthropic/project_summary.md`
- `applications/anthropic/measured_evidence_summary.md`
- `START_HERE.md`
- `docs/PROJECT_MASTER_TRUTH.md`
- `reports/final_truth_matrix.md`
- `reports/known_limits_v1.md`

## Evidence To Keep Secondary
- `TECHNICAL_REPORT.md`
- forecast or architecture-rationale reports
- any document that sounds larger than the current checkpoint-bound evidence

## Three-Minute Repo Pitch
MertFormer Titan is an offline-first AI systems repository. The current repo is a pilot-ready pre-training baseline, not a finished trained-model release. The strongest signals are a claim-safe verification surface, a train-readiness contract that currently reports TRAIN_ALLOWED with READY_REMOTE_BOOTSTRAP, an active recommended start lane of remote_bootstrap, a low-bit and multi-backend runtime lane with explicit fallback semantics, an offline assistant foundation built around governed local retrieval, and a chess proof lane that treats benchmark honesty as a feature rather than a liability. The main missing class is post-run evidence: a real training run, trained checkpoints, checkpoint-bound benchmarks, a trained demo bundle, and trained export measurements.

## Pair-Style Debugging Reminders
- narrate what you know versus what you are inferring
- localize the problem before proposing a fix
- say what signal would falsify your current hypothesis
- separate correctness fixes from performance fixes
- do not hide uncertainty; use it to guide the next measurement
