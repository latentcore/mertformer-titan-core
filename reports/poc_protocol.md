# Pilot / PoC Protocol

## Objective
Validate MertFormer Titan’s edge‑native coding capability and operational safety in a controlled pilot.

## Scope
- Offline or air‑gapped inference workflow
- Operator‑mode safety gates
- Benchmark outputs on agreed test set

## Duration
- Recommended: 2–4 weeks (adjustable per partner)

## Success Criteria (Example)
- Operator‑mode gates pass (kill‑switch, failure budget, restore drill)
- Stable inference without non‑finite events
- Targeted task success on pilot prompt set (customer‑defined)
- Benchmark outputs produced and reproducible

## Data & Security
- Customer provides sanitized prompt set
- No external network dependency required for inference
- Logs captured and shared per agreed policy

## Deliverables
- Benchmark output files (JSONL)
- Gate logs and manifest
- Short pilot report with results and recommendations

## Responsibilities
- MertFormer: setup, runbook, troubleshooting
- Partner: environment access, data approvals, security sign‑off

## Risks & Mitigations
- Risk: instability → Mitigation: kill‑switch and failure budget
- Risk: data constraints → Mitigation: offline workflows
- Risk: performance variance → Mitigation: agreed acceptance thresholds

## Acceptance
- Joint sign‑off based on success criteria
