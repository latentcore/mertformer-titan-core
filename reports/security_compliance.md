# Security & Compliance Overview

## Purpose
This document summarizes the security posture, controls, and compliance alignment targets for MertFormer Titan deployments. It is **not** a certification and does not claim formal compliance without third‑party audit.

## Scope
- Model training pipeline and operator‑mode gates
- Inference deployments (on‑device / edge)
- Logging, auditability, and data handling

## Data Handling
- Default design is **offline‑first** and **on‑device**.
- Training data sources are controlled and logged (reproducibility stamp).
- No external data exfiltration is required for inference.

## Access Control
- Principle of least privilege for training and deployment environments.
- Secrets loaded via environment variables and excluded from logs.

## Logging & Auditability
- Run manifests include git hash, config, seed, and dataset hashes.
- JSONL logs are chained with hash continuity for forensic integrity.
- Operator‑mode gates produce structured logs for evidence.

## Model Safety
- Non‑finite kill‑switch prevents unstable runs from continuing.
- Failure budget triggers pivot/debug when learning stalls.
- Checkpoint restore drills validate state integrity.

## Environment Isolation
- Supports air‑gapped and offline deployment workflows.
- Optional containerized runtime (Docker) for reproducible environments.

## Incident Response (Operational)
- Fail‑fast guardrails on NaN/instability.
- Clear audit trail for post‑mortem analysis.
- Manual kill/resume runbook available.

## Compliance Alignment Targets
(Alignment intent only; no certification claimed.)
- ISO 27001 controls: logging, access control, change management.
- GDPR principles: data minimization, purpose limitation, access control.
- Defense‑grade posture: offline inference, restricted data access.

## Out of Scope (Current)
- Formal certification audits (ISO/SOC/FedRAMP).
- Legal approvals or export compliance.

## Next Steps
- Integrate customer‑specific security requirements.
- Complete formal audit process if required.
