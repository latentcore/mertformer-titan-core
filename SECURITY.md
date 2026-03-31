# Security & Safety Policy

## Scope
This document defines the security, governance, and deployment boundaries for MertFormer Titan.

## Core Safety Boundaries
- Do not use the system for malicious, harmful, or unlawful purposes.
- Do not frame the system as an autonomous offensive or covert-surveillance tool.
- High-risk decisions require human approval and must remain auditable.

## Truth and Evidence Discipline
- No claim without evidence.
- `measured`, `target`, and `vision` statements must remain distinct.
- `verified`, `hypothesis`, and `creative_or_folklore` output modes must not be conflated.

## Readiness Guardrail
- 45K readiness is the primary ship gate for this pass.
- Runtime-invasive changes are allowed only when clearly non-invasive to the 45K path; otherwise they move to phase-2.
- Medium Refine is the official risk ceiling for this pass.

## Data and Secret Handling
- Respect dataset licenses, provenance, and retention boundaries.
- Keep secrets out of version control and release artifacts.
- Maintain audit-ready manifests for datasets, reports, and release bundles.

## Reporting
- Record incidents in `postmortems/` using the provided template.
- Update mitigation steps after resolution.

## Status
This policy is active for the Build 30 Max Closure pass.
