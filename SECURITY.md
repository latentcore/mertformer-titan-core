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

## Change Control
- Changes that could affect training behavior require explicit review and must remain auditable.
- Runtime-invasive changes are deferred unless clearly non-invasive to the canonical training path.

## Data and Secret Handling
- Respect dataset licenses, provenance, and retention boundaries.
- Keep secrets out of version control and release artifacts.
- Maintain audit-ready manifests for datasets, reports, and release bundles.

## Reporting a Vulnerability
- Report security vulnerabilities privately via GitHub Security Advisories (repository → **Security** tab → **Report a vulnerability**). Do not open a public issue.
- If Security Advisories are unavailable to you, email **mert.yunlu08@gmail.com** with `SECURITY` in the subject line. Please allow up to 7 days for an initial response.
- Operational incidents are recorded in `postmortems/` using the provided template; mitigation steps are updated after resolution.

## Status
This policy is active for the current pre-training (Build 30 V2) phase.
