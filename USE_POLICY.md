# Use Policy — MertFormer Titan

## Purpose
This policy defines acceptable and restricted use of the MertFormer Titan codebase, models, and artifacts.

## Allowed Use
- Research and evaluation in controlled environments
- Offline-first and edge-native experimentation
- Internal prototyping with compliance review
- Decision-support systems with clear human approval boundaries

## Restricted Use
- High-stakes deployment without human oversight
- Harmful autonomy, covert surveillance, or unauthorized tracking
- Any use that violates dataset licenses, privacy rules, or policy boundaries
- Presenting unverified capability as factual intelligence or measured performance

## Output Truth Modes
- `verified`: source-backed, measured, or artifact-backed statements
- `hypothesis`: forward-looking but not yet proven statements
- `creative_or_folklore`: explicitly labeled narrative or stylistic output
- Default mode is `verified`

## Data Handling
- Do not upload sensitive or regulated data without explicit approval
- Follow data minimization and retention policies
- Verify dataset additions against `datasets/LICENSES.md`, `datasets/SOURCES.md`, and `datasets/hashes.json`

## 45K Guardrail
- 45K readiness is the primary ship gate for this pass
- If a task increases risk to 45K readiness, reproducibility, or closure confidence, it moves to phase-2
- No open-ended architecture refactor before 45K

## Security and Governance
- Keep secrets out of version control
- Use preflight and readiness checks before training
- High-risk actions require human approval and auditable logs

## Enforcement
Violations may result in access revocation and reporting to appropriate owners.
