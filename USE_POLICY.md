# Use Policy — MertFormer Titan

## Purpose
This policy defines acceptable and restricted use of the MertFormer Titan
codebase, models, and artifacts.

## Allowed Use
- Research and evaluation in controlled environments
- On-device/edge experimentation
- Internal prototyping with compliance review

## Restricted Use
- Deployments that make high-stakes decisions without human oversight
- Any use that violates dataset licenses or privacy rules
- Automated systems that enable harm, harassment, or abuse

## Data Handling
- Do not upload sensitive or regulated data without explicit approval
- Follow data minimization and retention policies
- Verify new dataset additions against `datasets/LICENSES.md` and `datasets/SOURCES.md` before training

## Security
- Keep secrets (.env, tokens) out of version control
- Use preflight checks before running training jobs

## Enforcement
Violations may result in access revocation and reporting to appropriate owners.
