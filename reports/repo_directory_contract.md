# Repo Directory Contract

> Cross-reference (added 2026-07-19): `reports/folder_structure_policy.md` states a terser,
> overlapping version of the same generated-vs-source layout rule below. This file is the
> canonical, enforced version (see Enforcement); that one is a short-form summary. If they
> ever appear to disagree, this file wins — update the other to match rather than resolving
> the conflict by editing here first.

## Canonical Durable Areas
- `docs/`: official structural or explanatory documentation
- `reports/`: generated or curated closure, benchmark, readiness, and truth artifacts
- `artifacts/`: handoff, package, and delivery bundles
- `interfaces/`: canonical schemas and machine-readable contracts
- `scripts/`: real entrypoints and automation helpers
- `tests/`: verification surfaces
- `datasets/`: source notes, hashes, and repo-local validation inputs
- `adr/`: architecture decision records
- `runbooks/`, `checklists/`, `benchmarks/`, `configs/`, `knowledge/`: reserved canonical homes when used

## Generated-Content Rule
- Generated reports belong under `reports/`.
- Generated packages belong under `artifacts/` or purpose-built runtime/output roots.
- Temporary, cache, and debug clutter must not become durable repo truth.

## Enforcement
- `python3 scripts/sync_manifest.py --root . --manifest reports/release_manifest.json --structure docs/PROJECT_STRUCTURE.md --matrix reports/file_sync_matrix.json --sync-report reports/project_structure_sync_report.json --policy-report reports/policy_sync_report.json`
