# Update-First Policy

## Core Rule
- Audit what already exists before adding a replacement.
- If the current path is correct, keep it.
- If it is incomplete, extend it.
- If it is broken, repair it.
- If drift exists, synchronize it.

## Required Classification Before Change
- active path
- legacy path
- no-touch surface
- high-risk surface
- closure surface

## Forbidden Shortcuts
- blind rewrite without reading the current implementation
- duplicate entrypoint for the same responsibility
- key refactor without a measured or maintainability reason
- replacing a canonical path only because a new one feels cleaner

## Enforcement
- `scripts/verify_all.sh`
- `scripts/build_code_truth_audit.py`
- `scripts/build_closure_governance_pack.py`
