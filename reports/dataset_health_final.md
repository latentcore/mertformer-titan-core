# Dataset Health Final

- generated_utc: `2026-05-22T07:41:14Z`
- validation_rows: `1500`

## Stage Health

- `stage1`: rows=`41` seed_sources=`{'inline_or_primary': 41}`
- `stage2`: rows=`30` seed_sources=`{'inline_or_primary': 30}`
- `stage3`: rows=`8` seed_sources=`{'inline_or_primary': 8}`
- `stage4`: rows=`8` seed_sources=`{'datasets/validation.jsonl': 8}`
- `stage5`: rows=`12` seed_sources=`{'datasets/golden_samples.jsonl': 12}`

## Risk Boundary

- Dataset presence and parse health are green for the remote-bootstrap lane when its contract passes.
- Claim-grade dataset lineage, large-scale provenance, and post-run consumption journals remain post-run evidence.
