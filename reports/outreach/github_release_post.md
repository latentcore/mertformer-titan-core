# GitHub Release Post

## Title
MertFormer Titan Build30: canonical Kaggle closure lane + claim-safe post-train packaging

## Body
This release adds a single terminal-first Kaggle closure lane for Build30:

```bash
bash zero_touch_start.sh --kaggle-onefile --mode train-end --profile auto
```

What it standardizes:
- runtime GPU detection instead of hard-coded Kaggle entitlement claims
- normalized checkpoint contract (`latest.pt`, `best.pt`, `manifest.json`, `public_summary.json`, `stop_summary.json`)
- first-100-step loss snapshot
- canonical artifact index + sha256 manifest
- canonical evidence bundle zip
- auxiliary compare/text-understanding reports without narrating them as trained benchmark claims

Claim boundary:
- repo-side closure is green
- real trained-checkpoint evidence is still required for benchmark-grade claims
- Kaggle quota and accelerator availability remain account-dependent
