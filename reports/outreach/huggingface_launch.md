# Hugging Face Launch Draft

## Short Description
Build30 now ships with a canonical terminal-first Kaggle closure lane focused on reproducible packaging and claim-safe evidence.

## Suggested Model/Space Text
This project distinguishes `measured`, `target`, and `vision` claims. The current repo-side surface is training-ready, and the canonical Kaggle lane packages runtime evidence, checkpoint manifests, and closure artifacts. A real trained checkpoint is still required before benchmark-grade claims become valid.

## Canonical Command
```bash
bash zero_touch_start.sh --kaggle-onefile --mode train-end --profile auto
```

## Disclosure Notes
- Kaggle GPU type and quota are floating/account-dependent.
- Auxiliary compare/text surfaces are supportive evidence, not trained benchmark claims.
- Mobile/device measurements remain post-run evidence until produced on real hardware.
