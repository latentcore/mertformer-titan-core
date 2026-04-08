# Chess Onefile Profile Contract

## Canonical release-candidate profile
- `strength_4060_24h`
- support level: `baseline_supported`
- frozen long-run RTX 4060 train-start profile

## Supported portable baseline
- `production_5080`
- support level: `supported_portable_baseline`
- conservative repo-safe portable baseline

## Research-only profiles
- `strength_4060_24h_all_on_experimental`
- support level: `experimental`
- `strength_4060_24h_omni_max`
- support level: `experimental_high_risk`

## Intent
- profiles define operator-facing presets
- bundles define feature overlays
- explicit enable/disable flags override both

## Claim boundary
Profile names are execution presets only. They do not imply trained strength, benchmark superiority, or release readiness by themselves.

## Release boundary
- only `strength_4060_24h` is release-candidate eligible on the frozen chess lane
- `production_5080` remains supported for portability and shorter baselines
- research-only profiles can run and benchmark, but cannot satisfy the final release gate
