# Ablation: No MoE

**Purpose**: Establish a dense-only baseline by disabling MoE routing.

**Config change**:
- `use_moe: false`
- `num_experts_per_tok: 0` (if required by downstream code)

**Status**: Not run yet. Execute on training hardware and record results in `ablations/results.md`.
