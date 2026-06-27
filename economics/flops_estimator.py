"""Simple FLOPs estimator (approximate).

This is a rough-order tool for planning. It is **not** a precise profiler.

Note on MoE: the dense ``6 * params * tokens`` rule overestimates training FLOPs
for a sparse Mixture-of-Experts, where only the active experts run per token. Pass
``--active-params`` (the MoE-corrected active parameter count) to use that for the
training estimate instead of the dense ``--params`` default.
"""
from __future__ import annotations

import argparse

# design-target; measured ~3.67B params (config.py: 3,672,982,022). Override via --params for the real count.
DEFAULT_PARAMS = 2.64e9


def estimate_training_flops(params: float, tokens: float) -> float:
    return 6.0 * params * tokens


def estimate_inference_flops(params: float, tokens: float) -> float:
    return 2.0 * params * tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=float, default=DEFAULT_PARAMS, help="Parameter count (dense)")
    parser.add_argument(
        "--active-params",
        type=float,
        default=None,
        help="MoE-corrected active parameter count. When given, the TRAINING estimate "
             "uses this instead of --params (dense overestimates sparse-MoE compute).",
    )
    parser.add_argument("--tokens", type=float, required=True, help="Total tokens")
    args = parser.parse_args()

    train_params = args.active_params if args.active_params is not None else args.params
    train = estimate_training_flops(train_params, args.tokens)
    infer = estimate_inference_flops(args.params, args.tokens)

    print(f"Training FLOPs (approx): {train:.3e}")
    print(f"Inference FLOPs (approx): {infer:.3e}")


if __name__ == "__main__":
    main()
