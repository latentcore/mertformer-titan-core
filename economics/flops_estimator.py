"""Simple FLOPs estimator (approximate).

This is a rough-order tool for planning. It is **not** a precise profiler.
"""
from __future__ import annotations

import argparse

DEFAULT_PARAMS = 2.64e9


def estimate_training_flops(params: float, tokens: float) -> float:
    return 6.0 * params * tokens


def estimate_inference_flops(params: float, tokens: float) -> float:
    return 2.0 * params * tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=float, default=DEFAULT_PARAMS, help="Parameter count")
    parser.add_argument("--tokens", type=float, required=True, help="Total tokens")
    args = parser.parse_args()

    train = estimate_training_flops(args.params, args.tokens)
    infer = estimate_inference_flops(args.params, args.tokens)

    print(f"Training FLOPs (approx): {train:.3e}")
    print(f"Inference FLOPs (approx): {infer:.3e}")


if __name__ == "__main__":
    main()
