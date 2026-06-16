# Cost Model (Template)

This document provides a lightweight cost model for training and inference.

## Variables
- `P`: parameter count
- `T`: total tokens
- `F`: FLOPs per token (approx.)

## Approximation
- **Training FLOPs** ≈ `6 * P * T` (rough estimate)
- **Inference FLOPs** ≈ `2 * P * T`

Populate with real measurements after production runs.
