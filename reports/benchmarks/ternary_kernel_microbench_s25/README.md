# Ternary CPU-Kernel Microbenchmark — Galaxy S25 (NEON)

**Claim class: measured** · **Date: 2026-06-27** · **Device: physical Samsung Galaxy S25 (single-thread, CPU/NEON)**

## Scope boundary (read first)
This is a **standalone single-op microbenchmark** of one ternary matrix-multiply
kernel, hand-written as performance evidence. It is **NOT**:
- full-model tokens/sec,
- an NPU measurement (this is CPU/NEON, not the S25 NPU),
- integrated into the canonical `bitlinear.py` / `triton_fused_bitlinear.py` path.

It is **related, measured kernel evidence** — not main-architecture performance.
Treat it the same way the rest of this repo treats its boundaries
(measured / target / vision).

## Setup
- N=256, ITERS=8, seed=1453, weights ternary {-1,0,+1}, activations uniform[-1,1]
- Float baseline and ternary kernels use **identical** N/ITERS/data/seed
- Ground truth in **double precision**; **no `-ffast-math`**
- Single thread (S25 `nproc` path); numbers from the **physical device**, not qemu

## Results (best config = 4×8)

| Tier | Time (ms) | Speedup | Accuracy |
|------|-----------|---------|----------|
| float-naive (reference) | 426.5 | 1.00× | — |
| **NEON FMA exact** | 141.9 | **3.01×** | **bit-exact** (max diff vs float = 0.000e+00) |
| **NEON SDOT turbo** | 51.4 | **8.29×** | **approximate** (~0.4% rms, int8 activations) |

Two honesty notes:
1. **FMA tier is bit-exact** — identical math to the float baseline (x·w is exact
   for w ∈ {-1,0,+1}), so the speedup is real with zero error.
2. **SDOT tier is approximate** — it quantizes activations to int8 and uses
   `vdotq_s32` (16 MAC/instr). It is fast but **not** bit-exact (~0.4% rms). This is
   the low-bit deployment-mode number, reported as such.

## What this does NOT prove
Full-model t/s, NPU speed, end-to-end latency, that the 3.67B model runs at any
given rate, or production/mobile readiness. Capability still depends on the 45K
checkpoint-bound run.

## Reproducibility
Source: `ternary_matmul_arm.cpp` (in this directory). FMA tiers self-verify
`max|kernel − float_naive| = 0` at runtime; SDOT reports rms vs double ground
truth. Kernel logic was cross-compiled with both gcc-aarch64 and clang and run
under qemu-aarch64 before on-device measurement.

Build & run on device (CxxDroid):
```
g++ -O3 ternary_matmul_arm.cpp -o tern && ./tern
# optional extra: -O3 -march=armv8.2-a+dotprod
```
