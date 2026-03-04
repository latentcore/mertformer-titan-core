# MertFormer Titan One-Pager

## Summary
MertFormer Titan is a 2.64B design-target parameter edge-native coding model (latest measured runtime total: ~3.70B) designed to deliver near GPT-3.5 class capability at mobile compute cost. It combines 1.58-bit quantization, LiquidRouter MoE routing, and long-context attention for efficient on-device inference.

## Problem
Enterprise AI is expensive, cloud-dependent, and risky for privacy-sensitive workflows. Latency, data sovereignty, and operating cost block adoption in regulated and low-connectivity environments.

## Solution
A mobile-first, on-device coding model that minimizes inference cost while preserving practical coding capability. The architecture prioritizes energy efficiency, privacy, and operational stability.

## Product
- Edge-native coding model with on-device inference.
- Quantized 1.58-bit weights for large memory savings.
- LiquidRouter MoE for stable routing under low compute.
- Long-context attention for codebase-level tasks.

## Differentiators
- Hardware-aware architecture rather than post-quantization.
- Temporal routing (LiquidRouter) improves expert balance.
- Forensic run logging and reproducibility gates.

## Roadmap
- Operator-mode gates and failure budget enforcement.
- Master training run with telemetry-driven execution.
- Internal benchmarks on HumanEval and MBPP.

## Business Model
Licensing for enterprise on-device deployments and private inference stacks.

## Ask
Pilot partners for edge coding workflows and strategic deployment design.
