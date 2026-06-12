# Microsoft Founders Hub Application Draft

## Company
- Name: MertFormer
- Website: Private repository
- Stage: Prototype / Pre-train

## Product Summary
MertFormer Titan is a 2.64B design-target, edge-native coding model optimized for on-device inference (current build runtime total: ~3.67B). It targets near GPT-3.5 class coding capability while reducing memory and compute costs through 1.58-bit quantization and LiquidRouter MoE routing.

## Problem
Cloud-based AI is costly and unsuitable for privacy-sensitive or low-connectivity environments. Many enterprises cannot run coding copilots on-device due to hardware and energy constraints.

## Solution
A mobile-first architecture that brings capable code generation to edge devices, reducing latency and improving data sovereignty.

## Target Market
Enterprises with regulated data, defense-grade workflows, and low-connectivity environments requiring local inference.

## Differentiation
- Hardware-aware design instead of post-quantization.
- Temporal routing (LiquidRouter) reduces expert collapse.
- Reproducibility gates and forensic run logging.

## Traction
- Architecture and pipeline implemented.
- Internal gate suite and asset stack complete.
- Benchmark runners prepared (HumanEval, MBPP).

## Business Model
Licensing and enterprise deployment of on-device inference stacks.

## Team
- Founder: Systems-focused engineering lead.

## Funding
- Current funding: Not disclosed.
- Use of credits: Training infrastructure, benchmarking, and deployment validation.

## Risks and Mitigations
- Risk: Training cost and convergence.
- Mitigation: Failure budget and telemetry-driven execution gates.


Build 30 V2 refactor note: dedup pipeline, parallel MoE dispatch, CfC fast path, stricter training gates.
