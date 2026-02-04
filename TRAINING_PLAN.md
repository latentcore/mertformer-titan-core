# MertFormer Titan: Strategic Training Roadmap

This document outlines the execution plan for the Titan Training Run on Azure A100 infrastructure.

## Phase 0: Preflight & Safety Gates
**Goal:** Verify system integrity and readiness before launch.
- **Method:** `run.sh --test` + operator mode gate (full on training hardware).
- **Artifacts:** Preflight logs + operator gate logs.

## Phase 1: Distillation Run (Foundation)
**Goal:** Transfer knowledge from **Llama-3.3-70B-Instruct** (Teacher) to MertFormer (Student).
- **Dataset:** ~24B tokens (high-quality, KD-focused curriculum).
- **Method:** Offline logits distillation with **precomputed logits**.
- **Infrastructure:** 8x A100 (Founders Hub or equivalent).
- **Key config:** `max_steps=45000`, `max_seq_len=4096`.
- **Outcome:** Stable instruction-following and syntax-correct coding baseline.

## Phase 2: Agent Integration (The Swarm)
**Goal:** Specialize the model for multi-agent roles.
- **Dataset:** "Mert Archive" (15M Token Personal RAG) + Role-Specific Finetuning (QA, Security, Architect).
- **Method:** LoRA Adapters for each agent role.
- **Outcome:** Distinct agent personalities (e.g., The "Paranoid" Security Officer vs. The "Creative" Designer) emerging from the same base model.

## Phase 3: Performance Optimization (Sage Loop)
**Goal:** Production hardening and self-improvement.
- **Mechanism:** "Wisdom Loop" - Training on successful project logs and post-mortem reports.
- **Method:** Reinforcement Learning from Compiler Feedback (RLCF).
- **Outcome:** A self-healing system that stops making the same mistake twice.

## Phase 4: Evaluation & Benchmarks
**Goal:** Generate internal truth benchmarks after training.
- **HumanEval/MBPP:** Automatic run after training **if checkpoint exists**.
- **Artifacts:** JSONL outputs in `reports/benchmarks/`.

---
**Status:** READY FOR PHASE 1 LAUNCH (pending training hardware).
