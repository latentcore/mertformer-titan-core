# Systems and Performance Case Study

## One-Line Summary
This repo treats low-bit performance as a correctness-first systems problem with explicit backend routing, explicit fallback behavior, and explicit claim limits.

## Why This Matters
The strongest repo-side performance signal is not a giant speedup claim. It is the engineering discipline around:
- deterministic backend selection
- explicit parity surfaces
- explicit fallback paths
- test-covered kernel and dispatcher contracts
- measured vs projected boundaries that do not move when the story gets more ambitious

## Canonical Technical Surface
Primary code paths:
- `layers/bitlinear.py`
- `mertformer_sdk/kernels/dispatcher.py`
- `mertformer_sdk/kernels/triton_ternary.py`
- `mertformer_sdk/kernels/cpp/bitnet_cpu.cpp`
- `mertformer_sdk/kernels/metal/engine.py`

Primary verification surface:
- `tests/test_kernel_dispatcher.py`
- `tests/test_kernel_equivalence.py`
- `tests/test_cpp_kernel_loader.py`
- `reports/code_truth_delta_audit.md`

Primary command surface:
- `python3 -m pytest -q tests/test_kernel_dispatcher.py tests/test_kernel_equivalence.py tests/test_cpp_kernel_loader.py`
- `python3 scripts/bitnet_kernel_benchmark_standalone.py --shapes 2048x2048x2048,4096x2048x2048`

## What Was Actually Closed Repo-Side
1. BitLinear is wired as a real low-bit-aware layer, but it is not narrated as a universal speed claim.
2. Backend routing is explicit rather than hidden inside ad hoc runtime branches.
3. CPU remains a reference-safe parity surface.
4. Triton is kept in the repo as an experimental performance lane, not mislabeled as production-depth proof.
5. Metal, Vulkan, and NPU paths are carried as tested fallback surfaces where appropriate.
6. The repo documents exactly which surface is a benchmark path, which surface is a parity path, and which surface is still a fallback.

## Measured Repo-Side Evidence
- `reports/bench_cpp_report.json`: CPU reference benchmark completed with `avg_ms=0.0256958301179111`.
- `reports/bench_metal_report.json`: MPS surface recorded as `ok=true` in fallback-aware mode.
- `reports/bench_vulkan_report.json`: Vulkan surface recorded as `verified_fallback`.
- `reports/bench_npu_report.json`: NPU surface recorded as `verified_fallback`.
- `reports/bench_zero_copy_report.json`: zero-copy report mirrors the CPU reference result and marks the surface as repo-side verified.
- `reports/benchmark_compare_report.md`: smoke and pre-training benchmark compare is available, while trained-checkpoint compare is still post-run.

## What Changed in Engineering Terms
The important move here was not "make low-bit fast everywhere." The important move was:
- define the dispatcher contract
- preserve a safe reference path
- keep experimental acceleration paths opt-in
- measure what exists
- label what is still fallback or experimental

That is a better systems story than a premature headline speedup number.

## Interview-Ready Performance Story
If asked for the strongest performance story, the honest version is:

"I built and hardened a low-bit inference surface around BitLinear and an explicit backend dispatcher. Instead of pretending every backend was optimized, I kept CPU as the reference-safe parity path, treated Triton as experimental, marked Metal and other backends as tested fallback where appropriate, and put tests and reports around the routing behavior. The work was less about one magic kernel and more about making multi-backend behavior deterministic, measurable, and honest."

## Interview-Ready Debugging Story
If asked for the strongest systems-debugging story, the honest version is:

"I treated backend selection and fallback semantics as a debugging problem, not just a coding problem. The repo now has a traceable contract for why a backend was selected, what maturity that surface has, and which tests verify it. That means a throughput or correctness regression can be localized to the dispatcher, the backend implementation, or the claim layer instead of dissolving into vague model-level explanations."

## Five Metrics To Check When Throughput Drops
1. effective tokens or examples per second
2. backend selection and fallback reason
3. device memory headroom and allocation churn
4. kernel or step latency distribution rather than only average latency
5. data or orchestration stall time versus pure compute time

## Claim Boundary
This case study does not claim:
- trained end-to-end speedup from the real 45K run
- production-grade Triton performance
- measured mobile or edge latency superiority
- universal accelerator optimization depth

It claims the repo already contains a serious systems surface with explicit maturity labels and measured repo-side evidence.
