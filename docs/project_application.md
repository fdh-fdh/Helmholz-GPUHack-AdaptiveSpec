# Project Application: AdaptiveSpec

## Project Title

AdaptiveSpec: Runtime-Adaptive Mixed-Precision Speculative Decoding for LLM Inference on NVIDIA GH200

## One-Sentence Summary

AdaptiveSpec investigates whether speculative decoding for large language models can be accelerated by dynamically switching the draft model precision at runtime, balancing draft latency, acceptance rate, and GPU execution overhead on NVIDIA GH200.

## Project Motivation

Speculative decoding can improve LLM inference throughput by using a smaller draft model to propose tokens and a larger target model to verify them. In practice, the draft model creates a trade-off:

- Lower precision draft models can generate draft tokens faster, but may reduce acceptance rate.
- Higher precision draft models may improve acceptance rate, but add draft latency.

Most speculative decoding setups use a fixed draft precision. This project asks whether the draft precision should instead adapt during decoding based on runtime signals such as entropy, confidence, recent acceptance rate, context length, and latency.

## Core Research Question

Can runtime precision switching for the speculative decoding draft model improve end-to-end LLM inference speed compared with fixed-precision draft decoding?

The central GPU systems question is:

Can INT4, INT8, and FP16 draft execution be switched efficiently enough on GH200 that the gain from adaptive precision exceeds the overhead of switching, memory movement, and kernel scheduling?

## Why This Fits a GPU Hackathon

This project is primarily a GPU performance investigation rather than an application-layer LLM demo. It depends on measuring and understanding:

- Draft model latency across FP16, INT8, and INT4 execution.
- Target verification latency and acceptance behavior.
- GPU memory residency of multiple draft precisions.
- Tensor Core utilization and memory bandwidth utilization.
- Kernel launch overhead and precision-switching overhead.
- Whether speculative decoding is compute-bound, memory-bound, or scheduler-bound under different precisions.

The project is well suited for a GPU Hackathon because mentor support on profiling, quantized inference kernels, vLLM internals, and GH200-specific performance behavior would directly affect the quality of the results.

## Planned Technical Approach

AdaptiveSpec will be developed in five stages:

1. Build a minimal vLLM-based speculative decoding baseline.
2. Benchmark target-only inference and fixed draft-target speculative decoding.
3. Compare fixed FP16, INT8, and INT4 draft precision modes.
4. Run an oracle study to estimate whether adaptive precision has theoretical value.
5. Implement a lightweight runtime controller and profile it on GH200.

The initial adaptive controller will be rule-based:

```python
if entropy < T1 and recent_acceptance_rate > A1:
    precision = "INT4"
elif entropy < T2:
    precision = "INT8"
else:
    precision = "FP16"
```

The controller will only be considered successful if its overhead is much smaller than draft model forward latency.

## Current Status

The project is currently in the planning and initial prototype stage.

Completed:

- Defined the research question and GPU performance hypothesis.
- Designed the initial system architecture.
- Identified the metrics needed for baseline, static precision, oracle, and adaptive experiments.
- Created the initial repository structure and documentation plan.

In progress:

- vLLM speculative decoding baseline.
- Benchmark script for target-only and draft-target inference.
- Configuration files for target and draft models.

## Hackathon Goals

During the hackathon, the goal is to move from a baseline prototype to a credible GPU performance study.

Minimum target outcome:

- A reproducible target-only baseline.
- A reproducible fixed draft-target speculative decoding baseline.
- CSV benchmark outputs for latency, throughput, and GPU memory.
- Initial comparison between fixed draft precision modes.

Stretch target outcome:

- Oracle analysis showing when different draft precisions would have been optimal.
- A simple runtime adaptive precision controller.
- Nsight Systems or Nsight Compute traces explaining the main bottleneck.
- A short performance report with actionable conclusions.

## Success Criteria

The project will be considered successful if it can answer at least the following questions:

- Does speculative decoding improve end-to-end throughput for the selected model pair?
- How do FP16, INT8, and INT4 draft models differ in latency and acceptance behavior?
- Is precision switching overhead small enough to justify adaptive precision?
- Which bottleneck dominates on GH200: compute, memory bandwidth, kernel launch overhead, or model weight movement?

The ideal result is not simply to show that lower precision is faster. The goal is to identify when adaptive mixed precision is beneficial and when fixed precision is better.

## Expected Deliverables

- Runnable benchmark scripts.
- Model and experiment configuration files.
- Baseline benchmark CSV files.
- Static precision comparison table.
- Optional oracle analysis notebook or script.
- Optional adaptive controller prototype.
- Profiling notes and final performance report.

## Mentor Support Requested

We would especially benefit from mentor support in:

- Understanding vLLM speculative decoding internals.
- Choosing reliable ways to measure acceptance rate, TTFT, TPOT, and draft/verification latency.
- Evaluating whether INT4 and INT8 kernels are actually faster for the selected model sizes.
- Profiling Tensor Core utilization, memory bandwidth, and kernel launch overhead on GH200.
- Avoiding misleading benchmarks caused by warmup, batching, CUDA graph behavior, cache effects, or weight reloads.

## Risk and Mitigation

Risk: vLLM may not expose all desired internal metrics directly.

Mitigation: Start with end-to-end metrics, then add lightweight instrumentation or profiler traces where needed.

Risk: INT4 or INT8 draft execution may not provide the expected speedup for the selected model.

Mitigation: Treat this as a valid result and analyze whether the bottleneck is kernel support, memory movement, or acceptance degradation.

Risk: Runtime switching between precisions may be too expensive.

Mitigation: Run an oracle study first to estimate the maximum possible benefit before investing heavily in a controller.

## Initial Model Plan

The first experiments should use small enough models to run reliably before scaling:

- Target model: a compact instruction or base LLM suitable for vLLM speculative decoding.
- Draft model: a smaller compatible model from the same family when possible.
- Initial precision modes: FP16 first, then INT8, then INT4.

The exact model pair can be adjusted based on available GH200 memory, vLLM compatibility, and mentor guidance.

