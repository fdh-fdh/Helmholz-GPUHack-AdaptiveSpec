# Experiment Plan: AdaptiveSpec

## Goal

This document defines the initial experiment plan for AdaptiveSpec. The goal is to turn the project from a research idea into a reproducible GPU performance study with measurable baseline results.

## Main Hypothesis

Runtime-adaptive draft precision can improve speculative decoding performance when low-precision draft execution is fast enough and the acceptance-rate loss is small enough.

The key condition is:

```text
benefit from faster draft generation
>
loss from lower acceptance rate + precision-switching overhead
```

## Experiment Stages

### Stage 1: Target-Only Baseline

Purpose:

Establish the reference performance of target-only decoding.

Run:

```text
target model only
no draft model
fixed prompt set
fixed max output tokens
```

Metrics:

- Time to first token.
- Time per output token.
- End-to-end latency.
- Tokens per second.
- GPU memory usage.

Expected output:

```text
results/baseline_target_only.csv
```

Pass condition:

The script runs reproducibly and writes a CSV file with one row per prompt or one aggregate row per run.

### Stage 2: Fixed Speculative Decoding Baseline

Purpose:

Measure whether speculative decoding helps before adding quantization or adaptation.

Run:

```text
target model: FP16 or BF16
draft model: FP16 or BF16
fixed prompt set
same max output tokens as Stage 1
```

Metrics:

- Time to first token.
- Time per output token.
- End-to-end latency.
- Tokens per second.
- GPU memory usage.
- Acceptance rate if available.

Expected output:

```text
results/baseline_spec_decode.csv
```

Pass condition:

The speculative decoding run produces comparable output to the target-only baseline and records latency and throughput metrics.

### Stage 3: Static Draft Precision Profiling

Purpose:

Compare fixed precision choices for the draft model.

Experiment matrix:

| Run ID | Target Precision | Draft Precision | Adaptive |
| ------ | ---------------- | --------------- | -------- |
| S1     | FP16/BF16         | FP16/BF16        | No       |
| S2     | FP16/BF16         | INT8            | No       |
| S3     | FP16/BF16         | INT4            | No       |

Metrics:

- Draft latency.
- Target verification latency.
- Acceptance rate.
- Time per output token.
- Tokens per second.
- GPU memory usage.
- Kernel launch overhead where available.

Expected output:

```text
results/static_precision.csv
```

Pass condition:

At least FP16 draft and one lower-precision draft mode are measured with the same prompt set and comparable settings.

### Stage 4: Oracle Adaptive Study

Purpose:

Estimate whether adaptive precision has enough theoretical value before implementing a runtime controller.

Method:

For each prompt or decoding segment, use offline results to choose the best fixed precision. Compare the oracle result with the best single static precision.

Comparison:

```text
Static FP16 draft
Static INT8 draft
Static INT4 draft
Oracle adaptive draft
```

Expected output:

```text
results/oracle_adaptive.csv
```

Pass condition:

The oracle study shows whether adaptive precision could plausibly beat the best static mode.

### Stage 5: Runtime Adaptive Controller

Purpose:

Implement the smallest useful adaptive precision controller.

Initial signals:

- Draft token entropy.
- Top-1 probability.
- Top-1 / Top-2 margin.
- Recent acceptance rate.
- Consecutive rejection count.
- Context length.
- Recent draft latency.
- Recent verification latency.

Initial rule:

```python
if entropy < T1 and recent_acceptance_rate > A1:
    precision = "INT4"
elif entropy < T2:
    precision = "INT8"
else:
    precision = "FP16"
```

Expected output:

```text
results/adaptive_controller.csv
```

Pass condition:

Adaptive mode runs end-to-end and its overhead is measured separately from draft and verification latency.

## Prompt Set

Use a small, fixed prompt set first. The first version can contain 10 to 50 prompts covering:

- Short factual prompts.
- Medium reasoning prompts.
- Code generation prompts.
- Summarization prompts.
- Long-context prompts if supported.

Each prompt should have:

```text
prompt_id
category
prompt_text
max_output_tokens
```

For the first baseline, it is acceptable to hard-code a small prompt list in the benchmark script. Later, move prompts to a JSONL file.

## Minimum CSV Schema

Each benchmark row should include:

```text
run_id
timestamp
backend
target_model
draft_model
target_precision
draft_precision
adaptive
prompt_id
prompt_category
input_tokens
output_tokens
ttft_ms
tpot_ms
end_to_end_latency_ms
tokens_per_second
gpu_memory_peak_gb
acceptance_rate
notes
```

If a metric is unavailable, write `NA` rather than omitting the column.

## Reproducibility Rules

Use the same settings across comparable runs:

- Same prompt set.
- Same max output tokens.
- Same sampling settings.
- Same batch size.
- Same model pair.
- Same warmup policy.
- Same GPU environment where possible.

Recommended generation settings for baseline:

```text
temperature = 0.0
top_p = 1.0
max_output_tokens = 128
batch_size = 1
num_warmup_runs = 2
num_measured_runs = 5
```

## Profiling Plan

Start with end-to-end benchmarks, then add profiling.

Profiling questions:

- Is draft generation compute-bound or memory-bound?
- Does lower precision actually use faster kernels?
- Are Tensor Cores being used effectively?
- Does precision switching reload weights or move memory?
- Can multiple draft precision variants stay resident in GPU memory?
- Is kernel launch overhead significant for short decoding steps?
- Does CUDA Graph help or conflict with dynamic switching?

Tools:

- vLLM benchmark tools.
- PyTorch profiler.
- Nsight Systems.
- Nsight Compute.

## Initial Milestones

### Milestone 1: Application-Ready Repository

Deliverables:

- `docs/project_application.md`
- `docs/experiment_plan.md`
- Non-empty model configuration files.
- Minimal benchmark script skeleton.

### Milestone 2: Runnable Baseline

Deliverables:

- Target-only benchmark script.
- Speculative decoding benchmark script.
- CSV output for both modes.

### Milestone 3: First Static Precision Result

Deliverables:

- FP16 draft baseline.
- One low-precision draft result.
- A short result table and notes on bottlenecks.

### Milestone 4: Hackathon Profiling

Deliverables:

- Nsight or profiler trace.
- Bottleneck analysis.
- Recommendation on whether adaptive precision is worth implementing.

## Decision Criteria

After Stage 3, continue to adaptive control only if at least one of these is true:

- Lower precision improves draft latency without destroying acceptance rate.
- Different prompt categories prefer different draft precisions.
- The oracle study beats the best fixed precision mode.
- Profiling shows that precision switching overhead is likely small enough.

If none of these are true, the project should report that fixed precision is preferable for the tested setup.

