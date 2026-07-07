# AdaptiveSpec

**Runtime-Adaptive Mixed-Precision Speculative Decoding for LLM Inference on NVIDIA GH200**

AdaptiveSpec is a GPU systems project exploring whether speculative decoding for large language models can be accelerated by dynamically switching the draft model precision at runtime.

Instead of using a fixed draft precision such as FP16, INT8, or INT4 throughout generation, AdaptiveSpec studies whether the draft model can adapt its precision based on online decoding signals such as entropy, confidence, and recent acceptance rate.

---

## Motivation

Speculative decoding accelerates LLM inference by using a smaller **draft model** to propose tokens and a larger **target model** to verify them.

The key trade-off is:

```text
Low-precision draft model
→ faster draft generation
→ but possibly lower acceptance rate

High-precision draft model
→ better acceptance rate
→ but higher draft latency
```

This project asks:

> Can we dynamically choose the best draft precision during decoding to improve end-to-end inference speed?

---

## Core Idea

AdaptiveSpec compares four execution modes:

```text
1. Target-only baseline
2. Fixed FP16 draft speculative decoding
3. Fixed INT8 / INT4 draft speculative decoding
4. Runtime-adaptive mixed-precision draft speculative decoding
```

The adaptive controller selects draft precision using lightweight runtime signals:

```text
- Draft token entropy
- Top-1 probability
- Top-1 / Top-2 probability margin
- Recent acceptance rate
- Consecutive rejection count
- Context length
- Draft latency
- Verification latency
```

Output:

```text
INT4 / INT8 / FP16
```

---

## System Overview

```text
Input Prompt
    |
    v
Speculative Decoding Runtime
    |
    +-----------------------------+
    |                             |
    v                             v
Runtime Signals              GPU Metrics
entropy                      draft latency
confidence                   verify latency
acceptance history           memory usage
context length               kernel overhead
    |                             |
    +-------------+---------------+
                  |
                  v
        Precision Controller
                  |
      +-----------+-----------+
      |           |           |
      v           v           v
 INT4 Draft   INT8 Draft   FP16 Draft
      |           |           |
      +-----------+-----------+
                  |
                  v
           Draft Tokens
                  |
                  v
           Target Model
           Verification
                  |
                  v
         Acceptance Feedback
                  |
                  +-----> Controller
```

---

## Development Roadmap

### Stage 1: vLLM Speculative Decoding Baseline

Goal: build a minimal runnable baseline.

Tasks:

```text
- Run target-only inference
- Run draft-target speculative decoding
- Measure TTFT, TPOT, tokens/s, GPU memory
- Log acceptance rate if available
```

Expected output:

```text
results/baseline_target_only.csv
results/baseline_spec_decode.csv
```

---

### Stage 2: Static Precision Profiling

Goal: compare fixed draft precision modes.

Experiments:

```text
FP16 Draft + FP16 Target
INT8 Draft + FP16 Target
INT4 Draft + FP16 Target
```

Metrics:

```text
- Draft latency
- Target verification latency
- Acceptance rate
- TPOT
- tokens/s
- GPU memory usage
```

Expected table:

| Draft Precision | Draft Latency | Acceptance Rate | TPOT | Tokens/s |
| --------------- | ------------: | --------------: | ---: | -------: |
| INT4            |           TBD |             TBD |  TBD |      TBD |
| INT8            |           TBD |             TBD |  TBD |      TBD |
| FP16            |           TBD |             TBD |  TBD |      TBD |

---

### Stage 3: Oracle Study

Goal: prove whether adaptive precision has theoretical value.

The oracle selects the best precision offline for each prompt or decoding stage.

Comparison:

```text
Static INT4
Static INT8
Static FP16
Oracle Adaptive
```

If the oracle significantly outperforms the best static baseline, adaptive precision is worth implementing.

---

### Stage 4: Lightweight Runtime Controller

Goal: implement a simple adaptive controller.

Initial rule-based controller:

```python
if entropy < T1 and recent_acceptance_rate > A1:
    precision = "INT4"
elif entropy < T2:
    precision = "INT8"
else:
    precision = "FP16"
```

Later versions may use:

```text
- Decision tree
- Logistic regression
- Tiny MLP
```

The controller must be lightweight:

```text
controller overhead << draft forward latency
```

---

### Stage 5: GH200 Profiling and Optimization

Goal: use Hackathon resources to study whether adaptive precision works efficiently on NVIDIA GH200.

Key profiling questions:

```text
- How large is precision-switching overhead?
- Can INT4 / INT8 / FP16 draft weights stay resident in GPU memory?
- Does switching trigger weight reload or memory movement?
- Are INT4 kernels actually faster than INT8 / FP16?
- Is draft decoding memory-bound or compute-bound?
- What is the Tensor Core utilization?
- Can CUDA Graph reduce kernel launch overhead?
- How does KV cache behavior affect speculative decoding?
```

Tools:

```text
- Nsight Systems
- Nsight Compute
- vLLM benchmark tools
- PyTorch profiler
```

---

## Repository Structure

```text
AdaptiveSpec/
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_target_only.sh
│   ├── run_spec_decode_fp16.sh
│   ├── run_spec_decode_int8.sh
│   ├── run_spec_decode_int4.sh
│   └── run_adaptive_controller.sh
├── src/
│   ├── benchmark.py
│   ├── controller.py
│   ├── metrics.py
│   └── oracle_analysis.py
├── configs/
│   ├── target_model.yaml
│   ├── draft_fp16.yaml
│   ├── draft_int8.yaml
│   └── draft_int4.yaml
├── results/
│   ├── baseline/
│   ├── static_precision/
│   ├── oracle/
│   └── adaptive/
└── docs/
    ├── project_application.md
    ├── experiment_plan.md
    └── profiling_notes.md
```

---

## Main Metrics

AdaptiveSpec focuses on end-to-end inference performance:

```text
- TTFT: Time to First Token
- TPOT: Time per Output Token
- Tokens per second
- Acceptance rate
- Draft model latency
- Target verification latency
- GPU memory usage
- Kernel launch overhead
- Precision-switching overhead
- Tensor Core utilization
- Memory bandwidth utilization
```

---

## Expected Outcome

The goal is not only to show that INT4 or INT8 can be faster.

The real goal is to answer:

> When does adaptive mixed-precision speculative decoding actually improve end-to-end LLM inference speed?

By the end of the project, we aim to produce:

```text
- A working vLLM-based speculative decoding prototype
- Static INT4 / INT8 / FP16 benchmark results
- An oracle study for adaptive precision potential
- A lightweight runtime adaptive controller
- GH200 profiling results
- A reproducible performance report
```

---

## Hackathon Relevance

AdaptiveSpec is designed for GPU performance investigation.

The central Hackathon question is:

> Can runtime precision switching be implemented efficiently enough on GH200 to outperform fixed-precision speculative decoding?

This makes the project strongly connected to:

```text
- GPU kernel profiling
- Quantized inference
- Memory bandwidth analysis
- Tensor Core utilization
- Runtime scheduling overhead
- LLM inference acceleration
```

---

## One-Sentence Pitch

**AdaptiveSpec studies whether speculative decoding can dynamically switch draft-model precision at runtime to balance draft latency, acceptance rate, and GPU execution overhead on NVIDIA GH200.**

---

## Status

Current status:

```text
Planning / initial prototype stage
```

Next steps:

```text
1. Set up vLLM speculative decoding baseline
2. Benchmark target-only and draft-target inference
3. Add fixed FP16 / INT8 / INT4 draft profiling
4. Build oracle adaptive precision study
5. Implement lightweight runtime controller
6. Profile and optimize on GH200
```

---

## License

TBD.
