#!/usr/bin/env python3
"""Minimal vLLM benchmark entrypoint for AdaptiveSpec.

The script intentionally starts with end-to-end metrics. Internal speculative
decoding metrics such as acceptance rate can be added once the baseline is
stable on the hackathon system.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - only used before dependencies are installed.
    yaml = None


CSV_COLUMNS = [
    "run_id",
    "timestamp",
    "backend",
    "mode",
    "target_model",
    "draft_model",
    "target_precision",
    "draft_precision",
    "adaptive",
    "prompt_id",
    "prompt_category",
    "input_tokens",
    "output_tokens",
    "ttft_ms",
    "tpot_ms",
    "end_to_end_latency_ms",
    "tokens_per_second",
    "gpu_memory_peak_gb",
    "acceptance_rate",
    "notes",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if yaml is not None:
            data = yaml.safe_load(handle) or {}
        else:
            data = parse_simple_yaml(handle.read())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small nested YAML subset used by this repository's configs."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, raw_value = raw_line.strip().partition(":")
        if not separator:
            raise ValueError(f"Unsupported YAML line: {raw_line}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if raw_value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = coerce_scalar(raw_value.strip())
    return root


def coerce_scalar(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("prompt_id", "category", "prompt"):
                if key not in record:
                    raise ValueError(f"{path}:{line_number} missing required key: {key}")
            prompts.append(record)
    if not prompts:
        raise ValueError(f"{path} does not contain any prompts")
    return prompts


def cuda_peak_memory_gb() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "NA"
        return f"{torch.cuda.max_memory_allocated() / 1024**3:.3f}"
    except Exception:
        return "NA"


def count_input_tokens(llm: Any, prompt: str) -> str:
    try:
        token_ids = llm.get_tokenizer().encode(prompt)
        return str(len(token_ids))
    except Exception:
        return "NA"


def build_llm(mode: str, target_config: dict[str, Any], draft_config: dict[str, Any] | None) -> Any:
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed. Install requirements.txt in the target GPU environment."
        ) from exc

    target = target_config["target_model"]
    kwargs: dict[str, Any] = {
        "model": target["name"],
        "dtype": target.get("dtype", "auto"),
        "tensor_parallel_size": int(target.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(target.get("gpu_memory_utilization", 0.90)),
        "trust_remote_code": bool(target.get("trust_remote_code", False)),
    }

    if mode == "spec-decode":
        if not draft_config:
            raise ValueError("--draft-config is required for spec-decode mode")
        draft = draft_config["draft_model"]
        kwargs["speculative_model"] = draft["name"]
        kwargs["num_speculative_tokens"] = int(draft.get("num_speculative_tokens", 5))
        kwargs["speculative_model_dtype"] = draft.get("dtype", "auto")

    return LLM(**kwargs)


def benchmark(args: argparse.Namespace) -> None:
    target_config = load_yaml(Path(args.target_config))
    draft_config = load_yaml(Path(args.draft_config)) if args.draft_config else None

    benchmark_config = target_config.get("benchmark", {})
    prompt_path = Path(benchmark_config.get("prompt_file", "prompts/smoke_prompts.jsonl"))
    prompts = load_prompts(prompt_path)

    if args.dry_run:
        print(f"Loaded {len(prompts)} prompts from {prompt_path}")
        print(f"Mode: {args.mode}")
        print(f"Target model: {target_config['target_model']['name']}")
        if draft_config:
            print(f"Draft model: {draft_config['draft_model']['name']}")
        return

    generation = target_config.get("generation", {})
    max_tokens_default = int(generation.get("max_output_tokens", 128))
    warmups = int(benchmark_config.get("num_warmup_runs", 1))
    measured_runs = int(benchmark_config.get("num_measured_runs", 3))

    if args.mode == "spec-decode" and draft_config:
        output_csv = draft_config.get("benchmark", {}).get(
            "output_csv", "results/baseline_spec_decode.csv"
        )
    else:
        output_csv = benchmark_config.get("output_csv", "results/baseline_target_only.csv")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    llm = build_llm(args.mode, target_config, draft_config)

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=float(generation.get("temperature", 0.0)),
        top_p=float(generation.get("top_p", 1.0)),
        max_tokens=max_tokens_default,
    )

    warmup_prompts = [record["prompt"] for record in prompts[: max(1, min(warmups, len(prompts)))]]
    for prompt in warmup_prompts:
        llm.generate([prompt], sampling_params)

    run_id = args.run_id or f"{args.mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    rows: list[dict[str, str]] = []

    for measured_index in range(measured_runs):
        for record in prompts:
            prompt = record["prompt"]
            prompt_max_tokens = int(record.get("max_output_tokens", max_tokens_default))
            per_prompt_params = SamplingParams(
                temperature=float(generation.get("temperature", 0.0)),
                top_p=float(generation.get("top_p", 1.0)),
                max_tokens=prompt_max_tokens,
            )

            start = time.perf_counter()
            outputs = llm.generate([prompt], per_prompt_params)
            elapsed_ms = (time.perf_counter() - start) * 1000

            output = outputs[0].outputs[0]
            output_tokens = len(output.token_ids)
            tokens_per_second = output_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else math.nan
            tpot_ms = elapsed_ms / output_tokens if output_tokens else math.nan

            target = target_config["target_model"]
            draft = draft_config.get("draft_model", {}) if draft_config else {}
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "backend": target_config.get("backend", "vllm"),
                    "mode": args.mode,
                    "target_model": target["name"],
                    "draft_model": draft.get("name", "NA"),
                    "target_precision": target.get("dtype", "auto"),
                    "draft_precision": draft.get("precision_label", draft.get("dtype", "NA")),
                    "adaptive": "false",
                    "prompt_id": record["prompt_id"],
                    "prompt_category": record["category"],
                    "input_tokens": count_input_tokens(llm, prompt),
                    "output_tokens": str(output_tokens),
                    "ttft_ms": "NA",
                    "tpot_ms": f"{tpot_ms:.3f}" if not math.isnan(tpot_ms) else "NA",
                    "end_to_end_latency_ms": f"{elapsed_ms:.3f}",
                    "tokens_per_second": f"{tokens_per_second:.3f}"
                    if not math.isnan(tokens_per_second)
                    else "NA",
                    "gpu_memory_peak_gb": cuda_peak_memory_gb(),
                    "acceptance_rate": "NA",
                    "notes": f"measured_run={measured_index + 1}",
                }
            )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AdaptiveSpec baseline benchmarks.")
    parser.add_argument("--mode", choices=["target-only", "spec-decode"], required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--draft-config")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and prompts only.")
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
