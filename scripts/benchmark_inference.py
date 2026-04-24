#!/usr/bin/env python3
"""Benchmark vLLM inference on AMD MI300X for legal document workloads.

Measures: TTFT, throughput (tokens/s), latency at various context lengths.
Run after starting vLLM server.
"""
from __future__ import annotations

import argparse
import json
import time
from statistics import mean, stdev

import requests


def benchmark_single(
    base_url: str, model: str, prompt: str, max_tokens: int = 512
) -> dict:
    """Measure single request latency."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You extract legal facts from documents."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    
    start = time.perf_counter()
    resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=120)
    end = time.perf_counter()
    
    resp.raise_for_status()
    data = resp.json()
    
    usage = data.get("usage", {})
    total_tokens = usage.get("completion_tokens", 0)
    
    wall_time = end - start
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0
    
    return {
        "wall_time_s": round(wall_time, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": total_tokens,
        "tokens_per_sec": round(tokens_per_sec, 1),
    }


def run_benchmark(base_url: str, model: str, n_runs: int = 5):
    """Run benchmark suite across different context lengths."""
    
    # Generate prompts of varying lengths
    short_prompt = "Extract seller name from: This deed is executed by Ramesh Kumar."
    medium_prompt = short_prompt + " " + ("Additional context. " * 200)
    long_prompt = short_prompt + " " + ("Additional context with details about the property transaction. " * 1000)
    
    results = {}
    for label, prompt in [("short", short_prompt), ("medium", medium_prompt), ("long", long_prompt)]:
        timings = []
        for i in range(n_runs):
            try:
                r = benchmark_single(base_url, model, prompt)
                timings.append(r)
                print(f"  {label} run {i+1}: {r['wall_time_s']}s, {r['tokens_per_sec']} tok/s")
            except Exception as e:
                print(f"  {label} run {i+1}: FAILED ({e})")
        
        if timings:
            wall_times = [t["wall_time_s"] for t in timings]
            tps = [t["tokens_per_sec"] for t in timings]
            results[label] = {
                "mean_latency_s": round(mean(wall_times), 3),
                "std_latency_s": round(stdev(wall_times), 3) if len(wall_times) > 1 else 0,
                "mean_tokens_per_sec": round(mean(tps), 1),
                "n_runs": len(timings),
            }
    
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    
    print(f"Benchmarking {args.model} at {args.base_url}...")
    results = run_benchmark(args.base_url, args.model, args.runs)
    
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
