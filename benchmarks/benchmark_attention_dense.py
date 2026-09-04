#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import math
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "wan" / "modules" / "attention.py"
SPEC = importlib.util.spec_from_file_location("scail_attention", MODULE_PATH)
attention = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(attention)


def parse_bool(value):
    normalized = value.lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def time_cuda(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends):
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    samples_us = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
    peak_bytes = max(0, torch.cuda.max_memory_allocated() - baseline_memory)
    return statistics.median(samples_us), percentile(samples_us, 0.95), peak_bytes


def benchmark_case(batch, seqlen, dtype, causal, q_heads, kv_heads, head_dim, warmup, repeats):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    q = torch.randn(
        batch, seqlen, q_heads, head_dim,
        device="cuda", dtype=dtype, generator=generator,
    )
    k = torch.randn(
        batch, seqlen, kv_heads, head_dim,
        device="cuda", dtype=dtype, generator=generator,
    )
    v = torch.randn(
        batch, seqlen, kv_heads, head_dim,
        device="cuda", dtype=dtype, generator=generator,
    )
    lengths = torch.full((batch,), seqlen, device="cuda", dtype=torch.int32)

    def dense():
        return attention.flash_attention(q, k, v, causal=causal, version=2)

    def varlen():
        return attention.flash_attention(
            q, k, v,
            q_lens=lengths,
            k_lens=lengths,
            causal=causal,
            version=2,
        )

    dense_output = dense()
    varlen_output = varlen()
    torch.testing.assert_close(dense_output, varlen_output, atol=2e-2, rtol=2e-2)
    max_error = (dense_output - varlen_output).abs().max().item()
    mean_error = (dense_output - varlen_output).abs().float().mean().item()

    rows = []
    for backend, function in (("varlen", varlen), ("dense", dense)):
        median_us, p95_us, peak_bytes = time_cuda(function, warmup, repeats)
        rows.append({
            "backend": backend,
            "dtype": str(dtype).removeprefix("torch."),
            "causal": causal,
            "batch": batch,
            "q_len": seqlen,
            "k_len": seqlen,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "median_us": f"{median_us:.3f}",
            "p95_us": f"{p95_us:.3f}",
            "peak_allocated_bytes": peak_bytes,
            "max_abs_error": f"{max_error:.8g}",
            "mean_abs_error": f"{mean_error:.8g}",
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seqlen", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--dtype", nargs="+", choices=["float16", "bfloat16"], default=["float16", "bfloat16"])
    parser.add_argument("--causal", nargs="+", type=parse_bool, default=[False, True])
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", nargs="+", type=int, default=[8, 2])
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    rows = []
    for batch in args.batch:
        for seqlen in args.seqlen:
            for dtype_name in args.dtype:
                for causal in args.causal:
                    for kv_heads in args.kv_heads:
                        rows.extend(benchmark_case(
                            batch=batch,
                            seqlen=seqlen,
                            dtype=dtype_by_name[dtype_name],
                            causal=causal,
                            q_heads=args.q_heads,
                            kv_heads=kv_heads,
                            head_dim=args.head_dim,
                            warmup=args.warmup,
                            repeats=args.repeats,
                        ))

    fieldnames = list(rows[0])
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    writer = csv.DictWriter(
        sys.stdout, fieldnames=fieldnames, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
