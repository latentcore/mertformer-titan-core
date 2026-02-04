"""MertFormer SDK CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import load_model, generate, benchmark, enable_lowbit_kernels
from .export import export_onnx


def _cmd_info(args: argparse.Namespace) -> None:
    import torch
    info = {
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    print(json.dumps(info, indent=2))


def _cmd_run(args: argparse.Namespace) -> None:
    if args.lowbit:
        enable_lowbit_kernels(True)
    model, tokenizer, _ = load_model(ckpt=args.ckpt, device=args.device)
    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(output)


def _cmd_export(args: argparse.Namespace) -> None:
    export_onnx(
        ckpt=args.ckpt,
        output_dir=args.output_dir,
        bitpack=args.bitpack,
    )


def _cmd_benchmark(args: argparse.Namespace) -> None:
    results = benchmark(
        ckpt=args.ckpt,
        out_dir=args.output_dir,
        samples=args.samples,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="mertformer")
    sub = parser.add_subparsers(dest="command", required=True)

    info_p = sub.add_parser("info", help="Show environment info")
    info_p.set_defaults(func=_cmd_info)

    run_p = sub.add_parser("run", help="Generate text")
    run_p.add_argument("--prompt", required=True)
    run_p.add_argument("--ckpt", default="latest")
    run_p.add_argument("--device", default=None)
    run_p.add_argument("--max-new-tokens", type=int, default=128)
    run_p.add_argument("--temperature", type=float, default=0.7)
    run_p.add_argument("--top-p", type=float, default=0.9)
    run_p.add_argument("--lowbit", action="store_true")
    run_p.set_defaults(func=_cmd_run)

    export_p = sub.add_parser("export", help="Export ONNX")
    export_p.add_argument("--ckpt", default="latest")
    export_p.add_argument("--output-dir", default=str(Path("checkpoints") / "mertformer_titan_prod"))
    export_p.add_argument("--bitpack", action="store_true")
    export_p.set_defaults(func=_cmd_export)

    bench_p = sub.add_parser("benchmark", help="Run benchmarks")
    bench_p.add_argument("--ckpt", default="latest")
    bench_p.add_argument("--output-dir", default="reports/benchmarks")
    bench_p.add_argument("--samples", type=int, default=0)
    bench_p.add_argument("--max-new-tokens", type=int, default=256)
    bench_p.set_defaults(func=_cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
