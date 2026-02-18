"""MertFormer SDK CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import load_model, generate, benchmark, enable_lowbit_kernels
from .export import export_onnx
from .pilot import run_verify_all, build_pilot_report, write_pilot_report
from .kpi import collect_kpis, write_kpi_report


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
    model, tokenizer, _ = load_model(
        ckpt=args.ckpt,
        device=args.device,
        strict_checkpoint=not args.allow_random,
    )
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
        strict_checkpoint=not args.allow_random,
    )
    print(json.dumps(results, indent=2))


def _cmd_verify(args: argparse.Namespace) -> None:
    summary = run_verify_all(offline=True)
    print(json.dumps(summary, indent=2))
    if summary.get("status") != "pass":
        raise SystemExit(int(summary.get("exit_code") or 1))


def _cmd_pilot_report(args: argparse.Namespace) -> None:
    verify_summary = None
    if not args.skip_verify:
        verify_summary = run_verify_all(offline=True)
    else:
        verify_summary = {
            "status": "skipped",
            "exit_code": 0,
            "secret_scan_pass": None,
            "pytest_pass": None,
            "pytest_summary": {},
            "preflight_pass": None,
            "operator_gate_pass": None,
            "verify_script_pass": None,
            "operator_steps": {},
        }

    report = build_pilot_report(verify_summary=verify_summary)
    out_path = write_pilot_report(args.out, report)
    print(json.dumps(report, indent=2))
    print(f"pilot_report_written={out_path}")
    if verify_summary.get("status") == "fail":
        raise SystemExit(int(verify_summary.get("exit_code") or 1))


def _cmd_kpi_report(args: argparse.Namespace) -> None:
    payload = collect_kpis(
        project_root=Path("."),
        run_verify=not args.skip_verify,
        run_onnx_check=args.onnx_check,
    )
    out_path = write_kpi_report(args.out, payload)
    print(json.dumps(payload, indent=2))
    print(f"kpi_report_written={out_path}")


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
    run_p.add_argument("--allow-random", action="store_true", help="Allow running with random weights if checkpoint is missing.")
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
    bench_p.add_argument("--allow-random", action="store_true", help="Allow benchmark generation without a trained checkpoint.")
    bench_p.set_defaults(func=_cmd_benchmark)

    verify_p = sub.add_parser("verify", help="Run offline verify-all gate and print JSON summary")
    verify_p.set_defaults(func=_cmd_verify)

    report_p = sub.add_parser("pilot-report", help="Generate pilot_report_v1 JSON")
    report_p.add_argument("--out", required=True, help="Output JSON path")
    report_p.add_argument("--skip-verify", action="store_true", help="Skip running verify_all.sh and build report from filesystem signals only.")
    report_p.set_defaults(func=_cmd_pilot_report)

    kpi_p = sub.add_parser("kpi-report", help="Generate kpi_report_v1 JSON")
    kpi_p.add_argument("--out", required=True, help="Output JSON path")
    kpi_p.add_argument("--skip-verify", action="store_true", help="Skip running verify_all.sh")
    kpi_p.add_argument("--onnx-check", action="store_true", help="Run ONNX export smoke KPI")
    kpi_p.set_defaults(func=_cmd_kpi_report)

    args = parser.parse_args()
    try:
        args.func(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
