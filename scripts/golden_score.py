"""
Golden assertion scorer for Build30 training gates.

Writes:
- reports/benchmarks/golden_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _assertion_passes(assertion: dict[str, Any], completion: str) -> bool:
    kind = str(assertion.get("type", "contains_any")).strip().lower()
    if kind == "contains_any":
        values = [_to_text(v) for v in assertion.get("values", [])]
        return any(v and v in completion for v in values)
    if kind == "contains_all":
        values = [_to_text(v) for v in assertion.get("values", [])]
        return all(v in completion for v in values if v)
    if kind == "regex":
        pattern = _to_text(assertion.get("pattern"))
        if not pattern:
            return False
        return re.search(pattern, completion) is not None
    if kind == "min_length":
        n = int(assertion.get("value", 1))
        return len(completion.strip()) >= n
    return False


def load_predictions(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    for row in load_jsonl(path):
        rid = str(row.get("id", "")).strip()
        if not rid:
            continue
        mapping[rid] = _to_text(row.get("completion", ""))
    return mapping


def score_assertions(assertions_file: Path, predictions_file: Path, summary_file: Path) -> dict[str, Any]:
    cases = load_jsonl(assertions_file)
    predictions = load_predictions(predictions_file)

    total_cases = len(cases)
    case_pass = 0
    total_assertions = 0
    passed_assertions = 0

    for idx, case in enumerate(cases, start=1):
        rid = str(case.get("id", idx))
        completion = predictions.get(rid, "")
        checks = case.get("assertions", [])
        if not isinstance(checks, list):
            checks = []
        local_total = 0
        local_pass = 0
        for assertion in checks:
            if not isinstance(assertion, dict):
                continue
            local_total += 1
            if _assertion_passes(assertion, completion):
                local_pass += 1
        total_assertions += local_total
        passed_assertions += local_pass
        if local_total > 0 and local_pass == local_total:
            case_pass += 1

    assertion_score = (passed_assertions / total_assertions) if total_assertions else 0.0
    case_pass_rate = (case_pass / total_cases) if total_cases else 0.0
    summary: dict[str, Any] = {
        "status": "scored",
        "assertions_file": str(assertions_file),
        "predictions_file": str(predictions_file),
        "total_cases": int(total_cases),
        "case_pass": int(case_pass),
        "case_pass_rate": round(case_pass_rate, 6),
        "total_assertions": int(total_assertions),
        "passed_assertions": int(passed_assertions),
        "assertion_score": round(assertion_score, 6),
    }
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_model(assertions_file: Path, ckpt: Path, out_predictions: Path, max_new_tokens: int, samples: int) -> int:
    import torch
    from config.config import cfg
    from model.transformers import MertFormer
    from utils.tokenizer_resolver import load_tokenizer_from_identity, resolve_tokenizer

    cases = load_jsonl(assertions_file)
    if samples > 0:
        cases = cases[:samples]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if ckpt.exists():
        checkpoint = torch.load(ckpt, map_location=device)
        # TR: Tokenizer'i checkpoint kimliginden yukle; yoksa ACIK hata (sessiz
        #     teacher fallback YOK). EN: Tokenizer strictly from checkpoint
        #     identity; missing -> explicit error (same pattern as eval/gsm8k.py).
        tokenizer = load_tokenizer_from_identity(checkpoint.get("tokenizer_id"))
        cfg.vocab_size = len(tokenizer)
        model = MertFormer().to(device)
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(checkpoint.get("model", checkpoint))
    else:
        # Smoke-only path (no checkpoint -> no recorded identity).
        print(f"[golden_score] checkpoint not found ({ckpt}), using random weights (smoke).")
        tokenizer = resolve_tokenizer(cfg)
        cfg.vocab_size = len(tokenizer)
        model = MertFormer().to(device)
        model.resize_token_embeddings(len(tokenizer))

    model.eval()

    out_predictions.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_predictions.open("w", encoding="utf-8") as f:
        for case in cases:
            prompt = _to_text(case.get("prompt"))
            if not prompt:
                continue
            rid = str(case.get("id", written + 1))
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                # [H6 fix] stop on EOS; decode only the generated tokens.
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(
                output[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            f.write(
                json.dumps(
                    {"id": rid, "prompt": prompt, "completion": completion},
                    ensure_ascii=True,
                )
                + "\n"
            )
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assertions-file", type=str, default="datasets/golden_assertions.jsonl")
    parser.add_argument("--predictions", type=str, default="reports/benchmarks/golden_outputs.jsonl")
    parser.add_argument("--summary", type=str, default="reports/benchmarks/golden_summary.json")
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--ckpt", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--samples", type=int, default=0)
    args = parser.parse_args()

    assertions_file = Path(args.assertions_file)
    predictions_file = Path(args.predictions)
    summary_file = Path(args.summary)

    if not assertions_file.exists():
        raise FileNotFoundError(f"Assertions file not found: {assertions_file}")

    if args.run_model:
        generated = run_model(
            assertions_file=assertions_file,
            ckpt=Path(args.ckpt),
            out_predictions=predictions_file,
            max_new_tokens=args.max_new_tokens,
            samples=args.samples,
        )
        print(f"[golden_score] generated predictions: {generated}")

    summary = score_assertions(assertions_file, predictions_file, summary_file)
    print(
        f"[golden_score] assertion_score={summary['assertion_score']:.4f} "
        f"({summary['passed_assertions']}/{summary['total_assertions']}) -> {summary_file}"
    )


if __name__ == "__main__":
    main()
