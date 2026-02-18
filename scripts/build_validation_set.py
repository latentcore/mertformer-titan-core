#!/usr/bin/env python3
"""
Build a representative validation set for claim-grade evaluation.

Output format (jsonl):
{"text": "...", "source": "...", "lang": "..."}
"""

from __future__ import annotations

import argparse
import json
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent.parent


def _clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _fingerprint(text: str) -> str:
    return hashlib.sha1(text.lower().encode("utf-8")).hexdigest()


def _extract_gsm8k(ex: dict[str, Any]) -> str:
    q = str(ex.get("question", "")).strip()
    a = str(ex.get("answer", "")).strip()
    return f"Soru:\n{q}\n\nÇözüm:\n{a}" if q and a else ""


def _extract_mbpp(ex: dict[str, Any]) -> str:
    text = str(ex.get("text", "")).strip()
    code = str(ex.get("code", "")).strip()
    if text and code:
        return f"Task:\n{text}\n\nReference Code:\n{code}"
    return text or code


def _extract_wikipedia(ex: dict[str, Any]) -> str:
    return str(ex.get("text", "")).strip()


def _extract_instruc_turca(ex: dict[str, Any]) -> str:
    inp = str(ex.get("Input", "")).strip()
    out = str(ex.get("Output", "")).strip()
    if inp and out:
        return f"Girdi:\n{inp}\n\nYanıt:\n{out}"
    return inp or out


def _extract_oasst(ex: dict[str, Any]) -> str:
    return str(ex.get("text", "")).strip()


@dataclass
class SourceSpec:
    key: str
    dataset_id: str
    config: str | None
    split: str
    lang: str
    weight: float
    extractor: Callable[[dict[str, Any]], str]
    streaming: bool = True


SOURCES: list[SourceSpec] = [
    SourceSpec(
        key="gsm8k_main_test",
        dataset_id="openai/gsm8k",
        config="main",
        split="test",
        lang="en",
        weight=0.20,
        extractor=_extract_gsm8k,
    ),
    SourceSpec(
        key="mbpp_test",
        dataset_id="mbpp",
        config=None,
        split="test",
        lang="en",
        weight=0.15,
        extractor=_extract_mbpp,
    ),
    SourceSpec(
        key="wikipedia_tr_train",
        dataset_id="wikimedia/wikipedia",
        config="20231101.tr",
        split="train",
        lang="tr",
        weight=0.25,
        extractor=_extract_wikipedia,
    ),
    SourceSpec(
        key="instructurca_train",
        dataset_id="turkish-nlp-suite/InstrucTurca",
        config=None,
        split="train",
        lang="tr",
        weight=0.20,
        extractor=_extract_instruc_turca,
    ),
    SourceSpec(
        key="oasst_top1_train",
        dataset_id="OpenAssistant/oasst_top1_2023-08-25",
        config=None,
        split="train",
        lang="mixed",
        weight=0.20,
        extractor=_extract_oasst,
    ),
]


def _stream_source(spec: SourceSpec, seed: int):
    if spec.config:
        ds = load_dataset(
            spec.dataset_id,
            spec.config,
            split=spec.split,
            streaming=spec.streaming,
        )
    else:
        ds = load_dataset(
            spec.dataset_id,
            split=spec.split,
            streaming=spec.streaming,
        )
    if hasattr(ds, "shuffle"):
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    return ds


def _load_local_golden(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            txt = _clean_text(str(obj.get("prompt", "")))
            if txt:
                items.append(
                    {
                        "text": txt,
                        "source": "internal:golden_samples",
                        "lang": "mixed",
                    }
                )
    return items


def build_validation_set(
    *,
    target_size: int,
    seed: int,
    min_chars: int,
    max_chars: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[dict[str, str]] = []

    quotas = {s.key: max(1, int(round(target_size * s.weight))) for s in SOURCES}

    for spec in SOURCES:
        need = quotas[spec.key]
        try:
            ds = _stream_source(spec, seed=seed)
        except Exception as e:
            print(f"[warn] source failed: {spec.key} -> {e}")
            continue

        taken = 0
        for ex in ds:
            text = _clean_text(spec.extractor(ex))
            if not text:
                continue
            if len(text) < min_chars or len(text) > max_chars:
                continue
            fp = _fingerprint(text)
            if fp in seen:
                continue
            rows.append(
                {
                    "text": text,
                    "source": spec.key,
                    "lang": spec.lang,
                }
            )
            seen.add(fp)
            taken += 1
            if taken >= need:
                break
        print(f"[ok] {spec.key}: {taken}/{need}")

    # Fill from local golden prompts if needed.
    if len(rows) < target_size:
        golden = _load_local_golden(ROOT / "datasets" / "golden_samples.jsonl")
        rng.shuffle(golden)
        for item in golden:
            text = _clean_text(item["text"])
            if len(text) < min_chars or len(text) > max_chars:
                continue
            fp = _fingerprint(text)
            if fp in seen:
                continue
            rows.append(item)
            seen.add(fp)
            if len(rows) >= target_size:
                break

    # If still short, keep sampling from shuffled sources in round-robin.
    if len(rows) < target_size:
        extras = SOURCES[:]
        rng.shuffle(extras)
        for spec in extras:
            try:
                ds = _stream_source(spec, seed=seed + 1337)
            except Exception:
                continue
            for ex in ds:
                text = _clean_text(spec.extractor(ex))
                if not text:
                    continue
                if len(text) < min_chars or len(text) > max_chars:
                    continue
                fp = _fingerprint(text)
                if fp in seen:
                    continue
                rows.append(
                    {
                        "text": text,
                        "source": f"{spec.key}:extra",
                        "lang": spec.lang,
                    }
                )
                seen.add(fp)
                if len(rows) >= target_size:
                    break
            if len(rows) >= target_size:
                break

    rng.shuffle(rows)
    return rows[:target_size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "datasets" / "validation.jsonl"))
    parser.add_argument("--target-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1453)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=2400)
    args = parser.parse_args()

    rows = build_validation_set(
        target_size=max(1, int(args.target_size)),
        seed=int(args.seed),
        min_chars=max(1, int(args.min_chars)),
        max_chars=max(128, int(args.max_chars)),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"out": str(out), "count": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

