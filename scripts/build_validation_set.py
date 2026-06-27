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
import sys
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
            except Exception as e:
                print(f"[warn] _load_local_golden: skipping malformed JSONL line: {e}", file=sys.stderr)
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


DEFAULT_STAGE_PATHS = [
    ROOT / "datasets" / "stage1" / "stage1_data.jsonl",
    ROOT / "datasets" / "stage2" / "stage2_data.jsonl",
    ROOT / "datasets" / "stage3" / "stage3_data.jsonl",
    ROOT / "datasets" / "stage4_soul" / "stage4_data.jsonl",
    ROOT / "datasets" / "stage5_tools" / "stage5_data.jsonl",
]


def _training_fingerprinter():
    """TR: [H5] Eğitim pipeline'inin RollingDeduper fingerprint'ini (blake2b,
    normalize) YENİDEN kullan -> val-vs-train dışlaması, korpusun deduplandığı AYNI
    algoritmayla yapılır. EN: Reuse the training pipeline's RollingDeduper
    fingerprint so val-vs-train exclusion uses the SAME algorithm."""
    from scripts.data_pipeline import RollingDeduper
    return RollingDeduper(enabled=True)._fingerprint


def _load_training_fingerprints(stage_paths):
    """Set of training-row fingerprints from on-disk stage JSONLs (offline-safe;
    no network/re-download)."""
    fp = _training_fingerprinter()
    seen: set[int] = set()
    used: list[str] = []
    for p in stage_paths:
        if not p.exists():
            continue
        used.append(str(p))
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception as e:
                        print(f"[warn] _load_training_fingerprints: skipping malformed JSONL line in {p}: {e}", file=sys.stderr)
                        continue
                    t = str(obj.get("text", "") or "").strip()
                    if t:
                        seen.add(fp(t))
        except Exception as e:
            print(f"[warn] _load_training_fingerprints: failed reading stage file {p}: {e}", file=sys.stderr)
            continue
    return seen, used


def build_validation_set(
    *,
    target_size: int,
    seed: int,
    min_chars: int,
    max_chars: int,
    exclude_training: bool = True,
    stage_paths=None,
    offline_rebuild: bool = False,
    strict_offline: bool = False,
    current_val_path=None,
):
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    stage_paths = list(stage_paths) if stage_paths is not None else list(DEFAULT_STAGE_PATHS)

    # [H5] Training fingerprints for leakage exclusion (offline-safe).
    train_fp: set[int] = set()
    stage_files_used: list[str] = []
    if exclude_training:
        train_fp, stage_files_used = _load_training_fingerprints(stage_paths)
        if strict_offline and not stage_files_used:
            raise RuntimeError(
                "strict_offline + exclude_training requested but no stage JSONLs were "
                "found on disk; cannot certify non-leakage. Provide stage files."
            )
    tfp = _training_fingerprinter() if exclude_training else None
    excluded_leak = {}

    def _is_leak(text: str) -> bool:
        return bool(exclude_training and train_fp and tfp(text) in train_fp)

    if offline_rebuild:
        # TR: [H5] Networksuz yeniden kur: mevcut val + golden'dan training fingerprint
        #     ile dışlayarak sertifikalı (sızıntısız) altküme üret.
        # EN: [H5] Network-free rebuild: from existing val + golden, exclude training
        #     fingerprints to certify a leakage-free subset.
        pool = []
        if current_val_path and Path(current_val_path).exists():
            with Path(current_val_path).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pool.append(json.loads(line))
                    except Exception as e:
                        print(f"[warn] offline_rebuild: skipping malformed JSONL line in {current_val_path}: {e}", file=sys.stderr)
                        continue
        pool.extend(_load_local_golden(ROOT / "datasets" / "golden_samples.jsonl"))
        rng.shuffle(pool)
        for item in pool:
            text = _clean_text(item.get("text", ""))
            if not text or len(text) < min_chars or len(text) > max_chars:
                continue
            fp = _fingerprint(text)
            if fp in seen:
                continue
            if _is_leak(text):
                excluded_leak["offline_rebuild"] = excluded_leak.get("offline_rebuild", 0) + 1
                continue
            item["text"] = text
            rows.append(item)
            seen.add(fp)
            if len(rows) >= target_size:
                break
        rng.shuffle(rows)
        rows = rows[:target_size]
        provenance = {
            "mode": "offline_rebuild",
            "count": len(rows),
            "excluded_leak": excluded_leak,
            "stage_files": stage_files_used,
            "fingerprint_algo": "RollingDeduper.blake2b_normalized",
            "network_used": False,
        }
        return rows, provenance

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
            if _is_leak(text):
                excluded_leak[spec.key] = excluded_leak.get(spec.key, 0) + 1
                continue
            rows.append({"text": text, "source": spec.key, "lang": spec.lang})
            seen.add(fp)
            taken += 1
            if taken >= need:
                break
        print(f"[ok] {spec.key}: {taken}/{need} (excluded_leak={excluded_leak.get(spec.key, 0)})")

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
            if _is_leak(text):
                excluded_leak["golden"] = excluded_leak.get("golden", 0) + 1
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
                if _is_leak(text):
                    excluded_leak[f"{spec.key}:extra"] = excluded_leak.get(f"{spec.key}:extra", 0) + 1
                    continue
                rows.append({"text": text, "source": f"{spec.key}:extra", "lang": spec.lang})
                seen.add(fp)
                if len(rows) >= target_size:
                    break
            if len(rows) >= target_size:
                break

    rng.shuffle(rows)
    rows = rows[:target_size]
    provenance = {
        "mode": "networked",
        "count": len(rows),
        "excluded_leak": excluded_leak,
        "excluded_leak_total": sum(excluded_leak.values()),
        "stage_files": stage_files_used,
        "fingerprint_algo": "RollingDeduper.blake2b_normalized",
        "network_used": True,
    }
    return rows, provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "datasets" / "validation.jsonl"))
    parser.add_argument("--target-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1453)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=2400)
    parser.add_argument("--no-exclude-training", action="store_true",
                        help="Disable val-vs-train fingerprint exclusion (NOT recommended).")
    parser.add_argument("--strict-offline", action="store_true",
                        help="Fail if non-leakage cannot be certified against stage JSONLs.")
    parser.add_argument("--offline-rebuild", action="store_true",
                        help="Network-free: re-certify existing val + golden by training-fingerprint exclusion.")
    args = parser.parse_args()

    out = Path(args.out)
    rows, provenance = build_validation_set(
        target_size=max(1, int(args.target_size)),
        seed=int(args.seed),
        min_chars=max(1, int(args.min_chars)),
        max_chars=max(128, int(args.max_chars)),
        exclude_training=not args.no_exclude_training,
        offline_rebuild=args.offline_rebuild,
        strict_offline=args.strict_offline,
        current_val_path=out,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # [H5] Provenance artifact proving the val set is leakage-excluded.
    prov_path = ROOT / "datasets" / "validation_provenance.json"
    try:
        prov_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[warn] failed writing provenance artifact {prov_path}: {e}", file=sys.stderr)

    print(json.dumps({"out": str(out), "count": len(rows), "provenance": provenance}, ensure_ascii=False))
    # NOTE: after any val rebuild, run scripts/record_dataset_hashes.py because
    # datasets/hashes.json pins validation.jsonl's sha256.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

