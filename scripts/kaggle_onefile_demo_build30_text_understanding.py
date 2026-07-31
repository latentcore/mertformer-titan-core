from __future__ import annotations

"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - RULE-BASED EXTRACTION PoC (BUILD30 V2)
-------------------------------------------------------------------------------
Copyright 2026 Mert Yünlü
Licensed under the Apache License, Version 2.0 (see LICENSE).

NOTE (honest scope): Despite the "Mertformer Titan" / LLM framing, this PoC does
NOT perform any text "understanding" or model inference. The pipeline is a
deterministic regex/rule-based field extractor (see rule_based_answer below);
no model is trained or run. Treat the "understanding" wording as a project
label, not a measured capability.

NOTE (version label): The "BUILD30 V2" tag and __version__ below are
manually-maintained build labels embedded in this standalone onefile. There is
no single canonical version source wired into this script, so these strings can
fossilize relative to the repo's real version; update them by hand when bumping.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

import argparse
import json
import os
import random
import re
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _local_stamp() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S", time.localtime())


def _safe_write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    try:
        import hashlib
        hh = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                hh.update(b)
        return hh.hexdigest()
    except Exception as e:
        # Hash hesaplanamadı: sapmayı sessizce yutmamak için stderr'e uyari yaz;
        # geriye donus degeri olarak "error" korunuyor (cikti dogrulamasi bunu gate olarak işleyebilir).
        import sys
        print(f"[warn] _sha256_file başarısız ({path}): {type(e).__name__}: {e}", file=sys.stderr)
        return "error"


def _ensure_output_root() -> Path:
    env_root = os.environ.get("MERTFORMER_OUTPUT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (Path.home() / "Downloads" / "content" / "mertformer_outputs").resolve()


@dataclass
class TextPOCRecord:
    record_id: str
    context: str
    question: str
    answer: str
    summary: str


def build_synthetic_records(seed: int, count: int) -> List[TextPOCRecord]:
    rng = random.Random(seed)
    cities = ["Konya", "Kayseri", "Eskişehir", "Sivas", "Ankara", "Balıkesir", "Isparta"]
    units = ["52 Filo", "7 Birlik", "Kuzey Gözetleme", "Anka Takımı", "Bora Görev Gücü"]
    commanders = ["Ada Demir", "Mert Aydın", "Selin Kaya", "Aras Yıldız", "Nil Yılmaz"]
    codenames = ["KIPRA-7", "LODOS-3", "NEHİR-9", "KALKAN-4", "ŞAFAK-2"]
    payloads = ["görüntüleme", "haberleşme", "keşif", "sinyal izleme", "haritalama"]
    dates = ["14 Mart 2026", "15 Mart 2026", "16 Mart 2026", "17 Mart 2026", "18 Mart 2026"]

    questions = [
        ("Operasyon kodu nedir?", "Operasyon Kodu"),
        ("Komutan kim?", "Komutan"),
        ("Görev birimi nedir?", "Görev Birimi"),
        ("Hedef bölge neresi?", "Konum"),
        ("Görev tarihi nedir?", "Tarih"),
        ("Yük tipi nedir?", "Yük Tipi"),
    ]

    records: List[TextPOCRecord] = []
    for idx in range(count):
        codename = rng.choice(codenames)
        commander = rng.choice(commanders)
        unit = rng.choice(units)
        city = rng.choice(cities)
        payload = rng.choice(payloads)
        date = rng.choice(dates)

        q_text, q_key = rng.choice(questions)
        facts = {
            "Operasyon Kodu": codename,
            "Komutan": commander,
            "Görev Birimi": unit,
            "Konum": city,
            "Tarih": date,
            "Yük Tipi": payload,
        }

        sentences = [
            f"Operasyon Kodu: {codename}.",
            f"Komutan: {commander}.",
            f"Görev Birimi: {unit}.",
            f"Konum: {city}.",
            f"Tarih: {date}.",
            f"Yük Tipi: {payload}.",
            "Görev, sabah erken saatlerde planlandı ve kontrollü şekilde ilerledi.",
            "Hava koşulları orta seviyede olup rüzgar 8-12 knot aralığındaydı.",
            "Menzil ve yakıt planlaması önceden doğrulandı.",
            "Ekip, kritik kontrol noktalarını iki aşamada geçti.",
            "İletişim protokolü 3 kanallı yedekleme ile çalıştı.",
            "Görev raporu, operasyon sonunda güvenli kanalla aktarıldı.",
        ]
        rng.shuffle(sentences)
        context = " ".join(sentences)

        summary = (
            f"Operasyon {codename} kapsamında {unit} birimi {city} bölgesinde görev aldı. "
            f"Komutan {commander} yönetiminde {payload} yükü kullanıldı. "
            f"Görev tarihi {date} olarak kayda geçti."
        )

        record_id = f"tu_{seed}_{idx:05d}"
        answer = facts[q_key]
        records.append(
            TextPOCRecord(
                record_id=record_id,
                context=context,
                question=q_text,
                answer=answer,
                summary=summary,
            )
        )
    return records


def rule_based_answer(context: str, question: str) -> str:
    """Deterministic rule-based field extractor (NOT model understanding).

    This is the actual mechanism behind the "TEXT UNDERSTANDING" framing: a
    fixed question->field mapping plus a regex lookup over the context. No model
    inference or learned comprehension is involved; it is a rule-based baseline.
    """
    mapping = {
        "Operasyon kodu": "Operasyon Kodu",
        "Komutan": "Komutan",
        "Görev birimi": "Görev Birimi",
        "Hedef bölge": "Konum",
        "Görev tarihi": "Tarih",
        "Yük tipi": "Yük Tipi",
    }
    key = None
    for k in mapping:
        if k.lower() in question.lower():
            key = mapping[k]
            break
    if not key:
        return ""
    rx = re.compile(rf"{re.escape(key)}:\s*([^.]+)\.", re.IGNORECASE)
    m = rx.search(context)
    return m.group(1).strip() if m else ""


def score_records(records: List[TextPOCRecord]) -> Tuple[float, float]:
    if not records:
        return 0.0, 1.0
    correct = 0
    invalid = 0
    for rec in records:
        pred = rule_based_answer(rec.context, rec.question)
        if not pred:
            invalid += 1
        if pred.strip() == rec.answer.strip():
            correct += 1
    exact = (correct / len(records)) * 100.0
    invalid_ratio = invalid / len(records)
    return exact, invalid_ratio


def write_jsonl(path: Path, records: List[TextPOCRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def build_artifact_index(paths: List[Path]) -> Dict[str, Dict[str, str]]:
    return {p.name: {"sha256": _sha256_file(p), "path": str(p)} for p in paths if p.exists()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build30 text-understanding PoC (synthetic)")
    ap.add_argument("--quick", action="store_true", help="Use a smaller dataset for fast local runs")
    args = ap.parse_args()

    quick_env = os.environ.get("MERTFORMER_TEXT_POC_QUICK", "0") == "1"
    quick = bool(args.quick or quick_env)

    seed = int(os.environ.get("MERTFORMER_TEXT_POC_SEED", "42"))
    sizes = {
        "train": 18000,
        "val": 1200,
        "test": 1200,
        "unseen": 400,
    }
    if quick:
        sizes = {k: max(1, int(v / 10)) for k, v in sizes.items()}

    output_root = _ensure_output_root()
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / f"{run_id}_summary.json"
    compare_path = run_dir / f"{run_id}_compare.json"
    compare_md_path = run_dir / f"{run_id}_compare.md"
    compare_csv_path = run_dir / f"{run_id}_compare.csv"
    health_path = run_dir / f"{run_id}_health.txt"
    run_log_path = run_dir / f"{run_id}_run_log.jsonl"
    artifact_index_path = run_dir / f"{run_id}_artifact_index.json"
    evidence_zip_path = run_dir / f"{run_id}_evidence.zip"

    # Build datasets
    train = build_synthetic_records(seed, sizes["train"])
    val = build_synthetic_records(seed + 1, sizes["val"])
    test = build_synthetic_records(seed + 2, sizes["test"])
    unseen = build_synthetic_records(seed + 3, sizes["unseen"])

    # Log raw data (jsonl)
    write_jsonl(run_dir / f"{run_id}_train.jsonl", train)
    write_jsonl(run_dir / f"{run_id}_val.jsonl", val)
    write_jsonl(run_dir / f"{run_id}_test.jsonl", test)
    write_jsonl(run_dir / f"{run_id}_unseen.jsonl", unseen)

    # Score (rule-based baseline)
    exact_val, invalid_val = score_records(val)
    exact_test, invalid_test = score_records(test)
    exact_unseen, invalid_unseen = score_records(unseen)

    target_gate = float(os.environ.get("MERTFORMER_TEXT_POC_TARGET_EXACT", "95.0"))
    accuracy_gate_pass = bool(exact_test >= target_gate and exact_unseen >= target_gate)
    final_status = "gate_pass" if accuracy_gate_pass else "gate_fail"

    summary = {
        "schema": "text_understanding_poc_v1",
        "generated_at_utc": _utc_now(),
        "run_id": run_id,
        "sizes": sizes,
        "method": "rule_based_extraction",
        "exact_match_val": exact_val,
        "exact_match_test": exact_test,
        "exact_match_unseen": exact_unseen,
        "invalid_output_ratio_val": invalid_val,
        "invalid_output_ratio_test": invalid_test,
        "invalid_output_ratio_unseen": invalid_unseen,
        "target_exact_match_gate": target_gate,
        "accuracy_gate_pass": accuracy_gate_pass,
        "final_status": final_status,
    }
    _safe_write_json(summary_path, summary)

    compare = {
        "schema": "text_understanding_compare_v1",
        "generated_at_utc": _utc_now(),
        "run_id": run_id,
        "variant_results": [
            {
                "variant": "rule_based_baseline",
                "exact_match_val": exact_val,
                "exact_match_test": exact_test,
                "exact_match_unseen": exact_unseen,
                "invalid_output_ratio_val": invalid_val,
                "invalid_output_ratio_test": invalid_test,
                "invalid_output_ratio_unseen": invalid_unseen,
                "note": "Deterministic extraction; no model training performed.",
            }
        ],
        "gates": {
            "accuracy_gate_pass": accuracy_gate_pass,
        },
        "final_status": final_status,
    }
    _safe_write_json(compare_path, compare)

    compare_md = "\n".join(
        [
            "# Text Understanding PoC Compare",
            "",
            f"- run_id: {run_id}",
            f"- generated_at_utc: {_utc_now()}",
            f"- method: rule_based_extraction",
            "",
            "| Variant | ExactMatch(Test) | ExactMatch(Unseen) | Invalid(Test) | Invalid(Unseen) |",
            "|---|---:|---:|---:|---:|",
            f"| rule_based_baseline | {exact_test:.2f}% | {exact_unseen:.2f}% | {invalid_test:.2f} | {invalid_unseen:.2f} |",
            "",
            "## Gates",
            f"- accuracy_gate_pass: {accuracy_gate_pass}",
            f"- final_status: {final_status}",
            "",
        ]
    )
    compare_md_path.write_text(compare_md, encoding="utf-8")

    compare_csv_path.write_text(
        "variant,exact_match_test,exact_match_unseen,invalid_output_ratio_test,invalid_output_ratio_unseen\n"
        f"rule_based_baseline,{exact_test:.4f},{exact_unseen:.4f},{invalid_test:.4f},{invalid_unseen:.4f}\n",
        encoding="utf-8",
    )

    health_path.write_text(
        "\n".join(
            [
                f"RUN_ID={run_id}",
                f"FINAL_STATUS={final_status}",
                f"accuracy_gate_pass={accuracy_gate_pass}",
                f"exact_match_test={exact_test:.2f}",
                f"exact_match_unseen={exact_unseen:.2f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Run log (minimal)
    with run_log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "config", "sizes": sizes, "seed": seed, "quick": quick}) + "\n")
        f.write(json.dumps({"event": "eval", "exact_match_test": exact_test, "exact_match_unseen": exact_unseen}) + "\n")
        f.write(json.dumps({"event": "final", "final_status": final_status}) + "\n")

    artifact_index = build_artifact_index(
        [
            summary_path,
            compare_path,
            compare_md_path,
            compare_csv_path,
            health_path,
            run_log_path,
        ]
    )
    _safe_write_json(artifact_index_path, artifact_index)

    # Evidence zip (text-only)
    with zipfile.ZipFile(evidence_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in [summary_path, compare_path, compare_md_path, compare_csv_path, health_path, run_log_path, artifact_index_path]:
            zf.write(p, arcname=p.name)

    print(f"FINAL_STATUS: {final_status} run_id={run_id}")
    print(f"Artifacts: {run_dir}")
    return 0 if accuracy_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
