#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SitlEvent:
    run_id: int
    step: int
    offline_mode: bool
    sensor_fault: bool
    confidence: float
    action: str
    fail_safe: bool
    reason: str | None


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _policy_action(confidence: float, wind: float) -> str:
    if confidence < 0.55:
        return "hold"
    if wind > 0.8:
        return "stabilize"
    if wind < 0.3:
        return "advance"
    return "track"


def run_once(
    run_id: int,
    steps: int,
    fault_start: int,
    fault_duration: int,
    threshold: float,
    rng: random.Random,
) -> tuple[list[SitlEvent], dict]:
    events: list[SitlEvent] = []
    fallback_count = 0
    recovered = False

    for step in range(steps):
        sensor_fault = fault_start <= step < (fault_start + fault_duration)
        wind = rng.uniform(0.0, 1.0)
        base_conf = 0.78 - 0.25 * wind + rng.uniform(-0.03, 0.03)
        if sensor_fault:
            base_conf -= 0.35
        confidence = max(0.01, min(0.99, round(base_conf, 4)))

        action = _policy_action(confidence, wind)
        fail_safe = sensor_fault or confidence < threshold
        reason = None
        if fail_safe:
            fallback_count += 1
            reason = "sensor_fault" if sensor_fault else "low_confidence"
            action = "fallback_hover"
        elif step > (fault_start + fault_duration) and confidence >= threshold:
            recovered = True

        events.append(
            SitlEvent(
                run_id=run_id,
                step=step,
                offline_mode=True,
                sensor_fault=sensor_fault,
                confidence=confidence,
                action=action,
                fail_safe=fail_safe,
                reason=reason,
            )
        )

    summary = {
        "run_id": run_id,
        "steps": steps,
        "fault_window": [fault_start, fault_start + fault_duration - 1],
        "fail_safe_count": fallback_count,
        "fallback_triggered": fallback_count > 0,
        "recovered": recovered,
        "pass": fallback_count > 0 and recovered,
    }
    return events, summary


def write_outputs(out_dir: Path, events: list[SitlEvent], summaries: list[dict], args: argparse.Namespace) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "sitl_events.jsonl"
    summary_path = out_dir / "sitl_summary.json"
    report_path = out_dir / "sitl_report.md"

    with events_path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")

    all_green = all(item.get("pass") for item in summaries)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_id": args.pilot_id,
        "runs": summaries,
        "all_green": all_green,
        "config": {
            "runs": args.runs,
            "steps": args.steps,
            "seed": args.seed,
            "confidence_threshold": args.confidence_threshold,
            "fault_start": args.fault_start,
            "fault_duration": args.fault_duration,
            "offline_mode": True,
        },
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# SITL Demo Report",
        "",
        f"- Pilot ID: `{args.pilot_id}`",
        f"- Total runs: `{args.runs}`",
        f"- Steps per run: `{args.steps}`",
        f"- All green: `{all_green}`",
        "",
        "## Run Results",
    ]
    for s in summaries:
        report.append(
            f"- Run {s['run_id']}: pass={s['pass']}, fail_safe_count={s['fail_safe_count']}, recovered={s['recovered']}"
        )
    report.append("")
    report.append("## Evidence")
    report.append(f"- `{events_path.name}`")
    report.append(f"- `{summary_path.name}`")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    return {
        "events_path": str(events_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "all_green": all_green,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic SITL proof artifacts")
    parser.add_argument("--pilot-id", default="pilot_001")
    parser.add_argument("--out-root", default="reports/pilots")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--fault-start", type=int, default=45)
    parser.add_argument("--fault-duration", type=int, default=8)
    args = parser.parse_args()

    run_folder = Path(args.out_root) / args.pilot_id / f"sitl_{_iso_now()}"
    rng = random.Random(args.seed)

    all_events: list[SitlEvent] = []
    all_summaries: list[dict] = []
    for i in range(args.runs):
        events, summary = run_once(
            run_id=i + 1,
            steps=args.steps,
            fault_start=args.fault_start + (i * 2),
            fault_duration=args.fault_duration,
            threshold=args.confidence_threshold,
            rng=rng,
        )
        all_events.extend(events)
        all_summaries.append(summary)

    outputs = write_outputs(run_folder, all_events, all_summaries, args)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0 if outputs["all_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
