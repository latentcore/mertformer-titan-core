#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

# Ensure imports work both in main repo layout and defense-demo layout.
# - Main repo: <root>/layers
# - Defense demo: <root>/src/layers
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from layers.bitlinear import BitLinear
from layers.moe import LiquidRouter


ACTIONS = ("advance", "track", "stabilize", "hold")


@dataclass
class SitlEvent:
    run_id: int
    step: int
    offline_mode: bool
    sensor_fault: bool
    confidence: float
    wind: float
    policy_action: str
    policy_engine: str
    action: str
    fail_safe: bool
    reason: str | None


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class MertFormerSITLPolicy(nn.Module):
    """
    Lightweight controller using MertFormer primitives (BitLinear + LiquidRouter).
    This keeps the demo AI-driven without requiring a full checkpoint load.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_size: int = 48,
        num_experts: int = 4,
        top_k: int = 2,
        seed: int = 27,
    ) -> None:
        super().__init__()
        self.top_k = max(1, min(top_k, num_experts))
        self.encoder = BitLinear(input_dim, hidden_size, bias=False)
        self.router = LiquidRouter(hidden_size, num_experts)
        self.experts = nn.ModuleList(
            [BitLinear(hidden_size, hidden_size, bias=False) for _ in range(num_experts)]
        )
        self.action_head = BitLinear(hidden_size, len(ACTIONS), bias=False)

        torch.manual_seed(seed)
        for mod in [self.encoder, self.action_head, *self.experts]:
            if hasattr(mod, "weight") and mod.weight is not None:
                nn.init.xavier_uniform_(mod.weight)

    @torch.no_grad()
    def choose_action(self, state_vec: Sequence[float]) -> str:
        x = torch.tensor([list(state_vec)], dtype=torch.float32)
        h = torch.tanh(self.encoder(x))  # [1, H]
        router_logits = self.router(h.unsqueeze(1)).squeeze(1)  # [1, E]

        topv, topi = torch.topk(router_logits, k=self.top_k, dim=-1)
        gate = torch.softmax(topv, dim=-1).squeeze(0)  # [K]

        mixed = torch.zeros_like(h)
        for i in range(self.top_k):
            exp_idx = int(topi[0, i].item())
            mixed = mixed + gate[i] * torch.tanh(self.experts[exp_idx](h))

        action_logits = self.action_head(mixed).squeeze(0)  # [A]
        action_idx = int(torch.argmax(action_logits).item())
        return ACTIONS[action_idx]


def _baseline_action(confidence: float, wind: float) -> str:
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
    policy_engine: str,
    ai_policy: MertFormerSITLPolicy | None,
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

        step_norm = step / max(1, (steps - 1))
        state_vec = [confidence, wind, 1.0 if sensor_fault else 0.0, step_norm]
        if ai_policy is not None:
            policy_action = ai_policy.choose_action(state_vec)
        else:
            policy_action = _baseline_action(confidence, wind)

        action = policy_action
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
                wind=round(wind, 4),
                policy_action=policy_action,
                policy_engine=policy_engine,
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
        "policy_engine": policy_engine,
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
            "policy_engine": args.policy_engine,
            "ai_hidden_size": args.ai_hidden_size,
            "ai_experts": args.ai_experts,
            "ai_top_k": args.ai_top_k,
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
            f"- Run {s['run_id']}: pass={s['pass']}, fail_safe_count={s['fail_safe_count']}, recovered={s['recovered']}, policy={s['policy_engine']}"
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
    parser.add_argument(
        "--policy-engine",
        choices=["mertformer_liquidrouter", "baseline"],
        default="mertformer_liquidrouter",
        help="Use MertFormer components for action policy (default) or baseline rules.",
    )
    parser.add_argument("--ai-hidden-size", type=int, default=48)
    parser.add_argument("--ai-experts", type=int, default=4)
    parser.add_argument("--ai-top-k", type=int, default=2)
    args = parser.parse_args()

    run_folder = Path(args.out_root) / args.pilot_id / f"sitl_{_iso_now()}"
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    ai_policy = None
    policy_engine = args.policy_engine
    if args.policy_engine == "mertformer_liquidrouter":
        ai_policy = MertFormerSITLPolicy(
            hidden_size=args.ai_hidden_size,
            num_experts=args.ai_experts,
            top_k=args.ai_top_k,
            seed=args.seed,
        ).eval()
    else:
        policy_engine = "baseline"

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
            policy_engine=policy_engine,
            ai_policy=ai_policy,
        )
        all_events.extend(events)
        all_summaries.append(summary)

    outputs = write_outputs(run_folder, all_events, all_summaries, args)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0 if outputs["all_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
