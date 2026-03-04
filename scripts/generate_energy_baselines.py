#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    reports = ROOT / 'reports'
    reports.mkdir(parents=True, exist_ok=True)

    vm = psutil.virtual_memory()
    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'baseline',
        'cpu_percent': float(psutil.cpu_percent(interval=0.2)),
        'ram_total_gb': float(vm.total / (1024 ** 3)),
        'ram_used_gb': float((vm.total - vm.available) / (1024 ** 3)),
    }

    (reports / 'system_stats.jsonl').open('a', encoding='utf-8').write(json.dumps(payload) + '\n')
    (reports / 'energy_baseline.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (reports / 'latency_baseline.json').write_text(json.dumps({'timestamp_utc': payload['timestamp_utc'], 'mode': 'baseline'}, indent=2), encoding='utf-8')
    (reports / 'thermal_baseline.json').write_text(json.dumps({'timestamp_utc': payload['timestamp_utc'], 'mode': 'baseline_proxy'}, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
