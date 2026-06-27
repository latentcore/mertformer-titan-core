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

    # NOTE: This is a RESOURCE-USAGE proxy baseline only (CPU%/RAM via psutil).
    # No real power (watts/joules) or thermal (sensor temperature) is measured.
    # The "energy"/"thermal" filenames below are legacy report names kept for
    # downstream-consumer compatibility; the captured signals are NOT measured
    # energy or temperature -- treat them as resource proxies, not power data.
    vm = psutil.virtual_memory()
    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'baseline',
        'measurement_kind': 'resource_usage_proxy',
        'energy_measured': False,
        'thermal_measured': False,
        'note': 'CPU/RAM snapshot only; no real power or temperature sensor read',
        'cpu_percent': float(psutil.cpu_percent(interval=0.2)),
        'ram_total_gb': float(vm.total / (1024 ** 3)),
        'ram_used_gb': float((vm.total - vm.available) / (1024 ** 3)),
    }

    (reports / 'system_stats.jsonl').open('a', encoding='utf-8').write(json.dumps(payload) + '\n')
    (reports / 'energy_baseline.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (reports / 'latency_baseline.json').write_text(json.dumps({'timestamp_utc': payload['timestamp_utc'], 'mode': 'baseline'}, indent=2), encoding='utf-8')
    (reports / 'thermal_baseline.json').write_text(json.dumps({'timestamp_utc': payload['timestamp_utc'], 'mode': 'baseline_proxy', 'thermal_measured': False, 'note': 'no temperature sensor read; proxy placeholder only'}, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
