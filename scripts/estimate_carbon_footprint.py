#!/usr/bin/env python3
"""
Training-run carbon-footprint calculator.

[2026-07-12] BACKLOG I.7 #79: "Carbon footprint". Real arithmetic
(GPU-hours x TDP x PUE x grid carbon-intensity), no ML/checkpoint needed --
computable now, ready to plug in the real GPU-hours once the 45K run
actually happens (--gpu-hours defaults to an estimate from FACTS.json's own
compute-request numbers, override with the real wall-clock once known).

Methodology (standard, e.g. ML CO2 Impact / Strubell et al.):
    energy_kwh = num_gpus * gpu_hours * (tdp_watts / 1000) * pue
    co2_kg = energy_kwh * grid_carbon_intensity_kg_per_kwh

Grid carbon intensity varies wildly by region/provider -- this ships a small
table of commonly-cited reference values (global average, US average, a
renewables-heavy grid) rather than a single hardcoded "the" number, and
--carbon-intensity accepts a direct override for a known provider PUE/mix.

Usage:
    python scripts/estimate_carbon_footprint.py --gpu-type H100 --num-gpus 2 --gpu-hours 10
    python scripts/estimate_carbon_footprint.py --gpu-type H100 --num-gpus 8 --gpu-hours 240 --carbon-intensity 0.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "reports" / "carbon_footprint_estimate.json"

# TDP (thermal design power, watts) -- per-GPU, nameplate max, a conservative upper bound
# (real average utilization draw is typically lower; this errs toward overestimating).
GPU_TDP_WATTS = {
    "H100": 700,
    "H200": 700,
    "A100": 400,
    "B300": 1000,  # per public NVIDIA GB300/B300 platform figures at time of writing
}

# kg CO2e per kWh -- commonly-cited reference points, NOT a live/metered value for any
# specific provider. A real figure should come from the actual cloud provider's own
# sustainability disclosure for the specific datacenter region used.
GRID_CARBON_INTENSITY_REFERENCE = {
    "global_average": 0.475,
    "us_average": 0.386,
    "eu_average": 0.253,
    "renewables_heavy_grid": 0.05,
}

PUE_DEFAULT = 1.2  # datacenter Power Usage Effectiveness, a common modern-datacenter figure


def estimate(num_gpus: int, gpu_hours: float, tdp_watts: float, pue: float, carbon_intensity: float) -> dict:
    energy_kwh = num_gpus * gpu_hours * (tdp_watts / 1000.0) * pue
    co2_kg = energy_kwh * carbon_intensity
    return {
        "num_gpus": num_gpus,
        "gpu_hours_per_gpu": gpu_hours,
        "total_gpu_hours": num_gpus * gpu_hours,
        "tdp_watts": tdp_watts,
        "pue": pue,
        "carbon_intensity_kg_per_kwh": carbon_intensity,
        "energy_kwh": round(energy_kwh, 2),
        "co2_kg": round(co2_kg, 2),
        "co2_tonnes": round(co2_kg / 1000.0, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate the carbon footprint of a training run.")
    parser.add_argument("--gpu-type", choices=sorted(GPU_TDP_WATTS), default="H100")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--gpu-hours", type=float, required=True, help="wall-clock hours PER GPU")
    parser.add_argument("--pue", type=float, default=PUE_DEFAULT)
    parser.add_argument(
        "--carbon-intensity",
        type=float,
        default=None,
        help="kg CO2e/kWh override; defaults to reporting all reference grids",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    tdp = GPU_TDP_WATTS[args.gpu_type]

    if args.carbon_intensity is not None:
        scenarios = {"custom": estimate(args.num_gpus, args.gpu_hours, tdp, args.pue, args.carbon_intensity)}
    else:
        scenarios = {
            name: estimate(args.num_gpus, args.gpu_hours, tdp, args.pue, intensity)
            for name, intensity in GRID_CARBON_INTENSITY_REFERENCE.items()
        }

    summary = {
        "schema": "carbon_footprint_estimate_v1",
        "gpu_type": args.gpu_type,
        "scenarios": scenarios,
        "claim_boundary": (
            "Standard energy x grid-intensity arithmetic against reference grid-carbon "
            "figures, not a live/metered value for any specific cloud provider or "
            "datacenter. Not a capability or 'trained' claim."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n[estimate_carbon_footprint] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
