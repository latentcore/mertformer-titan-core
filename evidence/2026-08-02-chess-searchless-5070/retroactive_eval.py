#!/usr/bin/env python3
"""Run holdout + puzzles + elo against an already-trained checkpoint, without
retraining. The original run_20260802_194441 pipeline was stopped by the
operator before these stages had real data to evaluate against (holdout saw
0 batches, puzzles had no local puzzle DB, elo saw should_stop() already
true). This script re-runs exactly the same evaluation functions the
pipeline itself uses (chessformer.eval.holdout/puzzles/benchmark), loading
model weights from the saved checkpoint instead of training from scratch.

Usage:
    python retroactive_eval.py --run-id 20260802_194441 --checkpoint best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from chessformer.board import VOCAB_SIZE
from chessformer.config import ModelConfig, RunConfig, write_json
from chessformer.eval.benchmark import run_stockfish_benchmark
from chessformer.eval.holdout import evaluate_all_splits
from chessformer.eval.puzzles import evaluate_puzzles
from chessformer.model import ChessFormer
from chessformer.runtime import pick_device


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg_dict = dict(payload["model_config"])
    if isinstance(model_cfg_dict.get("liquid_layers_idx"), list):
        model_cfg_dict["liquid_layers_idx"] = tuple(model_cfg_dict["liquid_layers_idx"])
    model_cfg = ModelConfig(**model_cfg_dict)
    model = ChessFormer(model_cfg, VOCAB_SIZE).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    run_cfg = RunConfig.from_dict(payload["run_config"])
    return model, model_cfg, run_cfg, payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--workspace", default=str(ROOT / "workspace"))
    ap.add_argument("--skip-elo", action="store_true")
    ap.add_argument("--skip-holdout", action="store_true")
    ap.add_argument("--skip-puzzles", action="store_true")
    args = ap.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    run_dir = workspace / "runs" / f"run_{args.run_id}"
    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    out_dir = run_dir / "reports" / "retroactive_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    device_info = pick_device("auto")
    device = device_info.device
    print(f"device: {device_info.to_dict().get('name', device)}")

    model, model_cfg, run_cfg, payload = load_model_from_checkpoint(ckpt_path, device)
    print(f"loaded checkpoint: {ckpt_path}")
    print(f"trained step: {payload.get('state', {}).get('step')}")
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    data_dir = workspace / "data"
    data_root = data_dir / "packed"

    def progress(payload: dict) -> None:
        stage = payload.get("stage", "")
        status = payload.get("status")
        if status:
            print(f"  [{stage}] {status}", flush=True)

    if not args.skip_holdout:
        print("\n=== holdout ===")
        holdout_report = evaluate_all_splits(
            model, model_cfg, data_root, device,
            batch_size=max(32, int(run_cfg.batch_size or 64)),
            progress=progress, should_stop=lambda: False,
        )
        holdout_report["status"] = "completed"
        write_json(out_dir / "holdout.json", holdout_report)
        print(f"holdout -> {out_dir / 'holdout.json'}")

    if not args.skip_puzzles:
        print("\n=== puzzles ===")
        puzzle_path = data_dir / "lichess_db_puzzle.csv.zst"
        puzzles_report = evaluate_puzzles(
            model, device, puzzle_path,
            sample_size=int(run_cfg.puzzle_sample_size), seed=int(run_cfg.seed),
            progress=progress, should_stop=lambda: False,
        )
        write_json(out_dir / "puzzles.json", puzzles_report)
        print(f"puzzles -> {out_dir / 'puzzles.json'}")

    if not args.skip_elo:
        print("\n=== elo ===")
        elo_report = run_stockfish_benchmark(
            model, device,
            stockfish_path=run_cfg.stockfish_path,
            cache_dir=data_dir / "stockfish",
            allow_download=bool(run_cfg.allow_stockfish_download),
            total_games=int(run_cfg.elo_total_games),
            block_games=int(run_cfg.elo_block_games),
            start_elo=int(run_cfg.elo_start),
            movetime_ms=int(run_cfg.elo_movetime_ms),
            threads=int(run_cfg.stockfish_threads),
            hash_mb=int(run_cfg.stockfish_hash_mb),
            max_plies=int(run_cfg.elo_max_plies),
            progress=progress, should_stop=lambda: False,
            save_pgn_dir=out_dir / "games",
        )
        write_json(out_dir / "elo.json", elo_report)
        print(f"elo -> {out_dir / 'elo.json'}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
