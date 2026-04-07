from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.chess_5080_onefile as onefile

GUI_PATH = ROOT / "apps" / "chess_gui" / "play_mertformer_chess_web.py"


def load_gui_module():
    spec = importlib.util.spec_from_file_location("mertformer_chess_gui_local", GUI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load GUI module from {GUI_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_arena_state_passes_mode_and_teaching_level_to_contract(monkeypatch, tmp_path: Path) -> None:
    gui = load_gui_module()
    monkeypatch.setattr(gui, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(gui, "SESSION_LOG_PATH", tmp_path / "logs" / "arena_session.jsonl")
    monkeypatch.setattr(gui, "BENCHMARK_HISTORY_PATH", tmp_path / "logs" / "benchmark_history.json")

    captured: dict[str, str] = {}

    def fake_choose_move_trace(model, board, device, topk=5, *, mode="play", teaching_level="club"):
        captured["mode"] = mode
        captured["teaching_level"] = teaching_level
        return {
            "move": "e2e4",
            "value": 0.22,
            "latency_ms": 1.1,
            "raw_top1_is_legal": True,
            "raw_topk": ["e2e4"],
            "raw_topk_scores": [3.0],
            "masked_topk": ["e2e4"],
            "masked_topk_scores": [3.0],
            "confidence": {"score": 0.74, "gap": 0.31, "tier": "high"},
            "response_contract": {
                "best_move": "e2e4",
                "best_move_san": "e4",
                "evaluation": {"value": 0.22, "label": "pressing", "perspective": "side_to_move"},
                "principal_variation": ["e2e4"],
                "confidence": {"score": 0.74, "gap": 0.31, "tier": "high"},
                "teaching_tags": ["center_control"],
                "explanation_tr_short": "Türkçe öğretme modu: e4 merkez kontrolü sağlıyor.",
                "explanation_tr_long": "Uzun açıklama.",
                "mode": mode,
                "teaching_level": teaching_level,
            },
        }

    monkeypatch.setattr(onefile, "choose_move_trace", fake_choose_move_trace)

    summary = gui.ArenaSummary(
        run_id="test-run",
        rating_claim_status="no_claim",
        steps_completed=12,
        holdout_masked_accuracy=0.5,
        holdout_masked_top5=0.7,
        positions_total=1234,
        best_val_loss=1.23,
    )
    state = gui.ArenaState(
        onefile=onefile,
        model=object(),
        device=onefile.torch.device("cpu"),
        summary=summary,
        human_color=onefile.chess.WHITE,
    )

    payload = state.set_mode_preferences(mode="turkish_teach", teaching_level="advanced")
    assert payload["ui_mode"] == "turkish_teach"
    assert payload["teaching_level"] == "advanced"

    payload = state.new_game("black")
    assert captured["mode"] == "turkish_teach"
    assert captured["teaching_level"] == "advanced"
    assert payload["last_trace"]["response_contract"]["mode"] == "turkish_teach"
    assert payload["last_trace"]["response_contract"]["teaching_level"] == "advanced"
