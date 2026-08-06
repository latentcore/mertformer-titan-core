"""Runtime configuration for ChessFormer.

A single frozen dataclass replaces the upstream global ``config.config.cfg``
singleton. The architecture modules take this object explicitly, which is the
only structural difference from the canonical ``layers/*.py`` implementations --
the math is mirrored exactly (see ``chessformer/arch/PARITY.md``).
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --- Board tokenization ------------------------------------------------------
# 12 meta tokens + 64 square tokens. Fixed for every position; there is no
# padding and no variable length anywhere in the pipeline.
NUM_META_TOKENS = 12
NUM_SQUARE_TOKENS = 64
SEQ_LEN = NUM_META_TOKENS + NUM_SQUARE_TOKENS  # 76

# Cardinality of each meta token (index -> vocabulary size of that slot).
META_CARDINALITIES: Tuple[int, ...] = (
    2,   # 0  side to move
    2,   # 1  white O-O
    2,   # 2  white O-O-O
    2,   # 3  black O-O
    2,   # 4  black O-O-O
    9,   # 5  en-passant file (0 = none)
    16,  # 6  halfmove clock bucket
    32,  # 7  fullmove number bucket
    2,   # 8  side to move is in check
    32,  # 9  legal move count bucket
    40,  # 10 white material bucket
    40,  # 11 black material bucket
)
assert len(META_CARDINALITIES) == NUM_META_TOKENS

# Value head: DeepMind (arXiv:2402.04494) used 128 uniform bins over the
# win-probability range with a cross-entropy loss. We keep 128 bins and use
# HL-Gauss (arXiv:2403.03950) on top, which is the measured improvement over
# plain one-hot cross-entropy.
NUM_VALUE_BINS = 128
HL_GAUSS_SIGMA_RATIO = 0.75  # sigma expressed in units of bin width


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyper-parameters. Mirrors the canonical field names."""

    hidden_size: int = 512
    intermediate_size: int = 1536
    num_layers: int = 12
    num_heads: int = 8
    num_kv_heads: int = 4
    head_dim: Optional[int] = None  # defaults to hidden_size // num_heads

    max_seq_len: int = SEQ_LEN
    dropout: float = 0.0
    attention_dropout: float = 0.0
    ffn_dropout: float = 0.0
    rms_norm_eps: float = 1e-6

    # --- attention -----------------------------------------------------------
    # "bidirectional" is the correct setting for a board *state*; the canonical
    # LM path is causal. See PARITY.md "deliberate deviation".
    attention_mode: str = "bidirectional"
    rope_base: float = 10000.0
    rope_dim: Optional[int] = None
    use_rope: bool = True

    # --- BitNet --------------------------------------------------------------
    use_bitnet: bool = False

    # --- MoE -----------------------------------------------------------------
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_every_n_layers: int = 3
    moe_intermediate: Optional[int] = None
    router_temperature: float = 1.0
    router_jitter: float = 0.02
    router_jitter_boost: float = 0.10
    shared_expert_gate: float = 0.0
    z_loss_coef: float = 1e-4
    use_switch_loss: bool = True
    moe_capacity_enforce: bool = True
    moe_capacity_factor: float = 1.25
    moe_dispatch_mode: str = "parallel"
    moe_aux_loss_coef: float = 0.01

    # --- Liquid / CfC --------------------------------------------------------
    use_liquid: bool = False
    liquid_layers_idx: Tuple[int, ...] = ()
    liquid_every_n_layers: int = 0
    liquid_fast_path: bool = True

    # --- QINN ----------------------------------------------------------------
    use_qinn: bool = False
    qinn_every_n_layers: int = 4
    qinn_iters: int = 6

    # --- heads ---------------------------------------------------------------
    policy_head: str = "factorized"  # "factorized" | "pooled"
    policy_head_dim: int = 128
    num_value_bins: int = NUM_VALUE_BINS
    use_wdl_head: bool = True
    use_legality_head: bool = True

    # --- loss weights --------------------------------------------------------
    policy_loss_coef: float = 1.0
    value_loss_coef: float = 0.6
    wdl_loss_coef: float = 0.15
    legality_loss_coef: float = 0.05
    legality_pos_weight_cap: float = 64.0

    # --- memory ---------------------------------------------------------------
    use_gradient_checkpointing: bool = False

    def resolved_head_dim(self) -> int:
        if self.head_dim is not None:
            return int(self.head_dim)
        return int(self.hidden_size // self.num_heads)

    def resolved_moe_intermediate(self) -> int:
        if self.moe_intermediate is not None:
            return int(self.moe_intermediate)
        return int(self.intermediate_size)

    def resolved_liquid_layers(self) -> Tuple[int, ...]:
        if self.liquid_layers_idx:
            return tuple(sorted({int(i) for i in self.liquid_layers_idx if 0 <= int(i) < self.num_layers}))
        if self.use_liquid and self.liquid_every_n_layers > 0:
            return tuple(
                i for i in range(self.num_layers) if (i + 1) % self.liquid_every_n_layers == 0
            )
        if self.use_liquid:
            # Same 4/18, 10/18, 16/18 placement ratios the canonical repo uses.
            return tuple(
                sorted({int(round((self.num_layers - 1) * r)) for r in (4 / 18, 10 / 18, 16 / 18)})
            )
        return ()

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_size, num_layers and num_heads must all be > 0")
        hd = self.resolved_head_dim()
        if hd * self.num_heads != self.hidden_size:
            raise ValueError(
                f"head_dim*num_heads ({hd}*{self.num_heads}) must equal hidden_size ({self.hidden_size})"
            )
        if not (1 <= self.num_kv_heads <= self.num_heads):
            raise ValueError(f"num_kv_heads must be in [1, num_heads], got {self.num_kv_heads}")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads (GQA requirement)")
        rope_dim = hd if self.rope_dim is None else int(self.rope_dim)
        if self.use_rope:
            if not (0 < rope_dim <= hd):
                raise ValueError(f"rope_dim must be in (0, head_dim], got {rope_dim}")
            if rope_dim % 2 != 0:
                raise ValueError(f"rope_dim must be even, got {rope_dim}")
        if self.attention_mode not in {"bidirectional", "causal"}:
            raise ValueError(f"attention_mode must be bidirectional|causal, got {self.attention_mode}")
        if self.policy_head not in {"factorized", "pooled"}:
            raise ValueError(f"policy_head must be factorized|pooled, got {self.policy_head}")
        if self.moe_dispatch_mode not in {"parallel", "sequential"}:
            raise ValueError(f"moe_dispatch_mode must be parallel|sequential, got {self.moe_dispatch_mode}")
        if self.use_moe and not (1 <= self.num_experts_per_tok <= self.num_experts):
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        if self.num_value_bins < 2:
            raise ValueError("num_value_bins must be >= 2")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Size presets. Parameter counts are computed, not guessed -- see
# ``estimate_parameters`` and the exact count reported by the built model.
# ---------------------------------------------------------------------------
MODEL_PRESETS: Dict[str, ModelConfig] = {
    "tiny": ModelConfig(
        hidden_size=192, intermediate_size=512, num_layers=4, num_heads=4,
        num_kv_heads=2, policy_head_dim=64,
    ),
    "small": ModelConfig(
        hidden_size=384, intermediate_size=1024, num_layers=8, num_heads=6,
        num_kv_heads=3, policy_head_dim=96,
    ),
    "base": ModelConfig(
        hidden_size=512, intermediate_size=1536, num_layers=12, num_heads=8,
        num_kv_heads=4, policy_head_dim=128,
    ),
    "large": ModelConfig(
        hidden_size=768, intermediate_size=2304, num_layers=16, num_heads=12,
        num_kv_heads=4, policy_head_dim=160,
    ),
}


def estimate_parameters(cfg: ModelConfig, vocab_size: int) -> int:
    """Analytic parameter estimate, used for VRAM planning before allocation."""
    h = cfg.hidden_size
    hd = cfg.resolved_head_dim()
    per_layer = (
        h * (cfg.num_heads * hd)           # q_proj
        + 2 * h * (cfg.num_kv_heads * hd)  # k_proj, v_proj
        + (cfg.num_heads * hd) * h         # o_proj
        + 2 * hd                           # q_norm, k_norm
        + 2 * h                            # norm1, norm2
    )
    if cfg.use_moe and cfg.moe_every_n_layers > 0:
        moe_layers = sum(
            1 for i in range(cfg.num_layers) if (i + 1) % cfg.moe_every_n_layers == 0
        )
    else:
        moe_layers = 0
    dense_layers = cfg.num_layers - moe_layers
    ffn = 3 * h * cfg.intermediate_size
    moe_ffn = (cfg.num_experts + 1) * 3 * h * cfg.resolved_moe_intermediate()
    moe_router = h * cfg.num_experts * 2 + h * 4

    embeddings = (
        13 * h                                  # piece types (0 = empty)
        + NUM_SQUARE_TOKENS * h                 # square identity
        + NUM_META_TOKENS * h                   # meta slot identity
        + sum(META_CARDINALITIES) * h           # meta values
    )
    # Heads. NUM_PROMO_SLOTS == 5 (none + q/r/b/n); the factorized head also
    # carries one bias per move id.
    if cfg.policy_head == "factorized":
        heads = 2 * h * cfg.policy_head_dim + h * 5 + vocab_size
    else:
        heads = h * vocab_size + vocab_size
    heads += h * h + h + h * cfg.num_value_bins + cfg.num_value_bins  # value MLP
    if cfg.use_wdl_head:
        heads += h * 3 + 3
    if cfg.use_legality_head:
        legality_dim = max(32, cfg.policy_head_dim // 2)
        heads += 2 * h * legality_dim + h * 5 + vocab_size

    total = (
        embeddings
        + cfg.num_layers * per_layer
        + dense_layers * ffn
        + moe_layers * (moe_ffn + moe_router)
        + heads
        + h  # final norm
    )
    if cfg.use_liquid:
        total += len(cfg.resolved_liquid_layers()) * (4 * h * h + h + 2 * h)
    if cfg.use_qinn:
        n_q = sum(1 for i in range(cfg.num_layers) if (i + 1) % max(1, cfg.qinn_every_n_layers) == 0)
        total += n_q * h * h
    return int(total)


# Heuristic guard against training a model far larger than the data supports.
# For calibration: DeepMind's 270M model saw ~15e9 action-value pairs, i.e.
# ~55 training examples per parameter. We are nowhere near that regime, so we
# require a much weaker ratio and simply refuse presets that would be badly
# over-parameterized for the dataset actually being built. This is an explicit
# heuristic, not a measured scaling law -- it is recorded as such in the report
# and can be overridden from the GUI.
MIN_POSITIONS_PER_PARAM = 0.6


def autoscale_model_config(
    vram_bytes: int,
    vocab_size: int,
    *,
    preferred: Optional[str] = None,
    reserve_fraction: float = 0.55,
    target_positions: Optional[int] = None,
) -> Tuple[ModelConfig, str, Dict[str, Any]]:
    """Pick the largest preset that fits both the VRAM budget and the dataset.

    VRAM model (bf16 autocast + fp32 AdamW, the configuration we actually run):
    4 B fp32 master weight + 4 B grad + 8 B Adam moments = 16 B/param.
    ``reserve_fraction`` of total VRAM is left for activations, the CUDA
    context, cuBLAS workspaces and fragmentation.

    Data model: see ``MIN_POSITIONS_PER_PARAM``.
    """
    order = ["large", "base", "small", "tiny"]
    if preferred and preferred in MODEL_PRESETS:
        order = [preferred] + [k for k in order if k != preferred]
    budget = int(vram_bytes * (1.0 - reserve_fraction))
    param_cap = (
        int(target_positions / MIN_POSITIONS_PER_PARAM) if target_positions else None
    )

    considered = []
    for name in order:
        cfg = MODEL_PRESETS[name]
        params = estimate_parameters(cfg, vocab_size)
        need = params * 16
        fits_vram = need <= budget
        fits_data = param_cap is None or params <= param_cap
        considered.append({
            "preset": name,
            "params": params,
            "state_bytes": need,
            "fits_vram": fits_vram,
            "fits_data": fits_data,
        })
        if fits_vram and fits_data:
            return cfg, name, {
                "vram_bytes": int(vram_bytes),
                "budget_bytes": budget,
                "chosen": name,
                "estimated_params": params,
                "estimated_state_bytes": need,
                "reserve_fraction": reserve_fraction,
                "target_positions": target_positions,
                "param_cap_from_data": param_cap,
                "positions_per_param_rule": MIN_POSITIONS_PER_PARAM,
                "considered": considered,
                "mode": "auto",
            }

    cfg = MODEL_PRESETS["tiny"]
    return cfg, "tiny", {
        "vram_bytes": int(vram_bytes),
        "budget_bytes": budget,
        "chosen": "tiny",
        "estimated_params": estimate_parameters(cfg, vocab_size),
        "reserve_fraction": reserve_fraction,
        "target_positions": target_positions,
        "param_cap_from_data": param_cap,
        "considered": considered,
        "mode": "auto",
        "note": "no preset satisfied both the VRAM and dataset constraints; fell back to tiny",
    }


# ---------------------------------------------------------------------------
# Run configuration (data + training + evaluation)
# ---------------------------------------------------------------------------

LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
LICHESS_PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"


@dataclass
class RunConfig:
    """Everything the pipeline needs for one end-to-end run."""

    run_name: str = "chessformer"
    seed: int = 42
    device: str = "auto"

    # --- paths ---------------------------------------------------------------
    workspace: str = ""          # filled in by resolve_paths()
    data_dir: str = ""
    runs_dir: str = ""

    # --- data ----------------------------------------------------------------
    eval_db_url: str = LICHESS_EVAL_URL
    puzzle_db_url: str = LICHESS_PUZZLE_URL
    target_positions: int = 20_000_000
    min_eval_depth: int = 12
    max_pvs_per_position: int = 4
    # Soft-policy temperature in win-probability units. At 0.03 a move that is
    # ~5cp worse keeps ~86% of the best move's weight (near-equal moves stay
    # informative) while a ~100cp-worse move drops to ~5% (blunders are
    # suppressed). Measured in tests/test_data.py.
    policy_soft_tau: float = 0.03
    val_positions: int = 100_000
    test_positions: int = 100_000
    shard_positions: int = 1_000_000
    download_timeout_sec: int = 120
    download_retries: int = 4
    max_disk_gb: float = 60.0

    # --- model ---------------------------------------------------------------
    model_preset: str = "auto"
    model_overrides: Dict[str, Any] = field(default_factory=dict)

    # --- training ------------------------------------------------------------
    profile: str = "run_48h"
    max_wall_hours: float = 48.0
    max_steps: int = 0            # 0 = derive from wall clock via calibration
    batch_size: int = 0           # 0 = autotune
    grad_accum_steps: int = 1
    learning_rate: float = 3.0e-4
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    use_bf16: bool = True
    use_tf32: bool = True
    deterministic: bool = False
    compile_mode: str = "off"     # off | default | max-autotune
    num_workers: int = -1         # -1 = auto
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    ema_decay: float = 0.0

    checkpoint_every_steps: int = 2000
    checkpoint_every_minutes: float = 20.0
    keep_last_checkpoints: int = 3
    eval_every_steps: int = 5000
    log_every_steps: int = 50
    metric_sync_every_steps: int = 50

    # --- evaluation ----------------------------------------------------------
    puzzle_sample_size: int = 5000
    elo_total_games: int = 200
    elo_block_games: int = 20
    elo_start: int = 1500
    elo_movetime_ms: int = 100
    elo_max_plies: int = 300
    stockfish_path: str = ""
    stockfish_threads: int = 1
    stockfish_hash_mb: int = 64
    allow_stockfish_download: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RunConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})


# Wall-clock profiles. ``calibrate`` measures real throughput and then the step
# budget is *computed* rather than guessed -- see chessformer/train.py.
RUN_PROFILES: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "max_wall_hours": 0.05,
        "max_steps": 20,
        "target_positions": 20_000,
        "val_positions": 512,
        "test_positions": 512,
        "shard_positions": 20_000,
        "model_preset": "tiny",
        "batch_size": 32,
        "warmup_steps": 5,
        "checkpoint_every_steps": 10,
        "eval_every_steps": 10,
        "log_every_steps": 1,
        "metric_sync_every_steps": 1,
        "puzzle_sample_size": 64,
        "elo_total_games": 2,
        "elo_block_games": 2,
        "num_workers": 0,
        "persistent_workers": False,
    },
    "calibrate": {
        "max_wall_hours": 0.35,
        "max_steps": 0,
        "target_positions": 2_000_000,
        "model_preset": "auto",
        "warmup_steps": 200,
        "checkpoint_every_steps": 500,
        "eval_every_steps": 1000,
        "puzzle_sample_size": 1000,
        "elo_total_games": 40,
    },
    "run_12h": {
        "max_wall_hours": 12.0,
        "target_positions": 8_000_000,
        "model_preset": "small",
        "elo_total_games": 120,
    },
    "run_24h": {
        "max_wall_hours": 24.0,
        "target_positions": 16_000_000,
        "model_preset": "auto",
        "elo_total_games": 160,
    },
    "run_48h": {
        "max_wall_hours": 48.0,
        "target_positions": 32_000_000,
        "model_preset": "auto",
        "elo_total_games": 200,
    },
}


def apply_profile(cfg: RunConfig, profile: str) -> RunConfig:
    if profile not in RUN_PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {sorted(RUN_PROFILES)}")
    payload = cfg.to_dict()
    payload.update(RUN_PROFILES[profile])
    payload["profile"] = profile
    return RunConfig.from_dict(payload)


def resolve_paths(cfg: RunConfig, workspace: Optional[Path] = None) -> RunConfig:
    root = Path(cfg.workspace) if cfg.workspace else (workspace or Path.cwd())
    root = root.expanduser().resolve()
    cfg.workspace = str(root)
    cfg.data_dir = cfg.data_dir or str(root / "data")
    cfg.runs_dir = cfg.runs_dir or str(root / "runs")
    return cfg


def build_model_config(cfg: RunConfig, vram_bytes: int, vocab_size: int) -> Tuple[ModelConfig, Dict[str, Any]]:
    """Resolve the model config, autoscaling to VRAM when preset == 'auto'."""
    if cfg.model_preset == "auto":
        model_cfg, chosen, report = autoscale_model_config(
            vram_bytes, vocab_size, target_positions=int(cfg.target_positions)
        )
    else:
        if cfg.model_preset not in MODEL_PRESETS:
            raise ValueError(f"unknown model preset {cfg.model_preset!r}")
        model_cfg = MODEL_PRESETS[cfg.model_preset]
        chosen = cfg.model_preset
        report = {
            "mode": "explicit",
            "chosen": chosen,
            "vram_bytes": int(vram_bytes),
            "estimated_params": estimate_parameters(model_cfg, vocab_size),
        }
    if cfg.model_overrides:
        payload = model_cfg.to_dict()
        unknown = set(cfg.model_overrides) - set(payload)
        if unknown:
            raise ValueError(f"unknown model_overrides keys: {sorted(unknown)}")
        payload.update(cfg.model_overrides)
        if isinstance(payload.get("liquid_layers_idx"), list):
            payload["liquid_layers_idx"] = tuple(payload["liquid_layers_idx"])
        model_cfg = ModelConfig(**payload)
        report["overrides"] = dict(cfg.model_overrides)
        report["estimated_params"] = estimate_parameters(model_cfg, vocab_size)
    model_cfg.validate()
    return model_cfg, report


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomic JSON write (temp + os.replace), used everywhere we persist state."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
