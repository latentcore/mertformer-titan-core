"""Single source of truth for the MertFormer Titan runtime tokenizer.

TR: Egitim, eval ve demo tokenizer'i HER ZAMAN bu modulden alir. Boylece
    train/eval/demo arasinda tokenizer ailesi uyusmazligi (Llama BPE vs TR
    WordPiece) olusamaz.
EN: Train, eval and demo MUST obtain the tokenizer through this module so a
    tokenizer-family mismatch (Llama BPE vs Turkish WordPiece) can never arise
    between training and evaluation.

Policy (no half-half):
  - ``cfg.use_tr_tokenizer == False`` (default): the teacher tokenizer
    (``cfg.teacher_model_id``) is used everywhere -- teacher inputs, KD logit
    space, and eval. KD distillation only makes sense when student inputs share
    the teacher's vocabulary, so this is the safe default.
  - ``cfg.use_tr_tokenizer == True`` (opt-in): the local Turkish tokenizer is
    used everywhere -- teacher inputs, KD, and eval -- never a mix of the two.

There is intentionally NO silent ``gpt2`` / TR fallback when the teacher
tokenizer cannot be loaded: a wrong-but-loadable tokenizer is exactly the
failure mode this module exists to prevent. Callers get an explicit error.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_cfg(cfg):
    if cfg is not None:
        return cfg
    from config.config import cfg as _cfg

    return _cfg


def _tr_tokenizer_candidates(cfg) -> list[Path]:
    """Local Turkish-tokenizer artifact search paths (never the teacher BPE)."""
    candidates: list[Path] = []

    env_override = os.environ.get("TITAN_LOCAL_TOKENIZER_PATH", "").strip()
    if env_override:
        p = Path(env_override).expanduser()
        candidates.append(p if p.is_absolute() else PROJECT_ROOT / p)

    configured = str(getattr(cfg, "tr_tokenizer_id", "") or "").strip()
    if configured:
        p = Path(configured).expanduser()
        candidates.append(p if p.is_absolute() else PROJECT_ROOT / p)

    candidates.extend(
        [
            PROJECT_ROOT / "data" / "tokenizer" / "tr",
            PROJECT_ROOT / "tokenizer" / "tr",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _load_tr_tokenizer(cfg):
    from transformers import AutoTokenizer

    last_error: Optional[Exception] = None
    for path in _tr_tokenizer_candidates(cfg):
        if not path.exists() or not (path / "tokenizer.json").exists():
            continue
        try:
            print(f"🔤 Tokenizer (TR, opt-in): {path}")
            tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            tok._titan_name_or_path = str(path)
            return _ensure_pad_token(tok)
        except Exception as exc:  # noqa: BLE001 - surface the real cause below
            last_error = exc
            print(f"⚠️ TR tokenizer load failed for {path}: {exc}")
    if last_error is not None:
        raise RuntimeError(f"TR tokenizer candidates failed: {last_error}") from last_error
    raise FileNotFoundError(
        "use_tr_tokenizer=1 but no local Turkish tokenizer artifact was found "
        "(looked under data/tokenizer/tr, tokenizer/tr, tr_tokenizer_id, "
        "TITAN_LOCAL_TOKENIZER_PATH)."
    )


def _load_teacher_tokenizer(cfg):
    from transformers import AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    try:
        print(f"🔤 Tokenizer (teacher): {cfg.teacher_model_id}")
        tok = AutoTokenizer.from_pretrained(cfg.teacher_model_id, token=hf_token)
        tok._titan_name_or_path = cfg.teacher_model_id
        return _ensure_pad_token(tok)
    except Exception as exc:  # noqa: BLE001 - re-raised explicitly below
        # Offline escape hatch: a *local snapshot of the teacher tokenizer*
        # (NOT the Turkish WordPiece). Must be opt-in via an explicit env path so
        # we never silently substitute a different tokenizer family.
        snap = os.environ.get("TITAN_TEACHER_TOKENIZER_PATH", "").strip()
        if snap:
            p = Path(snap).expanduser()
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                print(f"🔤 Tokenizer (teacher snapshot): {p}")
                tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
                tok._titan_name_or_path = cfg.teacher_model_id
                return _ensure_pad_token(tok)
        raise RuntimeError(
            f"Teacher tokenizer '{cfg.teacher_model_id}' could not be loaded ({exc}). "
            "Provide HF_TOKEN for gated access, set TITAN_TEACHER_TOKENIZER_PATH to a "
            "local snapshot of the SAME teacher tokenizer, or opt in to the Turkish "
            "tokenizer with TITAN_USE_TR_TOKENIZER=1. No silent gpt2/TR fallback is "
            "allowed -- train and eval tokenizers must be identical."
        ) from exc


def resolve_tokenizer(cfg=None):
    """Return the one runtime tokenizer for train / eval / demo.

    The choice of tokenizer family is governed solely by ``cfg.use_tr_tokenizer``
    so that every entry point resolves the same tokenizer for a given config.
    """
    cfg = _resolve_cfg(cfg)
    if bool(getattr(cfg, "use_tr_tokenizer", False)):
        return _load_tr_tokenizer(cfg)
    return _load_teacher_tokenizer(cfg)


def tokenizer_name_or_path(tokenizer) -> str:
    return (
        getattr(tokenizer, "_titan_name_or_path", None)
        or getattr(tokenizer, "name_or_path", "")
        or ""
    )


def tokenizer_identity(tokenizer, cfg=None) -> dict:
    """Serializable identity stamped into checkpoints (``checkpoint['tokenizer_id']``).

    ``vocab_size`` uses ``len(tokenizer)`` so it includes added/special tokens
    (e.g. Llama-3: vocab_size attr 128000 but len 128256). The model embedding /
    lm_head are sized to this exact number.
    """
    cfg = _resolve_cfg(cfg)
    return {
        "name_or_path": tokenizer_name_or_path(tokenizer),
        "vocab_size": len(tokenizer),
        "tokenizer_class": type(tokenizer).__name__,
        "use_tr_tokenizer": bool(getattr(cfg, "use_tr_tokenizer", False)),
    }


def load_tokenizer_from_identity(identity: Optional[dict]):
    """Reload the exact tokenizer recorded in a checkpoint. No silent fallback.

    Raises ValueError if the checkpoint carries no tokenizer identity or if the
    reloaded tokenizer's vocab size no longer matches what training recorded --
    both indicate a train/eval tokenizer mismatch.
    """
    if not identity or not identity.get("name_or_path"):
        raise ValueError(
            "Checkpoint has no 'tokenizer_id'. Refusing to guess the tokenizer "
            "(silent teacher fallback is exactly the train/eval mismatch bug this "
            "guards against). Re-train with a build that records "
            "checkpoint['tokenizer_id']."
        )

    from transformers import AutoTokenizer

    name_or_path = identity["name_or_path"]
    local = Path(name_or_path)
    if local.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(local), local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, token=os.environ.get("HF_TOKEN")
        )
    tokenizer._titan_name_or_path = name_or_path
    tokenizer = _ensure_pad_token(tokenizer)

    expected = identity.get("vocab_size")
    if expected is not None and len(tokenizer) != expected:
        raise ValueError(
            f"Tokenizer vocab mismatch: checkpoint recorded {expected} tokens for "
            f"'{name_or_path}' but it now resolves to {len(tokenizer)}. Refusing to "
            "evaluate with a different tokenizer than training used."
        )
    return tokenizer
