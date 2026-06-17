"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - UTILS PACKAGE
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30 V2) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)

Public API surface for shared runtime utilities (logging, tokenizer policy,
liquid safeguards, numeric safety, dataset provenance).
==============================================================================
"""

from utils.dataset_registry import get_hf_revision, get_snapshot_sha256
from utils.liquid_safeguard import update_liquid_spike_state
from utils.logger import RunLogger, atomic_write_json, sha256_file, try_git_commit
from utils.safety import is_finite, kill_if_non_finite
from utils.tokenizer_resolver import (
    load_tokenizer_from_identity,
    resolve_tokenizer,
    tokenizer_identity,
    tokenizer_name_or_path,
)

__all__ = [
    "RunLogger",
    "atomic_write_json",
    "get_hf_revision",
    "get_snapshot_sha256",
    "is_finite",
    "kill_if_non_finite",
    "load_tokenizer_from_identity",
    "resolve_tokenizer",
    "sha256_file",
    "tokenizer_identity",
    "tokenizer_name_or_path",
    "try_git_commit",
    "update_liquid_spike_state",
]
