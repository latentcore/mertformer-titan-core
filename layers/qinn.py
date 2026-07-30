"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright 2026 Mert Yunlu
Licensed under the Apache License, Version 2.0 (see LICENSE).

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Status : PRE-TRAINING (UNVERIFIED)

NOTE (audit): inert / out-of-scope. UnitaryQINN is a speculative layer and is
disabled on the 45K training path via the `use_qinn` feature-flag
(config.config.cfg.use_qinn = False by default). The version label is sourced
from the SDK (mertformer_sdk.__version__) instead of a hand-maintained fossil
("BUILD30-V2") to avoid version drift.
==============================================================================
"""

try:
    from mertformer_sdk import __version__  # single source of truth
except Exception:  # pragma: no cover - SDK optional at import time
    __version__ = "unknown"
__author__ = "Mert Yünlü"

import torch
import torch.nn as nn
from typing import Optional

from config.config import cfg


def newton_schulz_inverse(
    mat: torch.Tensor, num_iters: int = 6
) -> torch.Tensor:
    """
    Computes approximate matrix inverse via Newton-Schulz method (GPU/MPS safe).

    Features:
    - Uses only matmul (no torch.linalg.solve / inv) -> CUDA / MPS safe
    - Runs in FP32, casts back to the input dtype at the end

    Args:
        mat (torch.Tensor): Square matrix [..., N, N]
        num_iters (int): Number of iterations
    Returns:
        torch.Tensor: Approximate matrix inverse
    Raises:
        AssertionError: If matrix is not square
    """
    # Convert to float32 just in case
    orig_dtype = mat.dtype
    mat = mat.float()

    # Dimensions
    *batch, n, m = mat.shape
    assert n == m, "TR: newton_schulz_inverse: Kare matris bekleniyor. / EN: Square matrix expected."

    # Identity matrix (can be broadcast to batch)
    I = torch.eye(n, device=mat.device, dtype=mat.dtype)
    if batch:
        I = I.expand(*batch, n, n)

    # Scaling (||A||_inf)
    # Goal: Normalize A to get ||I - A_scaled|| < 1
    mat_abs = mat.abs()
    row_sum = mat_abs.sum(dim=-1)  # (..., n)
    norm_inf = row_sum.max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # (..., 1, 1)
    # Protection against division by zero
    norm_inf = torch.clamp(norm_inf, min=1e-6)

    mat_scaled = mat / norm_inf

    # Initial guess: A_scaled^T (classic NS start)
    X = mat_scaled.transpose(-1, -2)

    # Newton-Schulz iteration: X_{k+1} = X_k (2I - A_scaled X_k)
    for _ in range(num_iters):
        AX = torch.matmul(mat_scaled, X)
        X = torch.matmul(X, (2.0 * I - AX))

    # True inverse: A^{-1} ~= X / norm_inf (A = norm_inf * mat_scaled)
    inv_mat = X / norm_inf

    return inv_mat.to(orig_dtype)


class UnitaryQINN(nn.Module):
    """
    Quantum-Inspired Unitary Layer - Orthogonal transformation via Cayley Transform.

    Definition:
        S = A - A^T           (skew-symmetric)
        U = (I - S)^{-1} (I + S)  ~= orthogonal/unitary

    Usage:
        x: (B, T, D) -> out = x @ U^T

    Note:
    - If cfg.use_qinn is False, the layer is fully bypassed
    - A is initialized with a small gaussian (for stability)
    """

    def __init__(self, dim: int, num_iters: int = 6, use_qinn: Optional[bool] = None) -> None:
        """
        UnitaryQINN initializer.

        Args:
            dim (int): Dimension
            num_iters (int): Newton-Schulz iteration count
            use_qinn (Optional[bool]): [21] Per-instance enable override. When None (default)
                the layer reads the global ``cfg.use_qinn``; pass True/False to make the
                layer self-contained (no implicit global-config dependency).
        """
        super().__init__()
        self.dim = dim
        self.num_iters = num_iters
        self._use_qinn_override = use_qinn

        # Small initialization: very small gaussian for stability
        self.A = nn.Parameter(torch.randn(dim, dim) * 1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - Applies unitary transformation (controlled by cfg.use_qinn).

        Args:
            x (torch.Tensor): Input tensor [Batch, Seq, Dim]
        Returns:
            torch.Tensor: Transformed tensor
        """
        # If QINN is disabled, return input as-is. Per-instance override wins over global cfg.
        enabled = self._use_qinn_override if self._use_qinn_override is not None else getattr(cfg, "use_qinn", False)
        if not enabled:
            return x

        orig_dtype = x.dtype
        x = x.float()

        # Skew-symmetric matrix: S = A - A^T
        S = self.A - self.A.t()  # (D, D)

        # Cayley matrices
        I = torch.eye(self.dim, device=S.device, dtype=S.dtype)
        M = I - S  # (I - S)
        P = I + S  # (I + S)

        # Calculate (I - S)^{-1} via Newton-Schulz
        M_inv = newton_schulz_inverse(M, num_iters=self.num_iters)  # (D, D)

        # U = (I - S)^{-1} (I + S)
        U = torch.matmul(M_inv, P)  # (D, D)

        # Safety: If NaN/Inf occurs rarely, fallback -> approach I
        if not torch.isfinite(U).all():
            U = torch.where(torch.isfinite(U), U, I)

        # x: (B, T, D) -> out = x @ U^T
        out = torch.matmul(x, U.t())

        return out.to(orig_dtype)
