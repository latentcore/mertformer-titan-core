"""ChessFormer: a chess policy/value network on the MertFormer Build30 trunk.

This is a trimmed subset of the ``chessformer`` package, vendored here so
``retroactive_eval.py`` (the script that produced this folder's
holdout/puzzle/elo reports) actually runs from a clone of this repo, not just
in the original, non-versioned ``ChessFormerAI`` folder it was developed in.
`chessformer` itself is an independent side-project package -- see
``DECISIONS.md``'s 2026-08-06 entry ("stays separate, not folded back") --
this is a reproducibility copy of its inference+eval code, not an adoption of
the package into this repo's own training pipeline or test suite. The full
package (including the training loop and GUI) lives at
https://github.com/latentcore/mertformer-titan-core (ChessFormerAI side) and
https://huggingface.co/Mert21779033/mertformer-chess-searchless (inference-only
copy, no ``eval``/``data``, for the "play it locally" use case).

Layout (this subset)
---------------------
``arch/``        mirrored canonical layers (see arch/__init__ and the parity tests)
``board.py``     tokenization, 4208-move UCI vocabulary, value binning
``config.py``    ``ModelConfig``/``RunConfig`` dataclasses and size presets
``model.py``     trunk + factorized policy head + HL-Gauss value head
``inference.py`` ``select_move`` -- the single-forward-pass move selector
``runtime.py``   device selection, precision policy, environment capture
``eval/``        holdout, Lichess puzzles, Stockfish UCI_Elo ladder -- what
                  ``retroactive_eval.py`` actually calls
``data/``        ``PackedChessDataset`` + packing/download code that
                  ``eval/holdout.py`` depends on to read the packed shards

Not included (not needed to reproduce these reports; see the full GitHub
repo): ``train.py`` (training loop), ``profiling.py``, ``pipeline.py``
(orchestration), ``gui/`` (local web dashboard). Running the eval stages for
real also requires the actual data on disk (packed shards, puzzle DB,
Stockfish binary) -- see ``dataset_provenance.json`` for exact sources; this
folder does not bundle multi-GB datasets, only the code and the results
already produced from them.
"""

__version__ = "1.0.0"

from .board import (  # noqa: F401
    ID_TO_MOVE,
    MOVE_TO_ID,
    MOVE_VOCAB,
    MOVE_VOCAB_HASH,
    VOCAB_SIZE,
)

__all__ = [
    "__version__",
    "MOVE_VOCAB",
    "MOVE_TO_ID",
    "ID_TO_MOVE",
    "VOCAB_SIZE",
    "MOVE_VOCAB_HASH",
]
