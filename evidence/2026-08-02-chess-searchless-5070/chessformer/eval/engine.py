"""Stockfish discovery, optional download and a strength-limited opponent.

The opponent is configured through Stockfish's own ``UCI_LimitStrength`` /
``UCI_Elo`` options rather than ``Skill Level``.

Why that matters: ``Skill Level`` is an internal 0-20 knob with no calibrated
rating attached to it. ``scripts/chess_5080_onefile.py`` built its whole ladder
on ``Skill Level`` and then attached hand-written ``anchor_elo_proxy`` numbers
to each level -- its own code comments admit these are "not calibrated against
an external rating pool". ``UCI_Elo`` is Stockfish's published strength-limiting
interface with a documented rating scale, so an Elo computed against it is at
least anchored to something the engine authors defined.

Downloads never happen implicitly: ``ensure_stockfish`` refuses unless the
caller passes ``allow_download=True``, which the GUI only sets after the
operator agrees.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess
import chess.engine

STOCKFISH_RELEASE_API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
STOCKFISH_HOME = "https://stockfishchess.org/download/"

# Stockfish clamps UCI_Elo to this range; going outside it silently saturates,
# which would corrupt an Elo estimate.
UCI_ELO_MIN = 1320
UCI_ELO_MAX = 3190


def find_stockfish(extra_paths: Optional[List[Path]] = None) -> Optional[Path]:
    """Look for a usable Stockfish binary without touching the network."""
    names = ["stockfish", "stockfish.exe"]
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)

    candidates: List[Path] = list(extra_paths or [])
    candidates += [
        Path.home() / "Desktop" / "stockfish",
        Path.home() / "Downloads" / "stockfish",
        Path.cwd() / "tools" / "stockfish",
    ]
    for root in candidates:
        if not root.exists():
            continue
        if root.is_file():
            return root
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            lowered = path.name.lower()
            if "stockfish" not in lowered:
                continue
            if path.suffix.lower() in {".zip", ".tar", ".gz", ".txt", ".md", ".json"}:
                continue
            if platform.system() == "Windows" and path.suffix.lower() != ".exe":
                continue
            return path
    return None


def _asset_score(name: str) -> int:
    lowered = name.lower()
    if any(bad in lowered for bad in ("source", "src", "android", "wasm", "armv7", "riscv")):
        return -1000
    system = platform.system()
    score = 0
    if system == "Windows":
        if "windows" in lowered or lowered.endswith(".exe") or "win" in lowered:
            score += 30
        if lowered.endswith(".zip"):
            score += 10
    elif system == "Linux":
        if "ubuntu" in lowered or "linux" in lowered:
            score += 30
        if lowered.endswith((".tar", ".zip")):
            score += 10
    elif system == "Darwin":
        if any(tag in lowered for tag in ("macos", "mac", "osx")):
            score += 30
        if lowered.endswith((".tar", ".zip")):
            score += 10
    # Prefer widely-compatible builds over ones needing very new CPU features.
    if "avx2" in lowered or "modern" in lowered:
        score += 6
    if "sse41" in lowered or "popcnt" in lowered:
        score += 4
    if "avx512" in lowered or "vnni" in lowered:
        score -= 4
    if "x86-64" in lowered or "x86_64" in lowered:
        score += 3
    return score


def _resolved_member_path(base: Path, member_name: str) -> Path:
    """Resolve an archive member's target path and reject any that escape ``base``.

    Guards against path traversal ("zip slip" / "tar slip") -- a malicious or
    corrupted archive entry (``../../etc/passwd``, an absolute path, or a
    symlink target outside ``base``) must never be allowed to write outside
    the intended extraction directory.
    """
    base = base.resolve()
    target = (base / member_name).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise RuntimeError(f"unsafe path in archive, escapes extraction dir: {member_name!r}")
    return target


def _safe_extract_tar(tf: "tarfile.TarFile", path: Path) -> None:
    base = Path(path).resolve()
    for member in tf.getmembers():
        _resolved_member_path(base, member.name)
        if member.issym() or member.islnk():
            # Reject symlinks outright -- their target is a separate string
            # that itself needs the same containment check, and Stockfish
            # release archives have no legitimate reason to contain one.
            raise RuntimeError(f"unsafe symlink member in archive: {member.name!r}")
    if sys.version_info >= (3, 12):
        tf.extractall(base, filter="data")
    else:
        # Every member's resolved path was already checked against `base`
        # above (path traversal, absolute paths) and symlinks were rejected
        # outright -- bandit's B202 can't see that static analysis, hence the
        # explicit suppression rather than a false "unvalidated" flag.
        tf.extractall(base)  # nosec B202 -- members validated above


def _safe_extract_zip(zf: "zipfile.ZipFile", path: Path) -> None:
    base = Path(path).resolve()
    for info in zf.infolist():
        _resolved_member_path(base, info.filename)
    # Same reasoning as _safe_extract_tar: every member path was validated
    # against `base` immediately above this call.
    zf.extractall(base)  # nosec B202 -- members validated above


def ensure_stockfish(
    cache_dir: Path,
    *,
    allow_download: bool = False,
    explicit_path: str = "",
    timeout: int = 180,
) -> Dict[str, Any]:
    """Return a usable Stockfish path, downloading only with explicit consent."""
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists():
            return {"status": "explicit", "path": str(path), "downloaded": False}
        return {"status": "not_found", "reason": f"explicit path does not exist: {path}"}

    local = find_stockfish([Path(cache_dir)])
    if local is not None:
        return {"status": "found_locally", "path": str(local), "downloaded": False}

    if not allow_download:
        return {
            "status": "not_found",
            "reason": "no local Stockfish and download consent was not given",
            "how_to_fix": (
                f"install Stockfish from {STOCKFISH_HOME} and put it on PATH, "
                "or enable the download option in the GUI"
            ),
        }

    import json
    import urllib.request

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "ChessFormerAI/1.0"}
    try:
        req = urllib.request.Request(STOCKFISH_RELEASE_API, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            release = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "error", "reason": f"release metadata fetch failed: {type(exc).__name__}: {exc}"}

    assets = [a for a in release.get("assets", []) if isinstance(a, dict)]
    if not assets:
        return {"status": "error", "reason": "release contained no assets"}
    best = max(assets, key=lambda a: _asset_score(str(a.get("name", ""))))
    if _asset_score(str(best.get("name", ""))) <= 0:
        return {"status": "error", "reason": f"no suitable asset for {platform.system()}"}

    name = str(best["name"])
    url = str(best.get("browser_download_url", ""))
    archive = cache_dir / name
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp, archive.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    except Exception as exc:
        return {"status": "error", "reason": f"download failed: {type(exc).__name__}: {exc}"}

    extract_dir = cache_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                _safe_extract_zip(zf, extract_dir)
        elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz")):
            with tarfile.open(archive) as tf:
                _safe_extract_tar(tf, extract_dir)
        else:
            shutil.copy2(archive, extract_dir / name)
    except Exception as exc:
        return {"status": "error", "reason": f"extract failed: {type(exc).__name__}: {exc}"}

    binary = find_stockfish([extract_dir])
    if binary is None:
        return {"status": "error", "reason": f"no binary found inside {name}"}
    if platform.system() != "Windows":
        with_suppress_chmod(binary)

    return {
        "status": "downloaded",
        "path": str(binary),
        "downloaded": True,
        "release_tag": release.get("tag_name", ""),
        "asset": name,
        "source": url,
    }


def with_suppress_chmod(path: Path) -> None:
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass


class StockfishOpponent:
    """A strength-limited Stockfish, used as a rating anchor."""

    def __init__(
        self,
        path: Path,
        *,
        uci_elo: int = 1500,
        movetime_ms: int = 100,
        threads: int = 1,
        hash_mb: int = 64,
    ) -> None:
        self.path = Path(path)
        self.movetime_ms = int(movetime_ms)
        self.threads = int(threads)
        self.hash_mb = int(hash_mb)
        self.requested_elo = int(uci_elo)
        self.effective_elo = int(min(UCI_ELO_MAX, max(UCI_ELO_MIN, uci_elo)))
        self.clamped = self.effective_elo != self.requested_elo
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def __enter__(self) -> "StockfishOpponent":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def open(self) -> None:
        if self._engine is not None:
            return
        self._engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
        self._engine.configure({
            "Threads": self.threads,
            "Hash": self.hash_mb,
            "UCI_LimitStrength": True,
            "UCI_Elo": self.effective_elo,
        })

    def set_elo(self, uci_elo: int) -> int:
        self.requested_elo = int(uci_elo)
        self.effective_elo = int(min(UCI_ELO_MAX, max(UCI_ELO_MIN, uci_elo)))
        self.clamped = self.effective_elo != self.requested_elo
        if self._engine is not None:
            self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.effective_elo})
        return self.effective_elo

    def move(self, board: chess.Board) -> Optional[str]:
        if self._engine is None:
            self.open()
        assert self._engine is not None
        try:
            result = self._engine.play(
                board, chess.engine.Limit(time=self.movetime_ms / 1000.0)
            )
        except chess.engine.EngineError:
            return None
        return result.move.uci() if result.move else None

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def describe(self) -> Dict[str, Any]:
        return {
            "engine": "stockfish",
            "path": str(self.path),
            "uci_limit_strength": True,
            "uci_elo_requested": self.requested_elo,
            "uci_elo_effective": self.effective_elo,
            "uci_elo_clamped": self.clamped,
            "uci_elo_range": [UCI_ELO_MIN, UCI_ELO_MAX],
            "movetime_ms": self.movetime_ms,
            "threads": self.threads,
            "hash_mb": self.hash_mb,
        }


def engine_version(path: Path) -> str:
    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(path))
        try:
            return str(engine.id.get("name", "unknown"))
        finally:
            engine.quit()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"
