#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ALWAYS_IGNORE_DIRS = {
    ".git",
}

VENV_DIRS = {
    ".titan-venv",
    ".lint-venv",
    ".venv",
    "venv",
    "env",
}

BASE_CACHE_DIRS = {
    "__pycache__",
}

TOOL_CACHE_DIRS = {
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}

FULL_CLEAN_DIRS = {
    ".cache",
    ".ipynb_checkpoints",
    ".tox",
    ".nox",
    ".hypothesis",
    ".pyre",
    ".pytype",
    ".benchmarks",
    ".eggs",
    ".vs",
    ".sass-cache",
    ".turbo",
}

BASE_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
}

BASE_FILE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}

FULL_CLEAN_FILE_NAMES = {
    ".coverage",
}


def _collect_targets(
    root: Path,
    include_tool_caches: bool,
    full_clean: bool,
    include_venv_caches: bool,
) -> tuple[list[Path], list[Path]]:
    dirs_to_remove: list[Path] = []
    files_to_remove: list[Path] = []

    cache_dirs = set(BASE_CACHE_DIRS)
    if include_tool_caches:
        cache_dirs.update(TOOL_CACHE_DIRS)
    if full_clean:
        cache_dirs.update(FULL_CLEAN_DIRS)

    file_suffixes = set(BASE_FILE_SUFFIXES)
    file_names = set(BASE_FILE_NAMES)
    if full_clean:
        file_names.update(FULL_CLEAN_FILE_NAMES)

    for current, dirs, files in os.walk(root, topdown=True):
        current_path = Path(current)

        # Keep .git pruned always; include venv trees only when explicitly requested.
        if include_venv_caches:
            dirs[:] = [d for d in dirs if d not in ALWAYS_IGNORE_DIRS]
        else:
            dirs[:] = [d for d in dirs if d not in ALWAYS_IGNORE_DIRS and d not in VENV_DIRS]

        for d in list(dirs):
            if d in cache_dirs:
                target = current_path / d
                dirs_to_remove.append(target)
                dirs.remove(d)

        for fname in files:
            fpath = current_path / fname
            if fname in file_names or fpath.suffix in file_suffixes:
                files_to_remove.append(fpath)

    # Deepest-first for safe recursive deletes.
    dirs_to_remove.sort(key=lambda x: len(x.parts), reverse=True)
    files_to_remove.sort()
    return dirs_to_remove, files_to_remove


def cleanup(
    root: Path,
    include_tool_caches: bool,
    full_clean: bool,
    include_venv_caches: bool,
    dry_run: bool,
    verbose: bool,
) -> dict[str, int]:
    dirs_to_remove, files_to_remove = _collect_targets(
        root,
        include_tool_caches=include_tool_caches,
        full_clean=full_clean,
        include_venv_caches=include_venv_caches,
    )

    if verbose:
        for p in files_to_remove:
            print(f"[cleanup:file] {p}")
        for p in dirs_to_remove:
            print(f"[cleanup:dir]  {p}")

    removed_dirs = 0
    removed_files = 0

    for f in files_to_remove:
        if dry_run:
            continue
        try:
            f.unlink(missing_ok=True)
            removed_files += 1
        except Exception:
            pass

    for d in dirs_to_remove:
        if dry_run:
            continue
        try:
            shutil.rmtree(d, ignore_errors=True)
            removed_dirs += 1
        except Exception:
            pass

    return {
        "dirs_found": len(dirs_to_remove),
        "files_found": len(files_to_remove),
        "dirs_removed": removed_dirs,
        "files_removed": removed_files,
    }


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run any command and always clean Python/build cache artifacts. "
            "Base mode cleans __pycache__/pyc; full mode also cleans common cache folders/files "
            "(.DS_Store, .cache, .ipynb_checkpoints, .tox, .nox, .hypothesis, .vs, etc.)."
        )
    )
    parser.add_argument("--root", default=".", help="Cleanup root directory (default: current directory)")
    parser.add_argument(
        "--include-tool-caches",
        action="store_true",
        help="Also remove .pytest_cache/.ruff_cache/.mypy_cache",
    )
    parser.add_argument(
        "--full-clean",
        action="store_true",
        help="Also remove broader cache artifacts (.DS_Store, .cache, .ipynb_checkpoints, .tox, .nox, .hypothesis, .vs, ...)",
    )
    parser.add_argument(
        "--include-venv-caches",
        action="store_true",
        help="Also scan venv folders (.titan-venv/.venv/venv/env) for cache cleanup",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print summary only; do not delete")
    parser.add_argument("--verbose", action="store_true", help="Print each discovered target")

    if "--" not in argv:
        parser.error("command is required; usage: run_and_clean_pycache.py [opts] -- <command> [args...]")

    sep = argv.index("--")
    ns = parser.parse_args(argv[:sep])
    cmd = argv[sep + 1 :]
    if not cmd:
        parser.error("missing command after --")
    return ns, cmd


def main(argv: list[str]) -> int:
    args, cmd = parse_args(argv)
    root = Path(args.root).resolve()

    print(f"[runner] cmd={' '.join(cmd)}")
    print(f"[runner] cleanup_root={root}")
    print(
        f"[runner] include_tool_caches={bool(args.include_tool_caches)} "
        f"full_clean={bool(args.full_clean)} "
        f"include_venv_caches={bool(args.include_venv_caches)} "
        f"dry_run={bool(args.dry_run)}"
    )

    rc = 1
    try:
        proc = subprocess.run(cmd)
        rc = int(proc.returncode)
    except KeyboardInterrupt:
        rc = 130
    except FileNotFoundError:
        print(f"[runner] command not found: {cmd[0]}", file=sys.stderr)
        rc = 127
    finally:
        report = cleanup(
            root,
            include_tool_caches=bool(args.include_tool_caches),
            full_clean=bool(args.full_clean),
            include_venv_caches=bool(args.include_venv_caches),
            dry_run=bool(args.dry_run),
            verbose=bool(args.verbose),
        )
        print(
            "[cleanup] "
            f"dirs_found={report['dirs_found']} files_found={report['files_found']} "
            f"dirs_removed={report['dirs_removed']} files_removed={report['files_removed']}"
            + (" (dry-run)" if args.dry_run else "")
        )

    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
