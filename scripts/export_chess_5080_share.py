#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
import marshal
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "scripts" / "chess_5080_onefile.py"
DESKTOP = Path.home() / "Desktop"
PREFIX = "MertFormer_Chess_5080_Delivery"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_obfuscated_wrapper(source_text: str, source_name: str) -> str:
    code_obj = compile(source_text, source_name, "exec")
    payload = base64.b85encode(lzma.compress(marshal.dumps(code_obj), preset=9)).decode("ascii")
    return f'''#!/usr/bin/env python3
"""Obfuscated share wrapper for {source_name}."""
from __future__ import annotations
import base64
import lzma
import marshal
import os

os.environ.setdefault("MERTFORMER_CHESS_SHARE_MODE", "1")
os.environ.setdefault("MERTFORMER_CHESS_SELF_DELETE", "1")
PAYLOAD = {payload!r}
exec(marshal.loads(lzma.decompress(base64.b85decode(PAYLOAD.encode("ascii")))), {{"__name__": "__main__", "__file__": __file__}})
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Export open + obfuscated RTX 5080 chess share bundle")
    parser.add_argument("--out-dir", help="Optional destination directory")
    args = parser.parse_args()

    if not SOURCE.exists():
        raise SystemExit(f"Source script missing: {SOURCE}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = Path(args.out_dir).expanduser() if args.out_dir else (DESKTOP / f"{PREFIX}_{stamp}")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    open_copy = bundle_dir / SOURCE.name
    share_copy = bundle_dir / "mertformer_chess_5080_share.py"
    readme = bundle_dir / "README_SHARE.md"
    manifest = bundle_dir / "share_manifest.json"
    zip_path = DESKTOP / f"{PREFIX}_{stamp}.zip"
    sha_path = DESKTOP / f"{PREFIX}_{stamp}.zip.sha256"

    source_text = SOURCE.read_text(encoding="utf-8")
    open_copy.write_text(source_text, encoding="utf-8")
    share_copy.write_text(build_obfuscated_wrapper(source_text, SOURCE.name), encoding="utf-8")
    readme.write_text(
        "# RTX 5080 Chess Share Bundle\n\n"
        "- `mertformer_chess_5080_onefile.py`: canonical readable copy\n"
        "- `mertformer_chess_5080_share.py`: obfuscated share wrapper\n"
        "- Run the share wrapper on the Windows RTX 5080 machine with PyCharm Run\n"
        "- After a successful run, the share wrapper may self-delete while outputs are zipped to Desktop\n",
        encoding="utf-8",
    )

    file_rows = []
    for path in (open_copy, share_copy, readme):
        file_rows.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    manifest.write_text(json.dumps({"generated_at": stamp, "files": file_rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle_dir.iterdir()):
            if path.is_dir():
                continue
            zf.write(path, arcname=path.name)
    sha_path.write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n", encoding="utf-8")

    print(json.dumps({
        "bundle_dir": str(bundle_dir),
        "zip_path": str(zip_path),
        "sha256_path": str(sha_path),
        "source": str(SOURCE),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
