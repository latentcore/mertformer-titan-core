#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_script(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(
        "\n".join(
            [
                "# Demo Script",
                "",
                "## Goals",
                "- Show start gates",
                "- Show kernel fallback matrix",
                "- Show release artifact checks",
                "",
                "## Sequence",
                "- Start gate outputs",
                "- Hardening reports",
                "- Demo checksum verification",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_ffmpeg(video_path: Path, with_drawtext: bool) -> subprocess.CompletedProcess[str]:
    ffmpeg_bin = subprocess.run(["bash", "-lc", "command -v ffmpeg"], capture_output=True, text=True)
    if ffmpeg_bin.returncode != 0 or not ffmpeg_bin.stdout.strip():
        return subprocess.CompletedProcess(args=["ffmpeg"], returncode=127, stdout="", stderr="ffmpeg_not_found")

    cmd = [
        ffmpeg_bin.stdout.strip(),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=0x101820:s=1280x720:d=8",
    ]
    if with_drawtext:
        cmd += [
            "-vf",
            "drawtext=text='MertFormer Titan Demo v1':x=(w-text_w)/2:y=(h-text_h)/2:fontsize=44:fontcolor=white",
        ]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path)]
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    artifacts = ROOT / "artifacts"
    reports = ROOT / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    script_path = reports / "demo_script.md"
    notes_path = reports / "demo_notes.md"
    video_path = artifacts / "demo_v1.mp4"
    checksum_path = reports / "demo_checksum.sha256"
    validation_path = reports / "demo_validation_report.json"

    ensure_script(script_path)

    p = _run_ffmpeg(video_path=video_path, with_drawtext=True)
    used_fallback = False
    if p.returncode != 0:
        # Fallback for ffmpeg builds without drawtext filter.
        p = _run_ffmpeg(video_path=video_path, with_drawtext=False)
        used_fallback = True

    if p.returncode != 0:
        validation_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "reason": "ffmpeg_failed",
                    "stderr_tail": p.stderr[-2000:],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 1

    digest = sha256(video_path)
    checksum_path.write_text(f"{digest}  {video_path.name}\n", encoding="utf-8")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    notes_path.write_text(
        "\n".join(
            [
                "# Demo Notes",
                "",
                f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
                f"Video: {video_path.relative_to(ROOT)}",
                f"Checksum: {digest}",
                f"Commit: {commit}",
                f"Generator mode: {'ffmpeg_color_fallback' if used_fallback else 'ffmpeg_drawtext'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    validation_path.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "ok": True,
                "used_fallback": used_fallback,
                "video_path": str(video_path.relative_to(ROOT)),
                "checksum_path": str(checksum_path.relative_to(ROOT)),
                "notes_path": str(notes_path.relative_to(ROOT)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"ok": True, "video": str(video_path.relative_to(ROOT)), "used_fallback": used_fallback}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
