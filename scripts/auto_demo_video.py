"""Optional demo video automation from demo script + logs.

Creates simple slide images and stitches into a video if ffmpeg is available.
This is opt-in and never required for training.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _load_sections(script_path: Path) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "Demo"
    current_lines: list[str] = []

    for raw in script_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line[3:].strip()
            current_lines = []
            continue
        if line.startswith("- "):
            current_lines.append(line[2:].strip())
        elif line:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    return sections


def _ensure_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
        return Image, ImageDraw, ImageFont
    except Exception:
        return None


def _render_slide(image_path: Path, title: str, bullets: list[str], size=(1280, 720)) -> None:
    Image, ImageDraw, ImageFont = _ensure_pillow()
    if Image is None:
        raise RuntimeError("Pillow not installed. Install with: pip install pillow")

    img = Image.new("RGB", size, color=(10, 14, 20))
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

    draw.text((60, 50), title, fill=(230, 230, 230), font=title_font)
    y = 120
    for bullet in bullets[:8]:
        draw.text((80, y), f"• {bullet}", fill=(200, 200, 200), font=body_font)
        y += 30

    img.save(image_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="reports/demo_video_script.md")
    parser.add_argument("--out-dir", default="reports/demo_video_auto")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds-per-slide", type=int, default=5)
    parser.add_argument("--resolution", default="1280x720")
    args = parser.parse_args()

    script_path = Path(args.script)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")

    sections = _load_sections(script_path)
    if not sections:
        raise SystemExit("No sections detected in demo script.")

    width, height = [int(x) for x in args.resolution.split("x")]
    slide_dir = out_dir / "slides"
    slide_dir.mkdir(parents=True, exist_ok=True)

    for idx, (title, bullets) in enumerate(sections, start=1):
        image_path = slide_dir / f"slide_{idx:02d}.png"
        _render_slide(image_path, title, bullets, size=(width, height))

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found. Slides generated at:", slide_dir)
        print("Install ffmpeg to auto-generate video.")
        return

    video_path = out_dir / "demo_video.mp4"
    seconds = int(args.seconds_per_slide)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-pattern_type",
            "glob",
            "-i",
            str(slide_dir / "slide_*.png"),
            "-c:v",
            "libx264",
            "-r",
            str(args.fps),
            "-pix_fmt",
            "yuv420p",
            "-t",
            str(seconds * len(sections)),
            str(video_path),
        ],
        check=False,
    )
    print("Video saved:", video_path)


if __name__ == "__main__":
    main()
