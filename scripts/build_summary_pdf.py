#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent.parent

FONT_REGULAR = "MertFormerSans"
FONT_BOLD = "MertFormerSansBold"

FONT_REGULAR_CANDIDATES = [
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
]
FONT_BOLD_CANDIDATES = [
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
]


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def register_fonts() -> tuple[str, str]:
    regular = _first_existing(FONT_REGULAR_CANDIDATES)
    bold = _first_existing(FONT_BOLD_CANDIDATES)
    if regular is None or bold is None:
        return "Helvetica", "Helvetica-Bold"

    pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(regular)))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(bold)))
    return FONT_REGULAR, FONT_BOLD


def normalize_markdown_line(line: str) -> str:
    # Convert markdown links to readable text in PDF.
    line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", line)
    # Strip inline code ticks for clean rendering.
    line = line.replace("`", "")
    return line


def extract_header_image(md_text: str) -> tuple[str | None, list[str]]:
    lines = md_text.splitlines()
    if not lines:
        return None, lines

    m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", lines[0].strip())
    if not m:
        return None, lines

    image_path = m.group(1).strip()
    return image_path, lines[1:]


def wrap_text(text: str, font: str, size: int, max_width: float) -> list[str]:
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    cur = words[0]
    for word in words[1:]:
        candidate = f"{cur} {word}"
        if pdfmetrics.stringWidth(candidate, font, size) <= max_width:
            cur = candidate
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def render_markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    header_image_rel, lines = extract_header_image(text)

    regular_font, bold_font = register_fonts()

    page_w, page_h = A4
    margin_x = 52
    margin_top = 46
    margin_bottom = 44
    max_width = page_w - (margin_x * 2)

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    y = page_h - margin_top

    if header_image_rel:
        img_path = (md_path.parent / header_image_rel).resolve()
        if img_path.exists():
            img = ImageReader(str(img_path))
            iw, ih = img.getSize()
            target_w = max_width
            target_h = target_w * (ih / iw)
            if target_h > 150:
                target_h = 150
                target_w = target_h * (iw / ih)
            c.drawImage(
                str(img_path),
                margin_x + ((max_width - target_w) / 2),
                y - target_h,
                width=target_w,
                height=target_h,
                preserveAspectRatio=True,
                mask="auto",
            )
            y -= target_h + 16

    in_code = False
    in_table = False
    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if line.strip().startswith("```"):
            in_code = not in_code
            continue

        if re.match(r"^\s*\|.*\|\s*$", line):
            in_table = True
        elif in_table and not line.strip():
            in_table = False

        if in_table and re.match(r"^\s*\|[\s:-]+\|\s*$", line):
            continue

        if line.strip() == "---":
            y -= 8
            c.setLineWidth(0.6)
            c.line(margin_x, y, page_w - margin_x, y)
            y -= 12
            continue

        if not line.strip():
            y -= 8
            continue

        text_line = normalize_markdown_line(line)

        if in_code:
            font = "Courier"
            size = 9
            indent = 10
        elif text_line.startswith("# "):
            font = bold_font
            size = 16
            text_line = text_line[2:].strip()
            indent = 0
        elif text_line.startswith("## "):
            font = bold_font
            size = 13
            text_line = text_line[3:].strip()
            indent = 0
        elif text_line.startswith("### "):
            font = bold_font
            size = 11
            text_line = text_line[4:].strip()
            indent = 0
        elif re.match(r"^\s*[-*]\s+", text_line):
            font = regular_font
            size = 10
            text_line = re.sub(r"^\s*[-*]\s+", "• ", text_line)
            indent = 12
        elif re.match(r"^\s*\d+\.\s+", text_line):
            font = regular_font
            size = 10
            indent = 10
        else:
            font = regular_font
            size = 10
            indent = 0

        wrapped = wrap_text(text_line, font, size, max_width - indent)
        line_h = int(size * 1.45)
        for wline in wrapped:
            if y < margin_bottom + line_h:
                c.showPage()
                y = page_h - margin_top
                c.setFont(regular_font, 10)
            c.setFont(font, size)
            c.drawString(margin_x + indent, y, wline)
            y -= line_h

    c.save()


def main() -> int:
    pairs = [
        (ROOT / "README_SUMMARY.md", ROOT / "README_SUMMARY.pdf"),
        (ROOT / "README_SUMMARY_TR.md", ROOT / "README_SUMMARY_TR.pdf"),
    ]
    for md_path, pdf_path in pairs:
        render_markdown_to_pdf(md_path, pdf_path)
        print(f"built: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
