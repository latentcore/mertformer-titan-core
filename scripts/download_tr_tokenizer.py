"""Download and cache the Turkish tokenizer (opt-in)."""
from __future__ import annotations

from pathlib import Path


def main() -> None:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise SystemExit(f"transformers not available: {exc}")

    model_id = "dbmdz/bert-base-turkish-128k-uncased"
    out_dir = Path("tokenizer/tr")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.save_pretrained(out_dir)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
