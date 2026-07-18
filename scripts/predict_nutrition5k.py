#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-photo inference for the Nutrition5k vision side-experiment.

Loads the trained checkpoint produced by train_nutrition5k.py and predicts
calories/mass/fat/carb/protein from one overhead-ish photo of a plate of food.

Usage:
    python predict_nutrition5k.py <path-to-photo.jpg>   -- predict this photo
    python predict_nutrition5k.py                       -- no path given: opens
                                                             a native file-picker
                                                             (double-click friendly;
                                                             Windows/macOS/Linux via
                                                             tkinter, stdlib only)

Must be run from the same folder as train_nutrition5k.py -- this script
imports it as a library (repo/vendor discovery, config shaping, model
architecture, and the exact same image preprocessing used during training),
so predictions are computed the same way the model was actually trained,
not a reimplementation that could silently drift from it.

Works both (a) right where training just ran (checkpoint under
nutrition5k_work/checkpoints/) and (b) from a delivered output ZIP extracted
on a completely different machine (checkpoint under ./checkpoints/, as
packaged by train_nutrition5k.py's build_zip()) -- checks both locations.

Not a claim of accuracy: this is the small, from-scratch, bounded side
experiment described in train_nutrition5k.py's own module docstring, not
the paper's much larger pretrained model. Treat the numbers as "what did
this experiment's model learn", not a nutrition-tracking tool.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import train_nutrition5k as trainer  # noqa: E402


def find_checkpoint() -> Optional[Path]:
    """Live-training location first, then the delivered-ZIP layout."""
    candidates = [
        trainer.CKPT_DIR / "nutrition5k_best.pt",
        SCRIPT_DIR / "checkpoints" / "nutrition5k_best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def pick_photo_via_dialog() -> Optional[Path]:
    """Native OS file-picker (stdlib tkinter, no extra dependency). Returns
    None if the user cancels or if no display/tkinter is available (headless
    machine, minimal Python build missing the optional _tkinter module)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Bir yemek fotoğrafı seç",
            filetypes=[("Görsel", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tüm dosyalar", "*.*")],
        )
        root.destroy()
    except Exception:
        return None
    return Path(chosen).resolve() if chosen else None


def main() -> int:
    if len(sys.argv) > 2:
        print("Usage: python predict_nutrition5k.py [path-to-photo.jpg]")
        return 2

    if len(sys.argv) == 2:
        photo_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        print("Fotoğraf yolu verilmedi -- dosya seçme penceresi açılıyor...")
        photo_path = pick_photo_via_dialog()
        if photo_path is None:
            print(
                "Dosya seçilmedi (iptal edildi veya bu ortamda pencere açılamıyor).\n"
                "Alternatif: python predict_nutrition5k.py <path-to-photo.jpg>"
            )
            return 2

    if not photo_path.exists():
        print(f"ERROR: file not found: {photo_path}")
        return 2

    ckpt_path = find_checkpoint()
    if ckpt_path is None:
        print(
            "ERROR: no trained checkpoint found in either "
            f"{trainer.CKPT_DIR / 'nutrition5k_best.pt'} or "
            f"{SCRIPT_DIR / 'checkpoints' / 'nutrition5k_best.pt'}. "
            "Run train_nutrition5k.py first (at least one epoch must complete), "
            "or make sure you extracted the full delivered output ZIP."
        )
        return 2

    trainer.bootstrap_dependencies()
    trainer.discover_or_vendor_repo()
    trainer.phase_config()

    import torch  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = trainer.build_model()
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    dummy_label = {t: 0.0 for t in trainer.TARGETS}
    ds = trainer.Nutrition5kDataset([("photo", str(photo_path), dummy_label)], trainer.IMAGE_SIZE)
    image_tensor, _ = ds[0]
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds, _ = model(image_tensor)

    print(f"\nTahmin ({photo_path.name}):")
    print(f"  Kalori  : {preds['calories'].item():.0f} kcal")
    print(f"  Kütle   : {preds['mass'].item():.0f} g")
    print(f"  Yağ     : {preds['fat'].item():.1f} g")
    print(f"  Karb    : {preds['carb'].item():.1f} g")
    print(f"  Protein : {preds['protein'].item():.1f} g")
    print(
        f"\n(checkpoint: {ckpt_path}, epoch {ckpt.get('epoch', '?')}, kaydedildiği "
        f"andaki val kalori MAE: {ckpt.get('best_val_calorie_mae', float('nan')):.1f} kcal)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
