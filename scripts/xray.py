"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - PROJECT AUDITOR (X-RAY)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Licensed under MIT License.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v27.0-FINAL (Locked & Sealed)
Status : PRODUCTION READY (LOCKED)
==============================================================================
"""

__version__ = "27.0-FINAL"
__author__ = "Mert"

import os
import sys
import time
import stat
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------------
# AYARLAR
# -------------------------------------------------------------------------

# HEDEF KONUM SEÇİMİ (Önce yeniye bakar, bulamazsa eskiye döner)
_PRIMARY_PATH = Path("/Applications/NİHAİ")
_FALLBACK_PATH = Path("/Users/mertyunlu/Downloads/NİHAİ")

if _PRIMARY_PATH.exists():
    ROOT_DIR = _PRIMARY_PATH
else:
    ROOT_DIR = _FALLBACK_PATH

DOWNLOADS_DIR = Path.home() / "Downloads"
BASE_FILENAME = "MertFormer_Smart_Dump"
TIMEOUT_SECONDS = 10

# BU UZANTILARIN SADECE İSMİ YAZILIR, İÇERİĞİ OKUNMAZ (Gereksizler Listesi)
SKIP_CONTENT_EXTENSIONS = {
    # Python Cache / Sistem
    ".pyc", ".pyo", ".pyd", ".DS_Store",
    # Resim / Medya
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".mp3", ".wav", ".mp4",
    # Sıkıştırılmış / Binary Veri
    ".zip", ".tar", ".gz", ".rar", ".7z", ".pdf", ".exe", ".dll", ".so",
    ".db", ".sqlite", ".bin", ".pkl", ".pt", ".pth", ".ckpt"
}


# -------------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# -------------------------------------------------------------------------

def timed_input(prompt, timeout):
    import select
    print(prompt, end='', flush=True)
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if r:
        return sys.stdin.readline().strip().lower()
    else:
        return None


def get_unique_filepath(directory: Path, base_name: str) -> Path:
    candidate = directory / f"{base_name}.txt"
    if not candidate.exists(): return candidate
    counter = 1
    while True:
        candidate = directory / f"{base_name}-{counter}.txt"
        if not candidate.exists(): return candidate
        counter += 1


def make_file_readonly(filepath: Path):
    os.chmod(filepath, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def log(message: str, file_handle=None):
    print(message)
    if file_handle: file_handle.write(message + "\n")


def is_text_file(path: Path) -> bool:
    """Dosyanın metin mi binary mi olduğunu kontrol eder."""
    # 1. Uzantı kontrolü (Hızlı eleme)
    if path.suffix.lower() in SKIP_CONTENT_EXTENSIONS:
        return False

    # 2. İçerik kontrolü (Kesin sonuç)
    try:
        with path.open("rb") as f:
            chunk = f.read(1024)
            return b"\0" not in chunk  # Null byte yoksa metindir
    except Exception:
        return False


# -------------------------------------------------------------------------
# TARAMA MOTORU
# -------------------------------------------------------------------------

def write_tree(path: Path, file_handle, prefix: str = ""):
    if not path.exists():
        log(f"HATA: {path} bulunamadı!", file_handle)
        return

    try:
        # Çıktı dosyasının kendisi hariç her şeyi listele
        entries = sorted(
            [e for e in path.iterdir() if e.name != file_handle.name],
            key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except PermissionError:
        log(prefix + "└── [ERİŞİM REDDEDİLDİ]", file_handle)
        return

    for idx, entry in enumerate(entries):
        connector = "└── " if idx == len(entries) - 1 else "├── "
        new_prefix = prefix + ("    " if idx == len(entries) - 1 else "│   ")

        if entry.is_dir():
            log(f"{prefix}{connector}📁 {entry.name}/", file_handle)
            write_tree(entry, file_handle, new_prefix)
        else:
            log(f"{prefix}{connector}📄 {entry.name}", file_handle)

            # --- İÇERİK KONTROLÜ ---
            try:
                # Eğer dosya metin ise (py, xml, txt, md...)
                if is_text_file(entry):
                    log("", file_handle)
                    log(f"{new_prefix}    " + "=" * 40, file_handle)
                    log(f"{new_prefix}    START: {entry.name}", file_handle)
                    log(f"{new_prefix}    " + "-" * 40, file_handle)

                    with entry.open("r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                        if not lines:
                            log(f"{new_prefix}    [DOSYA BOŞ]", file_handle)
                        for line in lines:
                            clean_line = line.rstrip('\n')
                            log(f"{new_prefix}    | {clean_line}", file_handle)

                    log(f"{new_prefix}    " + "=" * 40, file_handle)
                    log("", file_handle)

                # Eğer gereksiz/binary dosya ise (.pyc, .DS_Store...)
                else:
                    file_size = entry.stat().st_size
                    log(f"{new_prefix}    [İÇERİK GİZLENDİ: Binary/Sistem Dosyası ({file_size} bytes)]", file_handle)

            except Exception as e:
                log(f"{new_prefix}    [HATA: Dosya okunamadı - {e}]", file_handle)


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("🧠 MERTFORMER SMART AUDITOR (V18.0)")
    print("-" * 60)
    print(f"Hedef: {ROOT_DIR}")
    print("MOD: Tüm dosya yapısını gösterir, sadece METİN içeriklerini okur.")
    print("     (.pyc, .DS_Store gibi gereksizlerin içeriği atlanır.)")

    choice = timed_input(f"👉 Başlatılsın mı? (y/n, {TIMEOUT_SECONDS}s): ", TIMEOUT_SECONDS)

    if choice != 'y':
        print("\nİptal edildi.")
        return

    try:
        real_output_path = get_unique_filepath(DOWNLOADS_DIR, BASE_FILENAME)
        print(f"\nDosya oluşturuluyor: {real_output_path}")

        with open(real_output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(f"PROJECT SMART DUMP: {ROOT_DIR}\n")
            file_handle.write(f"GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            file_handle.write(f"{'=' * 60}\n\n")

            write_tree(ROOT_DIR, file_handle)

        print("\n" + "=" * 70)
        print("✅ İŞLEM TAMAMLANDI!")

        make_file_readonly(real_output_path)
        print(f"📄 Dosya kaydedildi: {real_output_path.name}")
        print(f"🔒 GÜVENLİK: Dosya Salt-Okunur (Read-Only) olarak kilitlendi.")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ HATA: {e}")


if __name__ == "__main__":
    main()