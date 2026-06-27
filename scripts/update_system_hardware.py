from __future__ import annotations

import datetime as _dt
import importlib.util as _importlib_util
import platform as _platform
import re as _re
import subprocess as _subprocess
import tempfile as _tempfile
from pathlib import Path


def _check_dependencies() -> bool:
    """Gercek kontrol: torch/transformers/accelerate import edilebiliyor mu?

    find_spec ile paketi yuklemeden varligini olcer; hicbiri yoksa False.
    """
    for _name in ("torch", "transformers", "accelerate"):
        try:
            if _importlib_util.find_spec(_name) is None:
                return False
        except Exception:
            return False
    return True


def _check_filesystem() -> bool:
    """Gercek kontrol: calisma dizinine gecici bir dosya yazip silmeyi dener."""
    try:
        with _tempfile.NamedTemporaryFile(
            dir=".", prefix=".titan_fs_check_", delete=True
        ) as _fh:
            _fh.write(b"ok")
            _fh.flush()
        return True
    except Exception:
        return False


def _check_tokenizer_cache() -> bool:
    """Gercek kontrol: tokenizer/tr dizini var mi?"""
    try:
        return Path("tokenizer/tr").exists()
    except Exception:
        return False


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def _run(cmd: list[str]) -> str:
    try:
        return _subprocess.check_output(cmd, text=True, stderr=_subprocess.DEVNULL)
    except Exception as _exc:  # noqa: BLE001 - best-effort probe; fallback stays
        import sys as _sys

        print(
            f"[warn] _run komutu basarisiz ({cmd[0] if cmd else '?'}): {_exc!r}",
            file=_sys.stderr,
        )
        return ""


def _parse_system_profiler() -> dict[str, str]:
    info: dict[str, str] = {}
    sp_hw = _run(["system_profiler", "SPHardwareDataType"])
    sp_gpu = _run(["system_profiler", "SPDisplaysDataType"])

    def _grab(pattern: str, text: str) -> str:
        m = _re.search(pattern, text)
        return m.group(1).strip() if m else ""

    info["device"] = _grab(r"Model Name:\s*(.+)", sp_hw) or "Mac"
    info["chip"] = _grab(r"Chip:\s*(.+)", sp_hw)
    info["cpu_cores"] = _grab(r"Total Number of Cores:\s*(.+)", sp_hw)
    info["memory"] = _grab(r"Memory:\s*(.+)", sp_hw)
    info["gpu"] = _grab(r"Chipset Model:\s*(.+)", sp_gpu)
    info["gpu_cores"] = _grab(r"Total Number of Cores:\s*(.+)", sp_gpu)
    info["os"] = _run(["sw_vers"]).strip().replace("\t", " ")
    return info


def _linux_info() -> dict[str, str]:
    info: dict[str, str] = {}
    lscpu = _run(["lscpu"])
    mem = _run(["free", "-h"])
    gpu = _run(["lspci"])

    def _grab(pattern: str, text: str) -> str:
        m = _re.search(pattern, text)
        return m.group(1).strip() if m else ""

    info["cpu"] = _grab(r"Model name:\s*(.+)", lscpu)
    info["cpu_cores"] = _grab(r"CPU\(s\):\s*(\d+)", lscpu)
    info["memory"] = _grab(r"Mem:\s*(\S+)", mem)
    info["gpu"] = _grab(r"VGA compatible controller: (.+)", gpu)
    return info


def _build_reports() -> tuple[str, str]:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system = _platform.system().lower()
    kernel = _platform.release()
    arch = _platform.machine()

    # Gercek (olculmus) bilesen kontrolleri.
    deps_ok = _check_dependencies()
    fs_ok = _check_filesystem()
    tok_ok = _check_tokenizer_cache()
    # Durum yalnizca zorunlu kontroller (bagimliliklar + dosya sistemi) gectiginde
    # 'VERIFIED'; tokenizer cache opt-in oldugu icin durumu etkilemez.
    verified = deps_ok and fs_ok
    status_en = "✅ LOCAL VERIFIED" if verified else "⚠️ LOCAL CHECKS INCOMPLETE"
    status_tr = "✅ YEREL DOĞRULANDI" if verified else "⚠️ YEREL KONTROLLER EKSİK"

    if system == "darwin":
        info = _parse_system_profiler()
        device = info.get("device") or "Mac"
        chip = info.get("chip") or "Apple Silicon"
        gpu = info.get("gpu") or "Apple GPU"
        gpu_cores = info.get("gpu_cores") or ""
        cpu_cores = info.get("cpu_cores") or "Unknown"
        memory = info.get("memory") or "Unknown"
        os_line = info.get("os") or "macOS"
        # GPU core sayisi SPDisplaysDataType'ta her macOS surumunde bulunmaz;
        # eslesmediyse '(N cores, ...)' iddiasini yazma, yalniz Metal'i belirt.
        if gpu_cores:
            gpu_line_en = f"- **GPU:** {gpu} ({gpu_cores} cores, Metal supported)\n"
            gpu_line_tr = f"- **GPU:** {gpu} ({gpu_cores} çekirdek, Metal destekli)\n"
        else:
            gpu_line_en = f"- **GPU:** {gpu} (Metal supported)\n"
            gpu_line_tr = f"- **GPU:** {gpu} (Metal destekli)\n"
        en = (
            "# 💻 TITAN SYSTEM & HARDWARE REPORT\n"
            f"**Date:** {now}\n"
            f"**Status:** {status_en}\n\n"
            "## 🖥️ Hardware Specification\n"
            f"- **Device:** {device} (Apple Silicon)\n"
            f"- **Chip:** {chip} ({cpu_cores})\n"
            f"{gpu_line_en}"
            f"- **Total RAM:** {memory}\n"
            f"- **OS:** {os_line}\n"
            f"- **Kernel:** Darwin {kernel} ({arch})\n\n"
            "## 🛠️ Components Checked\n"
            f"- {_mark(deps_ok)} **Dependencies:** Torch, Transformers, Accelerate (import probe)\n"
            f"- {_mark(fs_ok)} **Filesystem:** repo accessible, read/write\n"
            f"- {_mark(tok_ok)} **Tokenizer Cache:** `tokenizer/tr` present (opt-in)\n"
            "- ℹ️ **CPU/MPS Path:** safe fallback available (not exercised here)\n\n"
            "---\n"
            "*Generated locally from system_profiler/uname output. Serial/UUID values are intentionally omitted.*\n"
        )
        tr = (
            "# 💻 TITAN SİSTEM & DONANIM RAPORU\n"
            f"**Tarih:** {now}\n"
            f"**Durum:** {status_tr}\n\n"
            "## 🖥️ Donanım Özellikleri\n"
            f"- **Cihaz:** {device} (Apple Silicon)\n"
            f"- **Çip:** {chip} ({cpu_cores})\n"
            f"{gpu_line_tr}"
            f"- **Toplam RAM:** {memory}\n"
            f"- **İşletim Sistemi:** {os_line}\n"
            f"- **Kernel:** Darwin {kernel} ({arch})\n\n"
            "## 🛠️ Kontrol Edilen Bileşenler\n"
            f"- {_mark(deps_ok)} **Bağımlılıklar:** Torch, Transformers, Accelerate (import yoklaması)\n"
            f"- {_mark(fs_ok)} **Dosya Sistemi:** repo erişilebilir, okuma/yazma\n"
            f"- {_mark(tok_ok)} **Tokenizer Cache:** `tokenizer/tr` mevcut (opt-in)\n"
            "- ℹ️ **CPU/MPS Yolu:** güvenli fallback mevcut (burada çalıştırılmadı)\n\n"
            "---\n"
            "*system_profiler/uname çıktısından üretilmiştir. Serial/UUID bilgileri bilerek eklenmemiştir.*\n"
        )
        return en, tr

    linux = _linux_info()
    en = (
        "# 💻 TITAN SYSTEM & HARDWARE REPORT\n"
        f"**Date:** {now}\n"
        f"**Status:** {status_en}\n\n"
        "## 🖥️ Hardware Specification\n"
        f"- **Device:** Linux host\n"
        f"- **CPU:** {linux.get('cpu','Unknown')}\n"
        f"- **CPU Cores:** {linux.get('cpu_cores','Unknown')}\n"
        f"- **GPU:** {linux.get('gpu','Unknown')}\n"
        f"- **Total RAM:** {linux.get('memory','Unknown')}\n"
        f"- **Kernel:** {kernel} ({arch})\n\n"
        "## 🛠️ Components Checked\n"
        f"- {_mark(deps_ok)} **Dependencies:** Torch, Transformers, Accelerate (import probe)\n"
        f"- {_mark(fs_ok)} **Filesystem:** repo accessible, read/write\n"
        f"- {_mark(tok_ok)} **Tokenizer Cache:** `tokenizer/tr` present (opt-in)\n\n"
        "---\n"
        "*Generated from lscpu/free/lspci where available.*\n"
    )
    tr = (
        "# 💻 TITAN SİSTEM & DONANIM RAPORU\n"
        f"**Tarih:** {now}\n"
        f"**Durum:** {status_tr}\n\n"
        "## 🖥️ Donanım Özellikleri\n"
        f"- **Cihaz:** Linux host\n"
        f"- **CPU:** {linux.get('cpu','Unknown')}\n"
        f"- **CPU Çekirdek:** {linux.get('cpu_cores','Unknown')}\n"
        f"- **GPU:** {linux.get('gpu','Unknown')}\n"
        f"- **Toplam RAM:** {linux.get('memory','Unknown')}\n"
        f"- **Kernel:** {kernel} ({arch})\n\n"
        "## 🛠️ Kontrol Edilen Bileşenler\n"
        f"- {_mark(deps_ok)} **Bağımlılıklar:** Torch, Transformers, Accelerate (import yoklaması)\n"
        f"- {_mark(fs_ok)} **Dosya Sistemi:** repo erişilebilir, okuma/yazma\n"
        f"- {_mark(tok_ok)} **Tokenizer Cache:** `tokenizer/tr` mevcut (opt-in)\n\n"
        "---\n"
        "*lscpu/free/lspci çıktılarından üretilmiştir.*\n"
    )
    return en, tr


def main() -> None:
    en, tr = _build_reports()
    Path("reports/system_hardware.md").write_text(en, encoding="utf-8")
    Path("reports/system_hardware_TR.md").write_text(tr, encoding="utf-8")
    print("✅ system_hardware reports updated.")


if __name__ == "__main__":
    main()
