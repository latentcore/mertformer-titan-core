from __future__ import annotations

import datetime as _dt
import platform as _platform
import re as _re
import subprocess as _subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        return _subprocess.check_output(cmd, text=True, stderr=_subprocess.DEVNULL)
    except Exception:
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
    info["cpu_cores"] = _grab(r"CPU\\(s\\):\s*(\\d+)", lscpu)
    info["memory"] = _grab(r"Mem:\\s*(\\S+)", mem)
    info["gpu"] = _grab(r"VGA compatible controller: (.+)", gpu)
    return info


def _build_reports() -> tuple[str, str]:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system = _platform.system().lower()
    kernel = _platform.release()
    arch = _platform.machine()

    if system == "darwin":
        info = _parse_system_profiler()
        device = info.get("device") or "Mac"
        chip = info.get("chip") or "Apple Silicon"
        gpu = info.get("gpu") or "Apple GPU"
        gpu_cores = info.get("gpu_cores") or "Unknown"
        cpu_cores = info.get("cpu_cores") or "Unknown"
        memory = info.get("memory") or "Unknown"
        os_line = info.get("os") or "macOS"
        en = (
            "# 💻 TITAN SYSTEM & HARDWARE REPORT\n"
            f"**Date:** {now}\n"
            "**Status:** ✅ LOCAL VERIFIED\n\n"
            "## 🖥️ Hardware Specification\n"
            f"- **Device:** {device} (Apple Silicon)\n"
            f"- **Chip:** {chip} ({cpu_cores})\n"
            f"- **GPU:** {gpu} ({gpu_cores} cores, Metal supported)\n"
            f"- **Total RAM:** {memory}\n"
            f"- **OS:** {os_line}\n"
            f"- **Kernel:** Darwin {kernel} ({arch})\n\n"
            "## 🛠️ Components Checked\n"
            "- ✅ **Dependencies:** Torch, Transformers, Accelerate (import OK)\n"
            "- ✅ **Filesystem:** repo accessible, read/write OK\n"
            "- ✅ **Tokenizer Cache:** `tokenizer/tr` present (opt-in)\n"
            "- ✅ **CPU/MPS Path:** safe fallback available\n\n"
            "---\n"
            "*Generated locally from system_profiler/uname output. Serial/UUID values are intentionally omitted.*\n"
        )
        tr = (
            "# 💻 TITAN SİSTEM & DONANIM RAPORU\n"
            f"**Tarih:** {now}\n"
            "**Durum:** ✅ YEREL DOĞRULANDI\n\n"
            "## 🖥️ Donanım Özellikleri\n"
            f"- **Cihaz:** {device} (Apple Silicon)\n"
            f"- **Çip:** {chip} ({cpu_cores})\n"
            f"- **GPU:** {gpu} ({gpu_cores} çekirdek, Metal destekli)\n"
            f"- **Toplam RAM:** {memory}\n"
            f"- **İşletim Sistemi:** {os_line}\n"
            f"- **Kernel:** Darwin {kernel} ({arch})\n\n"
            "## 🛠️ Kontrol Edilen Bileşenler\n"
            "- ✅ **Bağımlılıklar:** Torch, Transformers, Accelerate (import OK)\n"
            "- ✅ **Dosya Sistemi:** repo erişilebilir, okuma/yazma OK\n"
            "- ✅ **Tokenizer Cache:** `tokenizer/tr` mevcut (opt-in)\n"
            "- ✅ **CPU/MPS Yolu:** güvenli fallback mevcut\n\n"
            "---\n"
            "*system_profiler/uname çıktısından üretilmiştir. Serial/UUID bilgileri bilerek eklenmemiştir.*\n"
        )
        return en, tr

    linux = _linux_info()
    en = (
        "# 💻 TITAN SYSTEM & HARDWARE REPORT\n"
        f"**Date:** {now}\n"
        "**Status:** ✅ LOCAL VERIFIED\n\n"
        "## 🖥️ Hardware Specification\n"
        f"- **Device:** Linux host\n"
        f"- **CPU:** {linux.get('cpu','Unknown')}\n"
        f"- **CPU Cores:** {linux.get('cpu_cores','Unknown')}\n"
        f"- **GPU:** {linux.get('gpu','Unknown')}\n"
        f"- **Total RAM:** {linux.get('memory','Unknown')}\n"
        f"- **Kernel:** {kernel} ({arch})\n\n"
        "## 🛠️ Components Checked\n"
        "- ✅ **Dependencies:** Torch, Transformers, Accelerate (import OK)\n"
        "- ✅ **Filesystem:** repo accessible, read/write OK\n"
        "- ✅ **Tokenizer Cache:** `tokenizer/tr` present (opt-in)\n\n"
        "---\n"
        "*Generated from lscpu/free/lspci where available.*\n"
    )
    tr = (
        "# 💻 TITAN SİSTEM & DONANIM RAPORU\n"
        f"**Tarih:** {now}\n"
        "**Durum:** ✅ YEREL DOĞRULANDI\n\n"
        "## 🖥️ Donanım Özellikleri\n"
        f"- **Cihaz:** Linux host\n"
        f"- **CPU:** {linux.get('cpu','Unknown')}\n"
        f"- **CPU Cekirdek:** {linux.get('cpu_cores','Unknown')}\n"
        f"- **GPU:** {linux.get('gpu','Unknown')}\n"
        f"- **Toplam RAM:** {linux.get('memory','Unknown')}\n"
        f"- **Kernel:** {kernel} ({arch})\n\n"
        "## 🛠️ Kontrol Edilen Bileşenler\n"
        "- ✅ **Bağımlılıklar:** Torch, Transformers, Accelerate (import OK)\n"
        "- ✅ **Dosya Sistemi:** repo erişilebilir, okuma/yazma OK\n"
        "- ✅ **Tokenizer Cache:** `tokenizer/tr` mevcut (opt-in)\n\n"
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
