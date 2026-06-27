"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - HARDWARE SENSE (MONITORING)
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

import platform
import psutil

# Config import - with fallback mechanism
try:
    from config.config import cfg
except ImportError:
    class cfg:  # type: ignore
        device = "cpu"


class HardwareSense:
    """
    Sistem RAM/CPU ve device bilgisini raporlar.

    NOT (inert / out-of-scope): orchestrator katmani 45K egitim yolunda
    kapalidir (feature-flag); egitim akisini etkilemez.

    scan() yalnizca psutil (virtual_memory / cpu_percent) ve platform
    bilgisini okuyan basit bir monitoring yardimcisidir; herhangi bir
    "kendi sinirlarini bilen AGI" yetenegi olculmemis/yoktur.
    """
    
    def __init__(self):
        self.system = platform.system()
        self.processor = platform.processor()
        self.device = getattr(cfg, "device", "cpu")

    def scan(self) -> str:
        """Scans system status and returns report."""
        mem = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent()
        ram_gb = round(mem.total / (1024 ** 3))
        ram_used = round(mem.used / (1024 ** 3), 1)

        return (
            f"[BEDEN DURUMU] OS: {self.system} | İşlemci: {self.processor} | "
            f"RAM: {ram_used}/{ram_gb} GB (%{mem.percent}) | CPU Yükü: %{cpu_usage} | "
            f"AI Device: {self.device}"
        )
    
    def get_available_memory_gb(self) -> float:
        """Returns available RAM in GB."""
        mem = psutil.virtual_memory()
        return round(mem.available / (1024 ** 3), 2)
    
    def is_low_memory(self, threshold_gb: float = 2.0) -> bool:
        """Checks for low memory condition."""
        return self.get_available_memory_gb() < threshold_gb
