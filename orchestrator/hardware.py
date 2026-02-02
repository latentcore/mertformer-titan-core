"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - HARDWARE SENSE (MONITORING)
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

import platform
import psutil

# TR: Config import - fallback mekanizmalı
# EN: Config import - with fallback mechanism
try:
    from config.config import cfg
except ImportError:
    class cfg:  # type: ignore
        device = "cpu"


class HardwareSense:
    """
    TR: AGI'nin kendi sınırlarını bilmesini sağlar.
    EN: Enables AGI to know its own limitations.
    TR: Sistem kaynakları, CPU/RAM kullanımı ve AI device bilgisi.
    EN: System resources, CPU/RAM usage and AI device information.
    """
    
    def __init__(self):
        self.system = platform.system()
        self.processor = platform.processor()
        self.device = getattr(cfg, "device", "cpu")

    def scan(self) -> str:
        """TR: Sistem durumunu tarar ve rapor döndürür. / EN: Scans system status and returns report."""
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
        """TR: Kullanılabilir RAM miktarını GB olarak döndürür. / EN: Returns available RAM in GB."""
        mem = psutil.virtual_memory()
        return round(mem.available / (1024 ** 3), 2)
    
    def is_low_memory(self, threshold_gb: float = 2.0) -> bool:
        """TR: Düşük bellek durumunu kontrol eder. / EN: Checks for low memory condition."""
        return self.get_available_memory_gb() < threshold_gb
