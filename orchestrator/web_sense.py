"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - WEB SENSE (SEARCH & RESEARCH)
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Module: orchestrator/web_sense.py - DuckDuckGo web search & URL reading helper.

NOTE (scope): Bu modül orchestrator/ altinda yer alır ve 45K eğitim yolunda
KAPALIDIR (inert / out-of-scope; feature-flag ile devre disi). Aşağıdaki
"Project / Version / Status" satirlari tum orchestrator dosyalarinda tekrar
eden ortak boilerplate banner'in fosilidir; bu modülun (DuckDuckGo arama)
gerçek işlevini tanimlamaz ve eğitim build surumunu temsil ETMEZ.

Project: (boilerplate banner - modül ile dogrudan ilgisiz)
Version: bkz. __version__ (legacy build etiketi, kanonik surum kaynagi değil)
Status : orchestrator yardimci modulu
==============================================================================
"""

# Legacy build etiketi (fosil); kanonik surum kaynagi değildir, yalniz geriye
# donuk uyumluluk icin korunuyor.
__version__ = "1.0-BUILD30-V2"
__author__ = "Mert Yünlü"

import logging
from typing import List, Callable, Optional

logger = logging.getLogger(__name__)

# Web libraries - optional
try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None


# Constants
MAX_WEB_RESULTS = 10
MAX_URL_CHARS = 5000


class WebSense:
    """
    Window to the world.
    Search and URL reading capabilities via DuckDuckGo.
    """
    
    def __init__(self):
        self.enabled = all([DDGS is not None, requests is not None, BeautifulSoup is not None])
        if self.enabled:
            print("🌐 Web Modülü Yüklendi.")
        else:
            print("🌐 Web Modülü devre dışı (eksik kütüphane).")

    def search(self, query: str, max_results: int = MAX_WEB_RESULTS) -> str:
        """Searches via DuckDuckGo and returns raw results."""
        if not self.enabled:
            return "Web Modülü aktif değil (gerekli kütüphaneler eksik)."

        try:
            results: List[str] = []
            with DDGS() as ddgs:  # type: ignore
                search_gen = ddgs.text(query, max_results=max_results)
                for r in search_gen:
                    title = r.get("title") or ""
                    body = r.get("body") or ""
                    href = r.get("href") or ""
                    block = f"BAŞLIK: {title}\nÖZET: {body}\nLİNK: {href}\n"
                    results.append(block)

            if not results:
                return "Web araması sonuç döndürmedi."

            return "\n".join(results)
        except Exception as e:
            logger.warning("Web araması başarısız (query=%r): %s", query, e)
            return f"Web Arama Hatası: {e}"

    def deep_research(self, query: str, llm_callback: Callable[[str, float], str]) -> str:
        """
        Performs search AND uses LLM to prepare a professional, referenced report.
        """
        raw_results = self.search(query, max_results=8)
        
        prompt = f"""
[SYS]
Sen Titan AGI'nin Web Araştırma Modülüsün.
Görevin: Kullanıcının sorusu için internetten bulunan ham verileri kullanarak
PROFESYONEL, AKADEMİK VE YAPISAL bir rapor oluşturmak.

Kurallar:
1. Asla sadece linkleri listeleme. Bilgiyi sentezle.
2. Referans ver. [1], [2] gibi.
3. Markdown formatını aktif kullan (Bold, Listeler, Tablolar).
4. Eğer sayısal veri varsa, ASCII tablosu oluştur.
5. Kullanıcıya doğrudan hitap et (Örn: "Araştırmalarıma göre...").
6. Asla "bilmiyorum" deme, elindeki veriyi en iyi şekilde sun.
[/SYS]

[SORU]
{query}
[/SORU]

[HAM VERİLER]
{raw_results}
[/HAM VERİLER]

[RAPOR]
"""
        return llm_callback(prompt.strip(), 0.5)

    def read_url(self, url: str) -> str:
        """Reads the given link."""
        if not self.enabled:
            return "URL okuma modülü aktif değil (gerekli kütüphaneler eksik)."

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=8)  # type: ignore
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")  # type: ignore
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
            if not text:
                text = resp.text
            text = text.replace("\n", " ").strip()
            if len(text) > MAX_URL_CHARS:
                text = text[:MAX_URL_CHARS] + "..."
            return text or "Sayfada anlamlı metin bulunamadı."
        except Exception as e:
            logger.warning("URL okuma başarısız (url=%r): %s", url, e)
            return f"Site Okuma Hatası: {e}"
