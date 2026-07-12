"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - AUDIO SENSE (STT & TTS)
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)

NOT: Bu modul orchestrator/ altinda yer alir; inert / out-of-scope'tur.
45K egitim yolunda kapali (feature-flag) -- TTS/STT yardimci arabirimi olup
egitim/cikarim ana hattini etkilemez.
==============================================================================
"""

from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

import subprocess
import logging
from typing import Optional

# Speech Recognition - optional
try:
    import speech_recognition as sr
except ImportError:
    sr = None

logger = logging.getLogger("TitanAudioSense")


class AudioSense:
    """
    Hearing and Speaking capability.
    macOS Native TTS + Google Speech Recognition.
    """
    
    def __init__(self):
        if sr is not None:
            print("🔊 Ses Modülü (MacOS Native + SpeechRecognition) Yüklendi.")
        else:
            print("🔊 Ses Modülü kısmi (sadece TTS). SpeechRecognition yok.")
        
        self.recognizer = None
        if sr is not None:
            try:
                self.recognizer = sr.Recognizer()
            except Exception as e:
                logger.warning("Mikrofon kütüphanesi başlatılamadı: %s", e)
                print(f"⚠️ Mikrofon kütüphanesi başlatılamadı: {e}")
                self.recognizer = None

    # --- MOUTH (TTS) ---
    def speak(self, text: str, voice: str = "Yelda") -> None:
        """Reads text aloud (TTS). Safe subprocess usage."""
        if not text:
            return
        
        # NOT: Shell enjeksiyon korumasi DEGIL. subprocess asagida liste
        # argumaniyla (shell=False) cagriliyor, bu yeterli korumayi saglar.
        # Bu temizlik yalniz 'say' ciktisinin okunusunu duzeltmek icin
        # kozmetiktir (tirnak/noktali virgul ayiklama, & -> ' ve ').
        safe_text = (
            text.replace('"', "")
            .replace("'", "")
            .replace(";", "")
            .replace("&", " ve ")
            .strip()
        )
        
        print(f"🗣️ [TTS]: {safe_text[:100]}...")
        
        try:
            # macOS 'say' command - non-blocking
            subprocess.Popen(
                ["say", "-v", voice, safe_text],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
        except FileNotFoundError:
            logger.warning("TTS: 'say' komutu bulunamadı (sadece macOS)")
        except Exception as e:
            logger.error(f"TTS Error: {e}")

    # --- EAR (STT) ---
    def listen(self, timeout: int = 5, phrase_time_limit: int = 10, language: str = "tr-TR") -> str:
        """Listens to microphone briefly."""
        if self.recognizer is None or sr is None:
            return ""

        try:
            with sr.Microphone() as source:
                print("\n👂 Dinliyorum (Konuşabilirsin)...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
                print("⏳ İşleniyor...")
                text = self.recognizer.recognize_google(audio, language=language)
                print(f"🗣️ Algılanan: {text}")
                return text
        except sr.WaitTimeoutError:
            print("⚠️ Ses gelmedi.")
            return ""
        except sr.UnknownValueError:
            print("⚠️ Anlaşılamadı.")
            return ""
        except Exception as e:
            logger.warning("Ses tanıma hatası: %s", e)
            print(f"⚠️ Ses tanıma hatası: {e}")
            return ""
    
    def is_stt_available(self) -> bool:
        """Is STT (Speech-to-Text) available?"""
        return self.recognizer is not None and sr is not None
    
    def is_tts_available(self) -> bool:
        """Is TTS (Text-to-Speech) available?"""
        try:
            # Check macOS 'say' command
            result = subprocess.run(
                ["which", "say"],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False
