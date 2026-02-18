"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - AUDIO SENSE (STT & TTS)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30"
__author__ = "Mert"

import subprocess
import logging
from typing import Optional

# TR: Speech Recognition - opsiyonel / EN: Speech Recognition - optional
try:
    import speech_recognition as sr
except ImportError:
    sr = None

logger = logging.getLogger("TitanAudioSense")


class AudioSense:
    """
    TR: Duyma ve Konuşma yeteneği.
    EN: Hearing and Speaking capability.
    TR: macOS Native TTS + Google Speech Recognition.
    EN: macOS Native TTS + Google Speech Recognition.
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
                print(f"⚠️ Mikrofon kütüphanesi başlatılamadı: {e}")
                self.recognizer = None

    # TR: --- AĞIZ (TTS) --- / EN: --- MOUTH (TTS) ---
    def speak(self, text: str, voice: str = "Yelda") -> None:
        """TR: Metni sesli okur (TTS). Güvenli subprocess kullanımı. / EN: Reads text aloud (TTS). Safe subprocess usage."""
        if not text:
            return
        
        # TR: Güvenlik için temizlik / EN: Cleanup for security
        safe_text = (
            text.replace('"', "")
            .replace("'", "")
            .replace(";", "")
            .replace("&", " ve ")
            .strip()
        )
        
        print(f"🗣️ [TTS]: {safe_text[:100]}...")
        
        try:
            # TR: macOS 'say' komutu - non-blocking / EN: macOS 'say' command - non-blocking
            subprocess.Popen(
                ["say", "-v", voice, safe_text],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
        except FileNotFoundError:
            logger.warning("TTS: 'say' komutu bulunamadı (sadece macOS)")
        except Exception as e:
            logger.error(f"TTS Error: {e}")

    # TR: --- KULAK (STT) --- / EN: --- EAR (STT) ---
    def listen(self, timeout: int = 5, phrase_time_limit: int = 10, language: str = "tr-TR") -> str:
        """TR: Mikrofonu kısa süre dinler. / EN: Listens to microphone briefly."""
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
            print(f"⚠️ Ses tanıma hatası: {e}")
            return ""
    
    def is_stt_available(self) -> bool:
        """TR: STT (Speech-to-Text) kullanılabilir mi? / EN: Is STT (Speech-to-Text) available?"""
        return self.recognizer is not None and sr is not None
    
    def is_tts_available(self) -> bool:
        """TR: TTS (Text-to-Speech) kullanılabilir mi? / EN: Is TTS (Text-to-Speech) available?"""
        try:
            # TR: macOS 'say' komutunu kontrol et / EN: Check macOS 'say' command
            result = subprocess.run(
                ["which", "say"],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False
