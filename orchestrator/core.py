"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - AGI ORCHESTRATOR CORE
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 27) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD27"
__author__ = "Mert"

import os
import sys
import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path

import torch

# TR: Yerel import'lar / EN: Local imports
from .paths import AGIPaths
from .hardware import HardwareSense
from .web_sense import WebSense
from .audio_sense import AudioSense
from .sense_engine import SenseEngine
from .memory import GodMemory, DocIndexer, RAGEngine

# TR: MertFormer import - fallback mekanizmalı
# EN: MertFormer import - with fallback mechanism
try:
    from config.config import cfg
    from model.transformers import MertFormer
    MERTFORMER_AVAILABLE = True
except ImportError:
    MertFormer = None
    MERTFORMER_AVAILABLE = False
    
    class cfg:  # type: ignore
        device = "cpu"
        save_dir = "checkpoints"
        model_name = "mertformer"
        vocab_size = 128256
        max_seq_len = 4096

# TR: Günlük kaydı / EN: Logging
logger = logging.getLogger("TitanOrchestrator")


class MertFormerOrchestrator:
    """
    TR: MertFormer Titan AGI Orkestratörü.
    EN: MertFormer Titan AGI Orchestrator.
    TR: Tüm sense modüllerini, hafızayı ve model inference'ı birleştirir.
    EN: Combines all sense modules, memory and model inference.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        load_model: bool = True,
        enable_voice: bool = False,
    ):
        """
        Args:
            device: TR: Hesaplama cihazı (None = otomatik) / EN: Compute device (None = auto)
            load_model: TR: MertFormer modelini yükle / EN: Load MertFormer model
            enable_voice: TR: Sesli yanıt (TTS) etkinleştir / EN: Enable voice response (TTS)
        """
        # TR: Cihaz seçimi / EN: Device selection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.enable_voice = enable_voice
        
        print(f"🚀 MertFormer Titan Orchestrator başlatılıyor...")
        print(f"   Device: {self.device}")
        
        # TR: Dizinleri oluştur / EN: Create directories
        AGIPaths.ensure_dirs()
        
        # TR: Sense modüllerini başlat / EN: Initialize sense modules
        self.hardware = HardwareSense()
        self.web = WebSense()
        self.audio = AudioSense()
        self.senses = SenseEngine(device=self.device)
        
        # TR: Hafıza ve RAG / EN: Memory and RAG
        self.memory = GodMemory(AGIPaths.MEMORY_FILE, self.senses)
        self.doc_indexer = DocIndexer(AGIPaths.DOC_DIR, AGIPaths.VECTOR_FILE, self.senses)
        self.rag = RAGEngine(self.memory, self.doc_indexer, self.senses)
        
        # TR: Model / EN: Model
        self.model = None
        self.tokenizer = None
        
        if load_model and MERTFORMER_AVAILABLE:
            self._load_model()
        
        print(f"✅ Orchestrator hazır!")
    
    def _load_model(self) -> None:
        """TR: MertFormer modelini yükle. / EN: Load MertFormer model."""
        try:
            print(f"🧠 MertFormer modeli yükleniyor...")
            
            # TR: Tokenizer / EN: Tokenizer
            from transformers import AutoTokenizer
            tokenizer_id = getattr(cfg, "teacher_model_id", "gpt2")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # TR: Model / EN: Model
            self.model = MertFormer().to(self.device)
            
            # TR: Checkpoint yükle (varsa) / EN: Load checkpoint (if exists)
            if AGIPaths.CHECKPOINT_FILE.exists():
                print(f"📂 Checkpoint yükleniyor: {AGIPaths.CHECKPOINT_FILE}")
                checkpoint = torch.load(AGIPaths.CHECKPOINT_FILE, map_location=self.device)
                
                # TR: İç içe state dict kontrolü / EN: Nested state dict check
                state_dict = checkpoint.get("model", checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"✅ Checkpoint yüklendi!")
            else:
                print(f"⚠️ Checkpoint bulunamadı, random weights kullanılıyor.")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            self.model = None
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """TR: Metin üret. / EN: Generate text."""
        if self.model is None or self.tokenizer is None:
            return "[TR: Model yüklenemedi, inference yapılamıyor / EN: Model not loaded, cannot perform inference]"
        
        # TR: Tokenize / EN: Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_seq_len - max_tokens,
        ).to(self.device)
        
        # TR: Üretim / EN: Generate
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # TR: Decode / EN: Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # TR: Prompt'u çıkar / EN: Remove prompt
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        
        return generated
    
    def chat(
        self,
        user_message: str,
        use_memory: bool = True,
        use_web: bool = False,
        temperature: float = 0.7,
    ) -> str:
        """
        TR: Kullanıcıyla sohbet.
        EN: Chat with user.
        
        Args:
            user_message: TR: Kullanıcı mesajı / EN: User message
            use_memory: TR: Hafızadan context al / EN: Get context from memory
            use_web: TR: Web araması yap / EN: Perform web search
            temperature: TR: Sampling sıcaklığı / EN: Sampling temperature
        """
        # TR: Hafızayı kaydet / EN: Save to memory
        self.memory.save("user", user_message, category="GENERAL", source="CHAT")
        
        # TR: Context oluştur / EN: Build context
        context_parts = []
        
        if use_memory:
            memory_context = self.memory.build_context_block(user_message)
            if memory_context:
                context_parts.append(memory_context)
        
        if use_web and self.web.enabled:
            web_results = self.web.search(user_message, max_results=3)
            context_parts.append(f"[WEB RESULTS]\n{web_results}\n[/WEB RESULTS]")
        
        # TR: Prompt oluştur / EN: Build prompt
        context = "\n\n".join(context_parts)
        prompt = f"""Sen MertFormer Titan, gelişmiş bir yapay zeka asistanısın.

{context}

Kullanıcı: {user_message}

Titan:"""
        
        # TR: Üretim / EN: Generate
        response = self.generate(prompt, temperature=temperature)
        
        # TR: Hafızaya kaydet / EN: Save to memory
        self.memory.save("assistant", response, category="GENERAL", source="CHAT")
        
        # TR: Sesli yanıt / EN: Voice response
        if self.enable_voice:
            self.audio.speak(response)
        
        return response
    
    def status(self) -> str:
        """TR: Sistem durumunu döndür. / EN: Return system status."""
        lines = [
            "📊 TITAN ORCHESTRATOR STATUS",
            "=" * 40,
            self.hardware.scan(),
            f"🧠 Model: {'Yüklü' if self.model else 'Yüklenmedi'}",
            f"💾 Hafıza: {len(self.memory.cache)} kayıt",
            f"📚 Doküman Chunks: {len(self.doc_indexer.chunks)}",
            f"🌐 Web: {'Aktif' if self.web.enabled else 'Devre Dışı'}",
            f"🔊 TTS: {'Aktif' if self.audio.is_tts_available() else 'Devre Dışı'}",
            "=" * 40,
        ]
        return "\n".join(lines)
    
    def repl(self) -> None:
        """TR: İnteraktif REPL döngüsü. / EN: Interactive REPL loop."""
        print("\n" + "=" * 60)
        print("🚀 MERTFORMER TITAN - Interactive Mode")
        print("   Komutlar: !status, !web <query>, !voice, !quit")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("Sen: ").strip()
                
                if not user_input:
                    continue
                
                # TR: Özel komutlar / EN: Special commands
                if user_input.lower() == "!quit":
                    print("👋 Görüşürüz!")
                    break
                elif user_input.lower() == "!status":
                    print(self.status())
                    continue
                elif user_input.lower() == "!voice":
                    self.enable_voice = not self.enable_voice
                    print(f"🔊 Sesli yanıt: {'Açık' if self.enable_voice else 'Kapalı'}")
                    continue
                elif user_input.lower().startswith("!web "):
                    query = user_input[5:].strip()
                    print(f"🌐 Web araması: {query}")
                    print(self.web.search(query))
                    continue
                
                # TR: Normal sohbet / EN: Normal chat
                response = self.chat(user_input, use_memory=True)
                print(f"\nTitan: {response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Görüşürüz!")
                break
            except Exception as e:
                print(f"⚠️ Hata: {e}")

# -----------------------------------------------------------------------------
# TR: GİRİŞ NOKTASI / EN: ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    """TR: CLI Giriş Noktası Orkestratör için. / EN: CLI Entry Point for Orchestrator."""
    try:
        # TR: Config'i yükle / EN: Load config to get model path or defaults
        from config.config import cfg
        
        print("🔧 Initializing MertFormer Titan Orchestrator...")
        titan = MertFormerOrchestrator(
            load_model=True, # Will auto-load defined model
            enable_voice=True
        )
        titan.repl()
        
    except KeyboardInterrupt:
        print("\n👋 Titan shutting down.")
    except Exception as e:
        print(f"\n❌ Titan Crash: {e} (Check if model checkpoints exist!)")
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    main()
