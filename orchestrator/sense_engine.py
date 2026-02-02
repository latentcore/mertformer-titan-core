"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - SENSE ENGINE (VISION & SEMANTIC)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v27.0-FINAL (Locked & Sealed)
Status : PRODUCTION READY (LOCKED)
==============================================================================
"""

__version__ = "27.0-FINAL"
__author__ = "Mert"

from typing import List, Optional
import torch
import torch.nn.functional as F

# TR: Transformers - opsiyonel / EN: Transformers - optional
try:
    from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
except ImportError:
    CLIPProcessor = CLIPModel = AutoTokenizer = AutoModel = None

try:
    from PIL import Image
except ImportError:
    Image = None


# TR: Sabitler / EN: Constants
DEFAULT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SenseEngine:
    """
    TR: Görme ve Anlam Motoru.
    EN: Vision and Semantic Engine.
    - TR: Text Embedding (MiniLM) / EN: Text Embedding (MiniLM)
    - TR: Vision (CLIP) - lazy load / EN: Vision (CLIP) - lazy load
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        print(f"👁️ Vision & Anlam Motoru Yükleniyor (embedding device={device})...")

        # TR: Metin Embedding (MiniLM) / EN: Text Embedding (MiniLM)
        self.tokenizer = None
        self.text_model = None
        self._init_text_encoder()

        # TR: Görüntü (CLIP) — lazy load / EN: Image (CLIP) — lazy load
        self.clip_model = None
        self.clip_proc = None
        self.vision_active = False
        
        if CLIPModel is not None and CLIPProcessor is not None and Image is not None:
            print("👁️ CLIP Vision hazır (lazy load, sadece !see çağrılınca yüklenecek).")
        else:
            print("⚠️ CLIP/PIL eksik, vision devre dışı.")

    def _init_text_encoder(self) -> None:
        """TR: Text embedding modelini başlat. / EN: Initialize text embedding model."""
        if AutoTokenizer is not None and AutoModel is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_EMB_MODEL)
                self.text_model = AutoModel.from_pretrained(DEFAULT_EMB_MODEL).to(self.device)
                self.text_model.eval()
                print(f"🧠 Text Embedding Modeli: {DEFAULT_EMB_MODEL}")
            except Exception as e:
                print(f"⚠️ Embedding modeli yüklenemedi: {e}")
        else:
            print("⚠️ transformers yok, metin embedding devre dışı.")

    def encode_text(self, text: str) -> List[float]:
        """TR: Metni vektöre dönüştürür. / EN: Converts text to vector."""
        if not text or self.tokenizer is None or self.text_model is None:
            return [0.0] * 384
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.text_model(**inputs)
            
            emb = outputs.last_hidden_state.mean(dim=1)
            emb = F.normalize(emb, p=2, dim=1)[0].cpu().tolist()
            return emb
        except Exception as e:
            print(f"⚠️ encode_text hatası: {e}")
            return [0.0] * 384

    def _ensure_clip_loaded(self) -> None:
        """TR: CLIP modelini sadece gerçekten gerektiğinde yükler. / EN: Loads CLIP model only when really needed."""
        if self.clip_model is not None and self.clip_proc is not None:
            return
        if CLIPModel is None or CLIPProcessor is None or Image is None:
            return
        
        try:
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device)
            self.clip_proc = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.clip_model.eval()
            self.vision_active = True
            print("👁️ CLIP Vision Modeli Yüklendi (lazy).")
        except Exception as e:
            print(f"⚠️ Vision modeli yüklenemedi (Sadece metin): {e}")
            self.clip_model = None
            self.clip_proc = None
            self.vision_active = False

    def see(self, image_path: str) -> str:
        """TR: Görüntüyü analiz eder. / EN: Analyzes the image."""
        self._ensure_clip_loaded()
        if not self.vision_active or Image is None:
            return "[GÖRSEL] Vision modülü aktif değil."
        
        try:
            image_path = image_path.strip().strip('"').strip("'")
            image = Image.open(image_path)
            return (
                f"[GÖRSEL ANALİZ] Dosya: {image_path} | "
                f"Boyut: {image.size} | Format: {image.format}"
            )
        except Exception as e:
            return f"[GÖRSEL HATA] {e}"
    
    def encode_image(self, image_path: str) -> Optional[List[float]]:
        """TR: Görüntüyü vektöre dönüştürür (CLIP). / EN: Converts image to vector (CLIP)."""
        self._ensure_clip_loaded()
        if not self.vision_active or Image is None:
            return None
        
        try:
            image_path = image_path.strip().strip('"').strip("'")
            image = Image.open(image_path)
            inputs = self.clip_proc(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            return F.normalize(image_features, p=2, dim=1)[0].cpu().tolist()
        except Exception as e:
            print(f"⚠️ encode_image hatası: {e}")
            return None
