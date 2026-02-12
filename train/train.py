"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
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

import glob
import json
import math
import os
import random
import sys
import time
import shutil
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed
try:
    import galore_torch
except ImportError:
    galore_torch = None
    
# Import DistillationManager
sys.path.append(str(Path(__file__).parent.parent))
from orchestrator.distillation_manager import DistillationManager

# -----------------------------------------------------------------------------
# TR: 0. PROJE KÖK DİZİN TESPİTİ / EN: 0. PROJECT ROOT DETECTION
# -----------------------------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent
for _ in range(4):
    if (project_root / "config").exists():
        break
    project_root = project_root.parent

sys.path.insert(0, str(project_root))
print(f"📍 PROJE ANA MERKEZİ: {project_root}")

# -----------------------------------------------------------------------------
# TR: İMPORTLAR / EN: IMPORTS
# -----------------------------------------------------------------------------
try:
    from config.config import cfg
    from model.transformers import MertFormer
    from utils.logger import RunLogger
except ImportError as e:
    print(f"❌ KRİTİK IMPORT HATASI: {e}")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
except Exception:
    print("❌ Transformers kütüphanesi eksik! 'pip install transformers' yap.")
    sys.exit(1)

try:
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
except ImportError:
    print("❌ Accelerate kütüphanesi eksik! 'pip install accelerate' yap.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# TR: 0.5. SAVE_DIR DÜZELTMESİ / EN: 0.5. SAVE_DIR FIX
# -----------------------------------------------------------------------------
if not hasattr(cfg, "save_dir"):
    cfg.save_dir = "checkpoints/mertformer_titan_prod"


# -----------------------------------------------------------------------------
# TR: 1. YARDIMCI FONİKSİYONLAR / EN: 1. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # TR: [PRO] Yeniden üretilebilirlik Modu / EN: [PRO] Reproducibility Mode
    if getattr(cfg, "deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            print("🔒 Deterministic Mode: ACTIVATED (Performance may slightly decrease)")
        except AttributeError:
            pass # Older pytorch versions


def validate_config(cfg, stage="pre"):
    """TR: Kritik config parametreleri için akıl sağlığı kontrolü (Aşamalı). / EN: Sanity check for critical config parameters (Staged)."""
    required = ["hidden_size", "num_layers", "max_seq_len", "model_name", "learning_rate"]
    
    # TR: [PRO] Öğretmen Sonrası Doğrulama (vocab_size gerektirir)
    # EN: [PRO] Post-Teacher Validation (requires vocab_size)
    if stage == "post":
        required.append("vocab_size")

    for k in required:
        if not hasattr(cfg, k):
            raise ValueError(f"❌ CRITICAL CONFIG ERROR: Missing key '{k}'")
    
    if cfg.learning_rate <= 0: raise ValueError("Learning Rate must be > 0")
    if getattr(cfg, "num_experts", 0) < 1: 
        # [PRO] Explicitly validate MoE config if use_moe flag exists, or just warn/error
        if getattr(cfg, "use_moe", False):
             raise ValueError("❌ CRITICAL: MoE enabled (use_moe=True) but num_experts < 1")
    if cfg.max_seq_len > 100000: print("⚠️ Warning: Unusual max_seq_len (>100k)")
    
    print(f"✅ Config Schema Verified ({stage}).")


def check_disk_space(min_gb: float = 10.0, path: Optional[Path] = None) -> bool:
    """
    TR: Diskte yeterli boş alan var mı kontrol eder.
    EN: Checks if there is enough free disk space.
    """
    target = path or project_root
    try:
        # TR: Disk boşluğu ölçümü
        # EN: Disk free space measurement
        total, used, free = shutil.disk_usage(str(target))
        free_gb = free / (1024 ** 3)
        return free_gb >= float(min_gb)
    except Exception:
        return True  # Fail-open to avoid breaking training


def get_gpu_memory_usage(device: Optional[int] = None) -> tuple[float, float]:
    """
    TR: GPU bellek kullanımını (allocated/reserved) GB olarak döndürür.
    EN: Returns GPU memory usage (allocated/reserved) in GB.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    # TR: Allocated/Reserved bellek raporu
    # EN: Allocated/Reserved memory report
    idx = device if device is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(idx) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
    return allocated, reserved


def get_student_device(accelerator=None):
    if accelerator:
        return accelerator.device
    return torch.device(cfg.device)


def get_teacher_device(accelerator=None):
    # Accelerate will handle device placement if initialized
    if accelerator:
        return accelerator.device
    if torch.cuda.is_available() and cfg.device == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def preflight_param_report(model: nn.Module):
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model Parametreleri: {n_params / 1e6:.2f} Milyon")
    print(f"📚 Trainable: {trainable / 1e6:.2f} Milyon")


def save_checkpoint_smart(model, optimizer, scheduler, step, cfg, keep_last_n=3, val_loss=None, best_val_loss=None):
    """
    Smart checkpoint saver - keeps only last N checkpoints + best checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        step: Current training step
        cfg: Configuration object
        keep_last_n: Number of recent checkpoints to keep
        val_loss: Current validation loss (optional)
        best_val_loss: Best validation loss seen so far (optional)
    
    Returns:
        float: Updated best_val_loss (if val_loss provided)
    """
    save_dir = project_root / cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"{cfg.model_name}_step_{step}.pt"
    latest_name = f"{cfg.model_name}_latest.pt"
    best_name = f"{cfg.model_name}_best.pt"

    save_path = save_dir / ckpt_name
    latest_path = save_dir / latest_name
    best_path = save_dir / best_name

    print(f"💾 Checkpoint Kaydediliyor: {ckpt_name}")

    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
        'config': str(cfg),
    }
    
    # Add validation loss to state if provided
    if val_loss is not None:
        state['val_loss'] = val_loss

    # Save regular checkpoint
    torch.save(state, save_path)
    torch.save(state, latest_path)
    
    # Save best checkpoint if this is the best so far
    if val_loss is not None and best_val_loss is not None:
        if val_loss < best_val_loss:
            print(f"🏆 NEW BEST! Val Loss: {val_loss:.4f} (Previous: {best_val_loss:.4f})")
            torch.save(state, best_path)
            best_val_loss = val_loss
        else:
            print(f"📊 Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")

    # Cleanup old checkpoints (but keep best.pt)
    search_pattern = str(save_dir / f"{cfg.model_name}_step_*.pt")
    all_ckpts = sorted(glob.glob(search_pattern), key=os.path.getmtime)

    if len(all_ckpts) > keep_last_n:
        files_to_delete = all_ckpts[:-keep_last_n]
        for f in files_to_delete:
            try:
                os.remove(f)
                print(f"🧹 Eski Checkpoint Silindi: {os.path.basename(f)}")
            except OSError as e:
                print(f"⚠️ Silme Hatası: {e}")
    
    return best_val_loss if val_loss is not None else best_val_loss



# -----------------------------------------------------------------------------
# TR: 1.5 INFERENCE SARLAYICI (ONNX TEMİZLEYİCİ)
# EN: 1.5 INFERENCE WRAPPER (ONNX CLEANER)
# -----------------------------------------------------------------------------
class MertFormerInferenceWrapper(nn.Module):
    """
    TR: Sadece Inference için hafifletilmiş model kılıfı. Aux loss'u atar.
    EN: Lightweight model wrapper for inference only. Discards aux loss.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        # TR: Aux loss'u yut, sadece logits döndür / EN: Swallow aux loss, return only logits
        logits, _, _ = self.model(input_ids)
        return logits


def export_to_onnx(model, save_dir, model_name, device):
    print(f"\n📦 ONNX DÖNÜŞÜMÜ BAŞLIYOR (CLEAN INFERENCE MODE)...")
    onnx_path = save_dir / f"{model_name}.onnx"

    # TR: Wrapper ile sar (Aux loss ve karmaşayı temizle) / EN: Wrap with wrapper (Clean aux loss and complexity)
    inference_model = MertFormerInferenceWrapper(model)
    inference_model.eval()

    # TR: [OPT] Küçük dummy girdi kullan (dinamik eksenler değişken uzunlukları idare eder)
    # EN: [OPT] Use small dummy input (dynamic axes handle variable lengths)
    dummy_len = min(cfg.max_seq_len, 32)
    dummy_input = torch.randint(0, cfg.vocab_size, (1, dummy_len)).to(device)

    try:
        torch.onnx.export(
            inference_model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=12, # [FIX] Legacy Stable Opset
            do_constant_folding=False, # [FIX] Disable folding to avoid graph capture errors
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        print(f"✅ ONNX BAŞARIYLA KAYDEDİLDİ: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX DÖNÜŞÜM HATASI: {e}")


# -----------------------------------------------------------------------------
# TR: 2. VERİ SETLERİ (CURRICULUM + VALIDATION)
# EN: 2. DATASETS (CURRICULUM + VALIDATION)
# -----------------------------------------------------------------------------
class ValidationJsonlDataset(IterableDataset):
    """
    TR: Validation için basit, sıralı (deterministik) okuyucu.
    EN: Simple, sequential (deterministic) reader for validation.
    """
    def __init__(self, path: Path, max_len: int, tokenizer):
        self.path = path
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if not text:
                        continue
                    enc = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_len,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    input_ids = enc["input_ids"].squeeze(0)
                    yield input_ids, input_ids
                except Exception:
                    continue


class CurriculumDataset(IterableDataset):
    """
    TR: Curriculum-aware dataset (RAM dostu Streaming + Worker Safe).
    EN: Curriculum-aware dataset (RAM-friendly Streaming + Worker Safe).
    """

    def __init__(self, stage_paths: List[Path], max_len: int, tokenizer, current_stage: int = 1):
        self.stage_paths = stage_paths
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.current_stage = current_stage
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # Dosya boyutlarını önceden hesapla (seek için)
        self.file_offsets = {}
        for p in stage_paths:
            if p.exists():
                with open(p, "rb") as f:
                    f.seek(0, 2)
                    self.file_offsets[str(p)] = f.tell()

    def set_stage(self, stage: int):
        """Update current curriculum stage."""
        self.current_stage = stage

    def __iter__(self):
        # 1. WORKER SEED SYNC (Çoklu işlemci çakışmasını önle)
        worker_info = get_worker_info()
        if worker_info is not None:
            # Her worker farklı bir seed alır
            random.seed(cfg.seed + worker_info.id)

        skipped_count = 0
        total_attempts = 0

        while True:
            # [FIX] Fallback guard: If not enough stage files, use all available
            n = len(self.stage_paths)
            if n < 4:
                # Fallback dataset or missing stages: disable curriculum
                active_paths = self.stage_paths
            elif self.current_stage == 1:
                active_paths = [self.stage_paths[0]]
            elif self.current_stage == 2:
                active_paths = [self.stage_paths[1]]
            elif self.current_stage == 3:
                # Stage 3: Mix all previous stages (10% Stage1, 10% Stage2, 80% Stage3)
                active_paths = self.stage_paths[:3]
            elif self.current_stage == 4:
                # Stage 4: Soul + Knowledge + Identity Mix
                active_paths = self.stage_paths
            else:
                active_paths = self.stage_paths

            # Select path based on stage logic or random choice from actives
            if self.current_stage == 3 and len(active_paths) > 1:
                r = random.random()
                if r < 0.1: path = active_paths[0]
                elif r < 0.2: path = active_paths[1]
                else: path = active_paths[2]
            else:
                path = random.choice(active_paths)

            if not path.exists():
                continue

            file_size = self.file_offsets.get(str(path), 0)
            if file_size == 0: continue

            total_attempts += 1

            # 2. VERİ KAYIP ALARMI (%5 eşik)
            if total_attempts % 5000 == 0 and total_attempts > 0:
                skip_rate = skipped_count / total_attempts
                if skip_rate > 0.05:
                    print(f"⚠️ DİKKAT: Veri atlama oranı yüksek: %{skip_rate*100:.1f} (JSON parse veya boş satır)")

            try:
                # [FIX] Binary mode seek is safe; text mode seek is fragile with UTF-8
                with open(path, "rb") as f:
                    # Rastgele bir bayt konumuna git
                    rand_pos = random.randint(0, max(0, file_size - 1000))
                    f.seek(rand_pos)
                    # İlk satır yarım olabilir, atla
                    f.readline()
                    # İkinci satırı al
                    line_bytes = f.readline()

                    if not line_bytes:
                        skipped_count += 1
                        continue

                    # [TITAN FIX] Decode with error handling (errors='ignore' for max stability)
                    try:
                        line = line_bytes.decode("utf-8", errors="ignore")
                    except Exception:
                        skipped_count += 1
                        continue

                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if not text:
                        skipped_count += 1
                        continue

                    enc = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_len,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    input_ids = enc["input_ids"].squeeze(0)
                    # TR: [FIX] RAM yükünü azaltmak için text yield'dan kaldırıldı / EN: [FIX] Removed text from yield to reduce RAM overhead
                    yield input_ids, input_ids
            except Exception:
                skipped_count += 1
                continue


class PrecomputedCurriculumDataset(IterableDataset):
    """
    TR: Önceden hesaplanmış logits ile deterministik curriculum dataset.
    EN: Deterministic curriculum dataset paired with precomputed logits.
    """

    def __init__(self, stage_info, max_len: int, tokenizer, distill_manager):
        self.stage_info = stage_info  # list of (stage_name, path)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.distill_manager = distill_manager
        self.current_stage = 1
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def set_stage(self, stage: int):
        self.current_stage = stage

    def _align_logits(self, logits: torch.Tensor, target_len: int) -> torch.Tensor:
        # TR: Logit uzunluğunu token uzunluğuna hizala
        # EN: Align logits length to token length
        # logits: [seq, vocab]
        if logits.dim() == 3 and logits.size(0) == 1:
            logits = logits.squeeze(0)
        if logits.dim() != 2:
            raise ValueError(f"Invalid logits shape: {tuple(logits.shape)}")
        seq_len = logits.size(0)
        if seq_len > target_len:
            logits = logits[:target_len]
        elif seq_len < target_len:
            pad = torch.zeros(target_len - seq_len, logits.size(1), dtype=logits.dtype)
            logits = torch.cat([logits, pad], dim=0)
        return logits

    def _iter_stage(self, stage_name: str, path: Path):
        # TR: Stage dataset + logits shard eşlemesi (deterministik)
        # EN: Deterministic pairing of stage dataset with logits shards
        logits_iter = iter(self.distill_manager.get_precomputed_loader(stage_name))
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if not text:
                        continue
                    enc = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_len,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    input_ids = enc["input_ids"].squeeze(0)
                    try:
                        t_logits = next(logits_iter)
                    except StopIteration:
                        return
                    t_logits = self._align_logits(t_logits, input_ids.size(0))
                    yield input_ids, input_ids, t_logits
                except Exception:
                    continue

    def __iter__(self):
        # TR: Offline logits için worker paralelliği kapalı
        # EN: Disable worker parallelism for offline logits alignment
        worker_info = get_worker_info()
        if worker_info is not None:
            raise RuntimeError("Precomputed logits require num_workers=0 for deterministic alignment.")

        current_stage = self.current_stage
        stage_name, path = self.stage_info[current_stage - 1]
        stage_iter = self._iter_stage(stage_name, path)

        while True:
            if self.current_stage != current_stage:
                current_stage = self.current_stage
                stage_name, path = self.stage_info[current_stage - 1]
                stage_iter = self._iter_stage(stage_name, path)

            try:
                yield next(stage_iter)
            except StopIteration:
                stage_iter = self._iter_stage(stage_name, path)


def collate_fn(batch):
    # TR: [FIX] 2 elemana sadeleştirildi (text kaldırıldı) / EN: [FIX] Simplified to 2 elements (removed text)
    if len(batch[0]) == 3:
        x, y, t = zip(*batch)
        return torch.stack(x), torch.stack(y), torch.stack(t)
    x, y = zip(*batch)
    return torch.stack(x), torch.stack(y)


# -----------------------------------------------------------------------------
# TR: 3. BİLGİ DAMITMA KAYBI / EN: 3. KD LOSS (Knowledge Distillation)
# -----------------------------------------------------------------------------
def kd_loss_safe(student_logits, teacher_logits, temp, mask=None):
    min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    s = student_logits[..., :min_vocab].float()
    t = teacher_logits[..., :min_vocab].float().to(s.device)
    T = float(temp)
    token_kl = F.kl_div(
        F.log_softmax(s / T, dim=-1),
        F.softmax(t / T, dim=-1),
        reduction="none",
    ).sum(dim=-1) * (T * T)
    if mask is not None:
        mask = mask.to(device=token_kl.device, dtype=torch.bool)
        if mask.shape != token_kl.shape:
            raise ValueError(f"KD mask shape mismatch: expected {token_kl.shape}, got {mask.shape}")
        if not bool(mask.any().item()):
            return token_kl.new_zeros(())
        token_kl = token_kl.masked_select(mask)
    return token_kl.mean()

# TR: V21.0 DÜZELTME: WSD Zamanlama (Isınma-Sabit-Azalma) Grokking için
# EN: V21.0 FIX: WSD Scheduler (Warmup-Stable-Decay) for Grokking
def get_wsd_schedule(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, stable_ratio=0.8):
    """TR: Isınma, X% sabit tut, sonra keskin azalt. / EN: Warmup up, hold stable for X%, then decay sharply."""
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        stable_steps = int(num_training_steps * stable_ratio)
        if current_step < stable_steps:
            return 1.0 # TR: Sabit Faz / EN: Stable Phase
        
        # TR: Azalma Fazı (1.0'dan min_lr_ratio'ya Kosinüs) / EN: Decay Phase (Cosine from 1.0 to min_lr_ratio)
        decay_steps = num_training_steps - stable_steps
        progress = float(current_step - stable_steps) / float(max(1, decay_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
    return LambdaLR(optimizer, lr_lambda)




# -----------------------------------------------------------------------------
# TR: 5. ÖĞRETMEN VE MODEL KURULUMU / EN: 5. TEACHER & MODEL SETUP
# -----------------------------------------------------------------------------
class TeacherBundle:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = get_teacher_device()

        print(f"👨‍🏫 Teacher Hazırlanıyor: {cfg.teacher_model_id}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if cfg.distill_alpha > 0.0:
                print(f"🔄 Teacher ({cfg.teacher_model_id}) Loading...")
                # V26.5 DDP FIX: Avoid device_map="auto" in DDP. 
                # Accelerate handles student, but teacher is static.
                # In DDP, each process should load teacher to its own device (or CPU offload).
                # device_map="auto" tries to use all GPUs, causing conflict in DDP.
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.teacher_model_id,
                    torch_dtype=torch.float16,
                    # [FIX] Use explicit string map for HF compatibility across versions/backends.
                    device_map={
                        "": (
                            f"cuda:{self.device.index if self.device.index is not None else torch.cuda.current_device()}"
                            if self.device.type == "cuda"
                            else "cpu"
                        )
                    },
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
                )
                self.model.eval()
                # Explicit move to device logic
                # For 4-bit, we don't manually .to(device), bitsandbytes handles it via Accelerator or requires manual 'device_map={"": device}' hack
                # Safest bet for 4bit + DDP: Let it sit, get_logits handles .to(device) or trust bnb
                
                print("✅ Teacher Model Yüklendi.")
            else:
                print("ℹ️ Distill Alpha 0, Teacher Model yüklenmedi (Sadece Tokenizer).")

        except OSError as e:
            if "gated repo" in str(e) or "token" in str(e) or "401" in str(e):
                print(f"⛔ TEACHER AUTH ERROR: {cfg.teacher_model_id} is gated/missing.")
                print("⚠️  Activating FALLBACK STRATEGY: Switching to Open Teacher (Mistral-7B)...")
                
                try:
                    # Fallback to Mistral (Open Weights)
                    fallback_id = "mistralai/Mistral-7B-v0.1"
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback_id)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        
                    if cfg.distill_alpha > 0.0:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            fallback_id, 
                            torch_dtype=torch.float16
                        ).to(self.device)
                        self.model.eval()
                        
                    print(f"✅ Fallback Teacher ({fallback_id}) Loaded Successfully.")
                    return # Initialize complete
                    
                except Exception as ex:
                    print(f"❌ Fallback Failed: {ex}")
                    # Final Fallback: Pure Student (No Teacher)
                    print("⚠️  SAFE MODE: Continuing without Teacher (Pure Student Training).")
                    self.model = None
                    cfg.distill_alpha = 0.0
                    
                    # [V27.1 FIX] TOKENIZER GUARANTEE
                    # Dataset needs a tokenizer even if teacher is missing.
                    # We try to use the student's tokenizer (gpt2 usually)
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                        print("✅ Loaded 'gpt2' tokenizer as failsafe for dataset.")
                    except Exception as tok_ex:
                        print(f"💀 Critical: No tokenizer available ({tok_ex}). Exiting.")
                        sys.exit(1)
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        print("✅ Loaded 'gpt2' tokenizer as last resort.")
                    except:
                        print("💀 Critical: No tokenizer available. Exiting.")
                        sys.exit(1)
            else:
                print(f"⚠️ Teacher Init Error: {e}")
                sys.exit(1)
        except Exception as e:
            print(f"⚠️ Teacher Init Error: {e}")
            sys.exit(1)

    @torch.no_grad()
    def get_logits(self, input_ids):
        if self.model is None:
            return None
        # Ensure input is on correct device (Accelerate handles this but good to be safe)
        if hasattr(self, 'device'):
             input_ids = input_ids.to(self.device)
        return self.model(input_ids).logits


def load_teacher_tokenizer():
    """
    TR: Öğretmen tokenizer'ını güvenli şekilde yükle.
    EN: Safely load the teacher tokenizer (without loading the teacher model).
    """
    try:
        tok = AutoTokenizer.from_pretrained(cfg.teacher_model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception as e:
        print(f"⚠️ Teacher tokenizer load failed: {e}. Falling back to gpt2.")
        tok = AutoTokenizer.from_pretrained("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok


# -----------------------------------------------------------------------------
# TR: 5.5 OPTİMİZER YARDIMCISI / EN: 5.5 OPTIMIZER HELPER
# -----------------------------------------------------------------------------
def rebuild_optimizer(model, opt, cfg):
    """
    TR: Optimizer gruplarını mevcut requires_grad durumuna göre yeniler.
    EN: Rebuilds optimizer param groups based on current requires_grad state.
    """
    print("🔧 REBUILDING OPTIMIZER GROUPS...")
    router_params, body_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "router" in n or "tau" in n or "shared_gate" in n:
            router_params.append(p)
        else:
            body_params.append(p)

    # [V27.0] GaLore & 8-bit Optimizer Logic
    if cfg.use_galore and galore_torch is not None:
        print(f"🚀 OPTIMIZER: Using GaLore + {'8-bit' if cfg.use_8bit_adam else 'Full'} AdamW")
        optim_cls = galore_torch.GaLoreAdamW8bit if cfg.use_8bit_adam else galore_torch.GaLoreAdamW
        
        # Helper to create param groups with GaLore specific settings
        opt.param_groups.clear()
        
        # Body (Rank projected)
        opt.add_param_group({
            'params': body_params,
            'lr': cfg.learning_rate,
            'weight_decay': cfg.weight_decay,
            'rank': 128, # GaLore Rank
            'update_proj_gap': 200,
            'scale': 0.25
        })
        
        # Router (Full rank sensitive) -> Keep standard or high rank
        opt.add_param_group({
            'params': router_params,
            'lr': cfg.learning_rate * 1.5,
            'weight_decay': 1e-4,
            'rank': 64, 
            'update_proj_gap': 200,
            'scale': 0.25
        })
        
    elif cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            print("🚀 OPTIMIZER: Using Standard 8-bit AdamW")
            # We need to re-init optimizer completely if class changes, 
            # but rebuild_optimizer assumes 'opt' instance. 
            # Limitation: changing optimizer class mid-flight is hard. 
            # We assume initial optimizer IS correct class.
            
            # Re-assign params
            opt.param_groups.clear()
            opt.add_param_group({'params': body_params, 'lr': cfg.learning_rate, 'weight_decay': cfg.weight_decay})
            opt.add_param_group({'params': router_params, 'lr': cfg.learning_rate * 1.5, 'weight_decay': 1e-4})
            
        except ImportError:
            print("⚠️ 8-bit Adam requested but bitsandbytes not found. Using Standard AdamW.")
            # Fallback standard logic
            opt.param_groups.clear()
            opt.add_param_group({'params': body_params, 'lr': cfg.learning_rate, 'weight_decay': cfg.weight_decay})
            opt.add_param_group({'params': router_params, 'lr': cfg.learning_rate * 1.5, 'weight_decay': 1e-4})
            
    else:
        # Standard Fallback
        opt.param_groups.clear()
        opt.add_param_group({
            'params': body_params,
            'lr': cfg.learning_rate,
            'weight_decay': cfg.weight_decay
        })
        opt.add_param_group({
            'params': router_params,
            'lr': cfg.learning_rate * 1.5, # Grokking Boost (Safe Mode: 1.5x)
            'weight_decay': 1e-4
        })

    print(f"✅ Optimizer Rebuilt: {len(body_params)} Body, {len(router_params)} Router params.")


# -----------------------------------------------------------------------------
# TR: 6. DONDURMA DESTEĞİ / EN: 6. FREEZE SUPPORT
# -----------------------------------------------------------------------------
def apply_freeze_policy(model, freeze_core_layers: bool):
    """TR: Çekirdek katmanları dondur, MoE Router ve Liquid Layers eğitilebilir kalsın. / EN: Freeze core layers, keep MoE Router and Liquid Layers trainable."""
    if not freeze_core_layers:
        return

    print("🔒 Applying freeze policy: Core layers frozen, MoE/Liquid trainable")

    for name, param in model.named_parameters():
        # TR: Varsayılan olarak her şeyi dondur / EN: Freeze everything by default
        param.requires_grad = False

        # TR: MoE router ve paylaşılan uzmanı aç / EN: Unfreeze MoE router and shared expert
        if 'router' in name or 'shared_expert' in name or 'shared_gate' in name:
            param.requires_grad = True

        # TR: Liquid katmanlarını aç / EN: Unfreeze Liquid layers
        if 'liquid' in name.lower() or 'tau' in name:
            param.requires_grad = True

        # TR: LM head ve embedding'leri aç (her zaman eğitilebilir) / EN: Unfreeze LM head and embeddings (always trainable)
        if 'lm_head' in name or 'tok_embeddings' in name:
            param.requires_grad = True

        # TR: [TITAN FIX]: LayerNorm'lar her zaman eğitilebilir olmalı / EN: [TITAN FIX]: LayerNorms should always be trainable
        if 'norm' in name.lower():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"📊 Freeze Status: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M parameters trainable")


# -----------------------------------------------------------------------------
# TR: 7. ANA EĞİTİM DÖNGÜSÜ / EN: 7. MAIN TRAIN LOOP
# -----------------------------------------------------------------------------
def train():
    # TR: Accelerate Başlatma / EN: Accelerate Init
    accelerator_project_config = ProjectConfiguration(project_dir=str(project_root), logging_dir=str(project_root / "logs"))
    # TR: V24.0: Ampere (A100) hızlandırması için TF32 / EN: V24.0: TF32 for Ampere (A100) speedup
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print("⚡ TensorFloat-32 (TF32) activated for A100.")
    
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum_steps,
        mixed_precision="bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16" if cfg.use_amp else "no",
        project_config=accelerator_project_config,
        log_with="all"
    )

    # TR: Her şeyi seed'le / EN: Seed everything
    set_seed(cfg.seed + accelerator.process_index)
    
    # TR: Sadece ana süreç doğrulaması / EN: Only main process validation
    if accelerator.is_main_process:
        validate_config(cfg, stage="pre")
    
    student_device = accelerator.device

    logs_dir = project_root / "logs"
    save_dir = project_root / cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    # Logger (Main Process Only)
    logger = None
    if accelerator.is_main_process:
        logger = RunLogger(
            cfg=cfg,
            log_dir=logs_dir,
            project_root=project_root,
            train_path=current_file,
            config_path=project_root / "config" / "config.py",
            also_csv=True
        )
        logger.log_meta()


    # V27.0 DISTILLATION MANAGER: Switch between Online (TeacherBundle) and Offline (Precomputed Logits)
    # TR: Offline logits modu (öğretmen model yüklemeden distill)
    # EN: Offline logits mode (distill without loading teacher model)
    use_offline_logits = getattr(cfg, "use_precomputed_logits", False)
    distill_manager = None
    teacher = None
    teacher_tokenizer = None

    if use_offline_logits:
        print("🚀 DISTILLATION: Usage of Pre-computed Logits ENABLED (Offline Mode)")
        print("   Teacher logits will be read from disk (no teacher VRAM load).")
        teacher_tokenizer = load_teacher_tokenizer()
        distill_manager = DistillationManager(cfg, teacher_tokenizer)
    else:
        teacher = TeacherBundle()
        teacher_tokenizer = teacher.tokenizer

    # CURRICULUM: Find stage datasets
    stage_info = [
        ("stage1", project_root / "datasets" / "stage1" / "stage1_data.jsonl"),
        ("stage2", project_root / "datasets" / "stage2" / "stage2_data.jsonl"),
        ("stage3", project_root / "datasets" / "stage3" / "stage3_data.jsonl"),
        ("stage4_soul", project_root / "datasets" / "stage4_soul" / "stage4_data.jsonl"),
        ("stage5_tools", project_root / "datasets" / "stage5_tools" / "stage5_data.jsonl"),
    ]
    stage_paths = [p for _, p in stage_info]

    # Fallback/Auto-Download logic
    offline_mode = os.getenv("TITAN_OFFLINE", "1") == "1"
    if not all(p.exists() for p in stage_paths):
        fallback_path = project_root / "datasets" / "training_data.jsonl"
        if offline_mode:
            if fallback_path.exists():
                stage_paths = [fallback_path]
                stage_info = [("fallback", fallback_path)]
                print("ℹ️ Offline mode: using local fallback dataset (datasets/training_data.jsonl).")
            else:
                raise FileNotFoundError(
                    "Offline mode is enabled (TITAN_OFFLINE=1) but required stage datasets are missing. "
                    "Provide local stage*.jsonl files or run data pipeline in online mode."
                )
        else:
            print("⚠️  Datasets not found. Launching Data Alchemy Engine...")
            import subprocess
            try:
                # Otomatik İndirme Tetikleyicisi
                alchemy_script = project_root / "scripts" / "data_pipeline.py"
                subprocess.check_call([sys.executable, str(alchemy_script)])
                print("✅ Data Alchemy Complete. Re-checking datasets...")

                # Tekrar kontrol et
                if not all(p.exists() for p in stage_paths):
                    # Belki sadece fallback oluştu, onu kontrol et
                    if fallback_path.exists():
                        stage_paths = [fallback_path]
                        stage_info = [("fallback", fallback_path)]
                        print("ℹ️ Using fallback dataset after Alchemy.")
                    else:
                        raise FileNotFoundError("Data Alchemy ran but datasets are still missing!")
            except Exception as e:
                print(f"❌ Data Pipeline Failed: {e}")
                sys.exit(1)

    # Curriculum dataset
    if use_offline_logits:
        stage_names = [name for name, _ in stage_info]
        if not distill_manager.has_precomputed_logits(stage_names):
            if offline_mode:
                raise RuntimeError(
                    "Precomputed logits are missing while TITAN_OFFLINE=1. "
                    "Offline mode cannot fall back to online teacher."
                )
            print("⚠️ Precomputed logits not found for all stages. Falling back to ONLINE teacher.")
            teacher = TeacherBundle()
            teacher_tokenizer = teacher.tokenizer
            use_offline_logits = False
        else:
            # TR: Logit'lerle senkron dataset (num_workers=0 zorunlu)
            # EN: Logit-synced dataset (requires num_workers=0)
            curriculum_ds = PrecomputedCurriculumDataset(stage_info, cfg.max_seq_len, teacher_tokenizer, distill_manager)

    if not use_offline_logits:
        curriculum_ds = CurriculumDataset(stage_paths, cfg.max_seq_len, teacher_tokenizer, current_stage=1)

    num_workers = getattr(cfg, "dataloader_num_workers", 4)
    prefetch_factor = getattr(cfg, "dataloader_prefetch_factor", 2)
    if use_offline_logits:
        num_workers = 0  # deterministic alignment with precomputed logits
        prefetch_factor = None
    dl = DataLoader(
        curriculum_ds,
        batch_size=cfg.micro_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available()
    )

    # [PRO] Validation Setup
    val_path = project_root / "datasets" / "validation.jsonl"
    if val_path.exists():
        print(f"🔍 Validation Dataset Found: {val_path}")
        # [PRO] Use Deterministic Dataset
        val_ds = ValidationJsonlDataset(val_path, cfg.max_seq_len, teacher_tokenizer)
        # num_workers=0 ensures main process does sequential read
        val_dl = DataLoader(val_ds, batch_size=cfg.micro_batch_size, collate_fn=collate_fn, num_workers=0)
    else:
        print("⚠️ Validation set not found (datasets/validation.jsonl). Using training stream for sanity check.")
        val_dl = dl # Fallback

    # V26.3 CRITICAL FIX: Epoch Mode Calculation MOVED UP
    # Must be done BEFORE Scheduler initialization!
    # [PRO] Only override max_steps if explicitly requested via EPOCH_MODE or if undefined
    # This allows test scripts to set max_steps=2 without interference.
    if getattr(cfg, "epoch_mode", True) and (not hasattr(cfg, "max_steps") or cfg.max_steps is None or cfg.max_steps > 1000):
        # V24.0: 12M samples (sweet spot for 1.5B model)
        NUM_EPOCHS = 3
        TOTAL_SAMPLES = 12_000_000  # Sweet spot: ~6B tokens
        cfg.max_steps = int((TOTAL_SAMPLES / (cfg.micro_batch_size * cfg.grad_accum_steps)) * NUM_EPOCHS)
        print(f"🔄 EPOCH MODE ACTIVATED: {NUM_EPOCHS} Epochs ({TOTAL_SAMPLES/1e6:.1f}M Samples) -> Max Steps: {cfg.max_steps}")
    else:
        print(f"ℹ️  Epoch Mode Skipped. Using provided max_steps: {cfg.max_steps}")

    cfg.vocab_size = teacher_tokenizer.vocab_size
    if accelerator.is_main_process:
        validate_config(cfg, stage="post")
    
    student = MertFormer()
    # Note: .to(device) is handled by accelerator.prepare, but explicit move is fine before prepare
    student.to(student_device)

    # -------------------------------------------------------------------------
    # V27.0: MAXIMUM PERFORMANCE - torch.compile with max-autotune
    # -------------------------------------------------------------------------
    use_compile = getattr(cfg, "use_torch_compile", True)
    compile_mode = getattr(cfg, "torch_compile_mode", "max-autotune")
    
    if torch.cuda.is_available() and hasattr(torch, 'compile') and use_compile:
        try:
            print(f"⚡ torch.compile ACTIVE - Mode: {compile_mode}")
            print("   First forward pass will take 10-15 minutes (kernel optimization)")
            print("   Subsequent steps will be 15-20% faster!")
            
            # max-autotune: Most aggressive optimization (best for training)
            # reduce-overhead: Balanced (good for inference)
            # default: Conservative (fallback)
            student = torch.compile(student, mode=compile_mode, fullgraph=False)
            print("✅ torch.compile successful!")
        except Exception as e:
            print(f"⚠️  torch.compile failed: {e}")
            print("   Falling back to eager mode (no performance loss, just no speedup)")
    else:
        if not torch.cuda.is_available():
            print("⚠️  torch.compile skipped (No CUDA)")
        elif not hasattr(torch, 'compile'):
            print("⚠️  torch.compile skipped (PyTorch < 2.0)")
        else:
            print("ℹ️  torch.compile disabled in config")


    # Apply freeze policy if enabled
    freeze_core = getattr(cfg, "freeze_core_layers", False)
    apply_freeze_policy(student, freeze_core)

    student.train()
    preflight_param_report(student)

    # GROKKING STRATEGY: Differential Learning Rates
    # Router ve Tau parametrelerine yüksek LR, düşük Decay vererek 
    # "yönlendirme mantığını" hızlı çözmesini (Grokking) sağlıyoruz.
    router_params = []
    body_params = []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if "router" in n or "tau" in n or "shared_gate" in n:
            router_params.append(p)
        else:
            body_params.append(p)

    print(f"🧠 Grokking Setup: {len(router_params)} Router Params | {len(body_params)} Body Params")

    opt = torch.optim.AdamW([
            {'params': body_params, 'lr': cfg.learning_rate, 'weight_decay': cfg.weight_decay},
            # Stabilization: Reduced LR multiplier and added weight decay for Router parameters
            {'params': router_params, 'lr': cfg.learning_rate * 1.5, 'weight_decay': 1e-4} 
        ])

    # V21.0 FIX: WSD Scheduler (Warmup-Stable-Decay) moved to global scope

    scheduler = get_wsd_schedule(
        opt,
        num_warmup_steps=int(cfg.max_steps * 0.1), # 10% Warmup
        num_training_steps=cfg.max_steps,
        min_lr_ratio=0.01
    )

    # [ACCELERATE PREPARE]
    # DDP Magic happens here!
    student, opt, dl, val_dl, scheduler = accelerator.prepare(
        student, opt, dl, val_dl, scheduler
    )
    
    if accelerator.is_main_process:
        print("✅ Accelerate Preparation Complete")

    # Mixed Precision handled by Accelerate (no manual scaler needed)
    scaler = None

    # Early Stopping & Monitoring
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = getattr(cfg, "early_stop_patience", 5)
    val_check_interval = getattr(cfg, "val_check_interval", 1000)

    # Gradient Norm Monitoring
    max_grad_norm_seen = 0.0
    grad_norm_history = []
    loss_history = [] # V26.1 FIX: Track Loss History for Signal-Based Curriculum
    grad_norm_collapse_threshold = 0.01  # If grad norm < 0.01, model stopped learning
    
    # V26.1 SAFEGUARD: Liquid Auto-Freeze State
    liquid_frozen_until = 0 # Step count until Liquid is unfrozen
    liquid_spike_counter = 0 # V26.11 SAFEGUARD: 3-Strike Rule

    # Curriculum Stage Tracking
    stage1_end = int(cfg.max_steps * 0.25)  # 0-25% (Logic)
    stage2_end = int(cfg.max_steps * 0.55)  # 25-55% (Knowledge)
    stage3_end = int(cfg.max_steps * 0.75)  # 55-75% (Language)
    stage4_end = int(cfg.max_steps * 0.85)  # 75-85% (Soul)
    # Stage 5 is 85-100% (Tools)
    current_curriculum_stage = 1

    global_step = 0
    early_stopped = False
    accum_loss = 0.0
    accum_count = 0  # [FIX] Counter for proper avg_loss calculation
    micro_step = 0
    start_time = time.time()
    tokens_processed = 0

    if accelerator.is_main_process:
        print(f"🚀 TITAN ONYX STORM TRAINING STARTING...")
        print(f"📊 Max Steps: {cfg.max_steps}")
        print(f"📚 Curriculum Stages:")
        print(f"   - Stage 1 (Logic):     Steps 0-{stage1_end} (25%)")
        print(f"   - Stage 2 (Knowledge): Steps {stage1_end}-{stage2_end} (30%)")
        print(f"   - Stage 3 (Language):  Steps {stage2_end}-{stage3_end} (20%)")
        print(f"   - Stage 4 (Soul):      Steps {stage3_end}-{stage4_end} (10%)")
        print(f"   - Stage 5 (Tools):     Steps {stage4_end}-{cfg.max_steps} (15%)")
        print(f"📊 Early Stopping: Patience={early_stop_patience} | Val Check: Every {val_check_interval} steps")

    try:
        dataloader_iter = iter(dl)
        while global_step < cfg.max_steps:
            # ---------------------------------------------------------------------
            # V26.1 FIX: Avg Loss Calculation for Safety
            # ---------------------------------------------------------------------
            if len(grad_norm_history) > 0: # Proxy for "at least 1 step done"
                 # (loss_history update happens at end of loop, so we check previous stats)
                 pass

            # ---------------------------------------------------------------------
            # V26.0 INTELLIGENT PILOT: Signal-Based Curriculum + Time Fallback
            # ---------------------------------------------------------------------
            
            # Use rolling average loss if available
            current_avg_loss = sum(loss_history[-100:]) / len(loss_history[-100:]) if loss_history else 999.0
            
            # Stage 1 -> 2
            if current_curriculum_stage == 1:
                # Loss < 1.5 signal OR Time Limit
                if (current_avg_loss < 1.5 and global_step > stage1_end * 0.5) or (global_step > stage1_end):
                    new_stage = 2
                else:
                    new_stage = 1
            # Stage 2 -> 3
            elif current_curriculum_stage == 2:
                # Loss < 1.2 signal OR Time Limit
                if (current_avg_loss < 1.2 and global_step > stage2_end * 0.5) or (global_step > stage2_end):
                    new_stage = 3
                else:
                    new_stage = 2
            # Stage 3 -> 4
            elif current_curriculum_stage == 3:
                # Loss < 1.0 signal OR Time Limit
                if (current_avg_loss < 1.0 and global_step > stage3_end * 0.5) or (global_step > stage3_end):
                    new_stage = 4
                else:
                    new_stage = 3
            # Stage 4 -> 5
            elif current_curriculum_stage == 4:
                # Loss < 0.9 signal OR Time Limit
                if (current_avg_loss < 0.9 and global_step > stage4_end * 0.5) or (global_step > stage4_end):
                    new_stage = 5
                else:
                    new_stage = 4
            else:
                new_stage = 5

            if new_stage != current_curriculum_stage:
                current_curriculum_stage = new_stage
                curriculum_ds.set_stage(new_stage)
                if accelerator.is_main_process:
                    print(f"📚 Curriculum Stage Updated: Stage {new_stage} (Signal: {current_avg_loss:.2f})")
            # ---------------------------------------------------------------------

            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dl)
                batch = next(dataloader_iter)

            t_logits = None
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                input_ids, labels, t_logits = batch
            else:
                input_ids, labels = batch

            # Accelerate handles device placement
            # input_ids = input_ids.to(student_device)
            # labels = labels.to(student_device)

            # ---------------------------------------------------------------------
            # V25.1 SAFEGUARD: Liquid Warmup (Freeze Early Steps)
            # ---------------------------------------------------------------------
            # ---------------------------------------------------------------------
            # V25.1 SAFEGUARD: Liquid Warmup (Freeze Early Steps)
            # V26.1 UPDATE: Liquid Re-Freeze Logic
            # ---------------------------------------------------------------------
            
            # 1. Warmup Phase
            if global_step < cfg.liquid_warmup_steps:
                 # Ensure Liquid is frozen
                 for n, p in student.named_parameters():
                     if "tau" in n or "liquid" in n:
                         p.requires_grad = False
                 
                 # V26.5 SAFEGUARD: Clear gradients for frozen params (Safer than LR=0)
                 for p in student.parameters():
                     if not p.requires_grad and p.grad is not None:
                         p.grad.detach_()
                         p.grad.zero_()
            elif global_step == cfg.liquid_warmup_steps:
                 # Unfreeze Logic - ONCE
                 if accelerator.is_main_process:
                    print(f"🔓 UNFREEZING LIQUID LAYERS at Step {global_step}!")
                 for n, p in student.named_parameters():
                     if "tau" in n or "liquid" in n:
                         p.requires_grad = True
                 
                 # Build 27 polish: Rebuild Optimizer to sync groups
                 # Note: With Accelerate, optimizer rebuilding is tricky. 
                 # We simply update params requires_grad, Accelerate/AdamW should handle it mostly.
                 # Full rebuild requires re-wrap with Accelerate. Skipping for safety in DDP.
                 pass 
                         
            # 2. Emergency Cooldown Phase (Refreeze)
            elif liquid_frozen_until > 0:
                 if global_step < liquid_frozen_until:
                     # Keep frozen
                     for n, p in student.named_parameters():
                        if "tau" in n or "liquid" in n:
                            p.requires_grad = False
                     # V26.5 SAFEGUARD: Clear gradients for frozen params
                     for p in student.parameters():
                         if not p.requires_grad and p.grad is not None:
                             p.grad.detach_()
                             p.grad.zero_()
                             
                 elif global_step == liquid_frozen_until:
                     print(f"🧊 LIQUID COOLDOWN OVER. Unfreezing at {global_step}...")
                     for n, p in student.named_parameters():
                         if "tau" in n or "liquid" in n:
                             p.requires_grad = True
                      
                     # Build 27 polish: Rebuild Optimizer to sync groups
                     rebuild_optimizer(student, opt, cfg)
                      
                     liquid_frozen_until = 0
            # ---------------------------------------------------------------------
            # ---------------------------------------------------------------------

            # ---------------------------------------------------------------------
            # [CRITICAL FIX] Correct DDP Accumulation using Context Manager
            # ---------------------------------------------------------------------
            with accelerator.accumulate(student):
                # Mixed Precision context handled by accelerate.accumulate or autocast
                # We use autocast for extra safety
                with accelerator.autocast():
                    s_logits, aux_loss, _ = student(input_ids)

                    # Teacher logits: offline (precomputed) or online (teacher model)
                    if t_logits is None and teacher is not None and teacher.model is not None:
                        # Teacher is already on correct device via device_map fix
                        t_logits = teacher.get_logits(input_ids)

                    shift_logits = s_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    pad_id = teacher_tokenizer.pad_token_id if teacher_tokenizer and teacher_tokenizer.pad_token_id is not None else 0
                    loss_ce = F.cross_entropy(
                        shift_logits.view(-1, cfg.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=pad_id,
                        label_smoothing=getattr(cfg, "label_smoothing", 0.0)
                    )

                    loss_distill = 0.0
                    if t_logits is not None:
                        t_logits = t_logits.to(shift_logits.device)
                        shift_t_logits = t_logits[..., :-1, :].contiguous()
                        kd_mask = shift_labels != pad_id
                        loss_distill = kd_loss_safe(shift_logits, shift_t_logits, cfg.teacher_temp, mask=kd_mask)

                    # Dynamic Distillation Alpha
                    progress_pct = global_step / cfg.max_steps
                    start_alpha = getattr(cfg, "distill_alpha", 0.6)
                    
                    if progress_pct < 0.3:
                        dyn_alpha = start_alpha
                    elif progress_pct > 0.8:
                        dyn_alpha = 0.15
                    else:
                        slope = (0.15 - start_alpha) / (0.8 - 0.3)
                        dyn_alpha = start_alpha + slope * (progress_pct - 0.3)
                    
                    alpha = dyn_alpha if t_logits is not None else 0.0
                    aux_coef = getattr(cfg, "router_aux_loss_coef", 0.01)
                    total_loss = (1.0 - alpha) * loss_ce + alpha * loss_distill + aux_coef * aux_loss

                # NaN Check
                if not torch.isfinite(total_loss):
                    print(f"⚠️ NaN detected, skipping")
                    opt.zero_grad()
                    continue

                # Track micro-batch stats (for accurate averages)
                accum_loss += total_loss.item()
                accum_count += 1
                tokens_processed += input_ids.numel()

                # Backward (Accelerate handles scaling and division by accum steps AUTOMATICALLY)
                accelerator.backward(total_loss)

                # Optimizer Step (Only runs when accumulation is complete)
                if accelerator.sync_gradients:
                    # Clipping
                    grad_norm = accelerator.clip_grad_norm_(student.parameters(), cfg.grad_clip)

                    opt.step()
                    scheduler.step()
                    opt.zero_grad()

                    global_step += 1

                    # Update Stats
                    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                    max_grad_norm_seen = max(max_grad_norm_seen, grad_norm_val)
                    grad_norm_history.append(grad_norm_val)
                    loss_history.append(total_loss.item())  # V26.2 FIX: Use total_loss.item()
                    if len(grad_norm_history) > 100:
                        grad_norm_history.pop(0)
                    if len(loss_history) > 100:
                        loss_history.pop(0)

                    # GRADIENT NORM COLLAPSE DETECTION
                    if len(grad_norm_history) >= 10:
                        avg_recent = sum(grad_norm_history[-10:]) / 10
                        if avg_recent < grad_norm_collapse_threshold:
                            print(f"⚠️  WARNING: Gradient norm collapse detected! Avg: {avg_recent:.6f}")
                            print(f"   Model may have stopped learning. Consider adjusting learning rate.")

                    # Engine Overheating Protection
                    if grad_norm_val > 10.0:
                        print(f"⚠️  CRITICAL: Gradient norm {grad_norm_val:.2f} exceeds safety threshold! Reducing clip threshold.")
                        cfg.grad_clip = max(cfg.grad_clip * 0.7, 0.1)

                    # Log Metrics (Main Process)
                    if global_step % cfg.log_interval == 0 and logger and accelerator.is_main_process:
                        metrics = {
                            "loss": total_loss.item(),
                            "ce": loss_ce.item(),
                            "distill": loss_distill.item() if isinstance(loss_distill, torch.Tensor) else loss_distill,
                            "aux": aux_loss.item(),
                            "lr": scheduler.get_last_lr()[0],
                            "grad_norm": grad_norm_val,
                            "alpha": alpha,
                            "stage": current_curriculum_stage
                        }
                        metrics["global_step"] = global_step
                        logger.log_step(metrics)

                    # Logging only on Main Process
                    if global_step % cfg.log_interval == 0 and accelerator.is_main_process:
                        dt = time.time() - start_time
                        tok_s = tokens_processed / max(dt, 1e-6)
                        # [FIX] Use counter for proper averaging
                        avg_loss = accum_loss / max(1, accum_count)
                        lr_now = scheduler.get_last_lr()[0]

                        distill_val = loss_distill.item() if isinstance(loss_distill, torch.Tensor) else 0.0
                        aux_val = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0.0
                        avg_grad_norm = sum(grad_norm_history[-10:]) / len(grad_norm_history[-10:]) if grad_norm_history else 0.0

                        print(f"Step {global_step} | Stage {current_curriculum_stage} | Loss: {avg_loss:.4f} | "
                              f"Tok/s: {tok_s:.0f} | LR: {lr_now:.2e} | GradNorm: {avg_grad_norm:.3f}")

                        # V27.0: Periodic Safety Checks
                        if global_step % 1000 == 0:
                            # Disk space check
                            if not check_disk_space(min_gb=50):
                                print("   ⚠️  Consider cleaning old checkpoints!")

                            # GPU memory report
                            if torch.cuda.is_available():
                                alloc, res = get_gpu_memory_usage()
                                print(f"   📊 GPU Memory: {alloc:.1f} GB / {res:.1f} GB")

                        log_data = {
                            "step": global_step,
                            "curriculum_stage": current_curriculum_stage,
                            "loss": avg_loss,
                            "ce": loss_ce.item(),
                            "kd": distill_val,
                            "aux": aux_val,
                            "tok_s": tok_s,
                            "lr": lr_now,
                            "grad_norm": avg_grad_norm,
                            "max_grad_norm": max_grad_norm_seen
                        }

                        # [V26.6 TELEMETRY] MoE Router Health (log-interval only)
                        moe_loads = []
                        moe_entropies = []
                        moe_overflows = []
                        # Handle DDP/Compile wrappers
                        model_ref = student.module if hasattr(student, "module") else student
                        model_ref = model_ref._orig_mod if hasattr(model_ref, "_orig_mod") else model_ref

                        for m in model_ref.modules():
                            if hasattr(m, "get_expert_load"):
                                moe_loads.append(m.get_expert_load())
                            if hasattr(m, "get_router_entropy"):
                                moe_entropies.append(m.get_router_entropy())
                            if hasattr(m, "last_capacity_overflow_ratio"):
                                moe_overflows.append(m.last_capacity_overflow_ratio)

                        if moe_loads:
                            # Stack: [Layers, Experts]
                            loads = torch.stack(moe_loads)  # (L, E)
                            # Metrics
                            max_load = loads.max().item()  # Worst case imbalance
                            avg_std = loads.std(dim=1).mean().item()  # Overall balance score (lower is better)
                            avg_entropy = (
                                torch.stack(moe_entropies).mean().item()
                                if moe_entropies
                                else float("nan")
                            )
                            avg_overflow = (
                                torch.stack(moe_overflows).mean().item()
                                if moe_overflows
                                else 0.0
                            )
                            collapse_alarm = float(getattr(cfg, "router_alarm_threshold", 0.40))

                            log_data["moe_max_load"] = max_load
                            log_data["moe_avg_std"] = avg_std
                            if not math.isnan(avg_entropy):
                                log_data["moe_load_entropy"] = avg_entropy
                            log_data["moe_capacity_overflow"] = avg_overflow

                            entropy_txt = f"{avg_entropy:.3f}" if not math.isnan(avg_entropy) else "n/a"
                            print(
                                f"   🧠 MoE Health: MaxLoad={max_load:.2f} | "
                                f"Balance(std)={avg_std:.3f} | Entropy={entropy_txt} | "
                                f"Overflow={avg_overflow:.3f}"
                            )

                            if max_load > collapse_alarm:
                                print(
                                    f"   ⚠️  EARLY IMBALANCE ALERT: Max Load {max_load:.2f} "
                                    f"(alarm>{collapse_alarm:.2f})"
                                )

                            # Router Collapse Warning + ACTION
                            if max_load > 0.85 and getattr(cfg, "active_experts", 1) < getattr(cfg, "num_experts", 4):
                                print(f"⚠️  ROUTER COLLAPSE DETECTED: Max Load {max_load:.2f} (Target: {1.0/cfg.num_experts:.2f})")
                                print(f"   🔧 ACTION: Jitter boost activated via MoE module (see moe.py collapse_detected)")
                                log_data["router_collapse"] = True

                        if logger:
                            logger.log_step(log_data)

                        accum_loss = 0.0
                        accum_count = 0  # [FIX] Reset counter
                        tokens_processed = 0
                        start_time = time.time()

                    # Validation & Early Stopping
                    if global_step % val_check_interval == 0 and global_step > 0:
                        student.eval()
                        val_loss_local = 0.0
                        val_samples_local = 0
                        val_steps = 10
                        try:
                            # [PRO] Use dedicated validation dataloader if available
                            val_iter = iter(val_dl)
                            for _ in range(val_steps):
                                try:
                                    val_input_ids, val_labels = next(val_iter)
                                except StopIteration:
                                    val_iter = iter(val_dl)
                                    val_input_ids, val_labels = next(val_iter)

                                val_input_ids = val_input_ids.to(student_device)
                                val_labels = val_labels.to(student_device)

                                with torch.no_grad():
                                    val_logits, _, _ = student(val_input_ids, use_cache=False)
                                    val_shift_logits = val_logits[..., :-1, :].contiguous()
                                    val_shift_labels = val_labels[..., 1:].contiguous()
                                    val_batch_loss = F.cross_entropy(
                                        val_shift_logits.view(-1, cfg.vocab_size),
                                        val_shift_labels.view(-1),
                                        ignore_index=teacher_tokenizer.pad_token_id if teacher_tokenizer and teacher_tokenizer.pad_token_id is not None else 0
                                    )
                                    val_loss_local += val_batch_loss.item()
                                    val_samples_local += 1
                        except Exception as e:
                            print(f"⚠️  Validation error: {e}, skipping early stopping check")
                            val_loss_local = 0.0
                            val_samples_local = 0

                        # DDP-safe global validation aggregation (all ranks agree on one loss value).
                        val_stats = torch.tensor(
                            [float(val_loss_local), float(val_samples_local)],
                            device=student_device,
                            dtype=torch.float32,
                        )
                        val_stats = accelerator.reduce(val_stats, reduction="sum")
                        val_loss_sum_global = float(val_stats[0].item())
                        val_samples_global = int(val_stats[1].item())

                        should_stop = False
                        if accelerator.is_main_process:
                            if val_samples_global > 0:
                                val_loss = val_loss_sum_global / max(1, val_samples_global)
                                print(f"📊 Validation Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")

                                # EVAL-DRIVEN EARLY STOPPING
                                if val_loss < best_val_loss:
                                    best_val_loss = val_loss
                                    patience_counter = 0
                                    print("✅ New best validation loss! Saving checkpoint...")
                                    unwrapped_model = accelerator.unwrap_model(student)
                                    best_val_loss = save_checkpoint_smart(
                                        unwrapped_model,
                                        opt,
                                        scheduler,
                                        global_step,
                                        cfg,
                                        keep_last_n=3,
                                        val_loss=val_loss,
                                        best_val_loss=best_val_loss,
                                    )
                                else:
                                    patience_counter += 1
                                    print(f"⏳ No improvement ({patience_counter}/{early_stop_patience})")
                                    if patience_counter >= early_stop_patience:
                                        print(f"🛑 Early stopping triggered! Best val loss: {best_val_loss:.4f}")
                                        should_stop = True
                            else:
                                print("⚠️  Validation produced zero usable batches; skipping early stopping check.")

                        stop_tensor = torch.tensor(
                            1 if should_stop else 0,
                            device=student_device,
                            dtype=torch.int64,
                        )
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            torch.distributed.broadcast(stop_tensor, src=0)
                        should_stop = bool(stop_tensor.item())

                        if should_stop:
                            early_stopped = True
                            student.train()
                            break

                        student.train()

                    if global_step % cfg.save_interval == 0 and accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(student)
                        best_val_loss = save_checkpoint_smart(unwrapped_model, opt, scheduler, global_step, cfg, keep_last_n=3, val_loss=None, best_val_loss=best_val_loss)

                micro_step += 1

        if accelerator.is_main_process:
            if early_stopped:
                if logger:
                    logger.finalize("early_stopped", extra={"best_val_loss": best_val_loss})
            else:
                if logger:
                    logger.finalize("completed")
                unwrapped_model = accelerator.unwrap_model(student)
                export_to_onnx(unwrapped_model, save_dir, cfg.model_name, student_device)

    except KeyboardInterrupt:
        print("\n🛑 Durduruldu. Kaydediliyor...")
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(student)
            best_val_loss = save_checkpoint_smart(unwrapped_model, opt, scheduler, global_step, cfg, keep_last_n=3, val_loss=None, best_val_loss=best_val_loss)
            if 'logger' in locals() and logger: logger.finalize("aborted")

    except Exception as e:
        print(f"\n💥 HATA: {e}")
        if 'accelerator' in locals() and accelerator.is_main_process and 'logger' in locals() and logger:
            logger.finalize("failed", extra={"error": str(e)})
        raise


if __name__ == "__main__":
    train()
