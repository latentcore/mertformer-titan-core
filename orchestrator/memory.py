"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - GOD MEMORY MODULE
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

import json
import time
import pathlib
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F

# TR: Yerel import için forward reference / EN: Forward reference for local import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .sense_engine import SenseEngine


# TR: Sabitler / EN: Constants
MAX_MEMORY_HITS = 20


class GodMemory:
    """
    TR: Kategorik Vektör Hafıza.
    EN: Categorical Vector Memory.
    TR: Quantized vektörlerle RAM-efficient depolama.
    EN: RAM-efficient storage with quantized vectors.
    """
    
    def __init__(self, path: pathlib.Path, senses: "SenseEngine"):
        self.path = path
        self.senses = senses
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """TR: Hafızayı dosyadan yükle. / EN: Load memory from file."""
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            self.cache.append(json.loads(line))
                        except Exception as e:
                            print(f"⚠️ Hafıza kaydı bozuk: {e}")
                print(f"💾 Hafıza yüklendi: {len(self.cache)} kayıt.")
            except Exception as e:
                print(f"⚠️ Hafıza yüklenemedi: {e}")

    def save(self, role: str, text: str, category: str = "GENERAL", source: str = "TEXT") -> None:
        """
        TR: Metni vektöre çevirir ve GodMemory'e kaydeder.
        EN: Converts text to vector and saves to GodMemory.
        TR: RAM tasarrufu için int8 quantization.
        EN: int8 quantization for RAM savings.
        """
        raw_vec = self.senses.encode_text(text)
        vec_tensor = torch.tensor(raw_vec, dtype=torch.float32)

        if vec_tensor.numel() == 0:
            q_list: List[int] = []
            scale = 1.0
        else:
            max_abs = float(vec_tensor.abs().max().item())
            if max_abs < 1e-8:
                scale = 1.0
                q_tensor = torch.zeros_like(vec_tensor, dtype=torch.int8)
            else:
                scale = max_abs / 127.0
                q_tensor = torch.clamp(
                    torch.round(vec_tensor / scale), -127, 127
                ).to(torch.int8)
            q_list = [int(v) for v in q_tensor.view(-1).tolist()]

        rec = {
            "role": role,
            "text": text,
            "category": category,
            "source": source,
            "vec_q": q_list,
            "vec_scale": scale,
            "ts": time.time(),
        }
        self.cache.append(rec)
        
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ Hafıza yazma hatası: {e}")

    def recall_raw(self, query: str, top_k: int = MAX_MEMORY_HITS) -> List[Dict[str, Any]]:
        """TR: Sorguya en yakın kayıtları döndürür. / EN: Returns records closest to query."""
        if not self.cache:
            return []

        # TR: Kategori tahmini / EN: Category prediction
        q_low = query.lower()
        if any(w in q_low for w in ["kod", "python", "hata", "traceback"]):
            cat = "CODE"
        elif "mert" in q_low:
            cat = "PERSONAL"
        else:
            cat = "GENERAL"

        q_vec = torch.tensor(self.senses.encode_text(query), dtype=torch.float32)

        candidates = [
            c for c in self.cache if c.get("category") in (cat, "GENERAL")
        ]
        if not candidates:
            candidates = self.cache

        if not candidates:
            return []

        vecs: List[torch.Tensor] = []
        dim = q_vec.numel()
        zero_vec = torch.zeros(dim, dtype=torch.float32)

        for c in candidates:
            # TR: Yeni format: vec_q + vec_scale (int8 quantize)
            # EN: New format: vec_q + vec_scale (int8 quantize)
            if "vec_q" in c:
                q_list = c.get("vec_q") or []
                scale = float(c.get("vec_scale", 1.0))
                if q_list:
                    q_tensor = torch.tensor(q_list, dtype=torch.float32)
                    if q_tensor.numel() != dim:
                        vecs.append(zero_vec)
                    else:
                        vecs.append(q_tensor * scale)
                else:
                    vecs.append(zero_vec)
            # TR: Eski format: vec (float list) / EN: Old format: vec (float list)
            else:
                base = c.get("vec", None)
                if isinstance(base, list) and base:
                    v = torch.tensor(base, dtype=torch.float32)
                    if v.numel() != dim:
                        vecs.append(zero_vec)
                    else:
                        vecs.append(v)
                else:
                    vecs.append(zero_vec)

        cand_vecs = torch.stack(vecs, dim=0)

        with torch.no_grad():
            q_norm = F.normalize(q_vec.unsqueeze(0), p=2, dim=1)
            c_norm = F.normalize(cand_vecs, p=2, dim=1)
            scores = torch.mv(c_norm, q_norm.squeeze(0))

        k = min(top_k, len(candidates))
        top_indices = torch.topk(scores, k=k).indices.tolist()
        return [candidates[i] for i in top_indices]

    def recall(self, query: str, top_k: int = MAX_MEMORY_HITS) -> str:
        """TR: Sorguya en yakın kayıtları string olarak döndürür. / EN: Returns records closest to query as string."""
        hits = self.recall_raw(query, top_k=top_k)
        if not hits:
            return ""

        lines: List[str] = []
        for h in hits:
            short = h.get("text", "")
            if len(short) > 200:
                short = short[:200] + "..."
            lines.append(
                f"- [{h.get('category')}/{h.get('source')}] "
                f"{h.get('role')}: {short}"
            )
        return "\n".join(lines)

    def build_context_block(self, query: str, top_k: int = MAX_MEMORY_HITS) -> str:
        """TR: V15.5 tarzı [MEMORY_CONTEXT_START] bloğu üretir. / EN: Produces V15.5 style [MEMORY_CONTEXT_START] block."""
        hits = self.recall_raw(query, top_k=top_k)
        if not hits:
            return ""
        
        ctx_lines: List[str] = ["[MEMORY_CONTEXT_START]"]
        for h in hits:
            ts_full = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(h["ts"]))
            role = h.get("role", "").upper()
            text = h.get("text", "")
            ctx_lines.append(f"- ({ts_full}) [{role}] {text}")
        
        ctx_lines.append("[MEMORY_CONTEXT_END]")
        return "\n".join(ctx_lines)

    def last_messages(self, limit: int = 10, category: Optional[str] = None) -> str:
        """TR: Son n kaydı döndürür. / EN: Returns last n records."""
        if not self.cache:
            return ""

        records = self.cache
        if category:
            records = [
                r for r in records
                if r.get("category") in (category, "GENERAL")
            ]

        records = sorted(records, key=lambda r: r.get("ts", 0.0))
        tail = records[-limit:]

        lines: List[str] = []
        for rec in tail:
            ts = rec.get("ts", time.time())
            ts_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
            role = rec.get("role", "unknown").upper()
            text = rec.get("text", "")
            if len(text) > 220:
                text = text[:220] + "..."
            cat = rec.get("category", "GENERAL")
            lines.append(f"- ({ts_str}) [{cat}] {role}: {text}")
        return "\n".join(lines)


class DocChunk:
    """TR: Doküman parçası. / EN: Document chunk."""
    
    def __init__(self, doc_id: str, source: str, text: str, vec: List[float]):
        self.doc_id = doc_id
        self.source = source
        self.text = text
        self.vec = vec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "text": self.text,
            "vec": self.vec,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DocChunk":
        return DocChunk(
            doc_id=d["doc_id"],
            source=d["source"],
            text=d["text"],
            vec=d["vec"],
        )


class DocIndexer:
    """TR: Doküman indeksleyici. / EN: Document indexer."""
    
    def __init__(self, doc_dir: pathlib.Path, vector_file: pathlib.Path, senses: "SenseEngine"):
        self.doc_dir = doc_dir
        self.vector_file = vector_file
        self.senses = senses
        self.doc_dir.mkdir(parents=True, exist_ok=True)
        self.vector_file.parent.mkdir(parents=True, exist_ok=True)
        self.chunks: List[DocChunk] = []

    def _iter_documents(self) -> List[pathlib.Path]:
        if not self.doc_dir.exists():
            return []
        files: List[pathlib.Path] = []
        for ext in ("*.txt", "*.md"):
            files.extend(self.doc_dir.rglob(ext))
        return files

    def build_index(self, max_chunk_chars: int = 800) -> None:
        """TR: Dokümanları indeksle. / EN: Index documents."""
        self.chunks = []
        doc_files = self._iter_documents()
        
        for path in doc_files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            
            text = text.strip()
            if not text:
                continue

            idx = 0
            chunk_id = 0
            while idx < len(text):
                chunk_text = text[idx: idx + max_chunk_chars]
                idx += max_chunk_chars
                vec = self.senses.encode_text(chunk_text)
                doc_id = f"{path.name}::chunk{chunk_id}"
                chunk_id += 1
                self.chunks.append(DocChunk(doc_id, str(path), chunk_text, vec))

        data = [c.to_dict() for c in self.chunks]
        with self.vector_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"📚 Doküman indeksi oluşturuldu: {len(self.chunks)} chunk")

    def load_index(self) -> None:
        """TR: Varolan indeksi yükle. / EN: Load existing index."""
        if not self.vector_file.exists():
            self.chunks = []
            return
        with self.vector_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = [DocChunk.from_dict(d) for d in data]

    def ready(self) -> bool:
        return bool(self.chunks)


class RAGEngine:
    """TR: Retrieval Augmented Generation Motoru. / EN: Retrieval Augmented Generation Engine."""
    
    def __init__(self, memory: GodMemory, indexer: DocIndexer, senses: "SenseEngine"):
        self.memory = memory
        self.indexer = indexer
        self.senses = senses

    def ensure_index(self) -> None:
        if not self.indexer.vector_file.exists():
            self.indexer.build_index()
        else:
            self.indexer.load_index()

    def search_docs(self, query: str, top_k: int = 5) -> List[DocChunk]:
        """TR: Dokümanlardan ilgili parçaları bul. / EN: Find relevant chunks from documents."""
        if not self.indexer.chunks:
            self.ensure_index()
        
        if not self.indexer.chunks:
            return []

        q_vec = torch.tensor(self.senses.encode_text(query), dtype=torch.float32)
        
        vecs = torch.tensor([c.vec for c in self.indexer.chunks], dtype=torch.float32)
        
        with torch.no_grad():
            q_norm = F.normalize(q_vec.unsqueeze(0), p=2, dim=1)
            v_norm = F.normalize(vecs, p=2, dim=1)
            scores = torch.mv(v_norm, q_norm.squeeze(0))
        
        k = min(top_k, len(self.indexer.chunks))
        top_indices = torch.topk(scores, k=k).indices.tolist()
        
        return [self.indexer.chunks[i] for i in top_indices]

    def hybrid_search(self, query: str, memory_k: int = 5, doc_k: int = 3) -> str:
        """TR: Hafıza + Doküman birleşik arama. / EN: Memory + Document hybrid search."""
        memory_context = self.memory.recall(query, top_k=memory_k)
        doc_hits = self.search_docs(query, top_k=doc_k)
        
        doc_context = ""
        if doc_hits:
            doc_lines = ["[DOC CONTEXT]"]
            for hit in doc_hits:
                doc_lines.append(f"- {hit.doc_id}: {hit.text[:300]}...")
            doc_lines.append("[/DOC CONTEXT]")
            doc_context = "\n".join(doc_lines)
        
        return f"{memory_context}\n\n{doc_context}".strip()
