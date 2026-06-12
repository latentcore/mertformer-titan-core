"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - AGI ORCHESTRATOR CORE
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

import sys
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

# Local imports
from .paths import AGIPaths
from .hardware import HardwareSense
from .web_sense import WebSense
from .audio_sense import AudioSense
from .sense_engine import SenseEngine
from .memory import GodMemory, DocIndexer, RAGEngine
from .swarm_runtime import SwarmRuntime
from .self_improvement_guard import SelfImprovementGuard
from .alignment_contracts import AlignmentContracts
from .compute_orchestrator import ComputeOrchestrator
from .verifier import SwarmVerifier

# MertFormer import - with fallback mechanism
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

# Logging
logger = logging.getLogger("TitanOrchestrator")


@dataclass(frozen=True)
class EpisodeBudget:
    """
    Runtime budget for TRIAD-OMEGA goal episodes.
    """
    max_iterations: int = 3
    max_tools: int = 2
    min_gate_confidence: float = 0.5
    max_uncertainty: float = 0.65
    allow_self_improvement: bool = False


@dataclass
class EpisodeResult:
    """
    Structured output for run_goal_episode.
    """
    goal: str
    pass_gate: bool
    final_response: str
    confidence: float
    uncertainty: float
    strategy: str
    tools_used: List[str] = field(default_factory=list)
    loops: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "pass_gate": self.pass_gate,
            "final_response": self.final_response,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "strategy": self.strategy,
            "tools_used": self.tools_used,
            "loops": self.loops,
            "notes": self.notes,
            "metadata": self.metadata,
        }


class MertFormerOrchestrator:
    """
    MertFormer Titan AGI Orchestrator.
    Combines all sense modules, memory and model inference.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        load_model: bool = True,
        enable_voice: bool = False,
    ) -> None:
        """
        Args:
            device: Compute device (None = auto)
            load_model: Load MertFormer model
            enable_voice: Enable voice response (TTS)
        """
        # Device selection
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
        
        # Create directories
        AGIPaths.ensure_dirs()
        
        # Initialize sense modules
        self.hardware = HardwareSense()
        self.web = WebSense()
        self.audio = AudioSense()
        self.senses = SenseEngine(device=self.device)
        
        # Memory and RAG
        self.memory = GodMemory(AGIPaths.MEMORY_FILE, self.senses)
        self.doc_indexer = DocIndexer(AGIPaths.DOC_DIR, AGIPaths.VECTOR_FILE, self.senses)
        self.rag = RAGEngine(self.memory, self.doc_indexer, self.senses)
        self.self_improvement_guard = SelfImprovementGuard()
        self.alignment_contracts = AlignmentContracts()
        self.compute_orchestrator = ComputeOrchestrator()
        
        # Model
        self.model = None
        self.tokenizer = None
        self.swarm = SwarmRuntime(generate_fn=self._swarm_generate_callback)
        self.verifier = SwarmVerifier()
        
        if load_model and MERTFORMER_AVAILABLE:
            self._load_model()
        
        # ================================================================
        # AGI COGNITIVE ARCHITECTURE
        # ================================================================
        from .reasoning_engine import ReasoningEngine
        from .tool_executor import ToolExecutor
        from .self_audit import SelfAuditor
        from .experience_store import ExperienceStore
        from .cognitive_loop import CognitiveLoop
        from .cognitive import WorldModel
        
        self.world_model = WorldModel(sense_engine=self.senses)
        self.reasoning = ReasoningEngine(generate_fn=self._swarm_generate_callback)
        self.tool_executor = ToolExecutor(
            memory=self.memory,
            web_sense=self.web,
            sense_engine=self.senses,
        )
        self.self_auditor = SelfAuditor(alignment_contracts=self.alignment_contracts)
        self.experience_store = ExperienceStore(
            store_path=AGIPaths.DATA_DIR / "experiences.jsonl",
            sense_engine=self.senses,
        )
        self.cognitive_loop = CognitiveLoop(
            generate_fn=self._swarm_generate_callback,
            reasoning_engine=self.reasoning,
            tool_executor=self.tool_executor,
            self_auditor=self.self_auditor,
            experience_store=self.experience_store,
            memory=self.memory,
            world_model=self.world_model,
        )
        
        print(f"✅ Orchestrator hazır! (AGI Cognitive Loop aktif)")

    def think(self, task: str, max_iterations: int = 5) -> dict:
        """
        AGI-style cognitive processing - Perceive -> Think -> Act -> Reflect.
        """
        result = self.cognitive_loop.run(task, max_iterations=max_iterations)
        return {
            "response": result.final_response,
            "strategy": result.strategy_used,
            "iterations": result.total_iterations,
            "confidence": result.confidence,
            "score": result.outcome_score,
            "tools_used": result.tools_used,
            "time_ms": result.total_time_ms,
        }

    @staticmethod
    def _default_tool_params(tool_id: str, goal: str, candidate_conclusion: str) -> Dict[str, Any]:
        if tool_id == "tool.calculate":
            return {"expression": "2 + 2"}
        if tool_id == "tool.verify_consistency":
            return {"text": candidate_conclusion, "reference": goal}
        if tool_id == "tool.search_local_docs":
            return {"query": goal, "top_k": 5}
        if tool_id == "tool.recall":
            return {"query": goal, "top_k": 5}
        if tool_id == "tool.memorize":
            return {"text": f"Goal episode: {goal}\nAnswer: {candidate_conclusion}", "category": "EPISODE"}
        return {"query": goal}

    def run_goal_episode(self, goal: str, budget: EpisodeBudget) -> EpisodeResult:
        """
        TRIAD-OMEGA episode:
          1) Hypothesis loop
          2) World loop
          3) Action loop
          4) Verifier loop
          5) Improvement loop
        """
        if not isinstance(budget, EpisodeBudget):
            budget = EpisodeBudget()

        notes: List[str] = []
        loops: Dict[str, Any] = {}
        tools_used: List[str] = []
        trace: List[Dict[str, Any]] = [{"stage": "goal", "output": goal}]

        # 1) Hypothesis Loop
        hypothesis_count = min(3, max(1, int(budget.max_iterations)))
        hypothesis_set = self.reasoning.generate_hypotheses(
            goal,
            strategy="auto",
            max_candidates=hypothesis_count,
        )
        best = hypothesis_set.best()
        loops["hypothesis"] = hypothesis_set.to_dict()
        trace.append(
            {
                "stage": "hypothesis",
                "output": best.conclusion,
                "strategy": best.strategy,
                "confidence": best.confidence,
            }
        )

        # 2) World Loop
        world_prediction = ""
        if self.world_model is not None:
            try:
                focus_entity = goal.split()[0] if goal.split() else "goal"
                focus_action = best.action_plan[0] if best.action_plan else "analyze"
                world_prediction = self.world_model.predict_next_state(focus_entity, focus_action)
            except Exception as exc:
                notes.append(f"world_loop_error:{exc}")
        loops["world"] = {"prediction": world_prediction}
        if world_prediction:
            trace.append({"stage": "world", "output": world_prediction})

        # 3) Action Loop (tool orchestration + memory write)
        tool_results: List[Dict[str, Any]] = []
        planned_tools = list(best.tool_calls)
        if self.memory is not None and "tool.memorize" not in planned_tools:
            planned_tools.append("tool.memorize")
        if not planned_tools:
            planned_tools = ["tool.verify_consistency"]

        for tool_id in planned_tools[: max(0, int(budget.max_tools))]:
            params = self._default_tool_params(tool_id, goal, best.conclusion)
            tool_res = self.tool_executor.execute(tool_id, params)
            tool_results.append(
                {
                    "tool_id": tool_id,
                    "success": tool_res.success,
                    "output": tool_res.output,
                    "error": tool_res.error,
                    "execution_time_ms": tool_res.execution_time_ms,
                    "governance_check": tool_res.governance_check,
                }
            )
            if tool_res.success:
                tools_used.append(tool_id)
            trace.append(
                {
                    "stage": "tool",
                    "tool_id": tool_id,
                    "output": tool_res.output or tool_res.error or "",
                    "blocked": (not tool_res.success) and (not bool(tool_res.governance_check)),
                }
            )

        final_response = best.conclusion
        successful_outputs = [r["output"] for r in tool_results if r["success"] and r.get("output")]
        if successful_outputs:
            final_response = f"{best.conclusion}\n\n[tool-context]\n{successful_outputs[0]}"
        loops["action"] = {"tool_results": tool_results, "final_response": final_response}
        trace.append({"stage": "response", "output": final_response})

        if self.memory is not None:
            try:
                self.memory.save(
                    "assistant",
                    final_response,
                    category="EPISODE",
                    source="TRIAD_OMEGA",
                )
                trace.append({"stage": "memory", "output": "episode_saved"})
            except Exception as exc:
                notes.append(f"memory_write_failed:{exc}")
                trace.append({"stage": "memory", "output": "episode_save_failed", "blocked": True})

        # 4) Verifier Loop (process + safety + uncertainty gate)
        gate = self.verifier.verify_episode(trace)
        safety_score = self.self_auditor.check_safety(final_response)
        if not safety_score.is_safe and gate.safety_pass:
            notes.append("self_audit_safety_block")

        pass_gate = (
            gate.pass_gate
            and gate.confidence >= float(budget.min_gate_confidence)
            and gate.uncertainty <= float(budget.max_uncertainty)
            and safety_score.is_safe
        )
        loops["verifier"] = {
            "pass_gate": pass_gate,
            "raw_pass_gate": gate.pass_gate,
            "confidence": gate.confidence,
            "uncertainty": gate.uncertainty,
            "consistency": gate.consistency,
            "safety_pass": gate.safety_pass and safety_score.is_safe,
            "notes": list(gate.notes),
            "gate_scores": gate.gate_scores,
        }

        # 5) Improvement Loop (guarded by metric gate)
        improvement_payload: Dict[str, Any] = {"applied": False, "reason": "disabled"}
        if bool(budget.allow_self_improvement):
            proposals = self.self_improvement_guard.propose(
                {
                    "health_score": gate.confidence,
                    "failure_budget_signal": max(0.0, gate.uncertainty),
                }
            )
            proposal = proposals[0]
            evaluation = {
                "delta_benchmark": 0.01 if pass_gate else -0.01,
                "delta_safety": 0.0 if safety_score.is_safe else -0.1,
                "cost_within_budget": True,
            }
            apply_result = self.self_improvement_guard.apply_if_safe(
                proposal,
                current_state={"goal": goal, "strategy": best.strategy},
                evaluation=evaluation,
            )
            improvement_payload = {
                "proposal": proposal.title,
                "applied": apply_result.applied,
                "reason": apply_result.reason,
                "rollback_id": apply_result.rollback_id,
            }
        loops["improvement"] = improvement_payload

        return EpisodeResult(
            goal=goal,
            pass_gate=pass_gate,
            final_response=final_response,
            confidence=gate.confidence,
            uncertainty=gate.uncertainty,
            strategy=best.strategy,
            tools_used=tools_used,
            loops=loops,
            notes=notes,
            metadata={
                "triad_omega": True,
                "hypothesis_count": len(hypothesis_set.hypotheses),
                "selected_hypothesis": best.candidate_id,
            },
        )

    def _swarm_generate_callback(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            return "[swarm-fallback] model unavailable"
        return self.generate(prompt, max_tokens=96, temperature=0.4, top_k=40, top_p=0.9)
    
    def _load_model(self) -> None:
        """Load MertFormer model."""
        try:
            print(f"🧠 MertFormer modeli yükleniyor...")
            
            # Tokenizer
            from transformers import AutoTokenizer
            tokenizer_id = getattr(cfg, "teacher_model_id", "gpt2")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Model
            self.model = MertFormer().to(self.device)
            
            # Load checkpoint (if exists)
            if AGIPaths.CHECKPOINT_FILE.exists():
                print(f"📂 Checkpoint yükleniyor: {AGIPaths.CHECKPOINT_FILE}")
                checkpoint = torch.load(AGIPaths.CHECKPOINT_FILE, map_location=self.device)
                
                # Nested state dict check
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
        """Generate text."""
        if self.model is None or self.tokenizer is None:
            return "[TR: Model yüklenemedi, inference yapılamıyor / EN: Model not loaded, cannot perform inference]"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_seq_len - max_tokens,
        ).to(self.device)
        
        # Generate
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt
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
        Chat with user.
        
        Args:
            user_message: User message
            use_memory: Get context from memory
            use_web: Perform web search
            temperature: Sampling temperature
        """
        # Save to memory
        self.memory.save("user", user_message, category="GENERAL", source="CHAT")
        
        # Build context
        context_parts = []
        
        if use_memory:
            memory_context = self.memory.build_context_block(user_message)
            if memory_context:
                context_parts.append(memory_context)
        
        if use_web and self.web.enabled:
            web_results = self.web.search(user_message, max_results=3)
            context_parts.append(f"[WEB RESULTS]\n{web_results}\n[/WEB RESULTS]")
        
        # Build prompt
        context = "\n\n".join(context_parts)
        prompt = f"""Sen MertFormer Titan, gelişmiş bir yapay zeka asistanısın.

{context}

Kullanıcı: {user_message}

Titan:"""
        
        # Generate
        response = self.generate(prompt, temperature=temperature)
        
        # Save to memory
        self.memory.save("assistant", response, category="GENERAL", source="CHAT")
        
        # Voice response
        if self.enable_voice:
            self.audio.speak(response)
        
        return response

    def run_swarm_task(self, task: str, mode: str = "nano") -> dict:
        """
        Deterministic swarm execution (nano|mid|omega).
        """
        report = self.swarm.run(task=task, mode=mode)
        return {
            "mode": report.mode,
            "task": report.task,
            "governance": report.governance,
            "selected_agents": report.selected_agents,
            "verification": report.verification,
            "telemetry": report.telemetry,
            "outputs": report.outputs,
        }

    def check_alignment(self, prompt: str) -> dict:
        violations = self.alignment_contracts.check_prompt(prompt)
        return {
            "pass_check": len(violations) == 0,
            "violations": [
                {"rule_id": v.rule_id, "message": v.message, "severity": v.severity}
                for v in violations
            ],
        }

    def propose_self_improvements(self) -> dict:
        snapshot = self.swarm.run("runtime telemetry snapshot", mode="nano").telemetry
        health = float(snapshot.get("health_report", {}).get("health_score", 1.0))
        failure_signal = float(snapshot.get("failure_budget", {}).get("failure_budget_signal", 0.0))
        telemetry = {"health_score": health, "failure_budget_signal": failure_signal}
        proposals = self.self_improvement_guard.propose(telemetry)
        return {
            "auto_apply": False,
            "requires_human_approval": True,
            "proposals": [
                {
                    "title": p.title,
                    "rationale": p.rationale,
                    "risk": p.risk,
                    "requires_human_approval": p.requires_human_approval,
                }
                for p in proposals
            ],
        }

    def compute_schedule(self, performance_priority: float = 0.6, energy_priority: float = 0.4) -> dict:
        return self.compute_orchestrator.schedule(
            {
                "performance_priority": float(performance_priority),
                "energy_priority": float(energy_priority),
            }
        )
    
    def status(self) -> str:
        """Return system status."""
        lines = [
            "📊 TITAN ORCHESTRATOR STATUS",
            "=" * 40,
            self.hardware.scan(),
            f"🧠 Model: {'Yüklü' if self.model else 'Yüklenmedi'}",
            "🤖 Swarm Modes: nano(3), mid(15), omega(45)",
            f"💾 Hafıza: {len(self.memory.cache)} kayıt",
            f"📚 Doküman Chunks: {len(self.doc_indexer.chunks)}",
            f"🧮 Compute Backend Suggestion: {self.compute_schedule().get('backend', 'unknown')}",
            f"🌐 Web: {'Aktif' if self.web.enabled else 'Devre Dışı'}",
            f"🔊 TTS: {'Aktif' if self.audio.is_tts_available() else 'Devre Dışı'}",
            "=" * 40,
        ]
        return "\n".join(lines)
    
    def repl(self) -> None:
        """Interactive REPL loop."""
        print("\n" + "=" * 60)
        print("🚀 MERTFORMER TITAN - Interactive Mode")
        print("   Komutlar: !status, !web <query>, !voice, !quit")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("Sen: ").strip()
                
                if not user_input:
                    continue
                
                # Special commands
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
                elif user_input.lower().startswith("!swarm "):
                    payload = user_input[7:].strip()
                    mode = "nano"
                    if payload.startswith("omega:"):
                        mode = "omega"
                        payload = payload[6:].strip()
                    elif payload.startswith("mid:"):
                        mode = "mid"
                        payload = payload[4:].strip()
                    print(self.run_swarm_task(payload, mode=mode))
                    continue
                
                # Normal chat
                response = self.chat(user_input, use_memory=True)
                print(f"\nTitan: {response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Görüşürüz!")
                break
            except Exception as e:
                print(f"⚠️ Hata: {e}")

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
def main() -> None:
    """CLI Entry Point for Orchestrator."""
    try:
        # Load config to get model path or defaults
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
