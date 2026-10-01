"""O-Laya Engine: Ontology-Grounded System 1 Decision Engine.

Provides the Jev/Laya-compatible `system_one` API extended with formal ontology grounding:
- `concept`: Grounds state to a verified ontology concept with ancestral lineage and consistency guarantees
- `subsumes`: Calibrated Bernoulli P(state is an instance of Concept C or any of its descendants)
- `choice`: Pick one option from a discrete set with calibrated probabilities
- `score`: Expected continuous level across an ordered rubric
- `noul`: Calibrated Bernoulli probability P(true) for yes/no statements
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from olaya.axioms import calibrate_and_project_distribution
from olaya.candidate_router import CandidateRouter
from olaya.model import OntologyDecisionModel
from olaya.ontology import Concept, Ontology, create_incident_ontology
from olaya.sequence import SequenceBundle, build_ontology_sequence


@dataclass
class ConceptAnswer:
    type: str = "concept"
    curie: str = ""
    label: str = ""
    lineage: List[str] = field(default_factory=list)
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    axioms_verified: bool = True


@dataclass
class SubsumesAnswer:
    type: str = "subsumes"
    concept: str = ""
    subsumes: float = 0.0  # Calibrated P(true)
    confidence: float = 0.0


@dataclass
class ChoiceAnswer:
    type: str = "choice"
    choice: str = ""
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class ScoreAnswer:
    type: str = "score"
    score: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class NoulAnswer:
    type: str = "noul"
    noul: float = 0.0  # P(true)


@dataclass
class SystemOneResult:
    model: str
    answers: Dict[str, Any]
    usage: Dict[str, int]
    latency_ms: float


class OLaya:
    """Ontology-Grounded Decision Model Engine."""

    def __init__(
        self,
        ontology: Ontology | None = None,
        model: OntologyDecisionModel | None = None,
        tokenizer: Any | None = None,
        device: str | None = None,
    ):
        self.ontology = ontology or create_incident_ontology()
        self.router = CandidateRouter(self.ontology, max_candidates=12)

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Tokenizer setup
        if tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = tokenizer

        # Model setup
        if model is None:
            # Initialize with lightweight backbone
            from transformers import BertConfig, BertModel

            v_size = max(len(self.tokenizer), getattr(self.tokenizer, "vocab_size", 30522)) + 256
            cfg = BertConfig(
                vocab_size=v_size,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=512,
                max_position_embeddings=512,
            )
            custom_encoder = BertModel(cfg)
            self.model = OntologyDecisionModel(hidden_size=256, custom_encoder=custom_encoder)
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

        # Special token IDs
        self.special_ids = {
            "cls": self.tokenizer.cls_token_id or 101,
            "sep": self.tokenizer.sep_token_id or 102,
            "mask": self.tokenizer.mask_token_id or 103,
            "pad": self.tokenizer.pad_token_id or 0,
        }

    def _encode_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def system_one(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> SystemOneResult:
        """Evaluate typed questions against input state in a single forward pass."""
        start_time = time.perf_counter()
        qids = list(questions.keys())
        if not qids:
            raise ValueError("system_one: At least one question required.")

        bundles: List[SequenceBundle] = []
        subgraphs: List[Ontology] = []
        total_tokens = 0

        for qid in qids:
            q = questions[qid]
            qtype = q.get("type", "concept")
            instructions = q.get("instructions", "")

            if qtype == "concept":
                root_curie = q.get("root")
                subgraph = self.router.extract_decision_subgraph(
                    state_text=str(state), root_curie=root_curie, max_candidates=10
                )
                active_concepts = list(subgraph.concepts.values())
                bundle = build_ontology_sequence(
                    tokenizer_encode=self._encode_text,
                    special_ids=self.special_ids,
                    state=state,
                    instructions=instructions,
                    active_concepts=active_concepts,
                    ontology=subgraph,
                    qtype_id=0,
                )
                bundles.append(bundle)
                subgraphs.append(subgraph)

            elif qtype == "subsumes":
                target_curie = q.get("concept", "")
                concept = self.ontology.concepts.get(target_curie)
                descendants = [self.ontology.concepts[d] for d in self.ontology.get_descendants(target_curie) if d in self.ontology.concepts]
                active_concepts = ([concept] if concept else []) + descendants[:6]
                subgraph = self.ontology.extract_subgraph([c.curie for c in active_concepts])
                bundle = build_ontology_sequence(
                    tokenizer_encode=self._encode_text,
                    special_ids=self.special_ids,
                    state=state,
                    instructions=instructions or f"Does state satisfy {target_curie}?",
                    active_concepts=active_concepts,
                    ontology=subgraph,
                    qtype_id=2,
                )
                bundles.append(bundle)
                subgraphs.append(subgraph)

            elif qtype == "choice":
                criteria = q.get("criteria", {})
                active_concepts = []
                for k, desc in criteria.items():
                    active_concepts.append(Concept(curie=k, label=k, description=desc or ""))
                subgraph = Ontology("choice_subgraph")
                for c in active_concepts:
                    subgraph.add_concept(c)
                bundle = build_ontology_sequence(
                    tokenizer_encode=self._encode_text,
                    special_ids=self.special_ids,
                    state=state,
                    instructions=instructions,
                    active_concepts=active_concepts,
                    ontology=subgraph,
                    qtype_id=0,
                )
                bundles.append(bundle)
                subgraphs.append(subgraph)

            elif qtype in ("score", "noul"):
                # Represent rubric or boolean as candidate slots
                if qtype == "score":
                    levels = q.get("criteria", ["level 0", "level 1", "level 2", "level 3"])
                    active_concepts = [Concept(curie=str(i), label=str(i), description=lvl) for i, lvl in enumerate(levels)]
                    q_id = 1
                else:
                    active_concepts = [
                        Concept(curie="false", label="false", description="Statement does not hold"),
                        Concept(curie="true", label="true", description="Statement holds"),
                    ]
                    q_id = 2

                subgraph = Ontology("rubric_subgraph")
                for c in active_concepts:
                    subgraph.add_concept(c)
                bundle = build_ontology_sequence(
                    tokenizer_encode=self._encode_text,
                    special_ids=self.special_ids,
                    state=state,
                    instructions=instructions,
                    active_concepts=active_concepts,
                    ontology=subgraph,
                    qtype_id=q_id,
                )
                bundles.append(bundle)
                subgraphs.append(subgraph)
            else:
                raise ValueError(f"Unknown question type: {qtype}")

        # Batch collate
        n = len(bundles)
        max_seq_len = max(len(b.input_ids) for b in bundles)
        max_markers = max(len(b.marker_positions) for b in bundles)

        pad_id = self.special_ids["pad"]
        input_ids = torch.full((n, max_seq_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_seq_len), dtype=torch.long)
        marker_pos = torch.zeros((n, max_markers), dtype=torch.long)
        marker_mask = torch.zeros((n, max_markers), dtype=torch.bool)
        qtypes = torch.zeros((n,), dtype=torch.long)

        for i, b in enumerate(bundles):
            seq_len = len(b.input_ids)
            input_ids[i, :seq_len] = torch.tensor(b.input_ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            num_m = len(b.marker_positions)
            if num_m > 0:
                marker_pos[i, :num_m] = torch.tensor(b.marker_positions, dtype=torch.long)
                marker_mask[i, :num_m] = True
            qtypes[i] = b.qtype
            total_tokens += seq_len

        # Execute single forward pass
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        marker_pos = marker_pos.to(self.device)
        marker_mask = marker_mask.to(self.device)
        qtypes = qtypes.to(self.device)

        with torch.no_grad():
            logits, act_probs = self.model(input_ids, attention_mask, marker_pos, marker_mask, qtypes)

        logits_np = logits.cpu().numpy()
        answers: Dict[str, Any] = {}

        for i, qid in enumerate(qids):
            b = bundles[i]
            subgraph = subgraphs[i]
            qtype = questions[qid].get("type", "concept")
            num_markers = len(b.marker_positions)
            raw_scores = logits_np[i, :num_markers]

            prob_map, confidence = calibrate_and_project_distribution(
                raw_logits=raw_scores,
                candidate_curies=b.candidate_curies,
                subgraph=subgraph,
                enforce_axioms=True,
            )

            if qtype == "concept":
                best_curie = max(prob_map.items(), key=lambda x: x[1])[0]
                concept_obj = self.ontology.concepts.get(best_curie)
                label = concept_obj.label if concept_obj else best_curie
                lineage = self.ontology.get_lineage(best_curie)
                answers[qid] = ConceptAnswer(
                    curie=best_curie,
                    label=label,
                    lineage=lineage,
                    probabilities={k: round(v, 4) for k, v in prob_map.items()},
                    confidence=round(confidence, 4),
                    axioms_verified=True,
                )

            elif qtype == "subsumes":
                # Probability of satisfying concept or descendants
                target_curie = questions[qid].get("concept", "")
                p_subsumes = prob_map.get(target_curie, 0.0)
                answers[qid] = SubsumesAnswer(
                    concept=target_curie,
                    subsumes=round(p_subsumes, 4),
                    confidence=round(confidence, 4),
                )

            elif qtype == "choice":
                best_choice = max(prob_map.items(), key=lambda x: x[1])[0]
                answers[qid] = ChoiceAnswer(
                    choice=best_choice,
                    probabilities={k: round(v, 4) for k, v in prob_map.items()},
                    confidence=round(confidence, 4),
                )

            elif qtype == "score":
                exp_score = sum(int(k) * p for k, p in prob_map.items() if k.isdigit())
                answers[qid] = ScoreAnswer(
                    score=round(exp_score, 4),
                    probabilities={k: round(v, 4) for k, v in prob_map.items()},
                    confidence=round(confidence, 4),
                )

            elif qtype == "noul":
                p_true = prob_map.get("true", 0.5)
                answers[qid] = NoulAnswer(noul=round(p_true, 4))

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return SystemOneResult(
            model="o-laya-incident-v1",
            answers=answers,
            usage={"input_tokens": total_tokens, "output_tokens": 0},
            latency_ms=round(latency_ms, 2),
        )
