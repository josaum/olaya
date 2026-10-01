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
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from olaya.axioms import calibrate_and_project_distribution, project_hierarchical_distribution, verify_axioms
from olaya.candidate_router import CandidateRouter
from olaya.model import OntologyDecisionModel
from olaya.ontology import Concept, Ontology, create_incident_ontology
from olaya.sequence import QTYPE_IDS, SequenceBundle, build_ontology_sequence, serialize_state

DEFAULT_SCORE_LEVELS = ["level 0", "level 1", "level 2", "level 3"]


@dataclass
class ConceptAnswer:
    """Grounded concept decision.

    ``candidate_probabilities`` is a distribution (sums to 1) over the most specific routed concepts
    (the leaves of the active subgraph); ``curie`` is its argmax. ``probabilities`` holds hierarchical
    marginals P(state is an instance of C) for every concept in the active subgraph, so ancestors
    carry the summed mass of their descendants and P(parent) >= P(child) by construction.
    """

    type: str = "concept"
    curie: str = ""
    label: str = ""
    lineage: list[str] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)
    candidate_probabilities: dict[str, float] = field(default_factory=dict)
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
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class ScoreAnswer:
    type: str = "score"
    score: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class NoulAnswer:
    type: str = "noul"
    noul: float = 0.0  # P(true)


@dataclass
class SystemOneResult:
    model: str
    answers: dict[str, Any]
    usage: dict[str, int]
    latency_ms: float


class OLaya:
    """Ontology-Grounded Decision Model Engine."""

    def __init__(
        self,
        ontology: Ontology | None = None,
        model: OntologyDecisionModel | None = None,
        tokenizer: Any | None = None,
        device: str | None = None,
        weights_path: str | Path | bool | None = None,
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
        self.calibrated_temperature = 1.0
        if model is None:
            # Check checkpoint config if available
            ckpt = None
            if weights_path is not False and weights_path != "":
                ckpt_file = Path(weights_path) if weights_path is not None else Path("models/olaya_decision_model.pt")
                if ckpt_file.is_file():
                    ckpt = torch.load(ckpt_file, map_location=self.device)
                    if isinstance(ckpt, dict):
                        self.calibrated_temperature = float(ckpt.get("temperature", 1.0))

            cfg_dict = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
            bb = cfg_dict.get("backbone")
            if bb and bb != "custom_bert":
                self.model = OntologyDecisionModel(backbone_name_or_config=bb)
            else:
                from transformers import BertConfig, BertModel
                default_v_size = max(len(self.tokenizer), getattr(self.tokenizer, "vocab_size", 30522)) + 256
                v_size = cfg_dict.get("vocab_size", default_v_size)
                h_size = cfg_dict.get("hidden_size", 256)
                cfg = BertConfig(
                    vocab_size=v_size,
                    hidden_size=h_size,
                    num_hidden_layers=cfg_dict.get("num_hidden_layers", 4),
                    num_attention_heads=cfg_dict.get("num_attention_heads", 4),
                    intermediate_size=cfg_dict.get("intermediate_size", 512),
                    max_position_embeddings=512,
                )
                custom_encoder = BertModel(cfg)
                self.model = OntologyDecisionModel(hidden_size=h_size, custom_encoder=custom_encoder)

            if ckpt is not None:
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                self.model.load_state_dict(state_dict)
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

        # Special token IDs (``is None`` checks: 0 is a valid token id)
        self.special_ids = {
            "cls": _token_id_or(self.tokenizer.cls_token_id, 101),
            "sep": _token_id_or(self.tokenizer.sep_token_id, 102),
            "mask": _token_id_or(self.tokenizer.mask_token_id, 103),
            "pad": _token_id_or(self.tokenizer.pad_token_id, 0),
        }

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def system_one(self, state: Any, questions: dict[str, dict[str, Any]]) -> SystemOneResult:
        """Evaluate typed questions against input state in a single forward pass."""
        start_time = time.perf_counter()
        qids = list(questions.keys())
        if not qids:
            raise ValueError("system_one: At least one question required.")

        bundles: list[SequenceBundle] = []
        subgraphs: list[Ontology] = []
        total_tokens = 0
        state_text = serialize_state(state)

        for qid in qids:
            q = questions[qid]
            qtype = q.get("type", "concept")
            instructions = q.get("instructions", "")

            if qtype == "concept":
                root_curie = q.get("root")
                if root_curie is not None and root_curie not in self.ontology.concepts:
                    raise ValueError(f"Question {qid!r}: unknown root concept {root_curie!r}.")
                subgraph = self.router.extract_decision_subgraph(
                    state_text=state_text, root_curie=root_curie, max_candidates=10
                )
                if not subgraph.concepts:
                    raise ValueError(f"Question {qid!r}: no candidate concepts (is the ontology empty?).")
                # Only the most specific routed concepts compete; their ancestors appear as
                # subClassOf context in the option text and receive mass through marginals.
                active_concepts = [c for c in subgraph.concepts.values() if not c.children]

            elif qtype == "subsumes":
                target_curie = q.get("concept", "")
                concept = self.ontology.concepts.get(target_curie)
                if concept is None:
                    raise ValueError(f"Question {qid!r}: unknown subsumes concept {target_curie!r}.")
                descendants = sorted(self.ontology.get_descendants(target_curie) - {target_curie})
                desc_labels = [self.ontology.concepts[d].label for d in descendants if d in self.ontology.concepts]
                desc_str = f" or descendants ({', '.join(desc_labels[:6])})" if desc_labels else ""

                false_desc = f"State is not subsumed by {concept.label} ({target_curie})"
                true_desc = f"State is subsumed by {concept.label} ({target_curie}){desc_str}"
                active_concepts = [
                    Concept(curie="false", label="false", description=false_desc),
                    Concept(curie="true", label="true", description=true_desc),
                ]
                subgraph = _flat_subgraph("subsumes_subgraph", active_concepts)
                default_inst = (
                    f"Does state satisfy {concept.label} ({target_curie}) or any of its descendants?"
                    if desc_labels
                    else f"Does state satisfy {concept.label} ({target_curie})?"
                )
                instructions = instructions or default_inst

            elif qtype == "choice":
                criteria = q.get("criteria")
                if isinstance(criteria, dict):
                    options = [(str(k), desc or "") for k, desc in criteria.items()]
                elif isinstance(criteria, (list, tuple)):
                    options = [(str(k), "") for k in dict.fromkeys(criteria)]
                else:
                    raise ValueError(f"Question {qid!r}: choice criteria must be a list or dict of options.")
                if not options:
                    raise ValueError(f"Question {qid!r}: choice requires at least one option.")
                active_concepts = [Concept(curie=k, label=k, description=desc) for k, desc in options]
                subgraph = _flat_subgraph("choice_subgraph", active_concepts)

            elif qtype == "score":
                levels = q.get("criteria", DEFAULT_SCORE_LEVELS)
                if not levels:
                    raise ValueError(f"Question {qid!r}: score requires at least one rubric level.")
                active_concepts = [Concept(curie=str(i), label=str(i), description=lvl) for i, lvl in enumerate(levels)]
                subgraph = _flat_subgraph("rubric_subgraph", active_concepts)

            elif qtype == "noul":
                active_concepts = [
                    Concept(curie="false", label="false", description="Statement does not hold"),
                    Concept(curie="true", label="true", description="Statement holds"),
                ]
                subgraph = _flat_subgraph("rubric_subgraph", active_concepts)

            else:
                raise ValueError(f"Unknown question type: {qtype}")

            bundle = build_ontology_sequence(
                tokenizer_encode=self._encode_text,
                special_ids=self.special_ids,
                state=state,
                instructions=instructions,
                active_concepts=active_concepts,
                ontology=subgraph,
                qtype_id=QTYPE_IDS[qtype],
                qtype_name=qtype,
            )
            bundles.append(bundle)
            subgraphs.append(subgraph)

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
        answers: dict[str, Any] = {}

        for i, qid in enumerate(qids):
            b = bundles[i]
            subgraph = subgraphs[i]
            qtype = questions[qid].get("type", "concept")
            num_markers = len(b.marker_positions)
            raw_scores = logits_np[i, :num_markers]

            if qtype == "concept":
                candidate_probs, marginals, confidence = project_hierarchical_distribution(
                    raw_logits=raw_scores,
                    candidate_curies=b.candidate_curies,
                    subgraph=subgraph,
                    enforce_axioms=True,
                )
                # Highest mass wins; ties go to the deeper concept, then the lexicographically smaller CURIE.
                best_curie = min(
                    candidate_probs,
                    key=lambda c: (-candidate_probs[c], -len(self.ontology.get_ancestors(c)), c),
                )
                concept_obj = self.ontology.concepts.get(best_curie)
                label = concept_obj.label if concept_obj else best_curie
                lineage = self.ontology.get_lineage(best_curie)
                answers[qid] = ConceptAnswer(
                    curie=best_curie,
                    label=label,
                    lineage=lineage,
                    probabilities={k: round(v, 4) for k, v in marginals.items()},
                    candidate_probabilities={k: round(v, 4) for k, v in candidate_probs.items()},
                    confidence=round(confidence, 4),
                    axioms_verified=verify_axioms(marginals, subgraph),
                )
                continue

            prob_map, confidence = calibrate_and_project_distribution(
                raw_logits=raw_scores,
                candidate_curies=b.candidate_curies,
                subgraph=subgraph,
                enforce_axioms=True,
            )

            if qtype == "subsumes":
                target_curie = questions[qid].get("concept", "")
                p_subsumes = prob_map.get("true", 0.5)
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


def _token_id_or(token_id: int | None, default: int) -> int:
    return default if token_id is None else token_id


def _flat_subgraph(name: str, concepts: list[Concept]) -> Ontology:
    subgraph = Ontology(name)
    for c in concepts:
        subgraph.add_concept(c)
    return subgraph
