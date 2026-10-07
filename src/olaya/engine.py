"""O-Laya Engine: Ontology-Grounded System 1 Decision Engine.

Provides the Jev/Laya-compatible `system_one` API extended with formal ontology grounding:
- `concept`: Grounds state to a verified ontology concept with ancestral lineage and consistency guarantees
- `subsumes`: Bernoulli P(state is an instance of Concept C or any of its descendants)
- `choice`: Pick one option from a discrete set with normalized probabilities
- `score`: Expected continuous level across an ordered rubric
- `noul`: Bernoulli probability P(true) for yes/no statements

Temperature rule
----------------
Candidate logits are softmaxed with a temperature. Without a fitted temperature the cardinality
heuristic :func:`olaya.axioms.temperature_by_cardinality` applies (the projections in ``olaya.axioms``
divide by it). When the loaded checkpoint carries a fitted ``temperature`` (see
``scripts/train_real_ontologies.py`` / ``scripts/calibrate_temperature.py``, which fit it on raw,
mask-excluded logits), that fitted value *replaces* the heuristic: the engine pre-multiplies the
candidate logits by ``T_card(k) / T_fitted`` so the projection's division by ``T_card(k)`` yields
``softmax(logits / T_fitted)`` exactly. Padded (non-candidate) slots never enter the softmax either way.

Confidence is normalized-entropy confidence; it is only empirically calibrated when such a fitted
temperature is loaded.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from olaya.axioms import (
    calibrate_and_project_distribution,
    project_hierarchical_distribution,
    temperature_by_cardinality,
    verify_axioms,
)
from olaya.candidate_router import CandidateRouter
from olaya.category_theory import extract_morphism_matrices
from olaya.model import OntologyDecisionModel
from olaya.ontology import Concept, Ontology, create_incident_ontology
from olaya.sequence import QTYPE_IDS, SequenceBundle, build_ontology_sequence, serialize_state

DEFAULT_SCORE_LEVELS = ["level 0", "level 1", "level 2", "level 3"]

# Provenance tag used when the engine runs on freshly initialised (random) weights.
UNTRAINED_MODEL_ID = "untrained-random-init"
# Provenance tag used when the caller injects a model object whose weights the engine did not load.
INJECTED_MODEL_ID = "injected-model"

_DISTRIBUTION_TOL = 1e-6


@dataclass
class ConceptAnswer:
    """Grounded concept decision.

    ``candidate_probabilities`` is a distribution (sums to 1) over the most specific routed concepts
    (the leaves of the active subgraph); ``curie`` is its argmax. ``probabilities`` holds hierarchical
    marginals P(state is an instance of C) for every concept in the active subgraph, so ancestors
    carry the summed mass of their descendants and P(parent) >= P(child) by construction.

    ``axioms_verified`` is always computed by the engine (``verify_axioms`` over the marginals); the
    dataclass default is ``False`` so a value of ``True`` is never a placeholder.
    """

    type: str = "concept"
    curie: str = ""
    label: str = ""
    lineage: list[str] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)
    candidate_probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    axioms_verified: bool = False


@dataclass
class SubsumesAnswer:
    type: str = "subsumes"
    concept: str = ""
    subsumes: float = 0.0  # P(true)
    confidence: float = 0.0
    axioms_verified: bool = False


@dataclass
class ChoiceAnswer:
    type: str = "choice"
    choice: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    axioms_verified: bool = False


@dataclass
class ScoreAnswer:
    type: str = "score"
    score: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    axioms_verified: bool = False


@dataclass
class NoulAnswer:
    type: str = "noul"
    noul: float = 0.0  # P(true)
    axioms_verified: bool = False


@dataclass
class SystemOneResult:
    """Result of one ``system_one`` call.

    ``model`` records weight provenance: ``checkpoint-sha256:<prefix>`` for a loaded checkpoint,
    ``untrained-random-init`` for freshly initialised weights, or ``injected-model`` for a model
    object supplied by the caller; an injected scorer appends ``+scorer:<name>``. ``act_probabilities`` maps each question id to the auxiliary
    act/escalate head's P(act) for that question's sequence (trained against escalation targets in
    ``olaya.train``; it is not an ontology-projected quantity).
    """

    model: str
    answers: dict[str, Any]
    usage: dict[str, int]
    latency_ms: float
    act_probabilities: dict[str, float] = field(default_factory=dict)


CANDIDATE_POLICIES = frozenset({"leaves", "routed"})


class OLaya:
    """Ontology-Grounded Decision Model Engine."""

    def __init__(
        self,
        ontology: Ontology | None = None,
        model: OntologyDecisionModel | None = None,
        tokenizer: Any | None = None,
        device: str | None = None,
        weights_path: str | Path | bool | None = None,
        router: Any | None = None,
        scorer: Any | None = None,
        candidate_policy: str = "leaves",
    ):
        self.ontology = ontology or create_incident_ontology()
        # ``candidate_policy`` decides which routed concepts compete in a ``concept`` question.
        # ``"leaves"`` (default, shared with the Rust / Node runtimes and the golden vectors) scores
        # only the most specific concepts of the routed subgraph; ``"routed"`` scores the routed
        # concepts themselves, so a class the router ranked but that has a routed descendant still
        # competes instead of surviving only as ancestor context. Marginals stay monotone either way
        # (a concept's marginal is the mass at or below it).
        if candidate_policy not in CANDIDATE_POLICIES:
            raise ValueError(f"candidate_policy must be one of {sorted(CANDIDATE_POLICIES)}, got {candidate_policy!r}")
        self.candidate_policy = candidate_policy
        # ``router`` is any object with ``extract_decision_subgraph(state_text, root_curie, max_candidates)``
        # over this ontology (``olaya.embedding_router`` provides embedding and hybrid routers); the
        # default stays the lexical router. ``scorer`` replaces the decision head's candidate logits
        # for ``concept`` questions only: ``scorer.score(state_text, candidate_curies)`` returns one
        # raw score per candidate and ``scorer.temperature`` is the temperature those scores are
        # softmaxed with (see ``_calibrated_logits``). The projection, lineage and axiom checks are
        # unchanged; every other question type still goes through the model.
        if router is not None and getattr(router, "ontology", self.ontology) is not self.ontology:
            raise ValueError("injected router must be built over the engine's ontology")
        self.router = router if router is not None else CandidateRouter(self.ontology, max_candidates=12)
        if scorer is not None:
            if not callable(getattr(scorer, "score", None)):
                raise TypeError("scorer must provide score(state_text, candidate_curies)")
            _validated_temperature(getattr(scorer, "temperature", None), Path("<scorer>"))
        self.scorer = scorer

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

        # Model setup. ``calibrated_temperature`` stays None unless the checkpoint carries a fitted
        # value (see the module docstring for how it is applied).
        self.calibrated_temperature: float | None = None
        self.model_id: str = UNTRAINED_MODEL_ID
        if model is None:
            ckpt = None
            ckpt_file: Path | None = None
            if weights_path is not False and weights_path != "":
                ckpt_file = Path(weights_path) if weights_path is not None else Path("models/olaya_decision_model.pt")
                if ckpt_file.is_file():
                    ckpt = torch.load(ckpt_file, map_location=self.device)
                    if isinstance(ckpt, dict) and "temperature" in ckpt:
                        self.calibrated_temperature = _validated_temperature(ckpt["temperature"], ckpt_file)
                else:
                    ckpt_file = None

            cfg_dict = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
            self.model = self._build_model(cfg_dict)

            if ckpt is not None:
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                _load_state_dict_strict(self.model, state_dict, ckpt_file)
                self.model_id = f"checkpoint-sha256:{_file_sha256(ckpt_file)[:12]}"
        else:
            self.model = model
            self.model_id = INJECTED_MODEL_ID

        if self.scorer is not None:
            self.model_id = f"{self.model_id}+scorer:{getattr(self.scorer, 'name', type(self.scorer).__name__)}"
        if self.candidate_policy != "leaves":
            self.model_id = f"{self.model_id}+candidates:{self.candidate_policy}"

        self.model.to(self.device)
        self.model.eval()

        # Special token IDs (``is None`` checks: 0 is a valid token id)
        self.special_ids = {
            "cls": _token_id_or(self.tokenizer.cls_token_id, 101),
            "sep": _token_id_or(self.tokenizer.sep_token_id, 102),
            "mask": _token_id_or(self.tokenizer.mask_token_id, 103),
            "pad": _token_id_or(self.tokenizer.pad_token_id, 0),
        }

    def _build_model(self, cfg_dict: dict[str, Any]) -> OntologyDecisionModel:
        """Instantiate the decision model from a checkpoint ``config`` block (or defaults)."""
        bb = cfg_dict.get("backbone")
        if bb and bb != "custom_bert":
            return OntologyDecisionModel(backbone_name_or_config=bb)

        from transformers import BertConfig, BertModel

        default_v_size = max(len(self.tokenizer), getattr(self.tokenizer, "vocab_size", 30522)) + 256
        h_size = cfg_dict.get("hidden_size", 256)
        cfg = BertConfig(
            vocab_size=cfg_dict.get("vocab_size", default_v_size),
            hidden_size=h_size,
            num_hidden_layers=cfg_dict.get("num_hidden_layers", 4),
            num_attention_heads=cfg_dict.get("num_attention_heads", 4),
            intermediate_size=cfg_dict.get("intermediate_size", 512),
            max_position_embeddings=cfg_dict.get("max_position_embeddings", 512),
        )
        return OntologyDecisionModel(hidden_size=h_size, custom_encoder=BertModel(cfg))

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _calibrated_logits(self, raw_scores: np.ndarray, temperature: float | None = None) -> np.ndarray:
        """Apply the temperature rule from the module docstring to the candidate-slot logits.

        ``raw_scores`` must already be restricted to the real candidate slots (no padding). With a
        fitted temperature ``T`` the returned logits are ``raw * T_card(k) / T``; the downstream
        projection divides by ``T_card(k)`` again, so the effective softmax is ``softmax(raw / T)``.
        ``temperature`` overrides the checkpoint's fitted value (used for an injected scorer).
        """
        if temperature is None:
            temperature = self.calibrated_temperature
        if temperature is None:
            return raw_scores
        k = len(raw_scores)
        return raw_scores * (temperature_by_cardinality(k) / temperature)

    def system_one(self, state: Any, questions: dict[str, dict[str, Any]]) -> SystemOneResult:
        """Evaluate typed questions against input state in a single forward pass."""
        start_time = time.perf_counter()
        qids = list(questions.keys())
        if not qids:
            raise ValueError("system_one: At least one question required.")

        bundles: list[SequenceBundle] = []
        subgraphs: list[Ontology] = []
        # Ontology used to build each bundle's morphism matrices. This mirrors the training
        # collate (``olaya.train.collate_decision_batch``): concept questions carry the full
        # ontology, flat questions carry none (all-zero adjacency).
        morphism_ontologies: list[Ontology | None] = []
        total_tokens = 0
        state_text = serialize_state(state)

        for qid in qids:
            q = questions[qid]
            qtype = q.get("type", "concept")
            instructions = q.get("instructions", "")
            morphism_ontology: Ontology | None = None

            if qtype == "concept":
                root_curie = q.get("root")
                if root_curie is not None and root_curie not in self.ontology.concepts:
                    raise ValueError(f"Question {qid!r}: unknown root concept {root_curie!r}.")
                routed_curies: list[str] | None = None
                if self.candidate_policy == "routed" and callable(getattr(self.router, "route_candidates", None)):
                    routed = self.router.route_candidates(state_text, root_curie=root_curie, top_k=10)
                    routed_curies = [c.curie for c, _ in routed if c.curie in self.ontology.concepts]
                    subgraph = self.ontology.extract_subgraph(routed_curies, include_parents=True)
                else:
                    subgraph = self.router.extract_decision_subgraph(
                        state_text=state_text, root_curie=root_curie, max_candidates=10
                    )
                if not subgraph.concepts:
                    raise ValueError(f"Question {qid!r}: no candidate concepts (is the ontology empty?).")
                if routed_curies:
                    # Every routed concept competes, whatever its depth; ancestors that were only
                    # pulled in as lineage context stay context.
                    active_concepts = [subgraph.concepts[c] for c in routed_curies]
                else:
                    # Only the most specific routed concepts compete; their ancestors appear as
                    # subClassOf context in the option text and receive mass through marginals.
                    active_concepts = [c for c in subgraph.concepts.values() if not c.children]
                morphism_ontology = self.ontology

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
            morphism_ontologies.append(morphism_ontology)

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
        # Morphism adjacency over candidate slots, built exactly as the training collate does.
        parent_adj = torch.zeros((n, max_markers, max_markers), dtype=torch.float32)
        child_adj = torch.zeros((n, max_markers, max_markers), dtype=torch.float32)
        disjoint_adj = torch.zeros((n, max_markers, max_markers), dtype=torch.float32)

        for i, b in enumerate(bundles):
            seq_len = len(b.input_ids)
            input_ids[i, :seq_len] = torch.tensor(b.input_ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            num_m = len(b.marker_positions)
            if num_m > 0:
                marker_pos[i, :num_m] = torch.tensor(b.marker_positions, dtype=torch.long)
                marker_mask[i, :num_m] = True
                if morphism_ontologies[i] is not None and b.candidate_curies:
                    _, p_m, c_m, d_m = extract_morphism_matrices(
                        b.candidate_curies, morphism_ontologies[i], max_k=num_m
                    )
                    parent_adj[i, :num_m, :num_m] = torch.from_numpy(p_m)
                    child_adj[i, :num_m, :num_m] = torch.from_numpy(c_m)
                    disjoint_adj[i, :num_m, :num_m] = torch.from_numpy(d_m)
            qtypes[i] = b.qtype
            total_tokens += seq_len

        # Execute single forward pass
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        marker_pos = marker_pos.to(self.device)
        marker_mask = marker_mask.to(self.device)
        qtypes = qtypes.to(self.device)

        with torch.no_grad():
            logits, act_probs = self.model(
                input_ids,
                attention_mask,
                marker_pos,
                marker_mask,
                qtypes,
                parent_adj=parent_adj.to(self.device),
                child_adj=child_adj.to(self.device),
                disjoint_adj=disjoint_adj.to(self.device),
            )

        logits_np = logits.cpu().numpy()
        act_np = act_probs.cpu().numpy()
        answers: dict[str, Any] = {}
        act_probabilities: dict[str, float] = {}

        for i, qid in enumerate(qids):
            b = bundles[i]
            subgraph = subgraphs[i]
            qtype = questions[qid].get("type", "concept")
            num_markers = len(b.marker_positions)
            # Only real candidate slots: padded slots (masked to -1e4 by the head) never enter the softmax.
            raw_scores = self._calibrated_logits(logits_np[i, :num_markers])
            act_probabilities[qid] = round(float(act_np[i, 1]), 4)

            if qtype == "concept" and self.scorer is not None:
                scored = np.asarray(self.scorer.score(state_text, b.candidate_curies), dtype=np.float64)
                if scored.shape != (num_markers,):
                    raise ValueError(
                        f"Question {qid!r}: scorer returned {scored.shape} scores for {num_markers} candidates."
                    )
                raw_scores = self._calibrated_logits(scored, temperature=float(self.scorer.temperature))

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
                    axioms_verified=_is_distribution(candidate_probs) and verify_axioms(marginals, subgraph),
                )
                continue

            prob_map, confidence = calibrate_and_project_distribution(
                raw_logits=raw_scores,
                candidate_curies=b.candidate_curies,
                subgraph=subgraph,
                enforce_axioms=True,
            )
            # Flat questions: the projection must return a proper distribution over the options and
            # satisfy the (trivially-flat) subsumption/disjointness axioms of their subgraph.
            axioms_ok = _is_distribution(prob_map) and verify_axioms(prob_map, subgraph)

            if qtype == "subsumes":
                target_curie = questions[qid].get("concept", "")
                p_subsumes = prob_map.get("true", 0.5)
                answers[qid] = SubsumesAnswer(
                    concept=target_curie,
                    subsumes=round(p_subsumes, 4),
                    confidence=round(confidence, 4),
                    axioms_verified=axioms_ok and _is_probability(p_subsumes),
                )

            elif qtype == "choice":
                best_choice = max(prob_map.items(), key=lambda x: x[1])[0]
                answers[qid] = ChoiceAnswer(
                    choice=best_choice,
                    probabilities={k: round(v, 4) for k, v in prob_map.items()},
                    confidence=round(confidence, 4),
                    axioms_verified=axioms_ok,
                )

            elif qtype == "score":
                exp_score = sum(int(k) * p for k, p in prob_map.items() if k.isdigit())
                answers[qid] = ScoreAnswer(
                    score=round(exp_score, 4),
                    probabilities={k: round(v, 4) for k, v in prob_map.items()},
                    confidence=round(confidence, 4),
                    axioms_verified=axioms_ok,
                )

            elif qtype == "noul":
                p_true = prob_map.get("true", 0.5)
                answers[qid] = NoulAnswer(
                    noul=round(p_true, 4),
                    axioms_verified=axioms_ok and _is_probability(p_true),
                )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return SystemOneResult(
            model=self.model_id,
            answers=answers,
            usage={"input_tokens": total_tokens, "output_tokens": 0},
            latency_ms=round(latency_ms, 2),
            act_probabilities=act_probabilities,
        )


def _token_id_or(token_id: int | None, default: int) -> int:
    return default if token_id is None else token_id


def _flat_subgraph(name: str, concepts: list[Concept]) -> Ontology:
    subgraph = Ontology(name)
    for c in concepts:
        subgraph.add_concept(c)
    return subgraph


def _is_probability(p: float, tol: float = _DISTRIBUTION_TOL) -> bool:
    return math.isfinite(p) and -tol <= p <= 1.0 + tol


def _is_distribution(prob_map: dict[str, float], tol: float = _DISTRIBUTION_TOL) -> bool:
    """True when every value is a probability and the values sum to one."""
    values = list(prob_map.values())
    if not values or not all(_is_probability(v, tol) for v in values):
        return False
    return abs(sum(values) - 1.0) <= tol * max(1, len(values))


def _validated_temperature(value: Any, ckpt_file: Path) -> float:
    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint {ckpt_file} has a non-numeric temperature: {value!r}") from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"Checkpoint {ckpt_file} has an invalid temperature {temperature!r}; expected > 0.")
    return temperature


def _load_state_dict_strict(model: torch.nn.Module, state_dict: Any, ckpt_file: Path | None) -> None:
    """Load weights strictly, naming the missing/unexpected keys when the checkpoint does not fit."""
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {ckpt_file} does not contain a state dict.")
    expected = set(model.state_dict())
    given = set(state_dict)
    missing = sorted(expected - given)
    unexpected = sorted(given - expected)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint {ckpt_file} does not match the model architecture "
            f"(missing {len(missing)} keys: {missing[:10]}; unexpected {len(unexpected)} keys: {unexpected[:10]}). "
            "Rebuild the engine with the checkpoint's stored config or re-export the checkpoint."
        )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:  # e.g. shape mismatches
        raise ValueError(f"Checkpoint {ckpt_file} does not match the model architecture: {exc}") from exc


def _file_sha256(path: Path | None) -> str:
    digest = hashlib.sha256()
    if path is None:
        return digest.hexdigest()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
