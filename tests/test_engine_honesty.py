"""The engine's advertised properties must be computed, not assumed.

Covers: fitted temperature actually applied, ``axioms_verified`` computed for every answer type,
morphism adjacency passed at inference exactly as in training, strict checkpoint loading, weight
provenance in ``SystemOneResult.model``, the act head being surfaced, and the group-disjoint split.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import torch

from olaya.axioms import calibrate_and_project_distribution, temperature_by_cardinality
from olaya.category_theory import extract_morphism_matrices
from olaya.data_gen import generate_synthetic_incidents
from olaya.engine import (
    INJECTED_MODEL_ID,
    UNTRAINED_MODEL_ID,
    ChoiceAnswer,
    ConceptAnswer,
    NoulAnswer,
    OLaya,
    ScoreAnswer,
    SubsumesAnswer,
    _flat_subgraph,
)
from olaya.ontology import Concept
from olaya.tokenizer import load_local_tokenizer
from olaya.train import create_trainable_model, incident_observations, split_incident_instances

STATE = {
    "service": "billing-worker",
    "error": "Deadlock detected while updating customer subscription balances in PostgreSQL cluster.",
}
QUESTIONS = {
    "triage": {"type": "concept", "root": "INCIDENT:ROOT"},
    "data": {"type": "subsumes", "concept": "INCIDENT:DATA"},
    "action": {"type": "choice", "criteria": ["restart_service", "retry_gateway", "page_oncall"]},
    "severity": {"type": "score", "criteria": ["low", "medium", "urgent", "critical"]},
    "page": {"type": "noul", "instructions": "Should on-call be paged?"},
}


@pytest.fixture(scope="module")
def small_weights(tokenizer):
    vocab_size = max(len(tokenizer), tokenizer.vocab_size) + 256
    torch.manual_seed(1)
    model = create_trainable_model(vocab_size=vocab_size, hidden_size=256)
    return {k: v.clone() for k, v in model.state_dict().items()}, vocab_size


def _write_checkpoint(path, state_dict, vocab_size, **extra):
    torch.save(
        {
            "model_state_dict": state_dict,
            "config": {"vocab_size": vocab_size, "hidden_size": 256, "backbone": "custom_bert"},
            **extra,
        },
        str(path),
    )
    return path


def _engine_from(path, tokenizer) -> OLaya:
    return OLaya(tokenizer=tokenizer, device="cpu", weights_path=path)


# --- 1. fitted temperature -------------------------------------------------------------------


def test_fitted_temperature_replaces_cardinality_heuristic(olaya, monkeypatch):
    raw = np.array([1.5, -0.25, 0.75])
    options = [Concept(curie=c, label=c) for c in ("a", "b", "c")]
    subgraph = _flat_subgraph("flat", options)

    monkeypatch.setattr(olaya, "calibrated_temperature", None)
    heuristic, _ = calibrate_and_project_distribution(olaya._calibrated_logits(raw), ["a", "b", "c"], subgraph)
    expected_heuristic = np.exp(raw / temperature_by_cardinality(3))
    expected_heuristic /= expected_heuristic.sum()
    assert [heuristic[c] for c in "abc"] == pytest.approx(expected_heuristic.tolist())

    monkeypatch.setattr(olaya, "calibrated_temperature", 2.0)
    fitted, _ = calibrate_and_project_distribution(olaya._calibrated_logits(raw), ["a", "b", "c"], subgraph)
    expected_fitted = np.exp(raw / 2.0)
    expected_fitted /= expected_fitted.sum()
    assert [fitted[c] for c in "abc"] == pytest.approx(expected_fitted.tolist())
    assert fitted != heuristic


def test_checkpoint_temperature_changes_output_distribution(tmp_path, tokenizer, small_weights):
    state_dict, vocab_size = small_weights
    sharp = _engine_from(_write_checkpoint(tmp_path / "t1.pt", state_dict, vocab_size, temperature=1.0), tokenizer)
    flat = _engine_from(_write_checkpoint(tmp_path / "t4.pt", state_dict, vocab_size, temperature=4.0), tokenizer)
    assert sharp.calibrated_temperature == pytest.approx(1.0)
    assert flat.calibrated_temperature == pytest.approx(4.0)

    # Mixed candidate counts in one batch so padded slots exist and must stay excluded.
    r_sharp = sharp.system_one(STATE, QUESTIONS)
    r_flat = flat.system_one(STATE, QUESTIONS)
    p_sharp = r_sharp.answers["action"].probabilities
    p_flat = r_flat.answers["action"].probabilities
    assert p_sharp != p_flat
    assert sum(p_flat.values()) == pytest.approx(1.0, abs=1e-3)
    assert r_flat.answers["action"].confidence < r_sharp.answers["action"].confidence
    assert r_flat.answers["triage"].confidence < r_sharp.answers["triage"].confidence

    # softmax(z / 4) has log-odds exactly one quarter of softmax(z / 1).
    a, b = max(p_sharp, key=p_sharp.get), min(p_sharp, key=p_sharp.get)
    lr_sharp = math.log(p_sharp[a] / p_sharp[b])
    lr_flat = math.log(p_flat[a] / p_flat[b])
    assert lr_flat == pytest.approx(lr_sharp / 4.0, rel=0.05, abs=1e-3)


def test_invalid_checkpoint_temperature_is_rejected(tmp_path, tokenizer, small_weights):
    state_dict, vocab_size = small_weights
    path = _write_checkpoint(tmp_path / "bad_t.pt", state_dict, vocab_size, temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        _engine_from(path, tokenizer)


# --- 2. axioms_verified is computed for every answer type -------------------------------------


def test_answer_dataclasses_never_default_to_verified():
    for answer in (ConceptAnswer(), SubsumesAnswer(), ChoiceAnswer(), ScoreAnswer(), NoulAnswer()):
        assert answer.axioms_verified is False


def test_flat_answers_report_verified_distributions(olaya):
    result = olaya.system_one(STATE, QUESTIONS)
    for qid in ("data", "action", "severity", "page"):
        assert result.answers[qid].axioms_verified is True, qid
    assert sum(result.answers["action"].probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert sum(result.answers["severity"].probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def test_flat_answers_detect_invalid_projection(olaya, monkeypatch):
    import olaya.engine as engine_module

    def broken_projection(raw_logits, candidate_curies, subgraph, enforce_axioms=True):
        # Not a distribution: every option keeps 0.7 mass.
        return dict.fromkeys(candidate_curies, 0.7), 0.5

    monkeypatch.setattr(engine_module, "calibrate_and_project_distribution", broken_projection)
    result = olaya.system_one(STATE, {k: v for k, v in QUESTIONS.items() if k != "triage"})
    for qid in ("data", "action", "severity", "page"):
        assert result.answers[qid].axioms_verified is False, qid


def test_concept_answer_detects_invalid_projection(olaya, monkeypatch):
    import olaya.engine as engine_module

    real = engine_module.project_hierarchical_distribution

    def broken_projection(raw_logits, candidate_curies, subgraph, enforce_axioms=True):
        candidate_probs, marginals, conf = real(raw_logits, candidate_curies, subgraph, enforce_axioms)
        # Break monotonicity: a leaf claims more mass than its own ancestors.
        leaf = next(iter(candidate_probs))
        marginals[leaf] = 1.5
        return candidate_probs, marginals, conf

    monkeypatch.setattr(engine_module, "project_hierarchical_distribution", broken_projection)
    result = olaya.system_one(STATE, {"triage": QUESTIONS["triage"]})
    assert result.answers["triage"].axioms_verified is False


# --- 3. morphism adjacency at inference matches the training collate ----------------------------


def test_concept_question_passes_candidate_adjacency(olaya, monkeypatch):
    layer = olaya.model.head.morphism_layer
    captured: list[dict[str, torch.Tensor | None]] = []
    original = layer.forward

    def spy(marker_states, **kwargs):
        captured.append(kwargs)
        return original(marker_states, **kwargs)

    monkeypatch.setattr(layer, "forward", spy)
    result = olaya.system_one(STATE, {"triage": QUESTIONS["triage"]})

    assert len(captured) == 1
    kwargs = captured[0]
    assert all(kwargs[name] is not None for name in ("parent_adj", "child_adj", "disjoint_adj"))
    assert float(kwargs["disjoint_adj"].sum()) > 0.0

    # Same construction as olaya.train.collate_decision_batch over the candidate slots.
    candidates = list(result.answers["triage"].candidate_probabilities)
    _, p_m, c_m, d_m = extract_morphism_matrices(candidates, olaya.ontology, max_k=len(candidates))
    np.testing.assert_array_equal(kwargs["parent_adj"][0].cpu().numpy(), p_m)
    np.testing.assert_array_equal(kwargs["child_adj"][0].cpu().numpy(), c_m)
    np.testing.assert_array_equal(kwargs["disjoint_adj"][0].cpu().numpy(), d_m)


def test_flat_questions_pass_zero_adjacency(olaya, monkeypatch):
    layer = olaya.model.head.morphism_layer
    captured: list[dict[str, torch.Tensor | None]] = []
    original = layer.forward
    monkeypatch.setattr(layer, "forward", lambda h, **kw: (captured.append(kw), original(h, **kw))[1])
    olaya.system_one(STATE, {"action": QUESTIONS["action"]})
    assert captured and all(float(t.sum()) == 0.0 for t in captured[0].values())


# --- 4. strict checkpoint loading --------------------------------------------------------------


def test_mismatched_checkpoint_names_keys(tmp_path, tokenizer, small_weights):
    state_dict, vocab_size = small_weights
    broken = dict(state_dict)
    del broken["head.classifier.bias"]
    broken["head.legacy_head.weight"] = torch.zeros(1)
    path = _write_checkpoint(tmp_path / "mismatch.pt", broken, vocab_size)
    with pytest.raises(ValueError, match=r"head\.classifier\.bias.*head\.legacy_head\.weight"):
        _engine_from(path, tokenizer)


def test_shape_mismatch_is_rejected(tmp_path, tokenizer, small_weights):
    state_dict, vocab_size = small_weights
    broken = dict(state_dict)
    broken["head.classifier.weight"] = torch.zeros(2, 256)
    path = _write_checkpoint(tmp_path / "shape.pt", broken, vocab_size)
    with pytest.raises(ValueError, match="does not match"):
        _engine_from(path, tokenizer)


# --- 5. weight provenance ----------------------------------------------------------------------


def test_model_id_reports_provenance(tmp_path, tokenizer, small_weights, olaya):
    state_dict, vocab_size = small_weights
    path = _write_checkpoint(tmp_path / "prov.pt", state_dict, vocab_size)
    engine = _engine_from(path, tokenizer)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    assert engine.model_id == f"checkpoint-sha256:{digest}"
    assert engine.system_one(STATE, {"page": QUESTIONS["page"]}).model == f"checkpoint-sha256:{digest}"

    assert olaya.model_id == UNTRAINED_MODEL_ID
    assert olaya.system_one(STATE, {"page": QUESTIONS["page"]}).model == UNTRAINED_MODEL_ID

    injected = OLaya(tokenizer=tokenizer, device="cpu", model=engine.model)
    assert injected.model_id == INJECTED_MODEL_ID


# --- 6. act head surfaced ----------------------------------------------------------------------


def test_act_probabilities_are_returned_per_question(olaya):
    result = olaya.system_one(STATE, QUESTIONS)
    assert set(result.act_probabilities) == set(QUESTIONS)
    assert all(0.0 <= p <= 1.0 for p in result.act_probabilities.values())


# --- 7. training split and tokenizer module -----------------------------------------------------


def test_split_incident_instances_is_group_disjoint():
    instances = generate_synthetic_incidents(num_samples=300, seed=7)
    splits = split_incident_instances(instances, seed=7)
    assert set(splits) == {"train", "validation", "calibration", "test"}
    assert all(splits.values())

    groups = {}
    for name, part in splits.items():
        groups[name] = {row.group_id for row in incident_observations(part)}
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (groups[a] & groups[b]), f"{a} and {b} share a template group"

    unique_sources = {row.source_id for row in incident_observations(instances)}
    assert sum(len(part) for part in splits.values()) == len(unique_sources)


def test_tokenizer_module_loads_bundled_tokenizer():
    tok = load_local_tokenizer()
    assert tok.mask_token_id is not None and tok.cls_token_id is not None
