"""Hermetic coverage for the decision-head training pipeline (olaya.train) on a tiny run."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from olaya.data_gen import generate_synthetic_incidents  # noqa: E402
from olaya.ontology import create_incident_ontology  # noqa: E402
from olaya.train import (  # noqa: E402
    DecisionDataset,
    build_training_samples,
    collate_decision_batch,
    incident_observations,
    split_incident_instances,
    train_olaya,
)


def _special(tok) -> dict[str, int]:
    return {"cls": tok.cls_token_id, "sep": tok.sep_token_id, "mask": tok.mask_token_id, "pad": tok.pad_token_id or 0}


def test_build_samples_cover_every_question_type(tokenizer):
    onto = create_incident_ontology()
    instances = generate_synthetic_incidents(num_samples=12, seed=3)
    samples = build_training_samples(
        instances, onto, lambda t: tokenizer.encode(t, add_special_tokens=False), _special(tokenizer), seed=3
    )
    kinds = {s.qtype_name for s in samples}
    assert {"concept", "subsumes", "choice", "score", "noul"} <= kinds
    for s in samples:
        assert 0 <= s.target_idx and s.act_target in (0, 1)
        assert len(s.bundle.input_ids) > 0

    ds = DecisionDataset(samples)
    assert len(ds) == len(samples) and ds[0] is samples[0]
    batch = collate_decision_batch([ds[i] for i in range(4)], pad_id=_special(tokenizer)["pad"])
    n, length = batch["input_ids"].shape
    assert n == 4 and batch["attention_mask"].sum() == sum(len(ds[i].bundle.input_ids) for i in range(4))
    m = batch["marker_pos"].shape[1]
    for key in ("subsumption_adj", "parent_adj", "child_adj", "disjoint_adj"):
        assert batch[key].shape == (4, m, m)
    concept = next(s for s in samples if s.qtype_name == "concept")
    one = collate_decision_batch([concept])
    from olaya.category_theory import extract_morphism_matrices

    expected = extract_morphism_matrices(
        concept.bundle.candidate_curies, onto, max_k=len(concept.bundle.marker_positions)
    )
    assert torch.equal(one["disjoint_adj"][0], torch.from_numpy(expected[3]))

    # A candidate set with a disjoint pair produces a non-zero disjointness morphism matrix.
    from olaya.sequence import QTYPE_IDS, build_ontology_sequence
    from olaya.train import TrainingSample

    pair = ["INCIDENT:DB_OUTAGE", "INCIDENT:INVOICE_MISCALC"]
    bundle = build_ontology_sequence(
        tokenizer_encode=lambda t: tokenizer.encode(t, add_special_tokens=False),
        special_ids=_special(tokenizer),
        state={"error": "db down"},
        instructions="Classify.",
        active_concepts=[onto.concepts[c] for c in pair],
        ontology=onto.extract_subgraph(pair),
        qtype_id=QTYPE_IDS["concept"],
        qtype_name="concept",
    )
    crafted = collate_decision_batch(
        [TrainingSample(bundle=bundle, target_idx=0, act_target=0, qtype_name="concept", ontology=onto)]
    )
    assert crafted["disjoint_adj"].sum() == 2  # symmetric pair


def test_incident_split_is_group_disjoint():
    instances = generate_synthetic_incidents(num_samples=80, seed=1)
    rows = incident_observations(instances)
    assert len(rows) == len(instances) and all(r.source_kind == "synthetic_incident" for r in rows)
    splits = split_incident_instances(instances, seed=1)
    assert set(splits) == {"train", "validation", "calibration", "test"}
    group = {r.source_id: r.group_id for r in rows}
    by_split = {
        name: {group[r.source_id] for r in incident_observations(part)} for name, part in splits.items() if part
    }
    names = list(by_split)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (by_split[a] & by_split[b]), (a, b)


def test_train_olaya_tiny_run_writes_a_loadable_checkpoint(tokenizer, tmp_path):
    from olaya.engine import OLaya

    out = tmp_path / "ckpt" / "model.pt"
    model, metrics = train_olaya(
        tokenizer=tokenizer, num_samples=40, epochs=1, batch_size=16, device="cpu", output_path=out, seed=0
    )
    assert set(metrics) == {
        "best_val_acc",
        "final_train_acc",
        "final_val_loss",
        "final_cat_loss",
        "final_mono_violation_rate",
        "final_disj_collision_rate",
    }
    assert 0.0 <= metrics["final_mono_violation_rate"] <= 1.0
    if metrics["best_val_acc"] > 0:
        ckpt = torch.load(out, map_location="cpu")
        assert ckpt["config"]["hidden_size"] == 256 and "val_mono_violation_rate" in ckpt
        engine = OLaya(tokenizer=tokenizer, device="cpu", weights_path=out)
        assert engine.model_id.startswith("checkpoint-sha256:")
        result = engine.system_one("database connection refused", {"q": {"type": "concept", "root": "INCIDENT:ROOT"}})
        assert result.answers["q"].axioms_verified is True
