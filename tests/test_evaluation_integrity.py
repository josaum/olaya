"""Evaluation boundaries must separate observations before task expansion."""

import json

import pytest

from olaya.evaluation import SourceObservation, dataset_manifest, split_observations
from olaya.ontology import Concept, Ontology
from olaya.real_ontology_train import expand_observations, generate_real_ontology_observations


def ontology():
    onto = Ontology(name="integrity")
    onto.add_concept(Concept("SECRET:ROOT", "Root"))
    for i in range(8):
        onto.add_concept(Concept(f"SECRET:{i}", f"Label {i}", parents={"SECRET:ROOT"}))
    return onto


def row(i, group=None, text=None, target="SECRET:0"):
    return SourceObservation(str(i), group or f"g{i}", target, text or f"observation {i}", "fixture")


def test_splits_deterministic_nonempty_and_disjoint():
    rows = [row(i) for i in range(20)]
    splits = split_observations(rows)
    assert splits == split_observations(list(reversed(rows)))
    assert [len(v) for v in splits.values()] == [12, 3, 2, 3]
    groups = [{r.group_id for r in values} for values in splits.values()]
    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            assert left.isdisjoint(right)


def test_duplicate_text_connects_transitive_groups_and_exact_rows_deduplicate():
    rows = [row(0, "a", "Same text"), row(1, "b", " SAME   TEXT "), row(2, "b", "second"), row(3, "c", "second")]
    rows += [row(i) for i in range(4, 12)]
    rows.append(rows[0])
    splits = split_observations(rows)
    locations = {r.source_id: name for name, values in splits.items() for r in values}
    assert len(locations) == 12
    assert len({locations[str(i)] for i in range(4)}) == 1


@pytest.mark.parametrize(
    "rows",
    [
        [row(0), row(0, text="changed")],
        [row(0, text="same"), row(1, text="same", target="SECRET:1")],
    ],
)
def test_conflicting_sources_fail_closed(rows):
    with pytest.raises(ValueError):
        split_observations(rows + [row(i) for i in range(2, 9)])


def test_insufficient_independent_groups_fail_closed():
    with pytest.raises(ValueError, match="four"):
        split_observations([row(i, "one") for i in range(20)])


def test_generator_excludes_unlabelled_templates_and_preserves_families():
    onto = ontology()
    observations = generate_real_ontology_observations(
        onto, domain="doid", num_samples=100, use_real_abox=False, synonym_aug_prob=0
    )
    assert len(observations) == 20
    assert len({r.group_id for r in observations}) == 5
    assert all(onto.concepts[r.target_curie].label in r.query_text for r in observations)
    splits = split_observations(observations)
    assert len({r.group_id for values in splits.values() for r in values}) == 5


def test_expansion_provenance_and_state_has_no_gold_curie():
    onto = ontology()
    observed_states = []

    def encode(text):
        if text.startswith("{"):
            observed_states.append(json.loads(text))
        return [10] * min(20, len(text.split()))

    source = row(0, text="Label 0 was observed")
    samples = expand_observations(onto, [source], "doid", encode, {"cls": 1, "sep": 2, "mask": 3, "pad": 0})
    assert len(samples) == 5
    assert observed_states
    assert all(state == {"domain": "doid", "query": source.query_text} for state in observed_states)
    assert all(
        (s.source_id, s.group_id, s.state_text, s.source_kind)
        == (source.source_id, source.group_id, source.query_text, source.source_kind)
        for s in samples
    )


def test_manifest_has_no_text_and_hash_changes_with_evidence():
    onto = ontology()
    rows = [row(i) for i in range(10)]
    manifest = dataset_manifest(onto, split_observations(rows))
    assert "observation " not in json.dumps(manifest)
    assert manifest == dataset_manifest(onto, split_observations(rows))
    changed = rows.copy()
    changed[0] = row(0, text="Different private evidence")
    assert manifest != dataset_manifest(onto, split_observations(changed))


def test_ontology_hash_does_not_depend_on_insertion_order():
    first = ontology()
    second = Ontology(name=first.name)
    for concept in reversed(list(first.concepts.values())):
        second.add_concept(concept)
    assert dataset_manifest(first, {}) == dataset_manifest(second, {})


def test_specificity_uses_longest_path_not_ancestor_count():
    from olaya.real_ontology_train import _specificity_depth

    onto = Ontology(name="multiple inheritance")
    for curie in ("rootA", "rootB", "rootC"):
        onto.add_concept(Concept(curie, curie))
    onto.add_concept(Concept("child", "Child", parents={"rootA", "rootB", "rootC"}))
    assert _specificity_depth(onto, "child") == 1
    onto.add_concept(Concept("grandchild", "Grandchild", parents={"child"}))
    assert _specificity_depth(onto, "grandchild") == 2
    onto.add_concept(Concept("deep", "Deep", parents={"grandchild"}))
    onto.add_concept(Concept("deeper", "Deeper", parents={"deep"}))
    assert _specificity_depth(onto, "deeper") == 3
    onto.concepts["rootA"].parents.add("child")
    with pytest.raises(ValueError, match="cycle"):
        _specificity_depth(onto, "child")


def test_noul_question_wording_does_not_encode_answer(monkeypatch):
    import olaya.real_ontology_train as generation

    onto = ontology()
    captured = []
    original = generation.build_ontology_sequence

    def capture(**kwargs):
        if kwargs["qtype_name"] == "noul":
            captured.append(kwargs["instructions"])
        return original(**kwargs)

    monkeypatch.setattr(generation, "build_ontology_sequence", capture)
    labels = set()
    for seed in range(20):
        samples = expand_observations(
            onto,
            [row(0, text="Label 0 observed")],
            "doid",
            lambda _: [10],
            {"cls": 1, "sep": 2, "mask": 3, "pad": 0},
            seed,
        )
        labels.add(samples[-1].target_idx)
        question = captured[-1]
        assert question.startswith("Is this entity accurately classified as ")
        assert "completely unrelated" not in question
        if samples[-1].target_idx == 1:
            assert "(SECRET:0)" in question
        else:
            assert "(SECRET:0)" not in question
            assert "(SECRET:ROOT)" not in question
    assert labels == {0, 1}


def test_expansion_is_stable_across_python_hash_seeds():
    import os
    import subprocess
    import sys

    program = """
import hashlib, json
from olaya.ontology import Ontology, Concept
from olaya.real_ontology_train import generate_real_ontology_observations, expand_observations
from olaya.evaluation import split_observations, dataset_manifest
onto = Ontology(name="hash seed integrity")
onto.add_concept(Concept("root", "Root"))
for i in range(20):
    onto.add_concept(Concept(f"child{i}", f"Child {i}", parents={"root"}))
onto.add_concept(Concept("multi", "Multi Parent", parents={"child0", "child1"}))
observations = generate_real_ontology_observations(onto, "schemaorg", 100, use_real_abox=False)
encode = lambda text: list(text.encode())
splits = split_observations(observations)
payload = {"manifest": dataset_manifest(onto, splits), "samples": {}}
for name, rows in splits.items():
    samples = expand_observations(onto, rows, "schemaorg", encode, {"cls": 1, "sep": 2, "mask": 3, "pad": 0})
    payload["samples"][name] = [(s.bundle.input_ids, s.bundle.candidate_curies, s.target_idx, s.source_id) for s in samples]
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", program], env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True
        )
        for seed in (1, 99)
    ]
    assert outputs[0] == outputs[1]
