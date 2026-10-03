import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.evaluate_lexical_baseline import evaluate_observations

from olaya.evaluation import SourceObservation
from olaya.ontology import Concept, Ontology


def observation(target, text, index=0):
    return SourceObservation(
        source_id=f"source-{index}",
        group_id="test-family",
        target_curie=target,
        query_text=text,
        source_kind="synthetic_template",
    )


def ontology():
    result = Ontology("tiny")
    result.add_concept(Concept("T:A", "Frozen seafood"))
    result.add_concept(Concept("T:B", "Office stationery"))
    result.add_concept(Concept("T:C", "Office stationery"))
    return result


def test_exact_rank_and_tie_abstention():
    result = evaluate_observations(
        ontology(),
        [
            observation("T:A", "Frozen seafood", 0),
            observation("T:B", "Office stationery", 1),
            observation("T:A", "zzzz quux", 2),
        ],
    )
    assert result["top1_accuracy_with_abstention"] == pytest.approx(1 / 3)
    assert result["recall_at_5"] == pytest.approx(2 / 3)
    assert result["macro_recall_at_5"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["abstention_counts"] == {"tied_top_score": 1, "no_matching_evidence": 1}
    assert result["observations"][0]["top5_curies"] == ["T:A"]
    assert result["observations"][1]["top5_curies"] == ["T:B", "T:C"]
    assert result["observations"][2]["top5_curies"] == []
    assert all("query_text" not in row for row in result["observations"])


def test_target_does_not_influence_retrieval():
    result = evaluate_observations(ontology(), [observation("T:B", "Frozen seafood")])
    assert result["observations"][0]["predicted_curie"] == "T:A"
    assert result["recall_at_5"] == 0
    assert result["top1_accuracy_with_abstention"] == 0


def test_empty_or_unknown_target_fails_closed():
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_observations(ontology(), [])
    with pytest.raises(ValueError, match="Unknown target"):
        evaluate_observations(ontology(), [observation("T:MISSING", "Frozen seafood")])


def test_cli_uses_only_held_out_partition_and_records_provenance(tmp_path):
    fixture = Ontology("generated-test-fixture")
    for index, label in enumerate(
        [
            "Frozen seafood",
            "Office stationery",
            "Tropical fruit",
            "Wireless headphones",
            "Garden tools",
            "Cooking utensils",
            "Cold beverages",
            "Leather shoes",
        ]
    ):
        fixture.add_concept(Concept(f"T:{index}", label))
    source = tmp_path / "ontology.json"
    output = tmp_path / "report.json"
    source.write_text(fixture.to_json())
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_lexical_baseline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--ontology",
            str(source),
            "--output",
            str(output),
            "--num-samples",
            "100",
            "--seed",
            "42",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    manifest = report["dataset_manifest"]["splits"]
    rows = report["metrics"]["observations"]
    assert report["ontology_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["evaluator_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert {row["source_id"] for row in rows} == set(manifest["test"]["source_ids"])
    assert report["metrics"]["test_observations"] == manifest["test"]["observations"]
    assert all(row["source_kind"] == "synthetic_template" for row in rows)
    assert all("query_text" not in row for row in rows)
    for partition in ("train", "validation", "calibration"):
        assert not set(manifest[partition]["source_ids"]) & {row["source_id"] for row in rows}
    assert report["metrics"]["top1_accuracy_with_abstention"] == 1.0
