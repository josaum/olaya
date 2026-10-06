"""Bind the Python reference kernel (spec steps 3-4) to the Lean 4 proof core.

``lean/`` proves that ``OlayaAxioms.project`` returns monotone, excluded, non-negative scores
(``project_verified``). This test runs the same verified definitions through the
``olaya-axioms-oracle`` executable on integer inputs and asserts the Python functions
``enforce_subsumption_monotonicity`` and ``enforce_disjointness_exclusion`` produce identical
values, so the proofs are about the algorithm the runtimes actually implement. Integers are exact
in float64, so equality is exact. Skipped when no Lean toolchain is installed; CI installs one.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from olaya.axioms import disjoint_pairs, enforce_disjointness_exclusion, enforce_subsumption_monotonicity, verify_axioms
from olaya.ontology import Concept, Ontology

ROOT = Path(__file__).resolve().parents[1]
LEAN_DIR = ROOT / "lean"
GOLDEN = ROOT / "spec" / "axioms_golden.json"
SCALE = 1_000_000_000  # fixed-point: 1.0 == 1e9
THRESHOLD = int(0.6 * SCALE)


def _lake() -> str | None:
    found = shutil.which("lake")
    if found:
        return found
    candidate = Path.home() / ".elan" / "bin" / "lake"
    return str(candidate) if candidate.exists() else None


LAKE = _lake()
pytestmark = pytest.mark.skipif(LAKE is None, reason="Lean toolchain (lake) not installed")


@pytest.fixture(scope="module")
def oracle() -> list[str]:
    subprocess.run([LAKE, "build", "olaya-axioms-oracle"], cwd=LEAN_DIR, check=True, capture_output=True)
    return [LAKE, "exe", "olaya-axioms-oracle"]


def run_oracle(oracle: list[str], values: list[int], parent_edges, disjoint_edges) -> dict:
    payload = {
        "values": values,
        "parent_edges": [list(e) for e in parent_edges],
        "disjoint_edges": [list(e) for e in disjoint_edges],
        "threshold": THRESHOLD,
    }
    result = subprocess.run(oracle, cwd=LEAN_DIR, input=json.dumps(payload), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def python_reference(values: list[int], parent_edges, disjoint_edges) -> tuple[list[int], list[tuple[int, int]], bool]:
    curies = [f"s{i}" for i in range(len(values))]
    ontology = Ontology("oracle")
    for curie in curies:
        ontology.add_concept(Concept(curie=curie, label=curie))
    for child, parent in parent_edges:
        ontology.add_subsumption(curies[child], curies[parent])
    for a, b in disjoint_edges:
        ontology.add_disjointness(curies[a], curies[b])
    probs = {curie: float(v) for curie, v in zip(curies, values, strict=True)}
    lifted = enforce_subsumption_monotonicity(probs, ontology)
    projected = enforce_disjointness_exclusion(lifted, ontology, threshold=float(THRESHOLD))
    pairs = [(curies.index(a), curies.index(b)) for a, b in disjoint_pairs(curies, ontology)]
    out = [int(projected[c]) for c in curies]
    assert all(float(int(projected[c])) == projected[c] for c in curies)
    return out, pairs, verify_axioms(projected, ontology, threshold=float(THRESHOLD))


def check_case(oracle, values, parent_edges, disjoint_edges):
    expected, pairs, py_verified = python_reference(values, parent_edges, disjoint_edges)
    got = run_oracle(oracle, values, parent_edges, pairs)
    assert got["projected"] == expected, (values, parent_edges, pairs)
    assert got["verified"] is True
    assert py_verified is True


def test_golden_structures_match_python(oracle):
    """Every golden case's edge structure, with its logits mapped to fixed-point scores."""
    cases = json.loads(GOLDEN.read_text())["cases"]
    for case in cases:
        k = len(case["logits"])
        lo = min(case["logits"])
        values = [int((x - lo + 1.0) / (max(case["logits"]) - lo + 1.0) * SCALE) for x in case["logits"]]
        check_case(oracle, values, case["parent_edges"], case["disjoint_edges"])
        assert k == len(values)


def test_random_dags_match_python(oracle):
    rng = random.Random(int(os.environ.get("OLAYA_ORACLE_SEED", "7")))
    for _ in range(40):
        k = rng.randint(1, 9)
        edges = set()
        for child in range(1, k):
            for parent in range(child):
                if rng.random() < 0.3:
                    edges.add((child, parent))
        parent_edges = sorted(edges)
        disjoint = set()
        for a in range(k):
            for b in range(a + 1, k):
                if (a, b) not in edges and (b, a) not in edges and rng.random() < 0.2:
                    disjoint.add((a, b))
        values = [rng.choice([0, rng.randint(0, SCALE)]) for _ in range(k)]
        check_case(oracle, values, parent_edges, sorted(disjoint))
