"""Regenerate spec/axioms_golden.json from the Python reference implementation.

Every runtime (Python, Rust, Node) loads this file in its tests and must reproduce the expected
probabilities and confidence for the flat calibrate-and-project kernel described in spec/AXIOMS.md.

Usage: python scripts/gen_axiom_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from olaya.axioms import calibrate_and_project_distribution, verify_axioms
from olaya.ontology import Concept, Ontology

OUT = Path(__file__).resolve().parent.parent / "spec" / "axioms_golden.json"

CASES = [
    {"name": "flat_three", "logits": [2.0, 1.0, 0.5], "parent_edges": [], "disjoint_edges": []},
    {"name": "single_candidate", "logits": [0.3], "parent_edges": [], "disjoint_edges": []},
    {
        "name": "chain_monotonicity_and_disjoint",
        "logits": [0.2, 0.3, 0.1, 2.0],
        "parent_edges": [[3, 1], [1, 0]],
        "disjoint_edges": [[1, 2]],
    },
    {"name": "dominant_disjoint", "logits": [3.0, 0.0, 0.0], "parent_edges": [], "disjoint_edges": [[0, 1]]},
    {"name": "disjoint_below_threshold", "logits": [1.0, 0.9, 0.8], "parent_edges": [], "disjoint_edges": [[0, 1]]},
    {
        "name": "disjoint_tie_after_lift_zeroes_later_slot",
        "logits": [0.0, 0.0, 4.0],
        "parent_edges": [[2, 0], [2, 1]],
        "disjoint_edges": [[0, 1]],
    },
    {
        "name": "disjoint_zeroes_loser_descendants",
        "logits": [3.0, 0.0, 1.0],
        "parent_edges": [[2, 1]],
        "disjoint_edges": [[0, 1]],
    },
    {
        "name": "twelve_candidates_temperature_1_25",
        "logits": [0.1 * i - 0.3 * (i % 3) for i in range(12)],
        "parent_edges": [[4, 0], [5, 0], [6, 1], [11, 6]],
        "disjoint_edges": [[0, 1]],
    },
    {
        "name": "deep_chain_of_ten",
        "logits": [0.0] * 9 + [5.0],
        "parent_edges": [[i + 1, i] for i in range(9)],
        "disjoint_edges": [],
    },
]


def slot_ontology(k: int, parent_edges, disjoint_edges) -> Ontology:
    onto = Ontology("golden")
    for i in range(k):
        onto.add_concept(Concept(curie=str(i), label=str(i), description=""))
    for child, parent in parent_edges:
        onto.add_subsumption(str(child), str(parent))
    for a, b in disjoint_edges:
        onto.add_disjointness(str(a), str(b))
    return onto


def main() -> None:
    cases = []
    for case in CASES:
        k = len(case["logits"])
        onto = slot_ontology(k, case["parent_edges"], case["disjoint_edges"])
        curies = [str(i) for i in range(k)]
        probs, conf = calibrate_and_project_distribution(np.array(case["logits"]), curies, onto)
        cases.append(
            {
                **case,
                "expected_probs": [round(probs[c], 8) for c in curies],
                "expected_confidence": round(conf, 8),
                "expected_axioms_verified": verify_axioms(probs, onto),
            }
        )
    payload = {
        "description": "Golden vectors for the flat calibrate-and-project kernel. See spec/AXIOMS.md.",
        "disjoint_threshold": 0.6,
        "tolerance": 1e-5,
        "cases": cases,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
