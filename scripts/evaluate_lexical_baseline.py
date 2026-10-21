"""Held-out, template-derived lexical retrieval baseline (no model training)."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from olaya.candidate_router import CandidateRouter, tokenize_normalize
from olaya.evaluation import SourceObservation
from olaya.ontology import Ontology


def evaluate_observations(ontology: Ontology, observations: list[SourceObservation]) -> dict[str, Any]:
    """Evaluate queries alone against every concept, without a gold candidate shortlist."""
    if not ontology.concepts or not observations:
        raise ValueError("Evaluation requires a nonempty ontology and test observation set")
    router = CandidateRouter(ontology)
    rows = []
    latencies = []
    for observation in observations:
        if observation.target_curie not in ontology.concepts:
            raise ValueError(f"Unknown target CURIE: {observation.target_curie}")
        start = time.perf_counter_ns()
        evidence = any(router.inverted_index.get(token) for token in tokenize_normalize(observation.query_text))
        ranked = router.route_candidates(observation.query_text, top_k=max(5, 2))
        tied = bool(evidence and len(ranked) > 1 and math.isclose(ranked[0][1], ranked[1][1], rel_tol=0, abs_tol=1e-12))
        prediction = ranked[0][0].curie if evidence and ranked and not tied else None
        # Router fallbacks are not lexical matches and do not count toward recall.
        top5 = [concept.curie for concept, _ in ranked[:5]] if evidence else []
        latencies.append((time.perf_counter_ns() - start) / 1e6)
        rows.append(
            {
                "source_id": observation.source_id,
                "group_id": observation.group_id,
                "source_kind": observation.source_kind,
                "target_curie": observation.target_curie,
                "predicted_curie": prediction,
                "top5_curies": top5,
                "top1_correct": prediction == observation.target_curie,
                "recall_at_5": observation.target_curie in top5,
                "abstention_reason": "no_matching_evidence" if not evidence else "tied_top_score" if tied else None,
            }
        )
    count = len(rows)
    by_target = defaultdict(list)
    for row in rows:
        by_target[row["target_curie"]].append(row)
    labels = sorted(set(by_target) | {row["predicted_curie"] for row in rows if row["predicted_curie"] is not None})
    class_metrics = {}
    for label in labels:
        tp = sum(row["target_curie"] == label and row["predicted_curie"] == label for row in rows)
        fp = sum(row["target_curie"] != label and row["predicted_curie"] == label for row in rows)
        fn = sum(row["target_curie"] == label and row["predicted_curie"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        class_metrics[label] = {
            "support": len(by_target[label]),
            "precision": precision,
            "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    latencies.sort()
    answered = sum(row["predicted_curie"] is not None for row in rows)
    reasons = Counter(row["abstention_reason"] for row in rows if row["abstention_reason"])
    return {
        "test_observations": count,
        "ontology_concepts": len(ontology.concepts),
        "observed_target_classes": sum(bool(values) for values in by_target.values()),
        "top1_accuracy_with_abstention": sum(row["top1_correct"] for row in rows) / count,
        "recall_at_5": sum(row["recall_at_5"] for row in rows) / count,
        "macro_recall_at_5": sum(
            sum(row["recall_at_5"] for row in values) / len(values) for values in by_target.values() if values
        )
        / sum(bool(values) for values in by_target.values()),
        "macro_precision": sum(value["precision"] for value in class_metrics.values()) / len(labels),
        "macro_recall": sum(value["recall"] for value in class_metrics.values()) / len(labels),
        "macro_f1": sum(value["f1"] for value in class_metrics.values()) / len(labels),
        "macro_label_policy": "Union of test target and non-abstained predicted classes; zero division = 0",
        "answered": answered,
        "coverage": answered / count,
        "abstained": count - answered,
        "abstention_counts": dict(reasons),
        "tie_policy": "Top1 abstains at equal scores within 1e-12; top5 uses router CURIE tie order",
        "per_class": class_metrics,
        "latency_ms": {
            "samples": count,
            "median": latencies[(count - 1) // 2],
            "p95": latencies[math.ceil(count * 0.95) - 1],
            "percentile_method": "nearest rank",
            "scope": "Query token evidence check, full ontology routing, tie/abstention and top5 selection; excludes index build, generation, split, metrics and file IO",
            "warmup": "none; includes first query",
        },
        "observations": rows,
    }


def main() -> None:
    from olaya.evaluation import dataset_manifest, split_observations
    from olaya.ontology_io import load_ontology
    from olaya.real_ontology_train import generate_real_ontology_observations

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain", default="generic", help="template family (generic, doid, schemaorg, gs1)")
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    ontology_bytes = args.ontology.read_bytes()
    ontology = load_ontology(args.ontology)
    observations = generate_real_ontology_observations(
        ontology, domain=args.domain, num_samples=args.num_samples, seed=args.seed
    )
    splits = split_observations(observations, seed=args.seed)
    test_observations = splits["test"]
    report = {
        "evaluation": "held_out_template_family_lexical_retrieval",
        "seed": args.seed,
        "requested_sample_budget": args.num_samples,
        "generated_observations": len(observations),
        "domain": args.domain,
        "python": sys.version,
        "platform": platform.platform(),
        "ontology_sha256": hashlib.sha256(ontology_bytes).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "router_sha256": hashlib.sha256(Path(inspect.getfile(CandidateRouter)).read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(
            Path(inspect.getfile(generate_real_ontology_observations)).read_bytes()
        ).hexdigest(),
        "splitter_sha256": hashlib.sha256(Path(inspect.getfile(split_observations)).read_bytes()).hexdigest(),
        "dataset_manifest": dataset_manifest(ontology, splits),
        "metrics": evaluate_observations(ontology, test_observations),
        "limitations": [
            "Targets and query terms derive from ontology templates; these are synthetic, not independently labeled observations.",
            "Held-out template families do not establish real-world accuracy, cross-ontology transfer or general language understanding.",
            "Uses query text only with full ontology retrieval, no injected gold target candidate or state metadata.",
            "No calibration, oracle-shortlist accuracy, decision-task accuracy or deployment performance is measured.",
            "Real ABox loading is disabled. Report omits raw query text.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "test_observations": len(test_observations)}))


if __name__ == "__main__":
    main()
