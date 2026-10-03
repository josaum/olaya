"""Source-level split integrity; derived tasks must be expanded after splitting."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from olaya.ontology import Ontology


def stable_id(*parts: str) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


@dataclass(frozen=True)
class SourceObservation:
    source_id: str
    group_id: str
    target_curie: str
    query_text: str
    source_kind: str


def split_observations(observations: list[SourceObservation], seed: int = 42) -> dict[str, list[SourceObservation]]:
    """Group-disjoint 60/15/10/15 split, with text-linked groups kept together.

    Fractions apply to connected source groups, not necessarily row counts.
    Require four independent groups rather than silently producing empty splits.
    """
    parents: dict[str, str] = {}
    seen_ids: dict[str, SourceObservation] = {}
    seen_text: dict[str, SourceObservation] = {}
    unique: list[SourceObservation] = []

    def root(group: str) -> str:
        parents.setdefault(group, group)
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    for row in observations:
        if not row.source_id or not row.group_id or not normalized_text(row.query_text):
            raise ValueError("Source IDs, group IDs and observation text must be nonempty")
        previous = seen_ids.get(row.source_id)
        if previous is not None and previous != row:
            raise ValueError("Duplicate source ID has inconsistent evidence or targets")
        text = normalized_text(row.query_text)
        prior = seen_text.get(text)
        if prior is not None:
            if prior.target_curie != row.target_curie:
                raise ValueError("Identical observation text has conflicting targets")
            a, b = root(prior.group_id), root(row.group_id)
            parents[max(a, b)] = min(a, b)
        root(row.group_id)
        seen_text[text] = row
        if previous is None:
            unique.append(row)
        seen_ids[row.source_id] = row
    groups: dict[str, list[SourceObservation]] = {}
    for row in unique:
        groups.setdefault(root(row.group_id), []).append(row)
    keys = sorted(groups)
    if len(keys) < 4:
        raise ValueError("Evaluation requires at least four independent source groups")
    random.Random(seed).shuffle(keys)
    # Reserve one group per partition, then allocate remaining groups by largest remainder.
    ratios = [0.6, 0.15, 0.1, 0.15]
    quotas = [len(keys) * ratio for ratio in ratios]
    counts = [max(1, int(q)) for q in quotas]
    while sum(counts) > len(keys):
        index = max((i for i in range(4) if counts[i] > 1), key=lambda i: counts[i] - quotas[i])
        counts[index] -= 1
    while sum(counts) < len(keys):
        index = max(range(4), key=lambda i: quotas[i] - counts[i])
        counts[index] += 1
    result = {}
    offset = 0
    for name, count in zip(("train", "validation", "calibration", "test"), counts, strict=True):
        result[name] = sorted(
            (row for key in keys[offset : offset + count] for row in groups[key]), key=lambda r: r.source_id
        )
        offset += count
    return result


def dataset_manifest(ontology: Ontology, splits: dict[str, list[SourceObservation]]) -> dict:
    """Hash evidence and ontology without exporting raw source text or metadata."""
    ontology_data = ontology.to_dict()
    ontology_data["concepts"] = sorted(ontology_data["concepts"], key=lambda concept: concept["curie"])
    if "properties" in ontology_data:
        ontology_data["properties"] = sorted(ontology_data["properties"], key=lambda prop: prop["curie"])
    ontology_payload = json.dumps(ontology_data, sort_keys=True, ensure_ascii=False)
    return {
        "schema_version": 1,
        "ontology_sha256": hashlib.sha256(ontology_payload.encode()).hexdigest(),
        "splits": {
            name: {
                "observations": len(rows),
                "groups": len({row.group_id for row in rows}),
                "source_ids": sorted({row.source_id for row in rows}),
                "group_ids": sorted({row.group_id for row in rows}),
                "evidence_sha256": stable_id(
                    *sorted(
                        stable_id(row.source_id, row.group_id, row.target_curie, row.query_text, row.source_kind)
                        for row in rows
                    )
                ),
            }
            for name, rows in splits.items()
        },
    }
