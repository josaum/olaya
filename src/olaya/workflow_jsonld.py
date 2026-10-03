"""Offline JSON-LD emission for TBox-bound workflow executions.

Predictions are linked to classes as decisions, not asserted as input membership.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any
from urllib.parse import quote, urlsplit
from uuid import uuid4

from olaya.ontology import Ontology

WF = "urn:olaya:workflow:"
PREFIXES = {
    "wf": WF,
    "prov": "http://www.w3.org/ns/prov#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


def absolute_iri(value: Any) -> str:
    if not isinstance(value, str) or not value or re.search(r"[\s<>\"{}|\\^`]", value):
        raise ValueError("Expected an absolute HTTP(S) or URN IRI")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https", "urn"}:
        raise ValueError("Expected an absolute HTTP(S) or URN IRI")
    if parts.scheme in {"http", "https"} and not parts.netloc:
        raise ValueError("HTTP(S) IRI requires an authority")
    if parts.scheme == "urn" and not re.match(r"^urn:[a-zA-Z0-9][a-zA-Z0-9-]{0,31}:.+$", value):
        raise ValueError("URN requires a namespace and identifier")
    return value


def validate_prefixes(prefixes: Any) -> dict[str, str]:
    if not isinstance(prefixes, dict):
        raise ValueError("prefixes must be a mapping")
    for name, namespace in prefixes.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError("Invalid prefix name")
        absolute_iri(namespace)
        if name in PREFIXES or name in {"http", "https", "urn"}:
            raise ValueError("Reserved prefix cannot be rebound")
    return dict(prefixes)


def expand_iri(value: str, prefixes: dict[str, str]) -> str:
    if not isinstance(value, str):
        raise ValueError("Identifier must be a string")
    if value.startswith(("https://", "http://", "urn:")):
        return absolute_iri(value)
    prefix, separator, suffix = value.partition(":")
    if not separator or not suffix or prefix not in prefixes:
        raise ValueError(f"Unbound compact IRI: {value!r}")
    return absolute_iri(prefixes[prefix] + suffix)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def ontology_digest(ontology: Ontology) -> str:
    data = ontology.to_dict()
    data["concepts"] = sorted(data["concepts"], key=lambda row: row["curie"])
    if "properties" in data:
        data["properties"] = sorted(data["properties"], key=lambda row: row["curie"])
    return canonical_hash(data)


def execution_id(document: dict[str, Any], run_id: str | None) -> str:
    """Keep generated execution resources distinct from definitions and class IRIs."""
    candidate = absolute_iri(run_id) if run_id is not None else f"urn:uuid:{uuid4()}"
    prefixes = document["prefixes"]
    reserved = {document["id"], document["tbox"], expand_iri(document["type"], prefixes)}
    for name, node in document["steps"].items():
        reserved.add(document["id"] + "/steps/" + quote(name, safe=""))
        if "emit" in node:
            reserved.add(expand_iri(node["emit"]["type"], prefixes))
            reserved.update(expand_iri(prop, prefixes) for prop in node["emit"].get("properties", {}))
        else:
            reserved.add(expand_iri(node["type"], prefixes))
            question = node["decide"]["question"]
            for key in ("root", "concept"):
                if key in question:
                    reserved.add(expand_iri(question[key], prefixes))
            for key in ("options", "criteria"):
                reserved.update(expand_iri(value, prefixes) for value in question.get(key, []))
    generated = {candidate, candidate + "/intent"}
    for index in range(len(document["steps"])):
        generated.update({candidate + "/steps/" + str(index), candidate + "/steps/" + str(index) + "/answer"})
    if reserved & generated:
        raise ValueError("Execution identifiers overlap workflow, TBox, or class definitions")
    return candidate


def _ref(iri: str) -> dict[str, str]:
    return {"@id": iri}


def build_trace(
    document: dict[str, Any],
    workflow_hash: str,
    tbox_hash: str,
    records: list[dict[str, Any]],
    terminal: dict[str, Any] | None,
    run_id: str | None,
    model_id: str,
) -> dict[str, Any]:
    """Build a self-contained context and linked execution graph, without input text."""
    prefixes = validate_prefixes(document["prefixes"])
    context: dict[str, Any] = {"@version": 1.1, **PREFIXES, **prefixes}
    workflow_id = absolute_iri(document["id"])
    tbox_id = absolute_iri(document["tbox"])
    run_id = execution_id(document, run_id)
    graph: list[dict[str, Any]] = [
        {
            "@id": tbox_id,
            "@type": ["owl:Ontology", "prov:Entity"],
            "wf:sha256": tbox_hash,
        },
        {
            "@id": workflow_id,
            "@type": ["wf:WorkflowDefinition", "prov:Entity"],
            "wf:declaredType": _ref(expand_iri(document["type"], prefixes)),
            "wf:tbox": _ref(tbox_id),
            "wf:sha256": workflow_hash,
        },
    ]
    step_ids = []
    for node_name, node in document["steps"].items():
        step_id = workflow_id + "/steps/" + quote(node_name, safe="")
        node_type = node["emit"]["type"] if "emit" in node else node["type"]
        graph.append(
            {
                "@id": step_id,
                "@type": ["wf:StepDefinition", "prov:Entity"],
                "wf:declaredType": _ref(expand_iri(node_type, prefixes)),
                "wf:inWorkflow": _ref(workflow_id),
                "wf:tbox": _ref(tbox_id),
            }
        )
    generated = []
    for index, record in enumerate(records):
        step_id = workflow_id + "/steps/" + quote(record["node"], safe="")
        activity_id = run_id + "/steps/" + str(index)
        step_ids.append(_ref(activity_id))
        activity = {
            "@id": activity_id,
            "@type": ["prov:Activity", "wf:DecisionActivity", expand_iri(record["type"], prefixes)],
            "wf:sequence": index,
            "wf:status": record["status"],
            "wf:reason": record.get("reason", ""),
            "wf:step": _ref(step_id),
            "wf:tbox": _ref(tbox_id),
            "prov:used": [_ref(step_id), _ref(tbox_id)],
            "prov:wasInformedBy": _ref(run_id),
        }
        if record.get("next"):
            activity["wf:nextStep"] = _ref(workflow_id + "/steps/" + quote(record["next"], safe=""))
        question = record.get("question", {})
        if question:
            activity["wf:primitive"] = question["type"]
            scope = question.get("root") or question.get("concept")
            if scope:
                activity["wf:scope"] = _ref(expand_iri(scope, prefixes))
            for key in ("options", "criteria"):
                values = question.get(key)
                if isinstance(values, list):
                    activity["wf:options"] = {"@list": [_ref(expand_iri(v, prefixes)) for v in values]}
        answer = record.get("answer")
        if answer is not None:
            answer_id = activity_id + "/answer"
            answer_node: dict[str, Any] = {
                "@id": answer_id,
                "@type": ["wf:DecisionRecord", "prov:Entity"],
                "prov:wasGeneratedBy": _ref(activity_id),
                "wf:status": record["status"],
            }
            selected = record.get("selected_type")
            if selected:
                answer_node["wf:predictedType"] = _ref(expand_iri(selected, prefixes))
            for key in ("confidence", "score", "noul", "subsumes", "axioms_verified"):
                value = answer.get(key)
                if isinstance(value, bool) or (isinstance(value, (int, float)) and math.isfinite(value)):
                    answer_node["wf:" + key] = value
            probabilities = answer.get("candidate_probabilities") or answer.get("probabilities")
            if isinstance(probabilities, dict):
                slots = []
                for slot, probability in probabilities.items():
                    if (
                        isinstance(probability, bool)
                        or not isinstance(probability, (int, float))
                        or not math.isfinite(probability)
                    ):
                        continue
                    item: dict[str, Any] = {"@type": "wf:CandidateProbability", "wf:probability": probability}
                    if question.get("type") == "concept" or question.get("type") == "choice":
                        item["wf:candidateType"] = _ref(expand_iri(slot, prefixes))
                    elif question.get("type") == "score":
                        criteria = question.get("criteria", [])
                        if str(slot).isdigit() and int(slot) < len(criteria):
                            item["wf:candidateType"] = _ref(expand_iri(criteria[int(slot)], prefixes))
                    else:
                        item["wf:option"] = str(slot)
                    slots.append(item)
                answer_node["wf:candidateProbabilities"] = slots
            graph.append(answer_node)
            activity["prov:generated"] = _ref(answer_id)
            generated.append(_ref(answer_id))
        graph.append(activity)
    if terminal is not None:
        intent_id = run_id + "/intent"
        intent: dict[str, Any] = {
            "@id": intent_id,
            "@type": ["wf:ActionIntent", "prov:Entity", expand_iri(terminal["type"], prefixes)],
            "prov:wasGeneratedBy": _ref(run_id),
            "wf:tbox": _ref(tbox_id),
            "wf:step": _ref(workflow_id + "/steps/" + quote(terminal["node"], safe="")),
        }
        for prop, value in terminal.get("properties", {}).items():
            intent[expand_iri(prop, prefixes)] = value
        graph.append(intent)
        generated.append(_ref(intent_id))
    graph.append(
        {
            "@id": run_id,
            "@type": ["prov:Activity", "wf:WorkflowRun", expand_iri(document["type"], prefixes)],
            "wf:modelIdentifier": str(model_id),
            "wf:workflow": _ref(workflow_id),
            "wf:tbox": _ref(tbox_id),
            "wf:steps": {"@list": step_ids},
            "wf:status": "completed" if terminal is not None else "failed",
            "prov:used": [_ref(workflow_id), _ref(tbox_id)],
            "prov:generated": generated,
        }
    )
    return {"@context": context, "@graph": graph}
