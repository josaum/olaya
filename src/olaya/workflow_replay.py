"""Replay the declared transition policy from emitted traces without model inference.

Hashes are consistency checks, not signatures. A recorded backend failure is a
replay marker, not evidence that the failure actually happened.
"""

from __future__ import annotations

import copy
import json
from typing import Any
from urllib.parse import quote

from olaya.ontology import Ontology
from olaya.workflow import CompiledWorkflow, run_workflow
from olaya.workflow_jsonld import PREFIXES, absolute_iri, expand_iri


def _ref(value: Any) -> str:
    if not isinstance(value, dict) or set(value) != {"@id"}:
        raise ValueError("Expected emitted IRI reference")
    return absolute_iri(value["@id"])


def _types(entity: dict) -> list[str]:
    types = entity.get("@type")
    if not isinstance(types, list) or any(not isinstance(value, str) for value in types):
        raise ValueError("Invalid emitted entity types")
    return types


def _native(iri: Any, classes: dict[str, str]) -> str:
    expanded = _ref(iri)
    if expanded not in classes:
        raise ValueError("Trace references an unknown TBox class")
    return classes[expanded]


def _raw_answer(entity: dict, activity: dict, question: dict, classes: dict[str, str], prefixes: dict) -> dict:
    kind = question["type"]
    raw: dict[str, Any] = {"type": kind}
    reason = activity.get("wf:reason")
    if reason == "backend_abstained":
        raw["abstained"] = True
    elif reason == "unsupported_evidence":
        raw["evidence_status"] = "unmatched"
    for key in ("confidence", "score", "noul", "subsumes", "axioms_verified"):
        if "wf:" + key in entity:
            raw[key] = entity["wf:" + key]
    slots = entity.get("wf:candidateProbabilities")
    if slots is not None:
        if not isinstance(slots, list) or not slots:
            raise ValueError("Invalid candidate probability slots")
        probabilities = {}
        criteria = {expand_iri(value, prefixes): str(index) for index, value in enumerate(question.get("criteria", []))}
        for slot in slots:
            if not isinstance(slot, dict) or slot.get("@type") != "wf:CandidateProbability":
                raise ValueError("Invalid candidate probability entity")
            if kind in {"choice", "concept"}:
                key = _native(slot.get("wf:candidateType"), classes)
            elif kind == "score":
                iri = _ref(slot.get("wf:candidateType"))
                if iri not in criteria:
                    raise ValueError("Unknown ordinal class")
                key = criteria[iri]
            else:
                key = slot.get("wf:option")
                if not isinstance(key, str):
                    raise ValueError("Invalid numeric probability option")
            if key in probabilities:
                raise ValueError("Duplicate probability slot")
            probabilities[key] = slot.get("wf:probability")
        raw["candidate_probabilities" if kind == "concept" else "probabilities"] = probabilities
    if kind in {"choice", "concept"}:
        selected = entity.get("wf:predictedType")
        if selected is not None:
            selected = _native(selected, classes)
        else:
            probabilities = raw.get("candidate_probabilities" if kind == "concept" else "probabilities", {})
            # Gated traces may retain distributions but omit a selected class.
            # Restore only an unambiguous argmax; tie policy runs before selection.
            if probabilities:
                try:
                    maximum = max(probabilities.values())
                    best = [key for key, value in probabilities.items() if value == maximum]
                except TypeError as exc:
                    raise ValueError("Invalid probability values") from exc
                selected = best[0] if len(best) == 1 else None
        if selected is not None:
            raw["choice" if kind == "choice" else "curie"] = selected
    elif kind == "subsumes":
        raw["concept"] = question["concept"]
    return raw


class _ReplayEngine:
    """Private response iterator. No model, tokenizer, network, or inference backend."""

    def __init__(self, ontology: Ontology, records: list[tuple[str, dict | None]], model_id: str):
        self.ontology = ontology
        self.records = records
        self.model_id = model_id
        self.index = 0
        self.mismatch = False

    def system_one(self, *, state: Any, questions: dict) -> dict:
        if self.index >= len(self.records):
            self.mismatch = True
            raise ValueError("Truncated replay")
        node, answer = self.records[self.index]
        self.index += 1
        if list(questions) != [node]:
            self.mismatch = True
            raise ValueError("Replay decision order mismatch")
        if answer is None:
            raise ValueError("Recorded backend failure marker")
        return {"model": self.model_id, "answers": {node: copy.deepcopy(answer)}}


def replay_workflow(compiled: CompiledWorkflow, trace: dict) -> dict:
    """Verify an emitted trace against compiled policy and emit a fresh replay run.

    This checks transition policy, typed output, and retained decision evidence.
    It cannot recover omitted inputs or certify model predictions/failures.
    """
    if not isinstance(compiled, CompiledWorkflow):
        raise ValueError("Expected compiled workflow")
    try:
        trace = json.loads(json.dumps(trace, allow_nan=False))
    except (TypeError, RecursionError) as exc:
        raise ValueError("Replay requires a finite JSON trace") from exc
    doc = compiled.document
    if not isinstance(trace, dict) or set(trace) != {"@context", "@graph"}:
        raise ValueError("Expected emitted trace document")
    if trace["@context"] != {"@version": 1.1, **PREFIXES, **doc["prefixes"]}:
        raise ValueError("Trace context differs from compiled mappings")
    graph = trace["@graph"]
    if not isinstance(graph, list) or len(graph) > 3 * len(doc["steps"]) + 4 + len(doc.get("resources", [])):
        raise ValueError("Invalid or oversized trace graph")
    entities = {}
    for entity in graph:
        if not isinstance(entity, dict):
            raise ValueError("Invalid graph entity")
        identity = absolute_iri(entity.get("@id"))
        if identity in entities:
            raise ValueError("Duplicate trace identity")
        _types(entity)
        entities[identity] = entity
    workflow = entities.get(doc["id"], {})
    tbox = entities.get(doc["tbox"], {})
    if workflow.get("wf:sha256") != compiled.workflow_hash or tbox.get("wf:sha256") != compiled.tbox_hash:
        raise ValueError("Workflow or TBox hash mismatch")
    runs = [entity for entity in graph if "wf:WorkflowRun" in entity["@type"]]
    if len(runs) != 1:
        raise ValueError("Trace requires exactly one workflow run")
    original_run = runs[0]
    if original_run.get("wf:executionMode") != "live":
        raise ValueError("Replay requires an original live emitted trace")
    run_id = original_run["@id"]
    model_id = original_run.get("wf:modelIdentifier")
    if not isinstance(model_id, str) or len(model_id) > 256:
        raise ValueError("Trace requires a reported model identifier")
    activities = [entity for entity in graph if "wf:DecisionActivity" in entity["@type"]]
    if len(activities) > len(doc["steps"]):
        raise ValueError("Too many decision activities")
    if original_run.get("wf:steps") != {
        "@list": [{"@id": run_id + "/steps/" + str(i)} for i in range(len(activities))]
    }:
        raise ValueError("Invalid ordered decision activity links")
    ontology = Ontology.from_json(json.loads(compiled.ontology_json))
    classes = {expand_iri(curie, doc["prefixes"]): curie for curie in ontology.concepts}
    node_ids = {doc["id"] + "/steps/" + quote(name, safe=""): name for name in doc["steps"]}
    records = []
    for index, activity in enumerate(activities):
        if (
            activity.get("@id") != run_id + "/steps/" + str(index)
            or type(activity.get("wf:sequence")) is not int
            or activity["wf:sequence"] != index
        ):
            raise ValueError("Out-of-order decision activities")
        node_id = _ref(activity.get("wf:step"))
        if node_id not in node_ids or "decide" not in doc["steps"][node_ids[node_id]]:
            raise ValueError("Activity references unknown decision node")
        name = node_ids[node_id]
        answer_id = _ref(activity.get("prov:generated"))
        if answer_id != activity["@id"] + "/answer" or answer_id not in entities:
            raise ValueError("Missing linked decision record")
        answer = entities[answer_id]
        if "wf:DecisionRecord" not in answer["@type"] or _ref(answer.get("prov:wasGeneratedBy")) != activity["@id"]:
            raise ValueError("Invalid decision record link")
        if activity.get("wf:status") == "error":
            records.append((name, None))
        else:
            records.append(
                (
                    name,
                    _raw_answer(answer, activity, doc["steps"][name]["decide"]["question"], classes, doc["prefixes"]),
                )
            )
    placeholders: dict = {}
    for node in doc["steps"].values():
        if "decide" in node:
            value = placeholders
            for part in node["decide"]["state"].split(".")[1:]:
                value = value.setdefault(part, {})
    engine = _ReplayEngine(ontology, records, model_id)
    expected = run_workflow(compiled, engine, placeholders, run_id=run_id)
    # Backend failures may precede model identity reporting; preserve the retained
    # identity as metadata, without claiming it generated the replay decisions.
    for entity in expected["@graph"]:
        if entity["@id"] == run_id:
            entity["wf:modelIdentifier"] = model_id
    if engine.mismatch or engine.index != len(records) or expected != trace:
        raise ValueError("Trace does not reproduce compiled transition policy")
    fresh_engine = _ReplayEngine(ontology, records, model_id)
    replay = run_workflow(compiled, fresh_engine, placeholders)
    for entity in replay["@graph"]:
        if "wf:WorkflowRun" in entity["@type"]:
            entity["wf:modelIdentifier"] = model_id
            entity["wf:executionMode"] = "replay"
            entity["wf:replayOf"] = {"@id": run_id}
    return replay
