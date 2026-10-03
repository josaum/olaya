"""Bounded declarative TBox workflows: typed decisions and trace-only terminal emission."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any
from urllib.parse import quote

from olaya.ontology import Ontology
from olaya.workflow_jsonld import PREFIXES, build_trace, execution_id, expand_iri, ontology_digest, validate_prefixes


@dataclass(frozen=True)
class CompiledWorkflow:
    canonical_json: str
    workflow_hash: str
    tbox_hash: str
    ontology_json: str

    @property
    def document(self) -> dict:
        return json.loads(self.canonical_json)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, RecursionError) as exc:
        raise ValueError("Expected finite canonical JSON") from exc


def _fields(obj: Any, allowed: set[str], required: set[str] = frozenset()) -> None:
    if not isinstance(obj, dict) or set(obj) - allowed or required - set(obj):
        raise ValueError("Invalid or unknown workflow fields")


def _number(value: Any, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected finite numeric value")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError("Numeric value outside allowed range")
    return float(value)


def compile_workflow(document: dict, ontology: Ontology) -> CompiledWorkflow:
    """Validate the whole graph and snapshot its workflow and supplied TBox."""
    ontology = Ontology.from_json(json.loads(ontology.to_json()))
    doc = json.loads(_canonical(document))
    _fields(
        doc,
        {"version", "id", "type", "prefixes", "tbox", "start", "steps", "resources"},
        {"version", "id", "type", "prefixes", "tbox", "start", "steps"},
    )
    if type(doc["version"]) is not int or doc["version"] != 1:
        raise ValueError("Unsupported workflow version")
    prefixes = validate_prefixes(doc["prefixes"])
    if not isinstance(prefixes, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in prefixes.items()
    ):
        raise ValueError("Invalid prefix map")
    for namespace in prefixes.values():
        expand_iri(namespace, {})
    expand_iri(doc["id"], {})
    expand_iri(doc["tbox"], {})

    def iri_index(values: Any) -> dict[str, str]:
        mapped = {}
        for value in values:
            iri = expand_iri(value, prefixes)
            if iri in mapped:
                raise ValueError("Ambiguous expanded TBox identifier")
            mapped[iri] = value
        return mapped

    classes = iri_index(ontology.concepts)
    properties_index = iri_index(ontology.properties)

    def cls(value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Unknown TBox class")
        resolved = classes.get(expand_iri(value, prefixes))
        if resolved is None:
            raise ValueError("Unknown TBox class")
        return resolved

    doc["type"] = cls(doc["type"])
    steps = doc["steps"]
    if not isinstance(steps, dict) or not 1 <= len(steps) <= 128 or any(not isinstance(n, str) or not n for n in steps):
        raise ValueError("Workflow requires 1..128 named nodes")
    if not isinstance(doc["start"], str) or doc["start"] not in steps:
        raise ValueError("Unknown start node")
    definition_ids = [doc["id"], doc["tbox"]] + [doc["id"] + "/steps/" + quote(name, safe="") for name in steps]
    if len(set(definition_ids)) != len(definition_ids):
        raise ValueError("Workflow definition identifiers collide")
    resources = doc.get("resources", [])
    if not isinstance(resources, list) or len(resources) > 128:
        raise ValueError("Resources must be a list of at most 128 declarations")
    resource_types = {}
    for resource in resources:
        _fields(resource, {"id", "type", "properties"}, {"id", "type"})
        resource["id"] = expand_iri(resource["id"], prefixes)
        resource["type"] = cls(resource["type"])
        if resource["id"] in resource_types or resource["id"] in definition_ids:
            raise ValueError("Declared resource identifiers collide")
        resource_types[resource["id"]] = resource["type"]

    def validate_properties(owner_type: str, properties: Any) -> dict:
        if not isinstance(properties, dict):
            raise ValueError("Typed resource properties must be a map")
        ancestors = ontology.get_ancestors(owner_type)
        normalized_properties = {}
        for prop, value in properties.items():
            prop = properties_index.get(expand_iri(prop, prefixes))
            if prop not in ontology.properties:
                raise ValueError("Unknown TBox property")
            if any(expand_iri(prop, prefixes).startswith(namespace) for namespace in PREFIXES.values()):
                raise ValueError("Property cannot modify reserved trace vocabulary")
            if prop in normalized_properties:
                raise ValueError("Repeated expanded property")
            definition = ontology.properties[prop]
            if not {cls(domain) for domain in definition.domains} <= ancestors:
                raise ValueError("Declared type violates property domain")
            if isinstance(value, dict):
                _fields(value, {"@id"}, {"@id"})
                value["@id"] = expand_iri(value["@id"], prefixes)
            elif not isinstance(value, (str, int, float, bool)):
                raise ValueError("Property values must be literals or IRI references")
            # Only known class ranges are checked. Datatype facets and unknown
            # ranges are not a datatype-reasoning implementation.
            known_ranges = set()
            for range_id in definition.ranges:
                try:
                    known_class = classes.get(expand_iri(range_id, prefixes))
                except ValueError:
                    known_class = None
                if known_class is not None:
                    known_ranges.add(known_class)
            if known_ranges:
                if not isinstance(value, dict) or value["@id"] not in resource_types:
                    raise ValueError("Class-range properties require a declared typed resource reference")
                target_ancestors = ontology.get_ancestors(resource_types[value["@id"]])
                if not known_ranges <= target_ancestors:
                    raise ValueError("Declared target violates property class range")
            normalized_properties[prop] = value
        return normalized_properties

    for resource in resources:
        resource["properties"] = validate_properties(resource["type"], resource.get("properties", {}))
    edges: dict[str, list[str]] = {}
    for name, node in steps.items():
        if not isinstance(node, dict):
            raise ValueError("Invalid workflow node")
        if "emit" in node:
            _fields(node, {"emit"}, {"emit"})
            emit = node["emit"]
            _fields(emit, {"type", "properties"}, {"type"})
            emit_type = cls(emit["type"])
            emit["type"] = emit_type
            emit["properties"] = validate_properties(emit_type, emit.get("properties", {}))
            edges[name] = []
            continue
        _fields(
            node,
            {"type", "decide", "accept", "route", "on_abstain", "on_error"},
            {"type", "decide", "route", "on_abstain", "on_error"},
        )
        node["type"] = cls(node["type"])
        decide = node["decide"]
        _fields(decide, {"state", "question"}, {"state", "question"})
        path = decide["state"]
        if not isinstance(path, str) or not path.startswith("input.") or any(not p for p in path.split(".")):
            raise ValueError("State must reference an input dotted path")
        question = decide["question"]
        _fields(question, {"type", "root", "concept", "options", "criteria", "instructions"}, {"type"})
        kind = question["type"]
        if not isinstance(kind, str) or kind not in {"concept", "choice", "score", "subsumes", "noul"}:
            raise ValueError("Unknown primitive")
        bindings = set(question) & {"root", "concept"}
        expected = {"concept"} if kind == "subsumes" else {"root"}
        if kind == "noul":
            if len(bindings) != 1:
                raise ValueError("Noul requires exactly one TBox binding")
        elif bindings != expected:
            raise ValueError("Primitive requires exactly its declared TBox binding")
        if not ("root" in question or "concept" in question):
            raise ValueError("Every decision must bind a TBox class")
        for binding in ("root", "concept"):
            if binding in question:
                question[binding] = cls(question[binding])
        if kind == "subsumes" and "concept" not in question:
            raise ValueError("Subsumes requires concept")
        if kind == "concept" and "root" not in question:
            raise ValueError("Concept requires root")
        if "instructions" in question and not isinstance(question["instructions"], str):
            raise ValueError("Instructions must be text")
        if kind in {"choice", "score"}:
            key = "options" if kind == "choice" else "criteria"
            values = question.get(key)
            if "root" not in question or not isinstance(values, list) or not values or len(set(values)) != len(values):
                raise ValueError("Ordered distinct TBox options and root required")
            values = [cls(value) for value in values]
            if len(set(values)) != len(values):
                raise ValueError("Repeated expanded option class")
            question[key] = values
            for value in values:
                if question["root"] not in ontology.get_ancestors(value):
                    raise ValueError("Option class lies outside bound root")
        if kind != "choice" and "options" in question or kind != "score" and "criteria" in question:
            raise ValueError("Irrelevant primitive fields")
        accept = node.get("accept", {})
        _fields(accept, {"min_confidence", "min_margin", "require_axioms"})
        for key in ("min_confidence", "min_margin"):
            if key in accept:
                _number(accept[key], 0, 1)
        if "require_axioms" in accept and type(accept["require_axioms"]) is not bool:
            raise ValueError("require_axioms must be boolean")
        routes = node["route"]
        if kind in {"concept", "choice"}:
            if not isinstance(routes, dict):
                raise ValueError("Categorical routes must be a class map")
            normalized_routes = {cls(value): destination for value, destination in routes.items()}
            if len(normalized_routes) != len(routes):
                raise ValueError("Repeated expanded route class")
            node["route"] = routes = normalized_routes
            for value in routes:
                if kind == "choice" and value not in question["options"]:
                    raise ValueError("Choice route not present in options")
                if kind == "concept" and question["root"] not in ontology.get_ancestors(value):
                    raise ValueError("Concept route outside bound root")
            destinations = list(routes.values())
        else:
            if not isinstance(routes, list):
                raise ValueError("Numeric routes must be interval lists")
            intervals = []
            for interval in routes:
                _fields(interval, {"gte", "lt", "to"}, {"to"})
                lower = _number(interval["gte"]) if "gte" in interval else -math.inf
                upper = _number(interval["lt"]) if "lt" in interval else math.inf
                if lower >= upper:
                    raise ValueError("Empty route interval")
                intervals.append((lower, upper))
            ordered = sorted(intervals)
            if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:], strict=False)):
                raise ValueError("Overlapping numeric routes")
            destinations = [r["to"] for r in routes]
        destinations += [node["on_abstain"], node["on_error"]]
        if any(not isinstance(to, str) or to not in steps for to in destinations):
            raise ValueError("Unknown transition")
        edges[name] = destinations
    visiting, visited = set(), set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError("Workflow cycles are forbidden")
        if name in visited:
            return
        visiting.add(name)
        for destination in edges[name]:
            visit(destination)
        visiting.remove(name)
        visited.add(name)

    visit(doc["start"])
    if visited != set(steps):
        raise ValueError("Unreachable workflow node")
    canonical = _canonical(doc)
    return CompiledWorkflow(
        canonical, hashlib.sha256(canonical.encode()).hexdigest(), ontology_digest(ontology), ontology.to_json()
    )


def _mapping(value: Any) -> dict:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, dict):
        raise ValueError("Backend returned invalid shape")
    return value


def _backend_question(question: dict, ontology: Ontology) -> dict:
    adapted = copy.deepcopy(question)

    def description(curie: str) -> str:
        concept = ontology.concepts[curie]
        return f"{concept.label}: {concept.description}" if concept.description else concept.label

    if question["type"] == "choice":
        adapted["criteria"] = {curie: description(curie) for curie in adapted.pop("options")}
    elif question["type"] == "score":
        adapted["criteria"] = [description(curie) for curie in question["criteria"]]
    elif question["type"] == "noul" and not question.get("instructions"):
        bound = question.get("concept", question.get("root"))
        adapted["instructions"] = f"Does the observation satisfy {description(bound)}?"
    return adapted


def _evaluate_answer(
    raw: dict, question: dict, accept: dict, ontology: Ontology
) -> tuple[dict, str | None, float | None, str | None]:
    kind = question["type"]
    if raw.get("type") != kind:
        raise ValueError("Backend answered a different primitive")
    sanitized = {"type": kind}
    # Explicit abstention must not expose arbitrary backend metadata.
    if raw.get("abstained") is True:
        return {**sanitized, "abstained": True}, None, None, "backend_abstained"
    if "abstained" in raw and type(raw["abstained"]) is not bool:
        raise ValueError("Invalid abstention flag")
    if "evidence_status" in raw and raw["evidence_status"] not in {"supported", "contradicted"}:
        return {**sanitized, "abstained": True}, None, None, "unsupported_evidence"
    confidence = raw.get("confidence")
    if confidence is not None:
        sanitized["confidence"] = _number(confidence, 0, 1)
    if "min_confidence" in accept and (confidence is None or confidence < accept["min_confidence"]):
        return sanitized, None, None, "insufficient_confidence"
    if "axioms_verified" in raw:
        if type(raw["axioms_verified"]) is not bool:
            raise ValueError("Invalid axiom flag")
        sanitized["axioms_verified"] = raw["axioms_verified"]
    if accept.get("require_axioms") and raw.get("axioms_verified") is not True:
        return sanitized, None, None, "axioms_unverified"
    selected = None
    numeric = None
    distribution = raw.get("candidate_probabilities" if kind == "concept" else "probabilities")
    margin = None
    if kind in {"choice", "concept", "score"}:
        if not isinstance(distribution, dict) or not distribution:
            raise ValueError("Missing decision distribution")
        probs = {}
        for key, value in distribution.items():
            if not isinstance(key, str):
                raise ValueError("Invalid probability slot")
            if kind == "choice" and key not in question["options"]:
                raise ValueError("Choice probability outside options")
            if kind == "concept" and (
                key not in ontology.concepts or question["root"] not in ontology.get_ancestors(key)
            ):
                raise ValueError("Concept probability outside root")
            if kind == "score" and key not in {str(i) for i in range(len(question["criteria"]))}:
                raise ValueError("Unknown ordinal probability slot")
            probs[key] = _number(value, 0, 1)
        if kind == "choice" and set(probs) != set(question["options"]):
            raise ValueError("Incomplete choice probability distribution")
        if kind == "score" and set(probs) != {str(i) for i in range(len(question["criteria"]))}:
            raise ValueError("Incomplete ordinal probability distribution")
        if abs(sum(probs.values()) - 1) > 0.01:
            raise ValueError("Invalid probability normalization")
        sanitized["candidate_probabilities" if kind == "concept" else "probabilities"] = probs
        ranked = sorted(probs.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
        if len(ranked) > 1 and math.isclose(ranked[0], ranked[1], abs_tol=1e-12):
            return sanitized, None, None, "tied_probability"
    if kind in {"choice", "concept"}:
        key = "choice" if kind == "choice" else "curie"
        selected = raw.get(key)
        if selected not in distribution:
            raise ValueError("Selection absent from probability distribution")
        if distribution[selected] < max(distribution.values()):
            raise ValueError("Selection disagrees with probability distribution")
        sanitized[key] = selected
    elif kind == "score":
        numeric = _number(raw.get("score"), 0, len(question["criteria"]) - 1)
        sanitized["score"] = numeric
        expected = sum(int(slot) * probability for slot, probability in sanitized["probabilities"].items())
        levels = len(question["criteria"])
        rounding_tolerance = 0.00005 * (1 + levels * (levels - 1) / 2) + 1e-12
        if abs(numeric - expected) > rounding_tolerance:
            raise ValueError("Score disagrees with ordinal distribution")
    else:
        numeric = _number(raw.get(kind), 0, 1)
        sanitized[kind] = numeric
        margin = abs(2 * numeric - 1)
        if math.isclose(numeric, 0.5, abs_tol=1e-12):
            return sanitized, None, None, "tied_probability"
        if kind == "subsumes" and raw.get("concept") != question["concept"]:
            raise ValueError("Subsumes answer changed bound concept")
        selected = question.get("concept", question.get("root")) if numeric > 0.5 else None
    if "min_margin" in accept and (margin is None or margin < accept["min_margin"]):
        return sanitized, None, None, "insufficient_margin"
    return sanitized, selected, numeric, None


def run_workflow(compiled: CompiledWorkflow, engine: Any, inputs: dict, *, run_id: str | None = None) -> dict:
    """Execute a validated acyclic snapshot; emit typed traces without state or errors."""
    if not isinstance(compiled, CompiledWorkflow):
        raise ValueError("Expected compiled workflow")
    if hashlib.sha256(compiled.canonical_json.encode()).hexdigest() != compiled.workflow_hash:
        raise ValueError("Workflow snapshot integrity failure")
    if not isinstance(inputs, dict) or len(_canonical(inputs).encode()) > 65536:
        raise ValueError("Inputs must be a JSON object within 64 KiB")
    run_id = execution_id(compiled.document, run_id)
    if ontology_digest(engine.ontology) != compiled.tbox_hash:
        raise ValueError("Engine TBox drifted from compiled snapshot")
    ontology = Ontology.from_json(json.loads(compiled.ontology_json))
    if ontology_digest(ontology) != compiled.tbox_hash:
        raise ValueError("TBox snapshot integrity failure")
    doc = compiled.document
    records = []
    terminal = None
    model_id = "not_reported"
    current = doc["start"]
    for _ in range(len(doc["steps"])):
        if ontology_digest(engine.ontology) != compiled.tbox_hash:
            raise ValueError("Engine TBox drifted during workflow execution")
        node = doc["steps"][current]
        if "emit" in node:
            terminal = {
                "node": current,
                "type": node["emit"]["type"],
                "properties": copy.deepcopy(node["emit"].get("properties", {})),
            }
            break
        question = node["decide"]["question"]
        record = {
            "node": current,
            "type": node["type"],
            "question": copy.deepcopy(question),
            "answer": {},
            "selected_type": None,
        }
        try:
            state = inputs
            for part in node["decide"]["state"].split(".")[1:]:
                if not isinstance(state, dict) or part not in state:
                    raise ValueError("Missing input reference")
                state = state[part]
            response = _mapping(
                engine.system_one(
                    state=copy.deepcopy(state), questions={current: _backend_question(question, ontology)}
                )
            )
            answers = _mapping(response.get("answers"))
            answer = _mapping(answers.get(current))
            if isinstance(response.get("model"), str):
                model_id = response["model"][:256]
            sanitized, selected, numeric, reason = _evaluate_answer(answer, question, node.get("accept", {}), ontology)
            record["answer"] = sanitized
            record["selected_type"] = selected
            if reason:
                next_node = node["on_abstain"]
                record.update(status="abstained", reason=reason)
            else:
                routes = node["route"]
                if isinstance(routes, dict):
                    next_node = routes.get(selected)
                else:
                    next_node = next(
                        (r["to"] for r in routes if r.get("gte", -math.inf) <= numeric < r.get("lt", math.inf)), None
                    )
                if next_node is None:
                    next_node = node["on_abstain"]
                    record.update(status="abstained", reason="uncovered_route")
                else:
                    record.update(status="accepted", reason="accepted")
        except Exception:
            next_node = node["on_error"]
            record.update(status="error", reason="backend_error", answer={}, selected_type=None)
        record["next"] = next_node
        records.append(record)
        current = next_node
    return build_trace(doc, compiled.workflow_hash, compiled.tbox_hash, records, terminal, run_id, model_id)
