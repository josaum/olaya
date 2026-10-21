"""Load a user-supplied ontology (Turtle, RDF/XML, N-Triples, JSON-LD or O-Laya JSON) into an :class:`Ontology`.

The engine is ontology-agnostic: the TBox is a runtime input, not something baked into the package.
:func:`load_ontology` is the single entry point and accepts

* an :class:`~olaya.ontology.Ontology` (returned unchanged),
* a ``dict`` (O-Laya JSON with a ``concepts`` list, or an expanded/compacted JSON-LD document),
* a path to a file (``.ttl``, ``.owl``, ``.rdf``, ``.xml``, ``.nt``, ``.n3``, ``.trig``, ``.jsonld``,
  ``.json``), or a directory, whose RDF files are merged into one graph (an import closure laid
  out on disk).

RDF parsing uses ``rdflib`` (``pip install 'leio-decision[rdf]'``); O-Laya JSON needs nothing.

Extraction (mirrors ``crates/olaya-core/src/bin/convert_ontology.rs`` where both apply):

* classes: ``owl:Class`` / ``rdfs:Class`` subjects plus every named class used in
  ``rdfs:subClassOf``, ``owl:disjointWith`` or ``owl:equivalentClass``; RDF/RDFS/OWL/XSD built-ins
  and ``owl:Thing`` are dropped;
* parents: named ``rdfs:subClassOf`` objects and the named operands of an ``owl:intersectionOf``
  that is equivalent to (or a superclass of) the class;
* disjointness: ``owl:disjointWith``, ``owl:AllDisjointClasses`` and ``owl:disjointUnionOf``
  (which also asserts the members as subclasses);
* label (``skos:prefLabel`` > ``rdfs:label``, English or untagged first), description
  (``skos:definition``, ``IAO:0000115``, ``rdfs:comment``, ``dcterms:description``; whitespace
  collapsed, truncated to 300 characters) and synonyms (``skos:altLabel``, oboInOwl exact/related
  synonyms, extra labels);
* ``metadata["iri"]`` keeps the full source IRI.

CURIEs default to ``prefix:Local`` from the graph's bound prefixes (``curie_style="prefixed"``);
``curie_style="local"`` reproduces the Rust converter (local name, OBO ``X_123`` -> ``X:123``).
Collisions fall back to the full IRI, so two classes never merge silently.

:func:`ontology_fingerprint` is a SHA-256 over the canonical TBox structure (CURIEs, parents,
disjointness, labels, descriptions, synonyms). It is independent of file formatting and serialisation
order, so a fine-tuned checkpoint can be bound to the ontology it was trained on.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from olaya.ontology import Ontology

RDF_FORMATS: dict[str, str] = {
    ".ttl": "turtle",
    ".turtle": "turtle",
    ".owl": "xml",
    ".rdf": "xml",
    ".xml": "xml",
    ".nt": "nt",
    ".n3": "n3",
    ".trig": "trig",
    ".jsonld": "json-ld",
    ".json-ld": "json-ld",
}
CURIE_STYLES = frozenset({"prefixed", "local"})
_MAX_DESCRIPTION = 300

_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_OWL = "http://www.w3.org/2002/07/owl#"
_XSD = "http://www.w3.org/2001/XMLSchema#"
_SKOS = "http://www.w3.org/2004/02/skos/core#"
_OBO_IN_OWL = "http://www.geneontology.org/formats/oboInOwl#"
_BUILTIN_NAMESPACES = (_RDF, _RDFS, _OWL, _XSD)

_LABEL_PREDICATES = (_SKOS + "prefLabel", _RDFS + "label")
_DESCRIPTION_PREDICATES = (
    _SKOS + "definition",
    "http://purl.obolibrary.org/obo/IAO_0000115",
    _RDFS + "comment",
    "http://purl.org/dc/terms/description",
    "http://purl.org/dc/elements/1.1/description",
)
_SYNONYM_PREDICATES = (
    _SKOS + "altLabel",
    _OBO_IN_OWL + "hasExactSynonym",
    _OBO_IN_OWL + "hasRelatedSynonym",
)


class OntologyFormatError(ValueError):
    """The source could not be recognised or parsed as an ontology."""


def load_ontology(
    source: Ontology | Mapping[str, Any] | str | Path,
    *,
    name: str | None = None,
    format: str | None = None,  # noqa: A002 - mirrors rdflib's keyword
    curie_style: str = "prefixed",
) -> Ontology:
    """Load ``source`` into an :class:`Ontology` (see the module docstring for what is accepted)."""
    if curie_style not in CURIE_STYLES:
        raise ValueError(f"curie_style must be one of {sorted(CURIE_STYLES)}, got {curie_style!r}")
    if isinstance(source, Ontology):
        return source
    if isinstance(source, Mapping):
        if _is_olaya_json(source):
            return _named(Ontology.from_json(dict(source)), name)
        graph = _new_graph()
        graph.parse(data=json.dumps(source), format="json-ld")
        return ontology_from_graph(graph, name=name or "jsonld_ontology", curie_style=curie_style)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Ontology file not found: {path}")
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in RDF_FORMATS)
        if not files:
            raise OntologyFormatError(f"no RDF files ({', '.join(sorted(RDF_FORMATS))}) in {path}")
        graph = _new_graph()
        for f in files:
            graph.parse(str(f), format=format or RDF_FORMATS[f.suffix.lower()])
        return ontology_from_graph(graph, name=name or path.name, curie_style=curie_style)

    suffix = path.suffix.lower()
    if format is None and suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, Mapping) and _is_olaya_json(data):
            return _named(Ontology.from_json(dict(data)), name)
        format = "json-ld"
    rdf_format = format or RDF_FORMATS.get(suffix)
    if rdf_format is None:
        raise OntologyFormatError(
            f"cannot infer the format of {path.name}; pass format= (one of {sorted(set(RDF_FORMATS.values()))})"
        )
    graph = _new_graph()
    try:
        graph.parse(str(path), format=rdf_format)
    except Exception as exc:  # rdflib raises many parser-specific types
        raise OntologyFormatError(f"failed to parse {path} as {rdf_format}: {exc}") from exc
    return ontology_from_graph(graph, name=name or _ontology_title(graph) or path.stem, curie_style=curie_style)


def ontology_from_graph(graph: Any, *, name: str = "rdf_ontology", curie_style: str = "prefixed") -> Ontology:
    """Extract the class hierarchy of an ``rdflib.Graph`` into an :class:`Ontology`."""
    from rdflib import RDF, BNode, URIRef
    from rdflib.collection import Collection

    owl = _ns(_OWL)
    rdfs = _ns(_RDFS)

    def named(node: Any) -> bool:
        return isinstance(node, URIRef) and not str(node).startswith(_BUILTIN_NAMESPACES)

    def rdf_list(head: Any) -> list[Any]:
        try:
            return list(Collection(graph, head))
        except Exception:  # malformed list
            return []

    classes: set[Any] = set()
    for cls_type in (owl["Class"], rdfs["Class"]):
        classes.update(s for s in graph.subjects(RDF.type, cls_type) if named(s))
    for pred in (rdfs["subClassOf"], owl["disjointWith"], owl["equivalentClass"]):
        for s, o in graph.subject_objects(pred):
            classes.update(n for n in (s, o) if named(n))
    for _, members in graph.subject_objects(owl["members"]):
        classes.update(m for m in rdf_list(members) if named(m))
    for s, members in graph.subject_objects(owl["disjointUnionOf"]):
        classes.update(n for n in [s, *rdf_list(members)] if named(n))

    curie_of = _curie_map(graph, classes, curie_style)
    parents: dict[Any, set[Any]] = {c: set() for c in classes}
    disjoint: dict[Any, set[Any]] = {c: set() for c in classes}

    def add_disjoint(a: Any, b: Any) -> None:
        if a in classes and b in classes and a != b:
            disjoint[a].add(b)
            disjoint[b].add(a)

    for c in classes:
        for sup in graph.objects(c, rdfs["subClassOf"]):
            if named(sup) and sup in classes and sup != c:
                parents[c].add(sup)
            elif isinstance(sup, BNode):
                parents[c].update(_intersection_operands(graph, sup, owl, rdf_list, classes, c))
        for eq in graph.objects(c, owl["equivalentClass"]):
            if isinstance(eq, BNode):
                parents[c].update(_intersection_operands(graph, eq, owl, rdf_list, classes, c))
        for d in graph.objects(c, owl["disjointWith"]):
            add_disjoint(c, d)
    for group in graph.subjects(RDF.type, owl["AllDisjointClasses"]):
        for members in graph.objects(group, owl["members"]):
            ms = [m for m in rdf_list(members) if m in classes]
            for i, a in enumerate(ms):
                for b in ms[i + 1 :]:
                    add_disjoint(a, b)
    for parent, members in graph.subject_objects(owl["disjointUnionOf"]):
        ms = [m for m in rdf_list(members) if m in classes]
        for m in ms:
            if parent in classes and m != parent:
                parents[m].add(parent)
        for i, a in enumerate(ms):
            for b in ms[i + 1 :]:
                add_disjoint(a, b)

    concepts = []
    for c in sorted(classes, key=lambda n: curie_of[n]):
        curie = curie_of[c]
        labels = _literals(graph, c, _LABEL_PREDICATES)
        label = labels[0] if labels else _local_name(str(c))
        synonyms = _dedupe([*labels[1:], *_literals(graph, c, _SYNONYM_PREDICATES)], exclude={label})
        descriptions = _literals(graph, c, _DESCRIPTION_PREDICATES)
        concepts.append(
            {
                "curie": curie,
                "label": label,
                "description": _clean_description(descriptions[0]) if descriptions else "",
                "synonyms": synonyms,
                "parents": sorted(curie_of[p] for p in parents[c]),
                "disjoint_with": sorted(curie_of[d] for d in disjoint[c]),
                "metadata": {"iri": str(c)},
            }
        )
    return Ontology.from_json({"name": name, "concepts": concepts})


def ontology_fingerprint(ontology: Ontology) -> str:
    """SHA-256 of the canonical TBox (format- and order-independent); see the module docstring."""
    canonical = [
        [
            c.curie,
            c.label,
            c.description,
            sorted(c.synonyms),
            sorted(c.parents),
            sorted(c.disjoint_with),
        ]
        for c in sorted(ontology.concepts.values(), key=lambda c: c.curie)
    ]
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- helpers --------------------------------------------------------------------------------------


def _new_graph() -> Any:
    try:
        from rdflib import Graph
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Loading Turtle / RDF / JSON-LD ontologies needs rdflib: pip install 'leio-decision[rdf]'"
        ) from exc
    return Graph()


def _ns(base: str) -> Any:
    from rdflib import Namespace

    return Namespace(base)


def _is_olaya_json(data: Mapping[str, Any]) -> bool:
    return isinstance(data.get("concepts"), list) and "@context" not in data and "@graph" not in data


def _named(onto: Ontology, name: str | None) -> Ontology:
    if name:
        onto.name = name
    return onto


def _ontology_title(graph: Any) -> str | None:
    from rdflib import RDF

    owl = _ns(_OWL)
    for onto in graph.subjects(RDF.type, owl["Ontology"]):
        titles = _literals(graph, onto, (_RDFS + "label", "http://purl.org/dc/terms/title"))
        if titles:
            return re.sub(r"\s+", "_", titles[0].strip())
    return None


def _intersection_operands(graph: Any, node: Any, owl: Any, rdf_list: Any, classes: set[Any], self_: Any) -> set[Any]:
    out: set[Any] = set()
    for members in graph.objects(node, owl["intersectionOf"]):
        out.update(m for m in rdf_list(members) if m in classes and m != self_)
    return out


def _literals(graph: Any, subject: Any, predicates: Iterable[str]) -> list[str]:
    """Literal values in predicate order; within a predicate English, then untagged, then the rest."""
    from rdflib import Literal, URIRef

    out: list[str] = []
    for pred in predicates:
        lits = [o for o in graph.objects(subject, URIRef(pred)) if isinstance(o, Literal)]
        lits.sort(
            key=lambda lit: (0 if (lit.language or "").startswith("en") else 1 if not lit.language else 2, str(lit))
        )
        out.extend(str(lit).strip() for lit in lits if str(lit).strip())
    return _dedupe(out)


def _dedupe(values: Iterable[str], exclude: set[str] | None = None) -> list[str]:
    seen = set(exclude or ())
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _clean_description(raw: str) -> str:
    cleaned = " ".join(line.strip() for line in raw.splitlines() if line.strip())
    return cleaned if len(cleaned) <= _MAX_DESCRIPTION else cleaned[: _MAX_DESCRIPTION - 3] + "..."


def _local_name(iri: str) -> str:
    for sep in ("#", "/", ":"):
        if sep in iri:
            tail = iri.rsplit(sep, 1)[1]
            if tail:
                return tail
    return iri


def _rust_curie(iri: str) -> str:
    """``extract_curie`` from ``crates/olaya-core/src/arrow_schema.rs``."""
    if "://" not in iri:
        return iri
    local = iri.rsplit("#", 1)[1] if "#" in iri else iri.rsplit("/", 1)[1] if "/" in iri else iri
    prefix, sep, ident = local.partition("_")
    return f"{prefix}:{ident}" if sep and prefix and ident else local


def _curie_map(graph: Any, classes: set[Any], curie_style: str) -> dict[Any, str]:
    proposals: dict[Any, str] = {}
    for c in classes:
        iri = str(c)
        if curie_style == "local":
            proposals[c] = _rust_curie(iri)
            continue
        try:
            prefix, _ns_uri, local = graph.namespace_manager.compute_qname(iri, generate=False)
        except Exception:  # no bound prefix covers this IRI
            prefix, local = "", ""
        proposals[c] = f"{prefix}:{local}" if prefix and local else _rust_curie(iri)
    counts: dict[str, int] = {}
    for curie in proposals.values():
        counts[curie] = counts.get(curie, 0) + 1
    return {c: (curie if counts[curie] == 1 else str(c)) for c, curie in proposals.items()}


__all__ = [
    "CURIE_STYLES",
    "RDF_FORMATS",
    "OntologyFormatError",
    "load_ontology",
    "ontology_fingerprint",
    "ontology_from_graph",
]
