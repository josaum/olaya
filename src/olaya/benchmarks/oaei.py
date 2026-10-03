"""OAEI Conference track (OntoFarm, ra1) ontology-alignment benchmark.

This module implements the *standard* OAEI alignment evaluation:

1. Parse the seven OntoFarm ontologies (classes and properties).
2. For every one of the 21 ontology pairs, a matcher scores every cross pair of
   entities of the same kind (class/class, property/property), keeps the
   mutual-best (stable 1:1) correspondences and applies a confidence threshold.
3. The resulting alignment is compared with the ra1 reference alignment and
   precision / recall / F-measure are computed, separately for classes (M1),
   properties (M2) and both (M3), with micro (absolute TP/FP/FN over the 21
   pairs) and macro (mean over pairs) aggregation, following the OAEI
   Conference track conventions.

Nothing here feeds the gold answer into the candidate set, and no metric is
derived from synthetic score distributions.  The only non-lexical step is an
optional alignment-coherence repair that uses the OWL 2 RL reasoner of the
repository to drop correspondences whose combination is logically incoherent
(a sub-class pair mapped onto a disjoint pair).  It is a decision over
competing candidate correspondences, which is the one place the axiom
machinery is legitimately applicable in a matching task.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olaya.ontology import Concept, ObjectProperty, Ontology
from olaya.owl2_rl import OWL2RLReasoner

# XML Namespaces used in OWL and OAEI Alignment RDF
NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
NS_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
NS_OWL = "http://www.w3.org/2002/07/owl#"
NS_ALIGN = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"

# The seven OntoFarm ontologies that have ra1 reference alignments, and the 21 pairs.
CONFERENCE_ONTOLOGIES = ("cmt", "conference", "confOf", "edas", "ekaw", "iasted", "sigkdd")
CONFERENCE_PAIRS = tuple((a, b) for i, a in enumerate(CONFERENCE_ONTOLOGIES) for b in CONFERENCE_ONTOLOGIES[i + 1 :])

MODALITIES = ("M1", "M2", "M3")  # classes, properties, classes + properties

_PROPERTY_TAGS = (
    "ObjectProperty",
    "DatatypeProperty",
    "FunctionalProperty",
    "InverseFunctionalProperty",
    "TransitiveProperty",
    "SymmetricProperty",
    "AnnotationProperty",
)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def clean_uri(uri: str | None) -> str:
    """Extract local name / fragment from URI."""
    if not uri:
        return ""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]


def camel_to_words(name: str) -> str:
    """Convert CamelCase or snake_case identifier to human-readable words."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", s)
    return s.replace("_", " ").replace("-", " ").strip()


STOP_WORDS = frozenset(
    {"a", "an", "the", "of", "has", "have", "is", "in", "by", "to", "and", "for", "with", "at", "on", "or"}
)


def stem(token: str) -> str:
    """Tiny deterministic suffix stripper (plurals and a few agentive/verbal forms).

    It is intentionally conservative: it only needs to fold obvious inflections such
    as ``papers``/``paper`` or ``reviews``/``review`` so that two ontologies naming the
    same notion in singular and plural still share a token.
    """
    t = token
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("sses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss") and not t.endswith("us"):
        return t[:-1]
    return t


def tokenize_identifier(text: str, *, use_stem: bool = True, drop_stop: bool = True) -> list[str]:
    """Split an identifier or phrase into normalised lowercase tokens."""
    words = re.findall(r"[A-Za-z0-9]+", camel_to_words(text))
    out: list[str] = []
    for w in words:
        w = w.lower()
        if drop_stop and w in STOP_WORDS:
            continue
        if use_stem:
            w = stem(w)
        if w:
            out.append(w)
    return out


def normalize_string(text: str) -> str:
    """Lowercase alphanumeric-only form used by the StringEquiv baseline."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def levenshtein_similarity(a: str, b: str) -> float:
    """1 - normalised Levenshtein distance; 1.0 for identical strings."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


# ---------------------------------------------------------------------------
# OWL parsing
# ---------------------------------------------------------------------------


def _entity_id(elem: ET.Element) -> str:
    return clean_uri(elem.attrib.get(f"{{{NS_RDF}}}ID") or elem.attrib.get(f"{{{NS_RDF}}}about"))


def _text_of(elem: ET.Element, tag: str) -> str:
    child = elem.find(tag)
    return child.text.strip() if child is not None and child.text else ""


def _referenced_classes(container: ET.Element, skip: str) -> list[str]:
    members = []
    for mc in container.findall(f".//{{{NS_OWL}}}Class"):
        mid = clean_uri(
            mc.attrib.get(f"{{{NS_RDF}}}ID")
            or mc.attrib.get(f"{{{NS_RDF}}}about")
            or mc.attrib.get(f"{{{NS_RDF}}}resource")
        )
        if mid and mid != "Thing" and mid != skip:
            members.append(mid)
    return members


class OWLParser:
    """Parser for the RDF/XML OWL ontologies used by the OAEI Conference track.

    Classes become :class:`~olaya.ontology.Concept` objects (with subsumption,
    disjointness, equivalence, union / intersection operands) and object *and*
    datatype properties become :class:`~olaya.ontology.ObjectProperty` objects
    (``metadata["kind"]`` is ``"object"`` or ``"datatype"``).
    """

    @staticmethod
    def parse(owl_file: str | Path) -> Ontology:
        root = ET.parse(owl_file).getroot()
        onto_name = Path(owl_file).stem
        ontology = Ontology(name=onto_name)

        classes: dict[str, Concept] = {}
        parents_map: dict[str, set[str]] = {}
        disjoints_map: dict[str, set[str]] = {}
        equivalences_map: dict[str, set[str]] = {}
        intersections_map: dict[str, list[list[str]]] = {}
        unions_map: dict[str, list[list[str]]] = {}
        disjoint_unions: list[tuple[str, list[str]]] = []

        for elem in root.iter(f"{{{NS_OWL}}}Class"):
            curie = _entity_id(elem)
            if not curie or curie.startswith("_") or curie == "Thing" or curie == "Nothing":
                continue
            label = _text_of(elem, f"{{{NS_RDFS}}}label")
            comment = _text_of(elem, f"{{{NS_RDFS}}}comment")

            for sc in elem.findall(f"{{{NS_RDFS}}}subClassOf"):
                res = clean_uri(sc.attrib.get(f"{{{NS_RDF}}}resource"))
                nested = sc.find(f"{{{NS_OWL}}}Class")
                if nested is not None:
                    res = _entity_id(nested)
                if res and res != "Thing" and res != curie:
                    parents_map.setdefault(curie, set()).add(res)
            for dj in elem.findall(f"{{{NS_OWL}}}disjointWith"):
                res = clean_uri(dj.attrib.get(f"{{{NS_RDF}}}resource"))
                nested = dj.find(f"{{{NS_OWL}}}Class")
                if nested is not None:
                    res = _entity_id(nested)
                if res and res != curie:
                    disjoints_map.setdefault(curie, set()).add(res)
            for eq in elem.findall(f"{{{NS_OWL}}}equivalentClass"):
                res = clean_uri(eq.attrib.get(f"{{{NS_RDF}}}resource"))
                if res and res != "Thing" and res != curie:
                    equivalences_map.setdefault(curie, set()).add(res)
                nested = eq.find(f"{{{NS_OWL}}}Class")
                if nested is not None:
                    nid = _entity_id(nested)
                    if nid and nid != "Thing" and nid != curie:
                        equivalences_map.setdefault(curie, set()).add(nid)
                    union = nested.find(f"{{{NS_OWL}}}unionOf")
                    if union is not None and (members := _referenced_classes(union, curie)):
                        unions_map.setdefault(curie, []).append(members)
                    inter = nested.find(f"{{{NS_OWL}}}intersectionOf")
                    if inter is not None and (members := _referenced_classes(inter, curie)):
                        intersections_map.setdefault(curie, []).append(members)
            for dju in elem.findall(f"{{{NS_OWL}}}disjointUnionOf"):
                members = _referenced_classes(dju, curie)
                if len(members) >= 2:
                    disjoint_unions.append((curie, members))

            if curie not in classes:
                classes[curie] = Concept(
                    curie=curie,
                    label=label or camel_to_words(curie),
                    description=comment,
                    metadata={"has_label": bool(label)},
                )
            else:
                if label and not classes[curie].metadata.get("has_label"):
                    classes[curie].label = label
                    classes[curie].metadata["has_label"] = True
                if comment and not classes[curie].description:
                    classes[curie].description = comment

        for c in classes.values():
            ontology.add_concept(c)
        for curie, parents in parents_map.items():
            for p in parents:
                if curie in classes and p in classes:
                    ontology.add_subsumption(curie, p)
        for curie, disjoints in disjoints_map.items():
            for d in disjoints:
                if curie in classes and d in classes:
                    ontology.add_disjointness(curie, d)
        for curie, equivs in equivalences_map.items():
            for e in equivs:
                if curie in classes and e in classes:
                    ontology.add_equivalence(curie, e)
        for curie, lists in unions_map.items():
            for ops in lists:
                ontology.add_union(curie, ops)
        for curie, lists in intersections_map.items():
            for ops in lists:
                ontology.add_intersection(curie, ops)
        for curie, members in disjoint_unions:
            ontology.add_disjoint_union(curie, members)

        for adc in root.iter(f"{{{NS_OWL}}}AllDisjointClasses"):
            members = [m for m in _referenced_classes(adc, "") if m in classes]
            for i, m1 in enumerate(members):
                for m2 in members[i + 1 :]:
                    ontology.add_disjointness(m1, m2)

        # Properties: object/datatype properties may be declared under their characteristic
        # tag (owl:FunctionalProperty, owl:InverseFunctionalProperty, ...) with rdf:type
        # giving the actual kind, and the same property can be declared more than once.
        props: dict[str, ObjectProperty] = {}
        for tag in _PROPERTY_TAGS:
            for pelem in root.iter(f"{{{NS_OWL}}}{tag}"):
                pcurie = _entity_id(pelem)
                if not pcurie:
                    continue
                types = {tag}
                for t in pelem.findall(f"{{{NS_RDF}}}type"):
                    types.add(clean_uri(t.attrib.get(f"{{{NS_RDF}}}resource")))
                prop = props.get(pcurie)
                if prop is None:
                    prop = ObjectProperty(curie=pcurie, label=camel_to_words(pcurie), metadata={"kind": "object"})
                    props[pcurie] = prop
                label = _text_of(pelem, f"{{{NS_RDFS}}}label")
                if label:
                    prop.label = label
                    prop.metadata["has_label"] = True
                comment = _text_of(pelem, f"{{{NS_RDFS}}}comment")
                if comment and not prop.description:
                    prop.description = comment
                for d in pelem.findall(f"{{{NS_RDFS}}}domain"):
                    for dres in [clean_uri(d.attrib.get(f"{{{NS_RDF}}}resource"))] + _referenced_classes(d, ""):
                        if dres and dres in classes:
                            prop.domains.add(dres)
                for r in pelem.findall(f"{{{NS_RDFS}}}range"):
                    rres = r.attrib.get(f"{{{NS_RDF}}}resource") or ""
                    if rres.startswith("http://www.w3.org/2001/XMLSchema#"):
                        prop.metadata["kind"] = "datatype"
                        prop.metadata["datatype"] = clean_uri(rres)
                    for rr in [clean_uri(rres)] + _referenced_classes(r, ""):
                        if rr and rr in classes:
                            prop.ranges.add(rr)
                for inv in pelem.findall(f"{{{NS_OWL}}}inverseOf"):
                    ires = clean_uri(inv.attrib.get(f"{{{NS_RDF}}}resource"))
                    for sub_tag in _PROPERTY_TAGS:
                        nested = inv.find(f"{{{NS_OWL}}}{sub_tag}")
                        if nested is not None:
                            ires = _entity_id(nested)
                    if ires:
                        prop.inverses.add(ires)
                for sp in pelem.findall(f"{{{NS_RDFS}}}subPropertyOf"):
                    spres = clean_uri(sp.attrib.get(f"{{{NS_RDF}}}resource"))
                    if spres:
                        prop.parents.add(spres)
                if "DatatypeProperty" in types:
                    prop.metadata["kind"] = "datatype"
                prop.is_transitive |= "TransitiveProperty" in types
                prop.is_functional |= "FunctionalProperty" in types
                prop.is_inverse_functional |= "InverseFunctionalProperty" in types
                prop.is_symmetric |= "SymmetricProperty" in types

        for prop in props.values():
            ontology.add_property(prop)
        return ontology


# ---------------------------------------------------------------------------
# Alignments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OAEIAlignment:
    """One correspondence ``entity1 relation entity2`` with a confidence measure."""

    entity1: str
    entity2: str
    relation: str = "="
    measure: float = 1.0

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.entity1, self.entity2, self.relation)


class OAEIAlignmentParser:
    """Parser for the OAEI RDF Alignment format (local names only)."""

    @staticmethod
    def parse(rdf_file: str | Path) -> list[OAEIAlignment]:
        root = ET.parse(rdf_file).getroot()
        alignments: list[OAEIAlignment] = []
        for cell in root.iter(f"{{{NS_ALIGN}}}Cell"):
            e1 = cell.find(f"{{{NS_ALIGN}}}entity1")
            e2 = cell.find(f"{{{NS_ALIGN}}}entity2")
            if e1 is None or e2 is None:
                continue
            u1 = clean_uri(e1.attrib.get(f"{{{NS_RDF}}}resource"))
            u2 = clean_uri(e2.attrib.get(f"{{{NS_RDF}}}resource"))
            rel = _text_of(cell, f"{{{NS_ALIGN}}}relation") or "="
            meas = _text_of(cell, f"{{{NS_ALIGN}}}measure")
            if u1 and u2:
                alignments.append(OAEIAlignment(u1, u2, rel, float(meas) if meas else 1.0))
        return alignments


def entity_kind(ontology: Ontology, name: str) -> str | None:
    """``"class"``, ``"property"`` or ``None`` for an entity local name."""
    if name in ontology.concepts:
        return "class"
    if name in ontology.properties:
        return "property"
    return None


def correspondence_kind(corr: OAEIAlignment, source: Ontology, target: Ontology) -> str:
    """Kind of a correspondence: ``class``, ``property`` or ``other`` (unknown entity)."""
    k1 = entity_kind(source, corr.entity1)
    k2 = entity_kind(target, corr.entity2)
    if k1 == "class" and k2 == "class":
        return "class"
    if k1 == "property" and k2 == "property":
        return "property"
    return "other"


# ---------------------------------------------------------------------------
# Precision / recall
# ---------------------------------------------------------------------------


@dataclass
class PRF:
    """Absolute counts with derived precision / recall / F-measures."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def f_beta(self, beta: float) -> float:
        p, r = self.precision, self.recall
        if p == 0.0 and r == 0.0:
            return 0.0
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r)

    @property
    def f1(self) -> float:
        return self.f_beta(1.0)

    def __add__(self, other: PRF) -> PRF:
        return PRF(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "f0.5": round(self.f_beta(0.5), 4),
            "f2": round(self.f_beta(2.0), 4),
        }


def evaluate_alignment(
    system: Iterable[OAEIAlignment],
    reference: Iterable[OAEIAlignment],
    source: Ontology,
    target: Ontology,
) -> dict[str, PRF]:
    """Compare a system alignment with a reference alignment.

    Returns counts for the three OAEI Conference modalities: ``M1`` (class
    correspondences only), ``M2`` (property correspondences only) and ``M3``
    (everything).  A correspondence is correct iff the same ``(entity1, entity2,
    relation)`` triple is in the reference; confidences are ignored once the
    alignment has been thresholded.
    """
    sys_keys = {c.key: c for c in system}
    ref_keys = {c.key: c for c in reference}

    def kind_of(corr: OAEIAlignment) -> str:
        return correspondence_kind(corr, source, target)

    out = {m: PRF() for m in MODALITIES}
    for key, corr in sys_keys.items():
        kind = kind_of(corr)
        hit = key in ref_keys
        for m in _modalities_for(kind):
            if hit:
                out[m].tp += 1
            else:
                out[m].fp += 1
    for key, corr in ref_keys.items():
        if key in sys_keys:
            continue
        for m in _modalities_for(kind_of(corr)):
            out[m].fn += 1
    return out


def _modalities_for(kind: str) -> tuple[str, ...]:
    if kind == "class":
        return ("M1", "M3")
    if kind == "property":
        return ("M2", "M3")
    return ("M3",)


# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------

ScoreFn = Callable[[str, str], float]


@dataclass
class EntityProfile:
    """Lexical profile of one ontology entity."""

    name: str
    kind: str
    tokens: tuple[str, ...]  # normalised name/label tokens
    normalized: str  # alphanumeric lowercase name/label
    context: tuple[str, ...]  # tokens from comment and neighbours (weaker evidence)


def build_profiles(ontology: Ontology, *, use_stem: bool = True, drop_stop: bool = True) -> list[EntityProfile]:
    """Build lexical profiles for all classes and properties of an ontology."""

    def tokenize(text: str) -> list[str]:
        return tokenize_identifier(text, use_stem=use_stem, drop_stop=drop_stop)

    profiles: list[EntityProfile] = []
    for curie, concept in sorted(ontology.concepts.items()):
        name_text = concept.label if concept.metadata.get("has_label") else curie
        context = tokenize(concept.description)
        for p in sorted(concept.parents):
            context.extend(tokenize(p))
        profiles.append(
            EntityProfile(
                name=curie,
                kind="class",
                tokens=tuple(tokenize(name_text)),
                normalized=normalize_string(name_text),
                context=tuple(context),
            )
        )
    for curie, prop in sorted(ontology.properties.items()):
        name_text = prop.label if prop.metadata.get("has_label") else curie
        context = tokenize(prop.description)
        for d in sorted(prop.domains):
            context.extend(tokenize(d))
        for r in sorted(prop.ranges):
            context.extend(tokenize(r))
        profiles.append(
            EntityProfile(
                name=curie,
                kind="property",
                tokens=tuple(tokenize(name_text)),
                normalized=normalize_string(name_text),
                context=tuple(context),
            )
        )
    return profiles


class Matcher:
    """Base class: ``score_all`` returns every cross pair with a confidence in [0, 1]."""

    name = "matcher"

    def score_all(self, source: Ontology, target: Ontology) -> list[OAEIAlignment]:
        raise NotImplementedError


class StringEquivMatcher(Matcher):
    """OAEI ``StringEquiv`` baseline: equality of normalised local names / labels."""

    name = "StringEquiv"

    def score_all(self, source: Ontology, target: Ontology) -> list[OAEIAlignment]:
        out = []
        for s in build_profiles(source):
            for t in build_profiles(target):
                if s.kind == t.kind and s.normalized and s.normalized == t.normalized:
                    out.append(OAEIAlignment(s.name, t.name, "=", 1.0))
        return out


@dataclass
class LexicalMatcherConfig:
    """Knobs of :class:`LexicalMatcher`.  Defaults are the configuration used for the
    published results; every switch was evaluated and kept only when it measurably
    helped (see ``benchmarks/results/oaei-conference-ra1.md``)."""

    use_stem: bool = True  # fold plurals / simple inflections (+0.00-0.01 F1)
    drop_stop: bool = False  # dropping stop words (has, of, ...) cost ~0.05 F1 on properties
    use_idf: bool = True  # IDF-weighted token overlap instead of plain Jaccard (+0.03 F1)
    token_metric: str = "weighted_jaccard"  # or "cosine" (slightly worse, threshold-sensitive)
    use_char_similarity: bool = True  # normalised Levenshtein on the concatenated name
    char_weight: float = 1.0  # weight of the char similarity once it passes ``char_min``
    char_min: float = 0.9  # only near-identical spellings count; ungated it cost ~0.1 F1
    context_weight: float = 0.2  # weight of comment/neighbour token overlap (+0.01-0.02 F1)
    match_properties: bool = True


class LexicalMatcher(Matcher):
    """Deterministic lexical matcher built from the repository's candidate-router
    ingredients: camelCase/underscore splitting, stop-word removal, light stemming,
    IDF-weighted token overlap, a character-level similarity for inflectional
    variants and a small context term from comments / neighbouring entities."""

    name = "olaya-lexical"

    def __init__(self, config: LexicalMatcherConfig | None = None):
        self.config = config or LexicalMatcherConfig()

    def score_all(self, source: Ontology, target: Ontology) -> list[OAEIAlignment]:
        cfg = self.config
        src = build_profiles(source, use_stem=cfg.use_stem, drop_stop=cfg.drop_stop)
        tgt = build_profiles(target, use_stem=cfg.use_stem, drop_stop=cfg.drop_stop)
        if not cfg.match_properties:
            src = [p for p in src if p.kind == "class"]
            tgt = [p for p in tgt if p.kind == "class"]

        # Document frequencies over the union of both ontologies (symmetric, deterministic).
        df: dict[str, int] = {}
        for p in src + tgt:
            for tok in set(p.tokens):
                df[tok] = df.get(tok, 0) + 1
        n_docs = len(src) + len(tgt)

        def weight(tok: str) -> float:
            if not cfg.use_idf:
                return 1.0
            return 1.0 + math.log((n_docs + 1.0) / (df.get(tok, 0) + 1.0))

        def token_sim(a: EntityProfile, b: EntityProfile) -> float:
            sa, sb = set(a.tokens), set(b.tokens)
            if not sa or not sb:
                return 0.0
            if cfg.token_metric == "cosine":
                dot = sum(weight(t) ** 2 for t in sa & sb)
                na = math.sqrt(sum(weight(t) ** 2 for t in sa))
                nb = math.sqrt(sum(weight(t) ** 2 for t in sb))
                return dot / (na * nb) if na and nb else 0.0
            inter = sum(weight(t) for t in sa & sb)
            union = sum(weight(t) for t in sa | sb)
            return inter / union if union else 0.0

        def context_sim(a: EntityProfile, b: EntityProfile) -> float:
            ca = set(a.tokens) | set(a.context)
            cb = set(b.tokens) | set(b.context)
            if not ca or not cb:
                return 0.0
            return len(ca & cb) / len(ca | cb)

        out: list[OAEIAlignment] = []
        for s in src:
            for t in tgt:
                if s.kind != t.kind:
                    continue
                score = token_sim(s, t)
                if cfg.use_char_similarity and s.normalized and t.normalized:
                    char_sim = levenshtein_similarity(s.normalized, t.normalized)
                    if char_sim >= cfg.char_min:
                        score = max(score, cfg.char_weight * char_sim)
                if cfg.context_weight > 0.0:
                    score = (1.0 - cfg.context_weight) * score + cfg.context_weight * context_sim(s, t)
                if score > 0.0:
                    out.append(OAEIAlignment(s.name, t.name, "=", round(score, 6)))
        return out


# ---------------------------------------------------------------------------
# Selection: mutual best + threshold, optional coherence repair
# ---------------------------------------------------------------------------


def select_mutual_best(scored: Iterable[OAEIAlignment]) -> list[OAEIAlignment]:
    """Keep correspondences where each entity is the other's best match.

    Ties are broken deterministically (higher measure, then entity names), so the
    result is a 1:1 alignment that does not depend on input order.
    """
    best1: dict[str, OAEIAlignment] = {}
    best2: dict[str, OAEIAlignment] = {}
    for c in sorted(scored, key=lambda c: (-c.measure, c.entity1, c.entity2)):
        if c.entity1 not in best1:
            best1[c.entity1] = c
        if c.entity2 not in best2:
            best2[c.entity2] = c
    kept = [c for c in best1.values() if best2.get(c.entity2) is c]
    return sorted(kept, key=lambda c: (-c.measure, c.entity1, c.entity2))


def apply_threshold(alignment: Iterable[OAEIAlignment], threshold: float) -> list[OAEIAlignment]:
    return [c for c in alignment if c.measure >= threshold]


def coherence_repair(alignment: list[OAEIAlignment], source: Ontology, target: Ontology) -> list[OAEIAlignment]:
    """Drop class correspondences that make the alignment logically incoherent.

    Two equivalence correspondences ``(a1 = b1)`` and ``(a2 = b2)`` are incoherent when
    ``a1 ⊑ a2`` in the source but ``b1`` and ``b2`` are disjoint in the target (or the
    symmetric situation), because the merged ontology would make ``a1``
    unsatisfiable.  The OWL 2 RL deductive closure of the repository is used to
    decide subsumption and disjointness; the lower-confidence correspondence of a
    conflicting pair is removed (ties: the lexicographically later one).
    """
    src_closure = OWL2RLReasoner(source).compute_closure()
    tgt_closure = OWL2RLReasoner(target).compute_closure()
    src_sub = {(c, p) for c, p in src_closure.subclasses}
    tgt_sub = {(c, p) for c, p in tgt_closure.subclasses}

    classes = [c for c in alignment if correspondence_kind(c, source, target) == "class"]
    others = [c for c in alignment if correspondence_kind(c, source, target) != "class"]
    ordered = sorted(classes, key=lambda c: (-c.measure, c.entity1, c.entity2))
    kept: list[OAEIAlignment] = []
    for cand in ordered:
        conflict = False
        for acc in kept:
            if cand.entity1 == acc.entity1 or cand.entity2 == acc.entity2:
                continue
            src_related = (cand.entity1, acc.entity1) in src_sub or (acc.entity1, cand.entity1) in src_sub
            tgt_related = (cand.entity2, acc.entity2) in tgt_sub or (acc.entity2, cand.entity2) in tgt_sub
            if (src_related and tgt_closure.are_disjoint(cand.entity2, acc.entity2)) or (
                tgt_related and src_closure.are_disjoint(cand.entity1, acc.entity1)
            ):
                conflict = True
                break
        if not conflict:
            kept.append(cand)
    merged = kept + others
    return sorted(merged, key=lambda c: (-c.measure, c.entity1, c.entity2))


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


@dataclass
class PairEvaluation:
    pair: str
    source: str
    target: str
    reference_size: int
    system_size: int
    results: dict[str, PRF]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": self.pair,
            "reference_size": self.reference_size,
            "system_size": self.system_size,
            **{m: self.results[m].to_dict() for m in MODALITIES},
        }


def aggregate(evals: list[PairEvaluation]) -> dict[str, Any]:
    """Micro (absolute counts) and macro (mean of per-pair scores) aggregates."""
    out: dict[str, Any] = {}
    for m in MODALITIES:
        micro = PRF()
        ps, rs, fs = [], [], []
        for e in evals:
            r = e.results[m]
            micro = micro + r
            if r.tp + r.fn == 0:  # no reference correspondences of this kind for the pair
                continue
            ps.append(r.precision)
            rs.append(r.recall)
            fs.append(r.f1)
        n = len(ps)
        out[m] = {
            "micro": micro.to_dict(),
            "macro": {
                "pairs": n,
                "precision": round(sum(ps) / n, 4) if n else 0.0,
                "recall": round(sum(rs) / n, 4) if n else 0.0,
                "f1": round(sum(fs) / n, 4) if n else 0.0,
            },
        }
    return out


@dataclass
class BenchmarkRun:
    matcher: str
    threshold: float
    coherence_repair: bool
    pairs: list[PairEvaluation]
    aggregates: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matcher": self.matcher,
            "threshold": self.threshold,
            "coherence_repair": self.coherence_repair,
            "aggregates": self.aggregates,
            "pairs": [p.to_dict() for p in self.pairs],
        }


class OAEIConferenceBenchmark:
    """Run a matcher over the 21 OntoFarm pairs and score it against ra1."""

    def __init__(self, ontologies_dir: str | Path, alignments_dir: str | Path):
        self.ontologies_dir = Path(ontologies_dir)
        self.alignments_dir = Path(alignments_dir)
        self._ontologies: dict[str, Ontology] = {}
        self._references: dict[tuple[str, str], list[OAEIAlignment]] = {}
        self._scores: dict[tuple[str, str, str], list[OAEIAlignment]] = {}

    # -- data access -------------------------------------------------------
    def ontology_path(self, name: str) -> Path:
        matches = [f for f in self.ontologies_dir.glob("*.owl") if f.stem.lower() == name.lower()]
        if not matches:
            raise FileNotFoundError(f"Ontology {name} not found in {self.ontologies_dir}")
        return matches[0]

    def reference_path(self, source: str, target: str) -> Path:
        candidates = [f for f in self.alignments_dir.glob("*.rdf") if f.stem.lower() == f"{source}-{target}".lower()]
        if not candidates:
            raise FileNotFoundError(f"Reference alignment {source}-{target} not found in {self.alignments_dir}")
        return candidates[0]

    def ontology(self, name: str) -> Ontology:
        key = name.lower()
        if key not in self._ontologies:
            self._ontologies[key] = OWLParser.parse(self.ontology_path(name))
        return self._ontologies[key]

    def reference(self, source: str, target: str) -> list[OAEIAlignment]:
        key = (source.lower(), target.lower())
        if key not in self._references:
            self._references[key] = [
                a for a in OAEIAlignmentParser.parse(self.reference_path(source, target)) if a.relation == "="
            ]
        return self._references[key]

    def dataset_hashes(self) -> dict[str, str]:
        files = sorted(self.ontologies_dir.glob("*.owl")) + sorted(self.alignments_dir.glob("*.rdf"))
        return {f"{f.parent.name}/{f.name}": hashlib.sha256(f.read_bytes()).hexdigest() for f in files}

    # -- matching ------------------------------------------------------------
    def scored_pairs(self, matcher: Matcher, source: str, target: str) -> list[OAEIAlignment]:
        key = (matcher.name, source.lower(), target.lower())
        if key not in self._scores:
            self._scores[key] = matcher.score_all(self.ontology(source), self.ontology(target))
        return self._scores[key]

    def align(
        self,
        matcher: Matcher,
        source: str,
        target: str,
        threshold: float,
        *,
        mutual_best: bool = True,
        repair: bool = False,
    ) -> list[OAEIAlignment]:
        scored = self.scored_pairs(matcher, source, target)
        alignment = select_mutual_best(scored) if mutual_best else sorted(scored, key=lambda c: c.key)
        alignment = apply_threshold(alignment, threshold)
        if repair:
            alignment = coherence_repair(alignment, self.ontology(source), self.ontology(target))
        return alignment

    def evaluate_pair(
        self, matcher: Matcher, source: str, target: str, threshold: float, *, repair: bool = False
    ) -> PairEvaluation:
        src, tgt = self.ontology(source), self.ontology(target)
        ref = self.reference(source, target)
        system = self.align(matcher, source, target, threshold, repair=repair)
        return PairEvaluation(
            pair=f"{source}-{target}",
            source=source,
            target=target,
            reference_size=len(ref),
            system_size=len(system),
            results=evaluate_alignment(system, ref, src, tgt),
        )

    def run(
        self,
        matcher: Matcher,
        threshold: float,
        *,
        repair: bool = False,
        pairs: Iterable[tuple[str, str]] = CONFERENCE_PAIRS,
    ) -> BenchmarkRun:
        evals = [self.evaluate_pair(matcher, s, t, threshold, repair=repair) for s, t in pairs]
        return BenchmarkRun(matcher.name, threshold, repair, evals, aggregate(evals))

    def sweep(
        self,
        matcher: Matcher,
        thresholds: Iterable[float],
        *,
        repair: bool = False,
        modality: str = "M3",
    ) -> tuple[BenchmarkRun, list[dict[str, Any]]]:
        """Run every threshold; return the best run by micro-F1 of ``modality`` and the curve."""
        best: BenchmarkRun | None = None
        curve = []
        for th in thresholds:
            run = self.run(matcher, th, repair=repair)
            agg = run.aggregates
            curve.append(
                {
                    "threshold": th,
                    **{m: {k: agg[m]["micro"][k] for k in ("precision", "recall", "f1")} for m in MODALITIES},
                }
            )
            f1 = agg[modality]["micro"]["f1"]
            if best is None or f1 > best.aggregates[modality]["micro"]["f1"]:
                best = run
        assert best is not None
        return best, curve


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run ``fn`` and return ``(result, elapsed_seconds)``."""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0
