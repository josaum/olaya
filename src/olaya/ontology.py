"""Ontology representation, subsumption DAG, and disjointness axioms for O-Laya."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class Concept:
    """An individual ontological class/concept with formal coordinates."""

    curie: str  # e.g., "INCIDENT:DATABASE_OUTAGE"
    label: str  # e.g., "Database Outage"
    description: str  # e.g., "Critical disruption to database connectivity or read/write operations"
    synonyms: List[str] = field(default_factory=list)
    parents: Set[str] = field(default_factory=set)  # direct superclasses (subClassOf)
    children: Set[str] = field(default_factory=set)  # direct subclasses
    disjoint_with: Set[str] = field(default_factory=set)  # mutually exclusive concepts
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "curie": self.curie,
            "label": self.label,
            "description": self.description,
            "synonyms": self.synonyms,
            "parents": sorted(list(self.parents)),
            "disjoint_with": sorted(list(self.disjoint_with)),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Concept:
        return cls(
            curie=data["curie"],
            label=data["label"],
            description=data.get("description", ""),
            synonyms=data.get("synonyms", []),
            parents=set(data.get("parents", [])),
            disjoint_with=set(data.get("disjoint_with", [])),
            metadata=data.get("metadata", {}),
        )


class Ontology:
    """Ontology substrate maintaining classes, subsumption DAG, and axiom constraints."""

    def __init__(self, name: str = "default_ontology"):
        self.name = name
        self.concepts: Dict[str, Concept] = {}
        self._ancestor_cache: Dict[str, Set[str]] = {}
        self._descendant_cache: Dict[str, Set[str]] = {}

    def add_concept(self, concept: Concept) -> Concept:
        """Add or update a concept in the ontology."""
        self.concepts[concept.curie] = concept
        self._invalidate_caches()
        return concept

    def add_subsumption(self, child_curie: str, parent_curie: str) -> None:
        """Assert child_curie subClassOf parent_curie."""
        if child_curie not in self.concepts:
            raise KeyError(f"Child concept {child_curie} not found in ontology")
        if parent_curie not in self.concepts:
            raise KeyError(f"Parent concept {parent_curie} not found in ontology")

        self.concepts[child_curie].parents.add(parent_curie)
        self.concepts[parent_curie].children.add(child_curie)
        self._invalidate_caches()

    def add_disjointness(self, curie_a: str, curie_b: str) -> None:
        """Assert curie_a disjointWith curie_b."""
        if curie_a not in self.concepts or curie_b not in self.concepts:
            raise KeyError(f"Concepts {curie_a} or {curie_b} not found")

        self.concepts[curie_a].disjoint_with.add(curie_b)
        self.concepts[curie_b].disjoint_with.add(curie_a)

    def _invalidate_caches(self) -> None:
        self._ancestor_cache.clear()
        self._descendant_cache.clear()

    def get_ancestors(self, curie: str) -> Set[str]:
        """Compute the transitive closure of superclasses (including self)."""
        if curie in self._ancestor_cache:
            return self._ancestor_cache[curie]

        if curie not in self.concepts:
            return set()

        ancestors: Set[str] = {curie}
        queue = deque([curie])
        while queue:
            curr = queue.popleft()
            concept = self.concepts.get(curr)
            if concept:
                for parent in concept.parents:
                    if parent not in ancestors:
                        ancestors.add(parent)
                        queue.append(parent)

        self._ancestor_cache[curie] = ancestors
        return ancestors

    def get_descendants(self, curie: str) -> Set[str]:
        """Compute the transitive closure of subclasses (including self)."""
        if curie in self._descendant_cache:
            return self._descendant_cache[curie]

        if curie not in self.concepts:
            return set()

        descendants: Set[str] = {curie}
        queue = deque([curie])
        while queue:
            curr = queue.popleft()
            concept = self.concepts.get(curr)
            if concept:
                for child in concept.children:
                    if child not in descendants:
                        descendants.add(child)
                        queue.append(child)

        self._descendant_cache[curie] = descendants
        return descendants

    def is_subsumed_by(self, child_curie: str, parent_curie: str) -> bool:
        """Return True if child_curie is a subclass of parent_curie (or same)."""
        return parent_curie in self.get_ancestors(child_curie)

    def are_disjoint(self, curie_a: str, curie_b: str) -> bool:
        """Check if curie_a and curie_b are disjoint (directly or inherited)."""
        anc_a = self.get_ancestors(curie_a)
        anc_b = self.get_ancestors(curie_b)
        for a in anc_a:
            concept_a = self.concepts.get(a)
            if concept_a and any(b in concept_a.disjoint_with for b in anc_b):
                return True
        return False

    def get_lineage(self, curie: str) -> List[str]:
        """Return ordered ancestral path from root to curie."""
        if curie not in self.concepts:
            return [curie]

        # Topological sort / shortest path to root
        lineage = []
        curr = curie
        visited = set()
        while curr and curr not in visited:
            visited.add(curr)
            lineage.append(curr)
            concept = self.concepts.get(curr)
            if not concept or not concept.parents:
                break
            # Pick primary parent (first)
            curr = next(iter(sorted(concept.parents)))

        return list(reversed(lineage))

    def extract_subgraph(self, target_curies: List[str], include_parents: bool = True) -> Ontology:
        """Extract a self-contained sub-ontology containing target curies and their lineage."""
        sub = Ontology(name=f"{self.name}_subgraph")
        # Ordered set: targets keep their given order, ancestors follow sorted, so the subgraph's
        # concept order never depends on set iteration (and hence on PYTHONHASHSEED).
        to_include: Dict[str, None] = {}

        for c in target_curies:
            if c in self.concepts:
                to_include[c] = None
        if include_parents:
            ancestors: Set[str] = set()
            for c in to_include:
                ancestors.update(self.get_ancestors(c))
            for c in sorted(ancestors):
                to_include.setdefault(c, None)

        # Add concepts
        for c in to_include:
            orig = self.concepts[c]
            new_c = Concept(
                curie=orig.curie,
                label=orig.label,
                description=orig.description,
                synonyms=list(orig.synonyms),
                parents=orig.parents.intersection(to_include),
                children=orig.children.intersection(to_include),
                disjoint_with=orig.disjoint_with.intersection(to_include),
                metadata=dict(orig.metadata),
            )
            sub.concepts[c] = new_c

        return sub

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "concepts": [c.to_dict() for c in self.concepts.values()],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str_or_dict: Any) -> Ontology:
        data = json.loads(json_str_or_dict) if isinstance(json_str_or_dict, str) else json_str_or_dict
        onto = cls(name=data.get("name", "loaded_ontology"))
        for c_data in data.get("concepts", []):
            concept = Concept.from_dict(c_data)
            onto.concepts[concept.curie] = concept

        # Populate children back-links
        for concept in onto.concepts.values():
            for p in concept.parents:
                if p in onto.concepts:
                    onto.concepts[p].children.add(concept.curie)

        onto._invalidate_caches()
        return onto


def create_incident_ontology() -> Ontology:
    """Create a standard enterprise IT / SaaS incident and triage ontology."""
    onto = Ontology(name="EnterpriseIncidentOntology")

    # Root
    onto.add_concept(Concept("INCIDENT:ROOT", "Operational Incident", "Any anomalous operational event."))

    # Level 1 Domains
    onto.add_concept(Concept("INCIDENT:INFRA", "Infrastructure Incident", "Hardware, compute, hypervisor, or OS issues."))
    onto.add_concept(Concept("INCIDENT:DATA", "Data & Storage Incident", "Database, cache, message broker, or pipeline issues."))
    onto.add_concept(Concept("INCIDENT:SECURITY", "Security & Access Incident", "Unauthorized access, breach, credential leak, or policy violation."))
    onto.add_concept(Concept("INCIDENT:BILLING", "Billing & Commercial Incident", "Invoicing, payment processing, subscription, or checkout errors."))
    onto.add_concept(Concept("INCIDENT:APPLICATION", "Application Error", "Client UI glitches, 5xx server exceptions, or logic bugs."))

    for dom in ["INCIDENT:INFRA", "INCIDENT:DATA", "INCIDENT:SECURITY", "INCIDENT:BILLING", "INCIDENT:APPLICATION"]:
        onto.add_subsumption(dom, "INCIDENT:ROOT")

    # Assert Level 1 Disjointness
    onto.add_disjointness("INCIDENT:SECURITY", "INCIDENT:BILLING")
    onto.add_disjointness("INCIDENT:INFRA", "INCIDENT:BILLING")

    # Level 2 Specifics under Data
    onto.add_concept(Concept("INCIDENT:DB_OUTAGE", "Database Outage", "Complete loss of primary DB or replica cluster connectivity.", synonyms=["db down", "postgres unreachable", "mysql connection refused"]))
    onto.add_concept(Concept("INCIDENT:DB_DEADLOCK", "Database Deadlock", "Concurrent transactions blocked waiting for conflicting locks.", synonyms=["lock wait timeout", "deadlock detected"]))
    onto.add_concept(Concept("INCIDENT:DATA_PIPELINE", "Data Pipeline Stall", "Kafka lag, Celery/Airflow failure, or consumer crash.", synonyms=["kafka lag", "dag failed"]))
    for d in ["INCIDENT:DB_OUTAGE", "INCIDENT:DB_DEADLOCK", "INCIDENT:DATA_PIPELINE"]:
        onto.add_subsumption(d, "INCIDENT:DATA")

    # Level 2 Specifics under Security
    onto.add_concept(Concept("INCIDENT:AUTH_BREACH", "Authentication Breach", "Compromised credentials, unauthorized token issuance, or brute-force attack.", synonyms=["api key leaked", "unauthorized login"]))
    onto.add_concept(Concept("INCIDENT:DDOS", "DDoS Attack", "High-volume distributed traffic overwhelming ingress points.", synonyms=["dos attack", "syn flood"]))
    for s in ["INCIDENT:AUTH_BREACH", "INCIDENT:DDOS"]:
        onto.add_subsumption(s, "INCIDENT:SECURITY")

    # Level 2 Specifics under Billing
    onto.add_concept(Concept("INCIDENT:PAYMENT_GATEWAY", "Payment Gateway Failure", "Stripe/Adyen/banking webhook failures or payment processing errors.", synonyms=["stripe 500", "payment decline storm"]))
    onto.add_concept(Concept("INCIDENT:INVOICE_MISCALC", "Invoice Calculation Error", "Incorrect tax, discount, or pricing tier billing error.", synonyms=["overcharged", "wrong tax"]))
    for b in ["INCIDENT:PAYMENT_GATEWAY", "INCIDENT:INVOICE_MISCALC"]:
        onto.add_subsumption(b, "INCIDENT:BILLING")

    # Disjointness between DB Outage and Invoice Error
    onto.add_disjointness("INCIDENT:DB_OUTAGE", "INCIDENT:INVOICE_MISCALC")

    return onto
