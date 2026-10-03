"""Ontology representation, subsumption DAG, and disjointness axioms for O-Laya."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObjectProperty:
    """An OWL 2 Object/Data Property with relational characteristics."""

    curie: str
    label: str = ""
    description: str = ""
    parents: set[str] = field(default_factory=set)  # rdfs:subPropertyOf
    children: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)  # rdfs:domain
    ranges: set[str] = field(default_factory=set)  # rdfs:range
    inverses: set[str] = field(default_factory=set)  # owl:inverseOf
    is_transitive: bool = False  # owl:TransitiveProperty
    is_functional: bool = False  # owl:FunctionalProperty
    is_inverse_functional: bool = False  # owl:InverseFunctionalProperty
    is_symmetric: bool = False  # owl:SymmetricProperty
    is_asymmetric: bool = False  # owl:AsymmetricProperty
    is_irreflexive: bool = False  # owl:IrreflexiveProperty
    disjoint_with: set[str] = field(default_factory=set)  # owl:propertyDisjointWith
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "curie": self.curie,
            "label": self.label,
            "description": self.description,
            "parents": sorted(list(self.parents)),
            "domains": sorted(list(self.domains)),
            "ranges": sorted(list(self.ranges)),
            "inverses": sorted(list(self.inverses)),
            "is_transitive": self.is_transitive,
            "is_functional": self.is_functional,
            "is_inverse_functional": self.is_inverse_functional,
            "is_symmetric": self.is_symmetric,
            "is_asymmetric": self.is_asymmetric,
            "is_irreflexive": self.is_irreflexive,
            "disjoint_with": sorted(list(self.disjoint_with)),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectProperty:
        return cls(
            curie=data["curie"],
            label=data.get("label", ""),
            description=data.get("description", ""),
            parents=set(data.get("parents", [])),
            domains=set(data.get("domains", [])),
            ranges=set(data.get("ranges", [])),
            inverses=set(data.get("inverses", [])),
            is_transitive=data.get("is_transitive", False),
            is_functional=data.get("is_functional", False),
            is_inverse_functional=data.get("is_inverse_functional", False),
            is_symmetric=data.get("is_symmetric", False),
            is_asymmetric=data.get("is_asymmetric", False),
            is_irreflexive=data.get("is_irreflexive", False),
            disjoint_with=set(data.get("disjoint_with", [])),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Concept:
    """An individual ontological class/concept with formal coordinates and OWL 2 RL semantics."""

    curie: str  # e.g., "INCIDENT:DATABASE_OUTAGE"
    label: str  # e.g., "Database Outage"
    description: str = ""  # e.g., "Critical disruption to database connectivity or read/write operations"
    synonyms: list[str] = field(default_factory=list)
    parents: set[str] = field(default_factory=set)  # direct superclasses (rdfs:subClassOf)
    children: set[str] = field(default_factory=set)  # direct subclasses
    disjoint_with: set[str] = field(default_factory=set)  # mutually exclusive concepts (owl:disjointWith)
    equivalent_to: set[str] = field(default_factory=set)  # equivalent classes (owl:equivalentClass)
    intersections: list[list[str]] = field(default_factory=list)  # operand lists for owl:intersectionOf
    unions: list[list[str]] = field(default_factory=list)  # operand lists for owl:unionOf
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "curie": self.curie,
            "label": self.label,
            "description": self.description,
            "synonyms": self.synonyms,
            "parents": sorted(list(self.parents)),
            "disjoint_with": sorted(list(self.disjoint_with)),
            "metadata": self.metadata,
        }
        if self.equivalent_to:
            d["equivalent_to"] = sorted(list(self.equivalent_to))
        if self.intersections:
            d["intersections"] = self.intersections
        if self.unions:
            d["unions"] = self.unions
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Concept:
        return cls(
            curie=data["curie"],
            label=data["label"],
            description=data.get("description", ""),
            synonyms=data.get("synonyms", []),
            parents=set(data.get("parents", [])),
            disjoint_with=set(data.get("disjoint_with", [])),
            equivalent_to=set(data.get("equivalent_to", [])),
            intersections=data.get("intersections", []),
            unions=data.get("unions", []),
            metadata=data.get("metadata", {}),
        )


class Ontology:
    """Ontology substrate maintaining classes, subsumption DAG, and OWL 2 RL axiom constraints."""

    def __init__(self, name: str = "default_ontology"):
        self.name = name
        self.concepts: dict[str, Concept] = {}
        self.properties: dict[str, ObjectProperty] = {}
        self.disjoint_unions: dict[str, list[list[str]]] = {}
        self._ancestor_cache: dict[str, set[str]] = {}
        self._descendant_cache: dict[str, set[str]] = {}
        self._equivalence_cache: dict[str, set[str]] = {}

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

    def add_equivalence(self, curie_a: str, curie_b: str) -> None:
        """Assert curie_a equivalentTo curie_b (W3C scm-eqc1: mutual subsumption)."""
        if curie_a not in self.concepts or curie_b not in self.concepts:
            raise KeyError(f"Concepts {curie_a} or {curie_b} not found in ontology")
        self.concepts[curie_a].equivalent_to.add(curie_b)
        self.concepts[curie_b].equivalent_to.add(curie_a)
        self.add_subsumption(curie_a, curie_b)
        self.add_subsumption(curie_b, curie_a)
        self._invalidate_caches()

    def add_disjoint_union(self, parent_curie: str, member_curies: list[str]) -> None:
        """Assert parent is the disjoint union of members (owl:disjointUnionOf)."""
        if parent_curie not in self.concepts:
            raise KeyError(f"Parent {parent_curie} not found in ontology")
        valid_members = [m for m in member_curies if m in self.concepts]
        self.disjoint_unions.setdefault(parent_curie, []).append(valid_members)
        for m in valid_members:
            self.add_subsumption(m, parent_curie)
        for i, m1 in enumerate(valid_members):
            for m2 in valid_members[i + 1 :]:
                self.add_disjointness(m1, m2)

    def add_intersection(self, target_curie: str, operand_curies: list[str]) -> None:
        """Assert target is equivalent to intersection of operands (owl:intersectionOf)."""
        if target_curie not in self.concepts:
            raise KeyError(f"Concept {target_curie} not found in ontology")
        valid = [op for op in operand_curies if op in self.concepts]
        self.concepts[target_curie].intersections.append(valid)
        for op in valid:
            self.add_subsumption(target_curie, op)

    def add_union(self, target_curie: str, operand_curies: list[str]) -> None:
        """Assert target is equivalent to union of operands (owl:unionOf)."""
        if target_curie not in self.concepts:
            raise KeyError(f"Concept {target_curie} not found in ontology")
        valid = [op for op in operand_curies if op in self.concepts]
        self.concepts[target_curie].unions.append(valid)
        for op in valid:
            self.add_subsumption(op, target_curie)

    def add_property(self, prop: ObjectProperty) -> ObjectProperty:
        """Add an object or data property."""
        self.properties[prop.curie] = prop
        return prop

    def add_subproperty(self, child_curie: str, parent_curie: str) -> None:
        """Assert child_curie subPropertyOf parent_curie."""
        if child_curie not in self.properties:
            raise KeyError(f"Property {child_curie} not found")
        if parent_curie not in self.properties:
            raise KeyError(f"Property {parent_curie} not found")
        self.properties[child_curie].parents.add(parent_curie)
        self.properties[parent_curie].children.add(child_curie)

    def get_equivalent_concepts(self, curie: str) -> set[str]:
        """Compute the transitive/symmetric closure of equivalent concepts."""
        if curie in self._equivalence_cache:
            return self._equivalence_cache[curie]
        if curie not in self.concepts:
            return set()
        equivs: set[str] = {curie}
        queue = deque([curie])
        while queue:
            curr = queue.popleft()
            concept = self.concepts.get(curr)
            if concept:
                for eq in concept.equivalent_to:
                    if eq not in equivs and eq in self.concepts:
                        equivs.add(eq)
                        queue.append(eq)
        self._equivalence_cache[curie] = equivs
        return equivs

    def _invalidate_caches(self) -> None:
        self._ancestor_cache.clear()
        self._descendant_cache.clear()
        self._equivalence_cache.clear()

    def get_ancestors(self, curie: str) -> set[str]:
        """Compute the transitive closure of superclasses (including self)."""
        if curie in self._ancestor_cache:
            return self._ancestor_cache[curie]

        if curie not in self.concepts:
            return set()

        ancestors: set[str] = {curie}
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

    def get_descendants(self, curie: str) -> set[str]:
        """Compute the transitive closure of subclasses (including self)."""
        if curie in self._descendant_cache:
            return self._descendant_cache[curie]

        if curie not in self.concepts:
            return set()

        descendants: set[str] = {curie}
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

    def get_lineage(self, curie: str) -> list[str]:
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

    def extract_subgraph(self, target_curies: list[str], include_parents: bool = True) -> Ontology:
        """Extract a self-contained sub-ontology containing target curies and their lineage."""
        sub = Ontology(name=f"{self.name}_subgraph")
        # Ordered set: targets keep their given order, ancestors follow sorted, so the subgraph's
        # concept order never depends on set iteration (and hence on PYTHONHASHSEED).
        to_include: dict[str, None] = {}

        for c in target_curies:
            if c in self.concepts:
                to_include[c] = None
        if include_parents:
            ancestors: set[str] = set()
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
                equivalent_to=orig.equivalent_to.intersection(to_include),
                intersections=[[op for op in ops if op in to_include] for ops in orig.intersections],
                unions=[[op for op in ops if op in to_include] for ops in orig.unions],
                metadata=dict(orig.metadata),
            )
            sub.concepts[c] = new_c

        # Copy applicable properties
        for pid, prop in self.properties.items():
            if prop.domains.intersection(to_include) or prop.ranges.intersection(to_include):
                sub.properties[pid] = ObjectProperty.from_dict(prop.to_dict())

        # Copy applicable disjoint unions
        for parent, partitions in self.disjoint_unions.items():
            if parent in to_include:
                filtered_parts = []
                for part in partitions:
                    filtered = [m for m in part if m in to_include]
                    if len(filtered) >= 2:
                        filtered_parts.append(filtered)
                if filtered_parts:
                    sub.disjoint_unions[parent] = filtered_parts

        return sub

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "concepts": [c.to_dict() for c in self.concepts.values()],
        }
        if self.properties:
            d["properties"] = [p.to_dict() for p in self.properties.values()]
        if self.disjoint_unions:
            d["disjoint_unions"] = self.disjoint_unions
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str_or_dict: Any) -> Ontology:
        data = json.loads(json_str_or_dict) if isinstance(json_str_or_dict, str) else json_str_or_dict
        onto = cls(name=data.get("name", "loaded_ontology"))
        for c_data in data.get("concepts", []):
            concept = Concept.from_dict(c_data)
            onto.concepts[concept.curie] = concept

        for p_data in data.get("properties", []):
            prop = ObjectProperty.from_dict(p_data)
            onto.properties[prop.curie] = prop

        onto.disjoint_unions = data.get("disjoint_unions", {})

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
    onto.add_concept(
        Concept("INCIDENT:INFRA", "Infrastructure Incident", "Hardware, compute, hypervisor, or OS issues.")
    )
    onto.add_concept(
        Concept("INCIDENT:DATA", "Data & Storage Incident", "Database, cache, message broker, or pipeline issues.")
    )
    onto.add_concept(
        Concept(
            "INCIDENT:SECURITY",
            "Security & Access Incident",
            "Unauthorized access, breach, credential leak, or policy violation.",
        )
    )
    onto.add_concept(
        Concept(
            "INCIDENT:BILLING",
            "Billing & Commercial Incident",
            "Invoicing, payment processing, subscription, or checkout errors.",
        )
    )
    onto.add_concept(
        Concept(
            "INCIDENT:APPLICATION", "Application Error", "Client UI glitches, 5xx server exceptions, or logic bugs."
        )
    )

    for dom in ["INCIDENT:INFRA", "INCIDENT:DATA", "INCIDENT:SECURITY", "INCIDENT:BILLING", "INCIDENT:APPLICATION"]:
        onto.add_subsumption(dom, "INCIDENT:ROOT")

    # Assert Level 1 Disjointness
    onto.add_disjointness("INCIDENT:SECURITY", "INCIDENT:BILLING")
    onto.add_disjointness("INCIDENT:INFRA", "INCIDENT:BILLING")

    # Level 2 Specifics under Data
    onto.add_concept(
        Concept(
            "INCIDENT:DB_OUTAGE",
            "Database Outage",
            "Complete loss of primary DB or replica cluster connectivity.",
            synonyms=["db down", "postgres unreachable", "mysql connection refused"],
        )
    )
    onto.add_concept(
        Concept(
            "INCIDENT:DB_DEADLOCK",
            "Database Deadlock",
            "Concurrent transactions blocked waiting for conflicting locks.",
            synonyms=["lock wait timeout", "deadlock detected"],
        )
    )
    onto.add_concept(
        Concept(
            "INCIDENT:DATA_PIPELINE",
            "Data Pipeline Stall",
            "Kafka lag, Celery/Airflow failure, or consumer crash.",
            synonyms=["kafka lag", "dag failed"],
        )
    )
    for d in ["INCIDENT:DB_OUTAGE", "INCIDENT:DB_DEADLOCK", "INCIDENT:DATA_PIPELINE"]:
        onto.add_subsumption(d, "INCIDENT:DATA")

    # Level 2 Specifics under Security
    onto.add_concept(
        Concept(
            "INCIDENT:AUTH_BREACH",
            "Authentication Breach",
            "Compromised credentials, unauthorized token issuance, or brute-force attack.",
            synonyms=["api key leaked", "unauthorized login"],
        )
    )
    onto.add_concept(
        Concept(
            "INCIDENT:DDOS",
            "DDoS Attack",
            "High-volume distributed traffic overwhelming ingress points.",
            synonyms=["dos attack", "syn flood"],
        )
    )
    for s in ["INCIDENT:AUTH_BREACH", "INCIDENT:DDOS"]:
        onto.add_subsumption(s, "INCIDENT:SECURITY")

    # Level 2 Specifics under Billing
    onto.add_concept(
        Concept(
            "INCIDENT:PAYMENT_GATEWAY",
            "Payment Gateway Failure",
            "Stripe/Adyen/banking webhook failures or payment processing errors.",
            synonyms=["stripe 500", "payment decline storm"],
        )
    )
    onto.add_concept(
        Concept(
            "INCIDENT:INVOICE_MISCALC",
            "Invoice Calculation Error",
            "Incorrect tax, discount, or pricing tier billing error.",
            synonyms=["overcharged", "wrong tax"],
        )
    )
    for b in ["INCIDENT:PAYMENT_GATEWAY", "INCIDENT:INVOICE_MISCALC"]:
        onto.add_subsumption(b, "INCIDENT:BILLING")

    # Disjointness between DB Outage and Invoice Error
    onto.add_disjointness("INCIDENT:DB_OUTAGE", "INCIDENT:INVOICE_MISCALC")

    return onto
