"""Forward-chaining W3C OWL 2 RL reasoner over the O-Laya ontology model (TBox + ABox).

The exact rule coverage, with the status of every rule in the W3C OWL 2 RL table, is in
``spec/OWL2RL.md``. In short:

- Schema rules over the class/property graph: scm-cls, scm-sco, scm-eqc1/2, scm-dco, scm-int, scm-uni,
  owl:disjointUnionOf, scm-spo, scm-dom1/2, scm-rng1/2, and unsatisfiability via self-disjointness
  (cls-nothing2 on the schema).
- Instance rules over named individuals: cax-sco, cax-eqc1/2, cax-dw, cls-int1, cls-nothing2, prp-dom,
  prp-rng, prp-spo1, prp-spo2 (property chains), prp-inv1/2, prp-symp, prp-trp, prp-fp, prp-ifp,
  prp-pdw, prp-asyp, prp-irp, and the equality rules eq-sym, eq-trans, eq-rep-s, eq-rep-o, eq-diff1.

Inconsistencies are collected in ``OWL2RLClosure.inconsistencies`` (never raised); every rule
application is counted in ``OWL2RLClosure.rule_firings``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from olaya.ontology import Ontology


@dataclass(frozen=True)
class Inconsistency:
    """One violated OWL 2 RL ``false``-conclusion rule, with the entities that triggered it."""

    rule: str  # W3C rule name, e.g. "cax-dw"
    entities: tuple[str, ...]  # individuals / classes / properties involved, in rule-argument order
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"rule": self.rule, "entities": list(self.entities), "message": self.message}


@dataclass
class OWL2RLClosure:
    """Fixed-point deductive closure under W3C OWL 2 RL inference rules."""

    subclasses: set[tuple[str, str]] = field(default_factory=set)  # (child, parent)
    equivalences: set[tuple[str, str]] = field(default_factory=set)  # (c1, c2) bidirectional
    disjointnesses: set[tuple[str, str]] = field(default_factory=set)  # (c1, c2) symmetric
    unsatisfiable_classes: set[str] = field(default_factory=set)  # classes subsumed by Nothing (C <= bot)
    property_subsumptions: set[tuple[str, str]] = field(default_factory=set)  # (subProp, superProp)
    property_domains: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    property_ranges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    property_inverses: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    intersections: dict[str, list[list[str]]] = field(default_factory=dict)
    unions: dict[str, list[list[str]]] = field(default_factory=dict)
    disjoint_unions: dict[str, list[list[str]]] = field(default_factory=dict)
    rule_firings: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    is_consistent: bool = True
    # ABox (instance-level) closure
    individual_types: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    property_assertions: set[tuple[str, str, str]] = field(default_factory=set)  # (subject, property, object)
    same_as: dict[str, frozenset[str]] = field(default_factory=dict)  # individual -> its owl:sameAs class
    inconsistencies: list[Inconsistency] = field(default_factory=list)

    def is_subclass_of(self, child: str, parent: str) -> bool:
        return child == parent or (child, parent) in self.subclasses

    def are_equivalent(self, c1: str, c2: str) -> bool:
        return c1 == c2 or (c1, c2) in self.equivalences

    def are_disjoint(self, c1: str, c2: str) -> bool:
        return (c1, c2) in self.disjointnesses or (c2, c1) in self.disjointnesses

    def is_unsatisfiable(self, c: str) -> bool:
        return c in self.unsatisfiable_classes

    def types_of(self, individual: str) -> set[str]:
        """All inferred classes of an individual (asserted, inherited, domain/range, via sameAs)."""
        return set(self.individual_types.get(individual, ()))

    def same_individuals(self, individual: str) -> frozenset[str]:
        """The owl:sameAs equivalence class of an individual (always contains the individual)."""
        return self.same_as.get(individual, frozenset({individual}))

    def are_same(self, a: str, b: str) -> bool:
        return a == b or b in self.same_individuals(a)

    def has_assertion(self, subject: str, prop: str, obj: str) -> bool:
        return (subject, prop, obj) in self.property_assertions

    def objects_of(self, subject: str, prop: str) -> set[str]:
        return {o for s, p, o in self.property_assertions if s == subject and p == prop}


class OWL2RLReasoner:
    """Forward-chaining rule reasoner computing the deductive closure under W3C OWL 2 RL."""

    def __init__(self, ontology: Ontology):
        self.ontology = ontology

    def materialize(self) -> OWL2RLClosure:
        """Alias for ``compute_closure()`` returning the materialized OWL 2 RL deductive closure."""
        return self.compute_closure()

    def compute_closure(self) -> OWL2RLClosure:
        """Run fixpoint forward-chaining over all OWL 2 RL inference rules."""
        closure = OWL2RLClosure()
        concepts = self.ontology.concepts
        properties = self.ontology.properties

        # -------------------------------------------------------------
        # STEP 1: Ingest Explicit Axioms
        # -------------------------------------------------------------
        # Ingest SubClassOf & Reflexivity (scm-cls: C <= C)
        for curie, concept in concepts.items():
            closure.subclasses.add((curie, curie))
            for parent in concept.parents:
                if parent in concepts:
                    closure.subclasses.add((curie, parent))
            for eq in concept.equivalent_to:
                if eq in concepts:
                    closure.equivalences.add((curie, eq))
                    closure.equivalences.add((eq, curie))
                    closure.subclasses.add((curie, eq))
                    closure.subclasses.add((eq, curie))
                    closure.rule_firings["scm-eqc1"] += 1
            for dj in concept.disjoint_with:
                if dj in concepts:
                    closure.disjointnesses.add((curie, dj))
                    closure.disjointnesses.add((dj, curie))

        # Ingest Intersections (scm-int: C == (C1 and ... and Cn) => C <= Ci)
        for curie, concept in concepts.items():
            for operands in concept.intersections:
                valid_ops = [op for op in operands if op in concepts]
                closure.intersections.setdefault(curie, []).append(valid_ops)
                for op in valid_ops:
                    closure.subclasses.add((curie, op))
                    closure.rule_firings["scm-int"] += 1

        # Ingest Unions (scm-uni: C == (C1 or ... or Cn) => Ci <= C)
        for curie, concept in concepts.items():
            for operands in concept.unions:
                valid_ops = [op for op in operands if op in concepts]
                closure.unions.setdefault(curie, []).append(valid_ops)
                for op in valid_ops:
                    closure.subclasses.add((op, curie))
                    closure.rule_firings["scm-uni"] += 1

        # Ingest Disjoint Unions (owl:disjointUnionOf)
        for parent, partitions in self.ontology.disjoint_unions.items():
            if parent not in concepts:
                continue
            for part in partitions:
                valid = [m for m in part if m in concepts]
                closure.disjoint_unions.setdefault(parent, []).append(valid)
                for m in valid:
                    closure.subclasses.add((m, parent))
                    closure.rule_firings["scm-dju-sub"] += 1
                for i, m1 in enumerate(valid):
                    for m2 in valid[i + 1 :]:
                        closure.disjointnesses.add((m1, m2))
                        closure.disjointnesses.add((m2, m1))
                        closure.rule_firings["scm-dju-disj"] += 1

        # Ingest Properties
        for pid, prop in properties.items():
            closure.property_subsumptions.add((pid, pid))
            for parent_prop in prop.parents:
                if parent_prop in properties:
                    closure.property_subsumptions.add((pid, parent_prop))
            for dom in prop.domains:
                if dom in concepts:
                    closure.property_domains[pid].add(dom)
            for rng in prop.ranges:
                if rng in concepts:
                    closure.property_ranges[pid].add(rng)
            for inv in prop.inverses:
                closure.property_inverses[pid].add(inv)
                closure.property_inverses[inv].add(pid)
            if prop.is_symmetric:
                closure.property_inverses[pid].add(pid)

        # -------------------------------------------------------------
        # STEP 2: Fixpoint Iteration over OWL 2 RL Rules
        # -------------------------------------------------------------
        changed = True
        iterations = 0
        max_iterations = 50

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            # Rule scm-sco: Transitivity of SubClassOf (A <= B and B <= C => A <= C)
            new_sub = set()
            parent_to_children = defaultdict(set)
            child_to_parents = defaultdict(set)
            for child, parent in closure.subclasses:
                parent_to_children[parent].add(child)
                child_to_parents[child].add(parent)

            for b in concepts:
                for a in parent_to_children[b]:
                    for c in child_to_parents[b]:
                        if (a, c) not in closure.subclasses:
                            new_sub.add((a, c))
                            closure.rule_firings["scm-sco"] += 1

            if new_sub:
                closure.subclasses.update(new_sub)
                changed = True

            # Rule scm-eqc2: SubClassOf anti-symmetry => Equivalence (A <= B and B <= A => A == B)
            new_eq = set()
            for a, b in list(closure.subclasses):
                if (b, a) in closure.subclasses and (a, b) not in closure.equivalences:
                    new_eq.add((a, b))
                    new_eq.add((b, a))
                    closure.rule_firings["scm-eqc2"] += 1

            if new_eq:
                closure.equivalences.update(new_eq)
                changed = True

            # Rule scm-dco: Disjointness inheritance through SubClassOf (A <= B and disjoint(B, C) => disjoint(A, C))
            new_disj = set()
            for b, c in list(closure.disjointnesses):
                for a in parent_to_children[b]:
                    if (a, c) not in closure.disjointnesses:
                        new_disj.add((a, c))
                        new_disj.add((c, a))
                        closure.rule_firings["scm-dco"] += 1

            if new_disj:
                closure.disjointnesses.update(new_disj)
                changed = True

            # Rule scm-spo: Transitivity of SubPropertyOf (P1 <= P2 and P2 <= P3 => P1 <= P3)
            new_psub = set()
            prop_parents = defaultdict(set)
            for p1, p2 in closure.property_subsumptions:
                prop_parents[p1].add(p2)
            for p1 in properties:
                for p2 in list(prop_parents[p1]):
                    for p3 in list(prop_parents[p2]):
                        if (p1, p3) not in closure.property_subsumptions:
                            new_psub.add((p1, p3))
                            closure.rule_firings["scm-spo"] += 1

            if new_psub:
                closure.property_subsumptions.update(new_psub)
                changed = True

            # Rule scm-dom1 & scm-dom2: Domain inheritance (P1 <= P2 and domain(P2, C) => domain(P1, C))
            for p1, p2 in list(closure.property_subsumptions):
                for c in list(closure.property_domains[p2]):
                    if c not in closure.property_domains[p1]:
                        closure.property_domains[p1].add(c)
                        closure.rule_firings["scm-dom1"] += 1
                        changed = True

            # Rule scm-rng1 & scm-rng2: Range inheritance (P1 <= P2 and range(P2, C) => range(P1, C))
            for p1, p2 in list(closure.property_subsumptions):
                for c in list(closure.property_ranges[p2]):
                    if c not in closure.property_ranges[p1]:
                        closure.property_ranges[p1].add(c)
                        closure.rule_firings["scm-rng1"] += 1
                        changed = True

        # -------------------------------------------------------------
        # STEP 3: Consistency & Unsatisfiability Detection (cls-nothing2, cxt-disjoint)
        # -------------------------------------------------------------
        # A class C is unsatisfiable if it is disjoint with itself (directly or via two superclasses A, B)
        for curie in concepts:
            # Direct self-disjointness or inherited
            if (curie, curie) in closure.disjointnesses:
                closure.unsatisfiable_classes.add(curie)
                closure.rule_firings["cls-nothing2"] += 1

        if closure.unsatisfiable_classes:
            closure.is_consistent = False

        # -------------------------------------------------------------
        # STEP 4: ABox instance rules (fixpoint) and inconsistency detection
        # -------------------------------------------------------------
        if self.ontology.individuals:
            _ABoxReasoner(self.ontology, closure).run()
            if closure.inconsistencies:
                closure.is_consistent = False

        return closure


def abox_consistency_report(ontology: Ontology, closure: OWL2RLClosure | None = None) -> list[Inconsistency]:
    """Return every ABox inconsistency (cax-dw, cls-nothing2, prp-pdw, prp-asyp, prp-irp, eq-diff1).

    The list is empty for a consistent ABox and sorted deterministically. Nothing is raised.
    """
    if closure is None:
        closure = OWL2RLReasoner(ontology).compute_closure()
    return list(closure.inconsistencies)


class _ABoxReasoner:
    """Instance-level OWL 2 RL rules over named individuals, run to a fixpoint.

    owl:sameAs is kept as a union-find partition; whenever two classes merge, every type and triple of
    one member is replicated onto all members (eq-rep-s / eq-rep-o), so later rules see a saturated
    ABox without having to reason modulo equality.
    """

    MAX_ITERATIONS = 500

    def __init__(self, ontology: Ontology, closure: OWL2RLClosure):
        self.onto = ontology
        self.closure = closure
        self.fire = closure.rule_firings
        self.types: dict[str, set[str]] = closure.individual_types
        self.triples: set[tuple[str, str, str]] = closure.property_assertions
        self._parent: dict[str, str] = {}
        self._different: set[tuple[str, str]] = set()

    # -- union-find -------------------------------------------------------
    def _find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def _members(self, x: str) -> set[str]:
        root = self._find(x)
        return {y for y in self._parent if self._find(y) == root}

    def _merge(self, a: str, b: str, rule: str) -> bool:
        """Record a sameAs(a, b) derived by `rule`; replicate facts across the merged class."""
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return False
        self.fire[rule] += 1
        before_a, before_b = self._members(a), self._members(b)
        self._parent[rb] = ra
        self.fire["eq-sym"] += 1
        # Every new cross pair beyond the asserted (a, b) is a transitive consequence
        self.fire["eq-trans"] += len(before_a) * len(before_b) * 2 - 2
        members = before_a | before_b
        # eq-rep-s on type triples and on subjects, eq-rep-o on objects
        union_types: set[str] = set()
        for m in members:
            union_types |= self.types.get(m, set())
        for m in members:
            missing = union_types - self.types.get(m, set())
            if missing:
                self.types.setdefault(m, set()).update(missing)
                self.fire["eq-rep-s"] += len(missing)
        for s, p, o in list(self.triples):
            if s in members:
                for m in members:
                    if m != s and (m, p, o) not in self.triples:
                        self.triples.add((m, p, o))
                        self.fire["eq-rep-s"] += 1
            if o in members:
                for m in members:
                    if m != o and (s, p, m) not in self.triples:
                        self.triples.add((s, p, m))
                        self.fire["eq-rep-o"] += 1
        return True

    # -- driver -------------------------------------------------------------
    def run(self) -> None:
        closure = self.closure
        onto = self.onto
        concepts = onto.concepts
        props = onto.properties

        # Ingest asserted ABox
        for curie, ind in onto.individuals.items():
            self._find(curie)
            self.types.setdefault(curie, set()).update(ind.types)
            for p, objs in ind.properties.items():
                for o in objs:
                    self.triples.add((curie, p, o))
                    self._find(o)
            for d in ind.different_from:
                self._different.add((curie, d))
                self._different.add((d, curie))
        for curie, ind in onto.individuals.items():
            for other in ind.same_as:
                self._merge(curie, other, "eq-asserted")

        # Static schema lookups
        superclasses: dict[str, set[str]] = defaultdict(set)
        for child, parent in closure.subclasses:
            if child != parent:
                superclasses[child].add(parent)
        equivalents: dict[str, set[str]] = defaultdict(set)
        for c1, c2 in closure.equivalences:
            if c1 != c2:
                equivalents[c1].add(c2)
        superprops: dict[str, set[str]] = defaultdict(set)
        for p1, p2 in closure.property_subsumptions:
            if p1 != p2:
                superprops[p1].add(p2)
        declared_inverses: dict[str, set[str]] = defaultdict(set)
        for pid, prop in props.items():
            for inv in prop.inverses:
                if inv != pid:
                    declared_inverses[pid].add(inv)
        chains: list[tuple[str, list[str]]] = [
            (pid, list(chain)) for pid, prop in props.items() for chain in prop.property_chains if len(chain) >= 2
        ]
        intersections: list[tuple[str, list[str]]] = [
            (curie, ops) for curie, parts in closure.intersections.items() for ops in parts if ops
        ]

        changed = True
        iterations = 0
        while changed and iterations < self.MAX_ITERATIONS:
            changed = False
            iterations += 1
            changed |= self._class_rules(superclasses, equivalents, intersections)
            changed |= self._property_rules(superprops, declared_inverses, chains, props)
            changed |= self._functional_rules(props)

        # Freeze sameAs classes
        for curie in list(self._parent):
            closure.same_as[curie] = frozenset(self._members(curie))

        self._detect_inconsistencies(concepts, props)

    # -- class rules --------------------------------------------------------
    def _class_rules(self, superclasses, equivalents, intersections) -> bool:
        changed = False
        for ind in list(self.types):
            current = self.types[ind]
            new: set[str] = set()
            for c in current:
                # cax-eqc1 / cax-eqc2: equivalent classes share instances. Checked before cax-sco because
                # add_equivalence also records mutual subsumption, which would otherwise mask the rule.
                for d in equivalents.get(c, ()):
                    if d not in current and d not in new:
                        new.add(d)
                        self.fire["cax-eqc1" if c < d else "cax-eqc2"] += 1
                # cax-sco: T(?c1, rdfs:subClassOf, ?c2), T(?x, rdf:type, ?c1) -> T(?x, rdf:type, ?c2)
                for d in superclasses.get(c, ()):
                    if d not in current and d not in new:
                        new.add(d)
                        self.fire["cax-sco"] += 1
            # cls-int1: x in every operand of an intersection -> x in the intersection class
            for target, ops in intersections:
                if target not in current and target not in new and all(op in current for op in ops):
                    new.add(target)
                    self.fire["cls-int1"] += 1
            if new:
                current.update(new)
                changed = True
        return changed

    # -- property rules -----------------------------------------------------
    def _property_rules(self, superprops, declared_inverses, chains, props) -> bool:
        changed = False
        new_triples: set[tuple[str, str, str]] = set()
        by_subject_prop: dict[tuple[str, str], set[str]] = defaultdict(set)
        for s, p, o in self.triples:
            by_subject_prop[(s, p)].add(o)

        def add(t: tuple[str, str, str], rule: str) -> None:
            if t not in self.triples and t not in new_triples:
                new_triples.add(t)
                self.fire[rule] += 1

        for s, p, o in list(self.triples):
            prop = props.get(p)
            # prp-dom / prp-rng
            for c in self.closure.property_domains.get(p, ()):
                if c not in self.types.setdefault(s, set()):
                    self.types[s].add(c)
                    self.fire["prp-dom"] += 1
                    changed = True
            for c in self.closure.property_ranges.get(p, ()):
                if c not in self.types.setdefault(o, set()):
                    self.types[o].add(c)
                    self.fire["prp-rng"] += 1
                    changed = True
            # prp-spo1
            for p2 in superprops.get(p, ()):
                add((s, p2, o), "prp-spo1")
            # prp-inv1: T(?p1, owl:inverseOf, ?p2), T(?x, ?p1, ?y) -> T(?y, ?p2, ?x)
            for p2 in declared_inverses.get(p, ()):
                add((o, p2, s), "prp-inv1")
            # prp-inv2: T(?p1, owl:inverseOf, ?p2), T(?x, ?p2, ?y) -> T(?y, ?p1, ?x)
            for p1, invs in declared_inverses.items():
                if p in invs:
                    add((o, p1, s), "prp-inv2")
            if prop is not None:
                # prp-symp
                if prop.is_symmetric:
                    add((o, p, s), "prp-symp")
                # prp-trp
                if prop.is_transitive:
                    for z in by_subject_prop.get((o, p), ()):
                        add((s, p, z), "prp-trp")
        # prp-spo2: property chain p1 o ... o pn subPropertyOf p
        for target, chain in chains:
            frontier: dict[str, set[str]] = {}
            for s, p, o in self.triples:
                if p == chain[0]:
                    frontier.setdefault(s, set()).add(o)
            for link in chain[1:]:
                next_frontier: dict[str, set[str]] = {}
                for start, mids in frontier.items():
                    for mid in mids:
                        for end in by_subject_prop.get((mid, link), ()):
                            next_frontier.setdefault(start, set()).add(end)
                frontier = next_frontier
            for start, ends in frontier.items():
                for end in ends:
                    add((start, target, end), "prp-spo2")
        if new_triples:
            self.triples.update(new_triples)
            changed = True
        return changed

    # -- functional / inverse-functional -> sameAs --------------------------
    def _functional_rules(self, props) -> bool:
        changed = False
        by_subject_prop: dict[tuple[str, str], set[str]] = defaultdict(set)
        by_object_prop: dict[tuple[str, str], set[str]] = defaultdict(set)
        for s, p, o in self.triples:
            by_subject_prop[(s, p)].add(o)
            by_object_prop[(o, p)].add(s)
        for (_s, p), objs in list(by_subject_prop.items()):
            if props.get(p) is not None and props[p].is_functional and len(objs) > 1:
                first, *rest = sorted(objs)
                for other in rest:
                    changed |= self._merge(first, other, "prp-fp")
        for (_o, p), subjs in list(by_object_prop.items()):
            if props.get(p) is not None and props[p].is_inverse_functional and len(subjs) > 1:
                first, *rest = sorted(subjs)
                for other in rest:
                    changed |= self._merge(first, other, "prp-ifp")
        return changed

    # -- false-conclusion rules ----------------------------------------------
    def _detect_inconsistencies(self, concepts, props) -> None:
        found: list[Inconsistency] = []
        closure = self.closure
        for ind in sorted(self.types):
            tps = sorted(self.types[ind])
            # cls-nothing2: T(?x, rdf:type, owl:Nothing) -> false (also classes proven unsatisfiable)
            for c in tps:
                if c == "owl:Nothing" or c in closure.unsatisfiable_classes:
                    found.append(
                        Inconsistency("cls-nothing2", (ind, c), f"{ind} is typed with unsatisfiable class {c}")
                    )
            # cax-dw: T(?c1, owl:disjointWith, ?c2), T(?x, rdf:type, ?c1), T(?x, rdf:type, ?c2) -> false
            for i, c1 in enumerate(tps):
                for c2 in tps[i + 1 :]:
                    if closure.are_disjoint(c1, c2):
                        found.append(
                            Inconsistency(
                                "cax-dw", (ind, c1, c2), f"{ind} is an instance of disjoint classes {c1} and {c2}"
                            )
                        )
        for s, p, o in sorted(self.triples):
            prop = props.get(p)
            if prop is None:
                continue
            # prp-pdw: T(?p1, owl:propertyDisjointWith, ?p2), T(?x, ?p1, ?y), T(?x, ?p2, ?y) -> false
            for q in sorted(prop.disjoint_with):
                if p < q and (s, q, o) in self.triples:
                    found.append(
                        Inconsistency(
                            "prp-pdw", (s, p, q, o), f"({s}, {o}) is related by disjoint properties {p} and {q}"
                        )
                    )
            # prp-asyp: T(?p, rdf:type, owl:AsymmetricProperty), T(?x, ?p, ?y), T(?y, ?p, ?x) -> false
            if prop.is_asymmetric and (o, p, s) in self.triples and s <= o:
                found.append(Inconsistency("prp-asyp", (p, s, o), f"asymmetric property {p} holds in both directions"))
            # prp-irp: T(?p, rdf:type, owl:IrreflexiveProperty), T(?x, ?p, ?x) -> false
            if prop.is_irreflexive and s == o:
                found.append(Inconsistency("prp-irp", (p, s), f"irreflexive property {p} relates {s} to itself"))
        # eq-diff1: T(?x, owl:sameAs, ?y), T(?x, owl:differentFrom, ?y) -> false
        for a, b in sorted(self._different):
            if a < b and self._find(a) == self._find(b):
                found.append(Inconsistency("eq-diff1", (a, b), f"{a} and {b} are both sameAs and differentFrom"))
        for inc in found:
            self.fire[inc.rule] += 1
        closure.inconsistencies.extend(found)
