"""Structural similarity propagation for the OAEI Conference matcher.

The lexical / embedding matchers of :mod:`olaya.benchmarks.oaei` score every entity
pair in isolation.  The published systems with the best *recall* on the Conference
track (GraphMatcher in particular) get it from the ontology graph: when two classes
have matching parents, siblings and properties they are probably a match even when
their names share nothing.  This module adds that evidence without any training.
Three candidates are implemented on the same graph (``rdfs:subClassOf``,
``owl:equivalentClass``, ``owl:disjointWith``, property ``rdfs:domain`` /
``rdfs:range``, ``owl:inverseOf``, ``rdfs:subPropertyOf``), all seeded by the
pairwise similarity matrix of a base matcher:

``flooding`` / ``flooding_relative``
    Similarity Flooding (Melnik, Garcia-Molina and Rahm, ICDE 2002).  Nodes of the
    pairwise connectivity graph are entity pairs ``(a, b)``; an edge ``(a, b) -> (x, y)``
    with label ``r`` exists when ``a -r-> x`` in the source and ``b -r-> y`` in the
    target.  Propagation coefficients are the inverse out-degree (forward) / in-degree
    (backward) of the pair node, and the fixpoint is the paper's variant *C*:
    ``sigma_{i+1} = normalise(sigma_0 + sigma_i + phi(sigma_0 + sigma_i))`` with
    ``phi`` the one-step propagation and normalisation by the global maximum.  On
    dense ``(n_s, n_t)`` matrices the whole step is a handful of ``A^T S B`` products
    per relation, so no explicit pair graph is built.  The final score is
    ``(1 - w) * seed + w * fixpoint``; ``flooding_relative`` first rescales the fixpoint
    to the paper's *relative similarity* (divide by the larger of the row and column
    maximum), which keeps it on the seed's ``[0, 1]`` scale.

``neighbour``
    Neighbour aggregation: ``sim_{i+1}(a, b) = (1 - w) * seed(a, b) + w * mean_r
    max_{a' in N_r(a), b' in N_r(b)} sim_i(a', b')`` where ``r`` ranges over the
    neighbourhood types both entities have (parents, children, siblings, equivalents,
    disjoints, the properties a class is domain / range of; the domains, ranges,
    inverses, super- and sub-properties of a property).  A contraction for ``w < 1``,
    iterated to convergence.

``smooth``
    Training-free message passing on the *embedding vectors* (the GraphMatcher idea
    without the learned attention): ``e' = normalise(e + beta * mean_{neighbours} e)``
    for a fixed number of rounds, then the usual cosine / CSLS matcher.  Neighbours are
    the union of all relations above, undirected.

Everything is deterministic numpy on matrices no larger than a few hundred by a few
hundred; one pair takes well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from olaya.benchmarks.oaei import (
    EmbeddingMatcher,
    EmbeddingMatcherConfig,
    HybridMatcher,
    HybridMatcherConfig,
    LexicalMatcher,
    LexicalMatcherConfig,
    Matcher,
    OAEIAlignment,
    build_entity_texts,
)
from olaya.ontology import Ontology

#: Directed relations extracted from an ontology, in the order they appear in the graph.
RELATIONS = ("subClassOf", "equivalentClass", "disjointWith", "domain", "range", "inverseOf", "subPropertyOf")

STRUCTURAL_METHODS = ("none", "flooding", "flooding_relative", "neighbour", "smooth")


# ---------------------------------------------------------------------------
# Graph extraction
# ---------------------------------------------------------------------------


@dataclass
class OntologyGraph:
    """Entities of one ontology (classes first, then properties, each sorted by name)
    with one dense directed adjacency matrix per relation."""

    names: list[str]
    kinds: list[str]  # "class" / "property", parallel to ``names``
    relations: dict[str, np.ndarray]  # relation -> (n, n) float32, ``M[a, x] = 1`` iff ``a -r-> x``

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.names)}

    def kind_vector(self) -> np.ndarray:
        return np.array([k == "class" for k in self.kinds], dtype=bool)

    def undirected(self, relations: tuple[str, ...] = RELATIONS) -> np.ndarray:
        """Symmetric 0/1 adjacency over the union of ``relations`` (no self loops)."""
        m = np.zeros((self.n, self.n), dtype=np.float32)
        for r in relations:
            a = self.relations[r]
            m = np.maximum(m, np.maximum(a, a.T))
        np.fill_diagonal(m, 0.0)
        return m

    def neighbourhoods(self) -> dict[str, np.ndarray]:
        """Named neighbourhood sets used by the ``neighbour`` method (0/1 matrices)."""
        sub = self.relations["subClassOf"]
        eq = self.relations["equivalentClass"]
        dj = self.relations["disjointWith"]
        dom = self.relations["domain"]
        rng = self.relations["range"]
        inv = self.relations["inverseOf"]
        subp = self.relations["subPropertyOf"]
        eye = np.eye(self.n, dtype=bool)
        siblings = ((sub @ sub.T) > 0) & ~eye  # share a direct parent
        return {
            "parents": sub > 0,
            "children": sub.T > 0,
            "siblings": siblings,
            "equivalents": eq > 0,
            "disjoints": dj > 0,
            "domain_of": dom.T > 0,  # class -> properties whose domain it is
            "range_of": rng.T > 0,
            "domains": dom > 0,  # property -> its domain classes
            "ranges": rng > 0,
            "inverses": inv > 0,
            "super_properties": subp > 0,
            "sub_properties": subp.T > 0,
        }


def build_graph(ontology: Ontology) -> OntologyGraph:
    """Extract the entity list and the directed relation matrices of ``ontology``.

    The entity order is the one of :func:`olaya.benchmarks.oaei.build_entity_texts`
    (classes sorted, then properties sorted), so the embedding rows line up.
    """
    texts = build_entity_texts(ontology)
    names = [t.name for t in texts]
    kinds = [t.kind for t in texts]
    idx = {name: i for i, name in enumerate(names)}
    n = len(names)
    rel = {r: np.zeros((n, n), dtype=np.float32) for r in RELATIONS}

    def put(r: str, a: str, b: str) -> None:
        if a in idx and b in idx and a != b:
            rel[r][idx[a], idx[b]] = 1.0

    for curie, concept in ontology.concepts.items():
        for p in concept.parents:
            put("subClassOf", curie, p)
        for e in concept.equivalent_to:
            put("equivalentClass", curie, e)
            put("equivalentClass", e, curie)
        for d in concept.disjoint_with:
            put("disjointWith", curie, d)
            put("disjointWith", d, curie)
    for curie, prop in ontology.properties.items():
        for d in prop.domains:
            put("domain", curie, d)
        for r in prop.ranges:
            put("range", curie, r)
        for i in prop.inverses:
            put("inverseOf", curie, i)
            put("inverseOf", i, curie)
        for p in prop.parents:
            put("subPropertyOf", curie, p)
    return OntologyGraph(names, kinds, rel)


def kind_mask(source: OntologyGraph, target: OntologyGraph) -> np.ndarray:
    """``(n_s, n_t)`` boolean: True where both entities are of the same kind."""
    ks, kt = source.kind_vector(), target.kind_vector()
    return ks[:, None] == kt[None, :]


# ---------------------------------------------------------------------------
# (a) Similarity Flooding
# ---------------------------------------------------------------------------


def _row_normalise(m: np.ndarray) -> np.ndarray:
    m = np.ascontiguousarray(m, dtype=np.float64)
    deg = m.sum(axis=1, keepdims=True)
    return np.divide(m, deg, out=np.zeros_like(m), where=deg > 0)


def similarity_flooding(
    seed: np.ndarray,
    source: OntologyGraph,
    target: OntologyGraph,
    *,
    iterations: int = 50,
    tolerance: float = 1e-4,
    relations: tuple[str, ...] = RELATIONS,
    normalise: str = "global",
) -> tuple[np.ndarray, int]:
    """Melnik et al. fixpoint (variant C) on dense matrices.  Returns ``(sigma, iterations_used)``.

    During the iteration ``sigma`` is normalised by its global maximum, as in the
    paper.  With ``normalise="relative"`` the returned matrix is the paper's *relative
    similarity* (section 4: ``sigma(x, y) / max(row max of x, column max of y)``), which
    is 1 for every pair that is the best for both of its entities and puts the fixpoint
    on the same ``[0, 1]`` scale as the seed; with ``"global"`` the global-max
    normalised fixpoint is returned as is.  Cross-kind pairs (class / property) are
    kept at zero throughout.
    """
    if normalise not in ("global", "relative"):
        raise ValueError(f"normalise must be 'global' or 'relative', got {normalise!r}")
    mask = kind_mask(source, target)
    sigma0 = np.where(mask, seed, 0.0).astype(np.float64)
    if sigma0.size == 0 or sigma0.max() <= 0.0:
        return sigma0, 0
    # Propagation operators, pre-transposed and contiguous so every step is a plain BLAS product.
    ops = []
    for r in relations:
        for sa, ta in ((source.relations[r], target.relations[r]), (source.relations[r].T, target.relations[r].T)):
            if sa.any() and ta.any():
                ops.append((np.ascontiguousarray(_row_normalise(sa).T), _row_normalise(ta)))

    def phi(s: np.ndarray) -> np.ndarray:
        out = np.zeros_like(s)
        for at, b in ops:
            out += at @ s @ b
        return out

    sigma = sigma0.copy()
    used = 0
    for i in range(1, iterations + 1):
        base = sigma0 + sigma
        nxt = base + phi(base)
        nxt = np.where(mask, nxt, 0.0)
        top = nxt.max()
        if top > 0:
            nxt = nxt / top
        delta = float(np.abs(nxt - sigma).max())
        sigma = nxt
        used = i
        if delta < tolerance:
            break
    if normalise == "relative":
        sigma = relative_similarity(sigma)
    return sigma, used


def relative_similarity(sigma: np.ndarray) -> np.ndarray:
    """``sigma(x, y) / max(max_y' sigma(x, y'), max_x' sigma(x', y))`` (0 where both maxima are 0)."""
    if sigma.size == 0:
        return sigma
    denom = np.maximum(sigma.max(axis=1, keepdims=True), sigma.max(axis=0, keepdims=True))
    return np.divide(sigma, denom, out=np.zeros_like(sigma), where=denom > 0)


# ---------------------------------------------------------------------------
# (b) Neighbour aggregation
# ---------------------------------------------------------------------------


def _padded_index(adj: np.ndarray) -> np.ndarray:
    """``(n, k_max)`` integer index of each row's neighbours, padded with ``n`` (a sentinel row)."""
    n = adj.shape[0]
    counts = adj.sum(axis=1)
    k = int(counts.max()) if n else 0
    out = np.full((n, max(k, 1)), n, dtype=np.int64)
    for i in range(n):
        nb = np.flatnonzero(adj[i])
        out[i, : len(nb)] = nb
    return out


def max_pooled_similarity(sim: np.ndarray, src_adj: np.ndarray, tgt_adj: np.ndarray) -> np.ndarray:
    """``out[a, b] = max_{a' in N(a), b' in N(b)} sim[a', b']`` (``-inf`` where either set is empty)."""
    n_s, n_t = sim.shape
    ext = np.full((n_s + 1, n_t + 1), -np.inf)
    ext[:n_s, :n_t] = sim
    ts = _padded_index(tgt_adj)  # (n_t, k_t)
    ss = _padded_index(src_adj)  # (n_s, k_s)
    # pooled over target neighbours: (n_s + 1, n_t)
    p = ext[:, ts].max(axis=2)
    # pooled over source neighbours: (n_s, n_t)
    return p[ss, :].max(axis=1)


def neighbour_aggregation(
    seed: np.ndarray,
    source: OntologyGraph,
    target: OntologyGraph,
    *,
    weight: float = 0.3,
    iterations: int = 10,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, int]:
    """``sim' = (1 - w) seed + w * mean_r maxpool_r(sim)``, iterated.  Returns ``(sim, iterations_used)``."""
    mask = kind_mask(source, target)
    seed = np.where(mask, seed, 0.0).astype(np.float64)
    if weight <= 0.0 or seed.size == 0:
        return seed, 0
    ns, nt = source.neighbourhoods(), target.neighbourhoods()
    types = [(ns[k], nt[k]) for k in ns if ns[k].any() and nt[k].any()]
    if not types:
        return seed, 0
    sim = seed.copy()
    used = 0
    for i in range(1, iterations + 1):
        total = np.zeros_like(sim)
        count = np.zeros_like(sim)
        for sa, ta in types:
            pooled = max_pooled_similarity(sim, sa, ta)
            present = np.isfinite(pooled)
            total += np.where(present, pooled, 0.0)
            count += present
        structural = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        # Pairs with no shared neighbourhood type fall back on the seed alone.
        nxt = np.where(count > 0, (1.0 - weight) * seed + weight * structural, seed)
        nxt = np.where(mask, nxt, 0.0)
        delta = float(np.abs(nxt - sim).max())
        sim = nxt
        used = i
        if delta < tolerance:
            break
    return sim, used


# ---------------------------------------------------------------------------
# (c) Embedding smoothing (training-free message passing)
# ---------------------------------------------------------------------------


def smooth_embeddings(vectors: np.ndarray, adjacency: np.ndarray, *, beta: float = 0.5, rounds: int = 1) -> np.ndarray:
    """``e' = normalise(e + beta * mean_{neighbours} e)``, ``rounds`` times.  Isolated entities are unchanged."""
    out = np.asarray(vectors, dtype=np.float32)
    if beta <= 0.0 or rounds <= 0 or out.size == 0:
        return out
    mean_op = _row_normalise(adjacency).astype(np.float32)
    for _ in range(rounds):
        out = out + beta * (mean_op @ out)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)
    return out


class SmoothedEmbeddingMatcher(EmbeddingMatcher):
    """:class:`EmbeddingMatcher` whose vectors are smoothed over the ontology graph before the cosine."""

    def __init__(
        self, encoder: Any, config: EmbeddingMatcherConfig | None = None, *, beta: float = 0.5, rounds: int = 1
    ):
        super().__init__(encoder, config)
        self.beta = beta
        self.rounds = rounds

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{super().name}+smooth(beta={self.beta:.2f},rounds={self.rounds})"

    def _encode(self, ontology: Ontology) -> tuple[list[Any], Any, Any]:
        key = ontology.name
        if key not in self._cache:
            texts, lab, full = super()._encode(ontology)
            adj = build_graph(ontology).undirected()
            lab_s = smooth_embeddings(lab, adj, beta=self.beta, rounds=self.rounds)
            full_s = lab_s if full is lab else smooth_embeddings(full, adj, beta=self.beta, rounds=self.rounds)
            self._cache[key] = (texts, lab_s, full_s)
        return self._cache[key]


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


def _default_hybrid() -> HybridMatcherConfig:
    return HybridMatcherConfig(
        combine="cascade",
        structure_weight=0.0,
        embedding=EmbeddingMatcherConfig(text="both", calibration="csls", csls_k=10),
    )


@dataclass
class StructuralMatcherConfig:
    """Knobs of :class:`StructuralMatcher`.

    ``method`` is one of :data:`STRUCTURAL_METHODS`.  ``weight`` is the mixing weight
    ``w`` of the ``flooding`` / ``neighbour`` methods and ``beta`` of ``smooth``;
    ``iterations`` the maximum fixpoint iterations (``flooding``, ``neighbour``) or the
    number of message-passing rounds (``smooth``).  ``hybrid`` configures the seed
    matcher (ignored when the matcher is built without an encoder: the seed is then
    the lexical matcher alone).
    """

    method: str = "neighbour"
    weight: float = 0.3
    iterations: int = 10
    tolerance: float = 1e-4
    hybrid: HybridMatcherConfig = field(default_factory=_default_hybrid)
    lexical: LexicalMatcherConfig = field(default_factory=LexicalMatcherConfig)


class StructuralMatcher(Matcher):
    """Seed similarity (hybrid, or lexical when ``encoder`` is ``None``) + structural propagation."""

    def __init__(
        self,
        encoder: Any | None,
        config: StructuralMatcherConfig | None = None,
        *,
        seed_cache: dict[tuple[str, str, str], dict[tuple[str, str], float]] | None = None,
    ):
        """``seed_cache`` may be shared between matchers with the same seed configuration so a
        grid over the structural knobs computes the (slow, pure-Python) lexical seed once per pair."""
        self.config = config or StructuralMatcherConfig()
        if self.config.method not in STRUCTURAL_METHODS:
            raise ValueError(f"unknown structural method {self.config.method!r}; expected one of {STRUCTURAL_METHODS}")
        self.encoder = encoder
        self.hybrid: HybridMatcher | None = None
        self.lexical = LexicalMatcher(self.config.lexical)
        if encoder is not None:
            self.hybrid = HybridMatcher(encoder, self.config.hybrid)
            if self.config.method == "smooth":
                self.hybrid.embedding = SmoothedEmbeddingMatcher(
                    encoder, self.config.hybrid.embedding, beta=self.config.weight, rounds=self.config.iterations
                )
        self.last_iterations: dict[tuple[str, str], int] = {}
        self._graphs: dict[str, OntologyGraph] = {}
        self._seed_cache = seed_cache if seed_cache is not None else {}

    @property
    def name(self) -> str:  # type: ignore[override]
        c = self.config
        base = self.hybrid.name if self.hybrid is not None else self.lexical.name
        if c.method == "none":
            return f"{base}+structure[none]"
        return f"{base}+structure[{c.method},w={c.weight:.2f},it={c.iterations}]"

    def graph(self, ontology: Ontology) -> OntologyGraph:
        if ontology.name not in self._graphs:
            self._graphs[ontology.name] = build_graph(ontology)
        return self._graphs[ontology.name]

    def seed_matrix(self, source: Ontology, target: Ontology) -> tuple[OntologyGraph, OntologyGraph, np.ndarray]:
        """Dense ``(n_s, n_t)`` seed similarity; zero for cross-kind pairs and unscored pairs."""
        gs, gt = self.graph(source), self.graph(target)
        seed_name = self.hybrid.name if self.hybrid is not None else self.lexical.name
        key = (seed_name, source.name, target.name)
        scores = self._seed_cache.get(key)
        if scores is None:
            if self.hybrid is not None:
                scores = self.hybrid.base_scores(source, target)
            else:
                scores = {(a.entity1, a.entity2): a.measure for a in self.lexical.score_all(source, target)}
            self._seed_cache[key] = scores
        seed = np.zeros((gs.n, gt.n), dtype=np.float64)
        si, ti = gs.index, gt.index
        for (s, t), v in scores.items():
            i, j = si.get(s), ti.get(t)
            if i is not None and j is not None:
                seed[i, j] = v
        return gs, gt, np.clip(seed, 0.0, 1.0)

    def similarity_matrix(self, source: Ontology, target: Ontology) -> tuple[OntologyGraph, OntologyGraph, np.ndarray]:
        gs, gt, seed = self.seed_matrix(source, target)
        c = self.config
        key = (source.name, target.name)
        if c.method in ("flooding", "flooding_relative"):
            sigma, used = similarity_flooding(
                seed,
                gs,
                gt,
                iterations=c.iterations,
                tolerance=c.tolerance,
                normalise="relative" if c.method == "flooding_relative" else "global",
            )
            sim = (1.0 - c.weight) * seed + c.weight * sigma
        elif c.method == "neighbour":
            sim, used = neighbour_aggregation(
                seed, gs, gt, weight=c.weight, iterations=c.iterations, tolerance=c.tolerance
            )
        else:  # "none" and "smooth" (smoothing already happened inside the embedding matcher)
            sim, used = seed, 0
        self.last_iterations[key] = used
        return gs, gt, np.clip(sim, 0.0, 1.0)

    def score_all(self, source: Ontology, target: Ontology) -> list[OAEIAlignment]:
        gs, gt, sim = self.similarity_matrix(source, target)
        out: list[OAEIAlignment] = []
        rows, cols = np.nonzero(sim > 0.0)
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
            out.append(OAEIAlignment(gs.names[i], gt.names[j], "=", round(float(sim[i, j]), 6)))
        return out


__all__ = [
    "RELATIONS",
    "STRUCTURAL_METHODS",
    "OntologyGraph",
    "SmoothedEmbeddingMatcher",
    "StructuralMatcher",
    "StructuralMatcherConfig",
    "build_graph",
    "kind_mask",
    "max_pooled_similarity",
    "neighbour_aggregation",
    "relative_similarity",
    "similarity_flooding",
    "smooth_embeddings",
]
