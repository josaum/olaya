"""Pairwise feature vectors for the learned OAEI matcher (:mod:`olaya.benchmarks.learned`).

For one ontology pair every cross pair of entities of the same kind (class/class,
property/property) gets a fixed-length feature vector built *only* from the two
ontologies and frozen encoders - no reference alignment is read here.  The groups
(feature names carry the group as a prefix, which is what the ablations in the
learned matcher key on):

``lex:*``
    the hand-tuned lexical matcher score, the IDF-weighted token Jaccard on its own,
    the normalised Levenshtein ratio on the normalised labels, head-noun equality,
    token containment in both directions, exact normalised-label equality and the
    comment / neighbour context Jaccard;
``emb:<enc>:*``
    label-only and full-description (label + comment + parents or domain / range)
    cosines of every encoder, raw and CSLS-calibrated (:func:`csls_calibrate`);
``rank:<score>:*``
    for the lexical score and every encoder's mean CSLS cosine: the log row / column
    rank of the pair, the margin to the best *other* entry in its row and column, and
    a mutual-best indicator;
``hybrid:*``
    the hand-tuned a-priori hybrid of ``oaei-conference-ra1`` (lexical mutual-best at 0.5,
    then the primary encoder's CSLS cosine for the entities left unmatched), without and
    with one round of parent / domain / range propagation (alpha 0.2) - the strongest
    single engineered signal, included so the learner can at least reproduce it;
``struct:*``
    with the base score ``mean(lex, primary encoder CSLS)``: max-pooled similarity of
    the parent, child, sibling, domain, range, domain-of and range-of neighbourhoods
    (:func:`olaya.benchmarks.structural.max_pooled_similarity`, with a presence flag
    for each), the converged neighbour-aggregated score
    (:func:`olaya.benchmarks.structural.neighbour_aggregation`), absolute depth
    difference and log child-count difference;
``meta:*``
    kind indicator and comment presence (both / either side).

The per-pair result (:class:`PairFeatures`) also carries the primary encoder's label
and full-text embeddings of every entity and the undirected adjacency of both graphs,
so the embedding-interaction models can be trained without re-encoding.  Everything
is cached on disk under ``$OLAYA_CACHE_DIR/learned-features`` keyed by the entity
texts, the relation structure, the encoder names and :data:`FEATURE_VERSION`, so a
cross-validation re-run only pays for the training.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from olaya.benchmarks.oaei import (
    EmbeddingMatcherConfig,
    HybridMatcher,
    HybridMatcherConfig,
    LexicalMatcher,
    LexicalMatcherConfig,
    build_entity_texts,
    build_profiles,
    csls_calibrate,
    levenshtein_similarity,
)
from olaya.benchmarks.structural import (
    OntologyGraph,
    build_graph,
    kind_mask,
    max_pooled_similarity,
    neighbour_aggregation,
)
from olaya.ontology import Ontology

#: Bump whenever the feature definition changes, so stale caches are ignored.
FEATURE_VERSION = 4

#: Neighbourhood types pooled for the structural features (names from ``OntologyGraph.neighbourhoods``).
STRUCT_NEIGHBOURHOODS = ("parents", "children", "siblings", "domains", "ranges", "domain_of", "range_of")

#: CSLS neighbourhood size (same as the hand-tuned matcher).
CSLS_K = 10


def cache_root() -> Path:
    base = os.environ.get("OLAYA_CACHE_DIR")
    root = Path(base).expanduser() if base else Path.home() / ".cache" / "olaya"
    return root / "learned-features"


@dataclass
class PairFeatures:
    """Feature vectors of every same-kind cross pair of one ontology pair."""

    source: str
    target: str
    src_names: list[str]  # entity order of ``build_graph`` (classes sorted, then properties sorted)
    tgt_names: list[str]
    src_kinds: list[str]
    tgt_kinds: list[str]
    rows: np.ndarray  # (n_pairs,) index into ``src_names``
    cols: np.ndarray  # (n_pairs,) index into ``tgt_names``
    feature_names: list[str]
    X: np.ndarray  # (n_pairs, n_features) float32
    src_emb_label: np.ndarray  # (n_s, d) primary encoder, label text
    src_emb_full: np.ndarray  # (n_s, d) primary encoder, full text
    tgt_emb_label: np.ndarray
    tgt_emb_full: np.ndarray
    src_adj: np.ndarray  # (n_s, n_s) undirected 0/1 adjacency
    tgt_adj: np.ndarray
    primary_encoder: str

    @property
    def n_pairs(self) -> int:
        return int(self.rows.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.src_names), len(self.tgt_names)

    def pair_names(self) -> list[tuple[str, str]]:
        return [
            (self.src_names[i], self.tgt_names[j]) for i, j in zip(self.rows.tolist(), self.cols.tolist(), strict=True)
        ]

    def pair_kinds(self) -> np.ndarray:
        """``True`` for class/class pairs, ``False`` for property/property pairs."""
        return np.array([self.src_kinds[i] == "class" for i in self.rows.tolist()], dtype=bool)

    def labels(self, reference: Sequence[Any]) -> np.ndarray:
        """0/1 vector: 1 iff ``(entity1, entity2)`` is an ``=`` correspondence of ``reference``."""
        ref = {(c.entity1, c.entity2) for c in reference if getattr(c, "relation", "=") == "="}
        return np.array([(a, b) in ref for a, b in self.pair_names()], dtype=np.float32)

    def to_matrix(self, values: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Scatter a per-pair vector back into the dense ``(n_s, n_t)`` matrix (cross-kind cells = ``fill``)."""
        m = np.full(self.shape, fill, dtype=np.float64)
        m[self.rows, self.cols] = values
        return m

    def columns(self, names: Sequence[str]) -> np.ndarray:
        idx = [self.feature_names.index(n) for n in names]
        return self.X[:, idx]

    # -- serialisation -------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "source": self.source,
            "target": self.target,
            "src_names": self.src_names,
            "tgt_names": self.tgt_names,
            "src_kinds": self.src_kinds,
            "tgt_kinds": self.tgt_kinds,
            "feature_names": self.feature_names,
            "primary_encoder": self.primary_encoder,
            "version": FEATURE_VERSION,
        }
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            meta=json.dumps(meta),
            rows=self.rows,
            cols=self.cols,
            X=self.X,
            src_emb_label=self.src_emb_label,
            src_emb_full=self.src_emb_full,
            tgt_emb_label=self.tgt_emb_label,
            tgt_emb_full=self.tgt_emb_full,
            src_adj=self.src_adj,
            tgt_adj=self.tgt_adj,
        )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> PairFeatures:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("version") != FEATURE_VERSION:
                raise ValueError("stale feature cache")
            return cls(
                source=meta["source"],
                target=meta["target"],
                src_names=meta["src_names"],
                tgt_names=meta["tgt_names"],
                src_kinds=meta["src_kinds"],
                tgt_kinds=meta["tgt_kinds"],
                rows=z["rows"],
                cols=z["cols"],
                feature_names=meta["feature_names"],
                X=z["X"],
                src_emb_label=z["src_emb_label"],
                src_emb_full=z["src_emb_full"],
                tgt_emb_label=z["tgt_emb_label"],
                tgt_emb_full=z["tgt_emb_full"],
                src_adj=z["src_adj"],
                tgt_adj=z["tgt_adj"],
                primary_encoder=meta["primary_encoder"],
            )


# ---------------------------------------------------------------------------
# Helpers on one ontology
# ---------------------------------------------------------------------------


def ontology_fingerprint(ontology: Ontology) -> str:
    """Hash of the entity texts and relation structure (what the features depend on)."""
    g = build_graph(ontology)
    texts = build_entity_texts(ontology)
    h = hashlib.sha256()
    for t in texts:
        h.update(f"{t.name}\t{t.kind}\t{t.label}\t{t.full}\n".encode())
    for r, m in sorted(g.relations.items()):
        h.update(r.encode())
        h.update(np.ascontiguousarray(m).tobytes())
    return h.hexdigest()


def _depths(adj_parent: np.ndarray) -> np.ndarray:
    """Longest root-to-node path length under ``adj_parent[a, p] = 1`` (cycle-safe, bounded by n)."""
    n = adj_parent.shape[0]
    depth = np.zeros(n, dtype=np.float64)
    parents = [np.flatnonzero(adj_parent[i]) for i in range(n)]
    for _ in range(n):
        new = np.array([max((depth[p] + 1.0 for p in parents[i]), default=0.0) for i in range(n)])
        if np.array_equal(new, depth):
            break
        depth = new
    return depth


def _entity_stats(ontology: Ontology, graph: OntologyGraph) -> dict[str, np.ndarray]:
    sub = graph.relations["subClassOf"]
    subp = graph.relations["subPropertyOf"]
    is_class = graph.kind_vector()
    parent_adj = np.where(is_class[:, None], sub, subp)
    depth = _depths(parent_adj)
    children = np.where(is_class, sub.sum(axis=0), subp.sum(axis=0)).astype(np.float64)
    has_comment = np.zeros(graph.n, dtype=np.float64)
    for i, name in enumerate(graph.names):
        ent = ontology.concepts.get(name) or ontology.properties.get(name)
        if ent is not None and getattr(ent, "description", ""):
            has_comment[i] = 1.0
    return {"depth": depth, "children": children, "comment": has_comment}


def _rank_features(score: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Row / column rank, margins to the best other entry and mutual-best flag of every cell."""
    s = np.where(mask, score, -np.inf)
    n_s, n_t = s.shape
    # rank 0 = best in its row / column (ties broken by position, deterministic)
    row_order = np.argsort(-s, axis=1, kind="stable")
    row_rank = np.empty_like(row_order)
    np.put_along_axis(row_rank, row_order, np.arange(n_t)[None, :].repeat(n_s, axis=0), axis=1)
    col_order = np.argsort(-s, axis=0, kind="stable")
    col_rank = np.empty_like(col_order)
    np.put_along_axis(col_rank, col_order, np.arange(n_s)[:, None].repeat(n_t, axis=1), axis=0)
    finite = np.where(np.isfinite(s), s, -1.0)
    row_top = np.sort(finite, axis=1)[:, ::-1]
    row_best = row_top[:, :1]
    row_second = row_top[:, 1:2] if n_t > 1 else np.full((n_s, 1), -1.0)
    col_top = np.sort(finite, axis=0)[::-1, :]
    col_best = col_top[:1, :]
    col_second = col_top[1:2, :] if n_s > 1 else np.full((1, n_t), -1.0)
    # margin = score - best other entry (positive iff the cell is the row / column maximum)
    row_margin = np.where(finite >= row_best, finite - row_second, finite - row_best)
    col_margin = np.where(finite >= col_best, finite - col_second, finite - col_best)
    mutual = ((row_rank == 0) & (col_rank == 0) & mask).astype(np.float64)
    return {
        "row_rank": np.log1p(row_rank.astype(np.float64)),
        "col_rank": np.log1p(col_rank.astype(np.float64)),
        "row_margin": row_margin,
        "col_margin": col_margin,
        "mutual": mutual,
    }


def _lexical_matrices(
    source: Ontology, target: Ontology, gs: OntologyGraph, gt: OntologyGraph
) -> dict[str, np.ndarray]:
    si, ti = gs.index, gt.index
    shape = (gs.n, gt.n)

    def to_matrix(cfg: LexicalMatcherConfig) -> np.ndarray:
        m = np.zeros(shape)
        for a in LexicalMatcher(cfg).score_all(source, target):
            m[si[a.entity1], ti[a.entity2]] = a.measure
        return m

    out = {
        "lex:score": to_matrix(LexicalMatcherConfig()),
        "lex:idf_jaccard": to_matrix(LexicalMatcherConfig(use_char_similarity=False, context_weight=0.0)),
    }
    sp = {p.name: p for p in build_profiles(source, use_stem=True, drop_stop=False)}
    tp = {p.name: p for p in build_profiles(target, use_stem=True, drop_stop=False)}
    lev = np.zeros(shape)
    head = np.zeros(shape)
    cont_ab = np.zeros(shape)
    cont_ba = np.zeros(shape)
    exact = np.zeros(shape)
    ctx = np.zeros(shape)
    for a, i in si.items():
        pa = sp[a]
        sa = set(pa.tokens)
        ca = sa | set(pa.context)
        for b, j in ti.items():
            pb = tp[b]
            if pa.kind != pb.kind:
                continue
            if pa.normalized and pb.normalized:
                lev[i, j] = levenshtein_similarity(pa.normalized, pb.normalized)
                exact[i, j] = float(pa.normalized == pb.normalized)
            sb = set(pb.tokens)
            if pa.tokens and pb.tokens:
                head[i, j] = float(pa.tokens[-1] == pb.tokens[-1])
            if sa and sb:
                cont_ab[i, j] = float(sa <= sb)
                cont_ba[i, j] = float(sb <= sa)
            cb = sb | set(pb.context)
            if ca and cb:
                ctx[i, j] = len(ca & cb) / len(ca | cb)
    out.update(
        {
            "lex:levenshtein": lev,
            "lex:head_equal": head,
            "lex:contain_ab": cont_ab,
            "lex:contain_ba": cont_ba,
            "lex:exact": exact,
            "lex:context_jaccard": ctx,
        }
    )
    return out


def _hybrid_matrices(
    source: Ontology, target: Ontology, gs: OntologyGraph, gt: OntologyGraph, encoder: Any
) -> dict[str, np.ndarray]:
    """The a-priori hand-tuned hybrid (cascade, text=both, CSLS k=10) with alpha 0 and 0.2."""
    si, ti = gs.index, gt.index
    out = {}
    for alpha, name in ((0.0, "hybrid:cascade"), (0.2, "hybrid:cascade_alpha02")):
        cfg = HybridMatcherConfig(
            combine="cascade",
            structure_weight=alpha,
            embedding=EmbeddingMatcherConfig(text="both", calibration="csls", csls_k=CSLS_K),
        )
        m = np.zeros((gs.n, gt.n))
        for a in HybridMatcher(encoder, cfg).score_all(source, target):
            m[si[a.entity1], ti[a.entity2]] = a.measure
        out[name] = m
    return out


def encoder_tag(name: str) -> str:
    """Short, filename-safe tag of an encoder name (``intfloat/e5-small-v2`` -> ``e5-small-v2``)."""
    return name.rsplit("/", 1)[-1].replace(":", "_")


def _embedding_matrices(
    source: Ontology, target: Ontology, encoders: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Per-encoder cosine matrices (raw and CSLS for label and full text) plus the entity vectors."""
    st, tt = build_entity_texts(source), build_entity_texts(target)
    feats: dict[str, np.ndarray] = {}
    means: dict[str, np.ndarray] = {}
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for name, enc in encoders.items():
        tag = encoder_tag(name)
        sl = np.asarray(enc.encode([t.label for t in st]), dtype=np.float32)
        sf = np.asarray(enc.encode([t.full for t in st]), dtype=np.float32)
        tl = np.asarray(enc.encode([t.label for t in tt]), dtype=np.float32)
        tf = np.asarray(enc.encode([t.full for t in tt]), dtype=np.float32)
        vectors[name] = {"src_label": sl, "src_full": sf, "tgt_label": tl, "tgt_full": tf}
        lab = (sl @ tl.T).astype(np.float64)
        full = (sf @ tf.T).astype(np.float64)
        lab_c = np.clip(csls_calibrate(lab, CSLS_K), 0.0, 1.0)
        full_c = np.clip(csls_calibrate(full, CSLS_K), 0.0, 1.0)
        feats[f"emb:{tag}:label_cos"] = lab
        feats[f"emb:{tag}:full_cos"] = full
        feats[f"emb:{tag}:label_csls"] = lab_c
        feats[f"emb:{tag}:full_csls"] = full_c
        means[tag] = 0.5 * (lab_c + full_c)
    return feats, means, vectors


def compute_pair_features(
    source: Ontology,
    target: Ontology,
    encoders: Mapping[str, Any],
    *,
    primary: str | None = None,
) -> PairFeatures:
    """Compute the feature vectors of every same-kind cross pair of ``source`` x ``target``.

    ``encoders`` maps an encoder name to an object with ``encode(texts) -> (n, d)``
    unit vectors; ``primary`` (default: the first) seeds the structural features and
    provides the stored entity embeddings.
    """
    if not encoders:
        raise ValueError("at least one encoder is required")
    primary = primary or next(iter(encoders))
    gs, gt = build_graph(source), build_graph(target)
    mask = kind_mask(gs, gt)
    feats: dict[str, np.ndarray] = {}

    feats.update(_lexical_matrices(source, target, gs, gt))
    emb_feats, emb_means, vectors = _embedding_matrices(source, target, encoders)
    feats.update(emb_feats)
    feats.update(_hybrid_matrices(source, target, gs, gt, encoders[primary]))

    # Rank / margin features of the lexical score and of each encoder's mean CSLS cosine.
    rank_bases = {"lex": feats["lex:score"], **emb_means}
    for tag, score in rank_bases.items():
        for k, v in _rank_features(score, mask).items():
            feats[f"rank:{tag}:{k}"] = v

    # Structural features on the base score mean(lex, primary CSLS).
    base = np.where(mask, 0.5 * (feats["lex:score"] + emb_means[encoder_tag(primary)]), 0.0)
    ns, nt = gs.neighbourhoods(), gt.neighbourhoods()
    for key in STRUCT_NEIGHBOURHOODS:
        pooled = max_pooled_similarity(base, ns[key], nt[key])
        present = np.isfinite(pooled)
        feats[f"struct:{key}"] = np.where(present, pooled, 0.0)
        feats[f"struct:{key}_present"] = present.astype(np.float64)
    agg, _ = neighbour_aggregation(base, gs, gt, weight=0.3, iterations=10)
    feats["struct:neighbour_agg"] = agg
    feats["struct:neighbour_gain"] = agg - base
    ss, ts = _entity_stats(source, gs), _entity_stats(target, gt)
    feats["struct:depth_diff"] = np.abs(ss["depth"][:, None] - ts["depth"][None, :])
    feats["struct:children_diff"] = np.abs(np.log1p(ss["children"])[:, None] - np.log1p(ts["children"])[None, :])
    feats["meta:is_class"] = np.broadcast_to(gs.kind_vector()[:, None].astype(np.float64), mask.shape).copy()
    feats["meta:comment_both"] = ss["comment"][:, None] * ts["comment"][None, :]
    feats["meta:comment_either"] = np.maximum(ss["comment"][:, None], ts["comment"][None, :])

    rows, cols = np.nonzero(mask)
    names = list(feats)
    X = np.stack([feats[n][rows, cols] for n in names], axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    v = vectors[primary]
    return PairFeatures(
        source=source.name,
        target=target.name,
        src_names=list(gs.names),
        tgt_names=list(gt.names),
        src_kinds=list(gs.kinds),
        tgt_kinds=list(gt.kinds),
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        feature_names=names,
        X=X,
        src_emb_label=v["src_label"],
        src_emb_full=v["src_full"],
        tgt_emb_label=v["tgt_label"],
        tgt_emb_full=v["tgt_full"],
        src_adj=gs.undirected().astype(np.float32),
        tgt_adj=gt.undirected().astype(np.float32),
        primary_encoder=primary,
    )


def pair_cache_key(source: Ontology, target: Ontology, encoder_names: Sequence[str], primary: str) -> str:
    h = hashlib.sha256()
    h.update(f"v{FEATURE_VERSION}|{ontology_fingerprint(source)}|{ontology_fingerprint(target)}|".encode())
    h.update(("|".join(encoder_names) + f"|primary={primary}").encode())
    return h.hexdigest()[:24]


def load_or_compute_pair_features(
    source: Ontology,
    target: Ontology,
    encoders: Mapping[str, Any],
    *,
    primary: str | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> tuple[PairFeatures, bool]:
    """Cached :func:`compute_pair_features`; returns ``(features, was_cached)``."""
    primary = primary or next(iter(encoders))
    root = cache_dir or cache_root()
    path = root / f"{source.name}-{target.name}-{pair_cache_key(source, target, list(encoders), primary)}.npz"
    if use_cache and path.exists():
        try:
            return PairFeatures.load(path), True
        except Exception:  # noqa: BLE001 - a corrupt / stale cache is recomputed
            pass
    pf = compute_pair_features(source, target, encoders, primary=primary)
    if use_cache:
        try:
            pf.save(path)
        except OSError:
            pass  # a read-only cache dir must not break the run
    return pf, False


def feature_groups(names: Sequence[str]) -> dict[str, list[str]]:
    """Group feature names by their prefix (``lex``, ``emb``, ``rank``, ``struct``, ``meta``)."""
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(n.split(":", 1)[0], []).append(n)
    return out


__all__ = [
    "CSLS_K",
    "FEATURE_VERSION",
    "STRUCT_NEIGHBOURHOODS",
    "PairFeatures",
    "cache_root",
    "compute_pair_features",
    "encoder_tag",
    "feature_groups",
    "load_or_compute_pair_features",
    "ontology_fingerprint",
    "pair_cache_key",
]
