"""Learned pairwise matcher for the OAEI Conference track, trained with leave-one-pair-out.

Protocol (supervised cross-validation on the track, the way trained systems such as
GraphMatcher are evaluated): for every held-out ontology pair a pairwise classifier is
trained on the reference alignments of the *other* pairs (positives = reference
correspondences, negatives = every other same-kind cross pair, class-weighted), then
the held-out pair is scored, decoded into a 1:1 alignment and evaluated against its
reference.  Decoder and confidence threshold are chosen by an *inner* split of the
training pairs only (:func:`inner_select`), never on the held-out pair; the held-out
TP / FP / FN are summed over folds into the usual micro P / R / F1.  Leave-one-
ontology-out (all six pairs of an ontology held out) is the stricter variant.

Models (all deterministic given the seed, all CPU-only):

``logreg``   logistic regression on standardised scalar features (:mod:`features`);
``mlp``      two-layer MLP (hidden 64) on the same features;
``gbt``      a small histogram gradient-boosted-trees implementation in numpy;
``emb_mlp``  the scalar features plus an interaction of the frozen encoder embeddings
             ``[p_a, p_b, |p_a - p_b|, p_a * p_b]`` (``p = W e`` a trained projection of
             the label + full-text vectors) fed to an MLP - the closest thing to
             GraphMatcher without a GNN;
``gcn_mlp``  the same, after one training-free GCN-style step that averages the
             neighbour embeddings into each entity vector (:func:`smooth_embeddings`).

Ablations are the same models on a subset of the feature groups.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from olaya.benchmarks.features import PairFeatures, encoder_tag, load_or_compute_pair_features
from olaya.benchmarks.oaei import (
    CONFERENCE_PAIRS,
    MODALITIES,
    OAEIAlignment,
    OAEIConferenceBenchmark,
    PairEvaluation,
    aggregate,
    apply_threshold,
    evaluate_alignment,
    select_mutual_best,
)
from olaya.benchmarks.structural import smooth_embeddings
from olaya.ontology import Ontology

DECODERS = ("mutual", "greedy", "hungarian")
THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20)) + (0.96, 0.97, 0.98, 0.99)  # 0.05 .. 0.99


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class PairDataset:
    """One ontology pair: its features, its reference alignment and the 0/1 labels."""

    pair: str
    source: Ontology
    target: Ontology
    features: PairFeatures
    reference: list[OAEIAlignment]
    y: np.ndarray

    @property
    def n_positive(self) -> int:
        return int(self.y.sum())


def build_dataset(
    bench: OAEIConferenceBenchmark,
    encoders: Mapping[str, Any],
    *,
    pairs: Iterable[tuple[str, str]] = CONFERENCE_PAIRS,
    primary: str | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> dict[str, PairDataset]:
    """Features + labels of every pair (features cached on disk, see :mod:`features`)."""
    out: dict[str, PairDataset] = {}
    for s, t in pairs:
        src, tgt = bench.ontology(s), bench.ontology(t)
        pf, _ = load_or_compute_pair_features(
            src, tgt, encoders, primary=primary, cache_dir=cache_dir, use_cache=use_cache
        )
        ref = bench.reference(s, t)
        out[f"{s}-{t}"] = PairDataset(f"{s}-{t}", src, tgt, pf, ref, pf.labels(ref))
    return out


# ---------------------------------------------------------------------------
# Feature selection / standardisation / negatives
# ---------------------------------------------------------------------------


def select_features(
    names: Sequence[str], *, keep_prefixes: Sequence[str] = (), drop_substrings: Sequence[str] = ()
) -> list[int]:
    """Column indices after keeping ``keep_prefixes`` (empty = all) and dropping ``drop_substrings``."""
    idx = []
    for i, n in enumerate(names):
        if keep_prefixes and not any(n.startswith(p) for p in keep_prefixes):
            continue
        if any(s in n for s in drop_substrings):
            continue
        idx.append(i)
    if not idx:
        raise ValueError("feature selection left no columns")
    return idx


@dataclass
class Standardiser:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> Standardiser:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


def hard_negative_rows(pf: PairFeatures, y: np.ndarray, *, top_k: int, n_random: int, seed: int) -> np.ndarray:
    """Row indices of every positive, the ``top_k`` per source / target entity under the lexical score and
    the primary encoder's CSLS cosines, and ``n_random`` random other pairs (deterministic)."""
    keep = np.zeros(pf.n_pairs, dtype=bool)
    keep |= y > 0
    tag = encoder_tag(pf.primary_encoder)
    score_cols = [
        i
        for i, n in enumerate(pf.feature_names)
        if n == "lex:score" or n in (f"emb:{tag}:label_csls", f"emb:{tag}:full_csls")
    ]
    for c in score_cols:
        m = pf.to_matrix(pf.X[:, c], fill=-np.inf)
        k_r = min(top_k, m.shape[1])
        k_c = min(top_k, m.shape[0])
        sel = np.zeros(m.shape, dtype=bool)
        if k_r > 0:
            np.put_along_axis(sel, np.argpartition(-m, k_r - 1, axis=1)[:, :k_r], True, axis=1)
        if k_c > 0:
            np.put_along_axis(sel, np.argpartition(-m, k_c - 1, axis=0)[:k_c, :], True, axis=0)
        keep |= sel[pf.rows, pf.cols]
    rng = np.random.default_rng(seed)
    rest = np.flatnonzero(~keep)
    if len(rest) and n_random > 0:
        keep[rng.choice(rest, size=min(n_random, len(rest)), replace=False)] = True
    return np.flatnonzero(keep)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """What to train.  ``kind`` in ``logreg`` / ``mlp`` / ``gbt`` / ``emb_mlp`` / ``gcn_mlp``."""

    name: str
    kind: str = "logreg"
    keep_prefixes: tuple[str, ...] = ()
    drop_substrings: tuple[str, ...] = ()
    negatives: str = "all"  # "all" or "hard"
    hard_top_k: int = 5
    hard_random: int = 300
    epochs: int = 600
    hidden: int = 64
    proj_dim: int = 64
    lr: float = 0.05
    weight_decay: float = 1e-4
    pos_weight_power: float = 0.5  # pos_weight = (n_neg / n_pos) ** power
    n_trees: int = 80
    max_depth: int = 3
    seed: int = 0


class Scorer:
    """A trained model: ``score(dataset) -> (n_pairs,)`` probabilities in ``[0, 1]``."""

    def score(self, pd: PairDataset) -> np.ndarray:
        raise NotImplementedError

    def importances(self) -> dict[str, float] | None:
        return None


def _pos_weight(y: np.ndarray, power: float) -> float:
    n_pos = max(1.0, float(y.sum()))
    n_neg = max(1.0, float(len(y) - y.sum()))
    return float((n_neg / n_pos) ** power)


#: Full-batch training on a few hundred thousand rows is memory-bound; with OpenMP on a 4-core
#: container, 4 intra-op threads are ~30x *slower* than 2 on these small tensors (spin / hand-off).
TORCH_THREADS = int(os.environ.get("OLAYA_TORCH_THREADS", "2"))


def _torch():
    import torch

    torch.set_num_threads(TORCH_THREADS)
    return torch


class TorchScalarScorer(Scorer):
    def __init__(self, spec: ModelSpec, columns: list[int], names: list[str], std: Standardiser, model: Any):
        self.spec, self.columns, self.names, self.std, self.model = spec, columns, names, std, model

    def score(self, pd: PairDataset) -> np.ndarray:
        torch = _torch()
        X = self.std(pd.features.X[:, self.columns])
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.from_numpy(X))).numpy().reshape(-1).astype(np.float64)

    def importances(self) -> dict[str, float] | None:
        if self.spec.kind != "logreg":
            return None
        w = self.model[0].weight.detach().numpy().reshape(-1)
        return {n: float(v) for n, v in zip(self.names, w, strict=True)}


def _train_torch(
    spec: ModelSpec, X: np.ndarray, y: np.ndarray, build: Callable[[int], Any], *, extra: tuple[np.ndarray, ...] = ()
) -> Any:
    torch = _torch()
    torch.manual_seed(spec.seed)
    model = build(X.shape[1])
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.float32))
    ex = [torch.from_numpy(e) for e in extra]
    pw = torch.tensor(_pos_weight(y, spec.pos_weight_power))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    for _ in range(spec.epochs):
        opt.zero_grad()
        out = model(xt, *ex) if ex else model(xt)
        loss = loss_fn(out.reshape(-1), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _scalar_net(torch: Any, spec: ModelSpec, d: int) -> Any:
    if spec.kind == "logreg":
        return torch.nn.Sequential(torch.nn.Linear(d, 1))
    return torch.nn.Sequential(
        torch.nn.Linear(d, spec.hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(spec.hidden, spec.hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(spec.hidden, 1),
    )


# -- embedding interaction model ------------------------------------------------------


def entity_vectors(pf: PairFeatures, *, gcn: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(n_s, 2d)`` / ``(n_t, 2d)`` label + full-text vectors, optionally neighbour-smoothed."""
    s = np.concatenate([pf.src_emb_label, pf.src_emb_full], axis=1)
    t = np.concatenate([pf.tgt_emb_label, pf.tgt_emb_full], axis=1)
    if gcn:
        s = smooth_embeddings(s, pf.src_adj, beta=0.5, rounds=1)
        t = smooth_embeddings(t, pf.tgt_adj, beta=0.5, rounds=1)
    return s.astype(np.float32), t.astype(np.float32)


class _InteractionNet:
    """Built lazily so that :mod:`torch` is only imported when a model is trained."""

    @staticmethod
    def build(torch: Any, spec: ModelSpec, d_scalar: int, d_emb: int) -> Any:
        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = torch.nn.Linear(d_emb, spec.proj_dim, bias=False)
                self.head = torch.nn.Sequential(
                    torch.nn.Linear(d_scalar + 4 * spec.proj_dim, 2 * spec.hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * spec.hidden, spec.hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(spec.hidden, 1),
                )

            def forward(self, x: Any, ea: Any, eb: Any) -> Any:
                pa, pb = self.proj(ea), self.proj(eb)
                z = torch.cat([x, pa, pb, (pa - pb).abs(), pa * pb], dim=1)
                return self.head(z)

        return Net()


class TorchEmbeddingScorer(Scorer):
    def __init__(self, spec: ModelSpec, columns: list[int], std: Standardiser, model: Any):
        self.spec, self.columns, self.std, self.model = spec, columns, std, model

    def score(self, pd: PairDataset) -> np.ndarray:
        torch = _torch()
        pf = pd.features
        X = self.std(pf.X[:, self.columns])
        es, et = entity_vectors(pf, gcn=self.spec.kind == "gcn_mlp")
        with torch.no_grad():
            out = self.model(torch.from_numpy(X), torch.from_numpy(es[pf.rows]), torch.from_numpy(et[pf.cols]))
            return torch.sigmoid(out).numpy().reshape(-1).astype(np.float64)


# -- gradient boosted trees (numpy) -------------------------------------------------


@dataclass
class _Tree:
    feature: list[int]
    threshold: list[int]  # bin index; go left iff bin <= threshold
    left: list[int]
    right: list[int]
    value: list[float]

    def predict(self, Xb: np.ndarray) -> np.ndarray:
        node = np.zeros(Xb.shape[0], dtype=np.int64)
        out = np.zeros(Xb.shape[0], dtype=np.float64)
        active = np.ones(Xb.shape[0], dtype=bool)
        feature = np.array(self.feature)
        threshold = np.array(self.threshold)
        left, right, value = np.array(self.left), np.array(self.right), np.array(self.value)
        while active.any():
            idx = np.flatnonzero(active)
            nd = node[idx]
            leaf = left[nd] < 0
            out[idx[leaf]] = value[nd[leaf]]
            active[idx[leaf]] = False
            inner = idx[~leaf]
            nd = nd[~leaf]
            go_left = Xb[inner, feature[nd]] <= threshold[nd]
            node[inner] = np.where(go_left, left[nd], right[nd])
        return out


class GradientBoostingScorer(Scorer):
    """Histogram gradient boosting for weighted logistic loss (depth-wise, 32 quantile bins)."""

    n_bins = 32
    reg_lambda = 1.0
    min_child_weight = 1.0

    def __init__(self, spec: ModelSpec, columns: list[int], names: list[str]):
        self.spec, self.columns, self.names = spec, columns, names
        self.edges: list[np.ndarray] = []
        self.trees: list[_Tree] = []
        self.base = 0.0
        self.lr = 0.1
        self.gain_by_feature = np.zeros(len(columns))

    def _bin(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape, dtype=np.int64)
        for j, e in enumerate(self.edges):
            out[:, j] = np.searchsorted(e, X[:, j], side="right")
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingScorer:
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        Xb = self._bin(X)
        w = np.where(y > 0, _pos_weight(y, self.spec.pos_weight_power), 1.0)
        p0 = float(np.clip((w * y).sum() / w.sum(), 1e-4, 1 - 1e-4))
        self.base = float(np.log(p0 / (1 - p0)))
        F = np.full(X.shape[0], self.base)
        rng = np.random.default_rng(self.spec.seed)
        for _ in range(self.spec.n_trees):
            p = 1.0 / (1.0 + np.exp(-F))
            g = w * (p - y)
            h = np.maximum(w * p * (1 - p), 1e-6)
            rows = rng.choice(X.shape[0], size=int(0.8 * X.shape[0]), replace=False)
            tree = self._grow(Xb, g, h, rows)
            self.trees.append(tree)
            F += self.lr * tree.predict(Xb)
        return self

    def _grow(self, Xb: np.ndarray, g: np.ndarray, h: np.ndarray, rows: np.ndarray) -> _Tree:
        tree = _Tree([], [], [], [], [])
        lam = self.reg_lambda

        def new_node(idx: np.ndarray) -> int:
            tree.feature.append(-1)
            tree.threshold.append(-1)
            tree.left.append(-1)
            tree.right.append(-1)
            tree.value.append(float(-g[idx].sum() / (h[idx].sum() + lam)))
            return len(tree.value) - 1

        root = new_node(rows)
        frontier = [(root, rows, 0)]
        while frontier:
            node, idx, depth = frontier.pop()
            if depth >= self.spec.max_depth or len(idx) < 2:
                continue
            G, H = g[idx].sum(), h[idx].sum()
            best = (0.0, -1, -1)
            for j in range(Xb.shape[1]):
                b = Xb[idx, j]
                gh = np.bincount(b, weights=g[idx], minlength=self.n_bins + 1)
                hh = np.bincount(b, weights=h[idx], minlength=self.n_bins + 1)
                GL, HL = np.cumsum(gh)[:-1], np.cumsum(hh)[:-1]
                GR, HR = G - GL, H - HL
                ok = (HL >= self.min_child_weight) & (HR >= self.min_child_weight)
                gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - G**2 / (H + lam))
                gain = np.where(ok, gain, -np.inf)
                k = int(np.argmax(gain))
                if gain[k] > best[0]:
                    best = (float(gain[k]), j, k)
            gain, j, k = best
            if j < 0:
                continue
            self.gain_by_feature[j] += gain
            mask = Xb[idx, j] <= k
            lidx, ridx = idx[mask], idx[~mask]
            left, right = new_node(lidx), new_node(ridx)
            tree.feature[node], tree.threshold[node] = j, k
            tree.left[node], tree.right[node] = left, right
            frontier.append((left, lidx, depth + 1))
            frontier.append((right, ridx, depth + 1))
        return tree

    def score(self, pd: PairDataset) -> np.ndarray:
        Xb = self._bin(pd.features.X[:, self.columns])
        F = np.full(Xb.shape[0], self.base)
        for t in self.trees:
            F += self.lr * t.predict(Xb)
        return 1.0 / (1.0 + np.exp(-F))

    def importances(self) -> dict[str, float] | None:
        tot = self.gain_by_feature.sum() or 1.0
        return {n: float(v / tot) for n, v in zip(self.names, self.gain_by_feature, strict=True)}


# -- training entry point -------------------------------------------------------------


def _stack_rows(
    spec: ModelSpec, train: Sequence[PairDataset], columns: list[int]
) -> tuple[np.ndarray, np.ndarray, list[tuple[PairDataset, np.ndarray]]]:
    xs, ys, picks = [], [], []
    for k, pd in enumerate(train):
        if spec.negatives == "hard":
            rows = hard_negative_rows(
                pd.features, pd.y, top_k=spec.hard_top_k, n_random=spec.hard_random, seed=spec.seed + k
            )
        else:
            rows = np.arange(pd.features.n_pairs)
        xs.append(pd.features.X[rows][:, columns])
        ys.append(pd.y[rows])
        picks.append((pd, rows))
    return np.concatenate(xs), np.concatenate(ys), picks


def train_model(spec: ModelSpec, train: Sequence[PairDataset]) -> Scorer:
    """Train ``spec`` on the pairs in ``train`` (labels come only from those pairs)."""
    names = train[0].features.feature_names
    columns = select_features(names, keep_prefixes=spec.keep_prefixes, drop_substrings=spec.drop_substrings)
    sel_names = [names[i] for i in columns]
    X, y, picks = _stack_rows(spec, train, columns)
    if spec.kind == "gbt":
        return GradientBoostingScorer(spec, columns, sel_names).fit(X, y)
    std = Standardiser.fit(X)
    Xs = std(X)
    if spec.kind in ("logreg", "mlp"):
        torch = _torch()
        model = _train_torch(spec, Xs, y, lambda d: _scalar_net(torch, spec, d))
        return TorchScalarScorer(spec, columns, sel_names, std, model)
    if spec.kind in ("emb_mlp", "gcn_mlp"):
        torch = _torch()
        ea, eb = [], []
        for pd, rows in picks:
            es, et = entity_vectors(pd.features, gcn=spec.kind == "gcn_mlp")
            ea.append(es[pd.features.rows[rows]])
            eb.append(et[pd.features.cols[rows]])
        ea_arr, eb_arr = np.concatenate(ea), np.concatenate(eb)
        d_emb = ea_arr.shape[1]
        model = _train_torch(
            spec, Xs, y, lambda d: _InteractionNet.build(torch, spec, d, d_emb), extra=(ea_arr, eb_arr)
        )
        return TorchEmbeddingScorer(spec, columns, std, model)
    raise ValueError(f"unknown model kind {spec.kind!r}")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def hungarian_max(score: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-weight assignment of a rectangular ``(n, m)`` score matrix (Jonker-Volgenant style
    shortest augmenting paths, rows <= columns after an optional transpose).  Returns ``(row, col)`` pairs."""
    if score.size == 0:
        return []
    transposed = score.shape[0] > score.shape[1]
    cost = -(score.T if transposed else score).astype(np.float64)
    n, m = cost.shape
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)  # p[j] = row assigned to column j (1-based), 0 = free
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = cost[i0 - 1, :] - u[i0] - v[1:]
            cand = ~used[1:]
            better = cand & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            free = np.where(cand, minv[1:], INF)
            j1 = int(np.argmin(free)) + 1
            delta = free[j1 - 1]
            upd = used
            u[p[upd]] += delta
            v[upd] -= delta
            minv[~upd] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] > 0]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)


def decode_all(pf: PairFeatures, probs: np.ndarray, decoder: str) -> list[OAEIAlignment]:
    """Un-thresholded 1:1 alignment of the per-pair probabilities (per kind), sorted by confidence.

    ``mutual``: each entity is the other's best match; ``greedy``: take pairs in decreasing order while
    both entities are free; ``hungarian``: maximum-weight assignment.  The threshold is applied
    afterwards (:func:`decode`), so a threshold sweep decodes once.
    """
    m = pf.to_matrix(probs, fill=-1.0)
    src_kinds, tgt_kinds = np.array(pf.src_kinds), np.array(pf.tgt_kinds)
    out: list[OAEIAlignment] = []
    for kind in ("class", "property"):
        ri, ci = np.flatnonzero(src_kinds == kind), np.flatnonzero(tgt_kinds == kind)
        if len(ri) == 0 or len(ci) == 0:
            continue
        block = m[np.ix_(ri, ci)]
        if decoder == "mutual":
            scored = [
                OAEIAlignment(pf.src_names[ri[r]], pf.tgt_names[ci[c]], "=", float(block[r, c]))
                for r, c in zip(*np.nonzero(block > 0.0), strict=True)
            ]
            pairs = [(pf.src_names.index(a.entity1), pf.tgt_names.index(a.entity2)) for a in select_mutual_best(scored)]
            out.extend(OAEIAlignment(pf.src_names[r], pf.tgt_names[c], "=", float(m[r, c])) for r, c in pairs)
            continue
        if decoder == "greedy":
            rr, cc = np.nonzero(block > 0.0)
            order = sorted(
                range(len(rr)), key=lambda i: (-block[rr[i], cc[i]], pf.src_names[ri[rr[i]]], pf.tgt_names[ci[cc[i]]])
            )
            taken_r, taken_c = np.zeros(len(ri), bool), np.zeros(len(ci), bool)
            for i in order:
                r, c = rr[i], cc[i]
                if taken_r[r] or taken_c[c]:
                    continue
                taken_r[r] = taken_c[c] = True
                out.append(OAEIAlignment(pf.src_names[ri[r]], pf.tgt_names[ci[c]], "=", float(block[r, c])))
            continue
        if decoder == "hungarian":
            for r, c in hungarian_max(np.where(block > 0.0, block, 0.0)):
                if block[r, c] > 0.0:
                    out.append(OAEIAlignment(pf.src_names[ri[r]], pf.tgt_names[ci[c]], "=", float(block[r, c])))
            continue
        raise ValueError(f"unknown decoder {decoder!r}")
    return sorted(out, key=lambda c: (-c.measure, c.entity1, c.entity2))


def decode(pf: PairFeatures, probs: np.ndarray, decoder: str, threshold: float) -> list[OAEIAlignment]:
    """Thresholded 1:1 alignment (:func:`decode_all` followed by the confidence cut-off)."""
    return apply_threshold(decode_all(pf, probs, decoder), threshold)


def evaluate_pair(
    pd: PairDataset,
    probs: np.ndarray,
    decoder: str,
    threshold: float,
    *,
    decoded: list[OAEIAlignment] | None = None,
) -> PairEvaluation:
    """Decode and score one pair (``decoded`` = a cached :func:`decode_all` result to skip the decoding)."""
    system = apply_threshold(decoded if decoded is not None else decode_all(pd.features, probs, decoder), threshold)
    return PairEvaluation(
        pair=pd.pair,
        source=pd.source.name,
        target=pd.target.name,
        reference_size=len(pd.reference),
        system_size=len(system),
        results=evaluate_alignment(system, pd.reference, pd.source, pd.target),
    )


def _micro_f1(evals: Iterable[PairEvaluation], modality: str) -> float:
    return aggregate(list(evals))[modality]["micro"]["f1"]


def best_operating_point(
    scored: Sequence[tuple[PairDataset, np.ndarray]],
    *,
    decoders: Sequence[str] = DECODERS,
    thresholds: Sequence[float] = THRESHOLDS,
    modality: str = "M3",
) -> tuple[str, float, float]:
    """``(decoder, threshold, micro_f1)`` maximising the micro-F1 of ``modality`` over ``scored``
    (ties: first decoder in ``decoders``, then lower threshold)."""
    best = ("", 0.0, -1.0)
    for dec in decoders:
        decoded = [(pd, decode_all(pd.features, pr, dec)) for pd, pr in scored]
        for th in thresholds:
            f1 = _micro_f1([evaluate_pair(pd, None, dec, th, decoded=d) for pd, d in decoded], modality)
            if f1 > best[2] + 1e-12:
                best = (dec, th, f1)
    return best


def inner_splits(train_pairs: Sequence[str], n_splits: int) -> list[tuple[list[str], list[str]]]:
    """Deterministic inner splits of the training pairs: split ``k`` holds out every ``n_splits``-th pair
    (offset ``k``) of the sorted list."""
    names = sorted(train_pairs)
    out = []
    for k in range(n_splits):
        held = [p for i, p in enumerate(names) if i % n_splits == k]
        out.append(([p for p in names if p not in held], held))
    return out


def inner_select(
    spec: ModelSpec,
    data: Mapping[str, PairDataset],
    train_pairs: Sequence[str],
    *,
    n_splits: int = 4,
    modality: str = "M3",
    decoders: Sequence[str] = DECODERS,
    thresholds: Sequence[float] = THRESHOLDS,
) -> tuple[str, float, float]:
    """Choose decoder and threshold on the training pairs only: train on the inner-train pairs of every
    inner split, score the inner-held-out pairs, and pick the operating point with the best pooled
    micro-F1.  ``train_pairs`` must not contain the outer held-out pair(s) - this function never sees them."""
    scored: list[tuple[PairDataset, np.ndarray]] = []
    for inner_train, inner_held in inner_splits(train_pairs, n_splits):
        scorer = train_model(spec, [data[p] for p in inner_train])
        scored.extend((data[p], scorer.score(data[p])) for p in inner_held)
    return best_operating_point(scored, decoders=decoders, thresholds=thresholds, modality=modality)


# ---------------------------------------------------------------------------
# Cross-validation driver
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    model: str
    folds: int
    chosen: dict[str, dict[str, Any]]
    pairs: list[PairEvaluation]
    aggregates: dict[str, Any]
    importances: dict[str, float] | None = None
    seconds: float = 0.0
    fold_scores: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def micro(self, modality: str) -> dict[str, Any]:
        return self.aggregates[modality]["micro"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "folds": self.folds,
            "chosen": self.chosen,
            "aggregates": self.aggregates,
            "pairs": [p.to_dict() for p in self.pairs],
            "importances": self.importances,
            "seconds": round(self.seconds, 1),
        }


def cross_validate_model(
    spec: ModelSpec,
    data: Mapping[str, PairDataset],
    folds: Mapping[str, tuple[list[str], list[str]]],
    *,
    inner_splits_n: int = 4,
    modality: str = "M3",
    decoders: Sequence[str] = DECODERS,
    thresholds: Sequence[float] = THRESHOLDS,
    log: Callable[[str], None] | None = None,
) -> CVResult:
    """Nested CV: for every fold, pick decoder / threshold by :func:`inner_select` on the train pairs,
    train the final model on all train pairs and score the held-out pairs under that operating point."""
    t0 = time.perf_counter()
    evals: list[PairEvaluation] = []
    chosen: dict[str, dict[str, Any]] = {}
    imps: list[dict[str, float]] = []
    fold_scores: dict[str, np.ndarray] = {}
    for fold, (train, test) in folds.items():
        assert not set(train) & set(test), "train / test pairs overlap"
        dec, th, inner_f1 = inner_select(
            spec, data, train, n_splits=inner_splits_n, modality=modality, decoders=decoders, thresholds=thresholds
        )
        scorer = train_model(spec, [data[p] for p in train])
        imp = scorer.importances()
        if imp:
            imps.append(imp)
        for p in test:
            probs = scorer.score(data[p])
            fold_scores[p] = probs
            evals.append(evaluate_pair(data[p], probs, dec, th))
        chosen[fold] = {"decoder": dec, "threshold": th, "inner_f1": round(inner_f1, 4)}
        if log:
            e = aggregate([x for x in evals if x.pair in test])
            log(
                f"  [{spec.name}] fold {fold}: {dec} @ {th:.2f} (inner F1 {inner_f1:.3f}) -> "
                f"M1 {e['M1']['micro']['f1']:.3f} M3 {e['M3']['micro']['f1']:.3f}"
            )
    importances = None
    if imps:
        keys = imps[0].keys()
        importances = {k: float(np.mean([d[k] for d in imps])) for k in keys}
    return CVResult(
        model=spec.name,
        folds=len(folds),
        chosen=chosen,
        pairs=evals,
        aggregates=aggregate(evals),
        importances=importances,
        seconds=time.perf_counter() - t0,
        fold_scores=fold_scores,
    )


def summary_row(res: CVResult) -> dict[str, dict[str, float]]:
    return {m: {k: res.micro(m)[k] for k in ("precision", "recall", "f1")} for m in MODALITIES}


__all__ = [
    "DECODERS",
    "THRESHOLDS",
    "CVResult",
    "GradientBoostingScorer",
    "ModelSpec",
    "PairDataset",
    "Scorer",
    "Standardiser",
    "best_operating_point",
    "build_dataset",
    "cross_validate_model",
    "decode",
    "decode_all",
    "entity_vectors",
    "evaluate_pair",
    "hard_negative_rows",
    "hungarian_max",
    "inner_select",
    "inner_splits",
    "select_features",
    "summary_row",
    "train_model",
]
