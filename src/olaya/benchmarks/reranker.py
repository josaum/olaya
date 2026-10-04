"""Cross-encoder reranker for OAEI alignment candidates.

The matchers in :mod:`olaya.benchmarks.oaei` score every cross pair with a lexical /
embedding similarity, keep the mutual-best 1:1 correspondences and threshold them.
This module adds a *reranking* stage on top of that pipeline:

1. the base matcher's scores are reduced to a candidate set (top-``k`` per source
   entity and top-``k`` per target entity);
2. a small cross-encoder (sequence classification over ``text(A) [SEP] text(B)``,
   where ``text`` is the same :func:`~olaya.benchmarks.oaei.build_entity_texts`
   description the embedding matcher uses) scores every candidate;
3. the base score and the cross-encoder probability are fused as
   ``(1 - w) * base + w * ce`` and the usual mutual-best + threshold selection runs
   on the fused candidate scores.

The cross-encoder is fine-tuned on the OAEI *Anatomy* track (mouse / human
reference alignment: positives from the reference, hard negatives from the
top-``k`` non-reference candidates), so the Conference track remains a clean test
set.  Nothing in here reads a Conference reference alignment except the evaluation
helpers, which are the same honest protocol as the base benchmark (a-priori config,
full grid, leave-one-pair-out cross-validation).

Training and scoring need ``torch`` / ``transformers``; everything else (pair
building, splits, fusion, grid / CV bookkeeping) is dependency-free and unit-tested.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from olaya.benchmarks.oaei import (
    MODALITIES,
    PRF,
    EntityText,
    GridResults,
    Matcher,
    OAEIAlignment,
    OAEIAlignmentParser,
    OAEIConferenceBenchmark,
    OWLParser,
    PairEvaluation,
    aggregate,
    apply_threshold,
    build_entity_texts,
    csls_calibrate,
    evaluate_alignment,
    select_mutual_best,
)
from olaya.ontology import Ontology

DEFAULT_BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BASE_REVISION = "main"
ANATOMY_FILES = ("mouse.owl", "human.owl", "reference.rdf")


# ---------------------------------------------------------------------------
# Entity text
# ---------------------------------------------------------------------------


def pair_text(text: str) -> str:
    """Light normalisation of an entity description before it enters the cross-encoder.

    The anatomy labels use underscores as word separators (``Spermatic_Cord``) while the
    Conference ontologies use spaces; the cross-encoder vocabulary knows words, not
    underscores, so they are mapped to spaces.  Nothing else is changed.
    """
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def entity_texts(ontology: Ontology) -> dict[str, EntityText]:
    """``{local name: EntityText}`` for every class and property (same text as the embedding matcher)."""
    return {t.name: t for t in build_entity_texts(ontology)}


def pair_texts(
    pairs: Iterable[tuple[str, str]], source_texts: dict[str, EntityText], target_texts: dict[str, EntityText]
) -> list[tuple[str, str]]:
    """Cross-encoder inputs ``(text(A), text(B))`` for the given ``(source, target)`` local names."""
    return [(pair_text(source_texts[s].full), pair_text(target_texts[t].full)) for s, t in pairs]


# ---------------------------------------------------------------------------
# Candidates, pairs, splits
# ---------------------------------------------------------------------------

Scores = dict[tuple[str, str], float]


def candidate_pairs(scores: Scores, k: int) -> set[tuple[str, str]]:
    """Top-``k`` pairs per source entity plus top-``k`` per target entity (union).

    Ties are broken by the entity names so the set does not depend on dict order.
    """
    if k <= 0:
        return set()
    by_source: dict[str, list[tuple[float, str]]] = {}
    by_target: dict[str, list[tuple[float, str]]] = {}
    for (s, t), v in scores.items():
        by_source.setdefault(s, []).append((v, t))
        by_target.setdefault(t, []).append((v, s))
    out: set[tuple[str, str]] = set()
    for s, lst in by_source.items():
        lst.sort(key=lambda x: (-x[0], x[1]))
        out.update((s, t) for _, t in lst[:k])
    for t, lst in by_target.items():
        lst.sort(key=lambda x: (-x[0], x[1]))
        out.update((s, t) for _, s in lst[:k])
    return out


def topk_from_matrix(sim: np.ndarray, rows: Sequence[str], cols: Sequence[str], k: int) -> Scores:
    """Top-``k`` per row and per column of a dense similarity matrix, as a sparse score dict.

    Used for the Anatomy track, where the dense 2743 x 3304 cross product does not fit a
    Python dict comfortably but a numpy matrix does.
    """
    out: Scores = {}
    if sim.size == 0 or k <= 0:
        return out
    kk = min(k, sim.shape[1])
    idx = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
    for i, r in enumerate(rows):
        for j in idx[i]:
            out[(r, cols[j])] = float(sim[i, j])
    kk = min(k, sim.shape[0])
    idx = np.argpartition(-sim, kk - 1, axis=0)[:kk, :]
    for j, c in enumerate(cols):
        for i in idx[:, j]:
            out[(rows[i], c)] = float(sim[i, j])
    return out


@dataclass(frozen=True)
class Pair:
    """One training / validation example."""

    source: str
    target: str
    label: int  # 1 = in the reference alignment
    base: float = 0.0  # base-matcher score (0 when the pair is a reference pair outside the candidates)


def build_pairs(
    reference: Iterable[OAEIAlignment],
    scores: Scores,
    k: int,
    *,
    sources: set[str] | None = None,
    include_missing_positives: bool = True,
) -> list[Pair]:
    """Positives from the reference, hard negatives from the top-``k`` non-reference candidates.

    ``sources`` restricts the pairs to the given source entities (used to split train /
    validation by source entity so that no entity leaks across the split).  Reference pairs
    outside the candidate set are kept as positives (``include_missing_positives``) with
    base score 0; the candidate recall ceiling is reported separately.
    """
    ref = {(a.entity1, a.entity2) for a in reference if a.relation == "="}
    cands = candidate_pairs(scores, k)
    pairs: list[Pair] = []
    for s, t in sorted(ref):
        if sources is not None and s not in sources:
            continue
        if (s, t) in cands or include_missing_positives:
            pairs.append(Pair(s, t, 1, scores.get((s, t), 0.0)))
    for s, t in sorted(cands):
        if (s, t) in ref:
            continue
        if sources is not None and s not in sources:
            continue
        pairs.append(Pair(s, t, 0, scores[(s, t)]))
    return pairs


def split_by_source(
    reference: Iterable[OAEIAlignment], *, fraction: float = 0.8, seed: int = 0
) -> tuple[set[str], set[str]]:
    """Deterministic train / validation split of the *source entities* of a reference alignment."""
    return split_sources({a.entity1 for a in reference}, fraction=fraction, seed=seed)


def split_sources(names: Iterable[str], *, fraction: float = 0.8, seed: int = 0) -> tuple[set[str], set[str]]:
    """Deterministic (seeded shuffle) split of entity names into ``(train, validation)`` sets."""
    sources = sorted(set(names))
    rng = random.Random(seed)
    rng.shuffle(sources)
    n_train = int(round(fraction * len(sources)))
    return set(sources[:n_train]), set(sources[n_train:])


def pairs_digest(pairs: Iterable[Pair]) -> str:
    """SHA-256 of the sorted ``source\\ttarget\\tlabel`` lines (the training-data hash in the manifest)."""
    lines = sorted(f"{p.source}\t{p.target}\t{p.label}" for p in pairs)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Fusion and selection
# ---------------------------------------------------------------------------


def fuse(base: float, ce: float, weight: float) -> float:
    """``(1 - w) * base + w * ce``; ``w = 0`` is the base matcher, ``w = 1`` the cross-encoder alone."""
    return (1.0 - weight) * base + weight * ce


def fuse_scores(base: Scores, ce: Scores, weight: float) -> Scores:
    """Fused scores for the candidate pairs in ``ce`` (pairs without a cross-encoder score are dropped)."""
    return {p: fuse(base.get(p, 0.0), v, weight) for p, v in ce.items()}


def select(scores: Scores, threshold: float) -> list[OAEIAlignment]:
    """Mutual-best 1:1 selection then threshold, exactly as the base benchmark does."""
    scored = [OAEIAlignment(s, t, "=", round(v, 6)) for (s, t), v in sorted(scores.items()) if v > 0.0]
    return apply_threshold(select_mutual_best(scored), threshold)


def config_label(base: str, k: int, weight: float, threshold: float) -> str:
    return f"base={base},k={k},w={weight:.2f},th={threshold:.2f}"


# ---------------------------------------------------------------------------
# Ranking metrics for the held-out validation
# ---------------------------------------------------------------------------


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """ROC AUC by the rank statistic (ties get half credit); 0.5 when one class is missing."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        return 0.5
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for q in order[i : j + 1]:
            ranks[q] = r
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels, strict=True) if y == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def hits_at_1(pairs: Sequence[Pair], scores: Sequence[float]) -> float:
    """Fraction of source entities with a positive whose best-scored pair is a positive."""
    best: dict[str, tuple[float, int]] = {}
    has_pos: set[str] = set()
    for p, s in zip(pairs, scores, strict=True):
        if p.label == 1:
            has_pos.add(p.source)
        cur = best.get(p.source)
        if cur is None or s > cur[0] or (s == cur[0] and p.label > cur[1]):
            best[p.source] = (s, p.label)
    if not has_pos:
        return 0.0
    return sum(1 for s in has_pos if best[s][1] == 1) / len(has_pos)


def validation_report(
    pairs: Sequence[Pair], ce: Sequence[float], weights: Sequence[float] = (0.0, 0.5, 1.0)
) -> dict[str, Any]:
    """AUC and hits@1 of the base score, the cross-encoder and their fusions on labelled pairs."""
    labels = [p.label for p in pairs]
    base = [p.base for p in pairs]
    out: dict[str, Any] = {
        "pairs": len(pairs),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "base": {"auc": round(auc(labels, base), 4), "hits@1": round(hits_at_1(pairs, base), 4)},
        "cross_encoder": {"auc": round(auc(labels, list(ce)), 4), "hits@1": round(hits_at_1(pairs, list(ce)), 4)},
        "fused": {},
    }
    for w in weights:
        fused = [fuse(b, c, w) for b, c in zip(base, ce, strict=True)]
        out["fused"][f"w={w:.2f}"] = {"auc": round(auc(labels, fused), 4), "hits@1": round(hits_at_1(pairs, fused), 4)}
    return out


# ---------------------------------------------------------------------------
# Anatomy track data
# ---------------------------------------------------------------------------


@dataclass
class AnatomyData:
    directory: Path
    mouse: Ontology
    human: Ontology
    reference: list[OAEIAlignment]

    def hashes(self) -> dict[str, str]:
        return {name: file_sha256(self.directory / name) for name in ANATOMY_FILES}


def load_anatomy(directory: str | Path) -> AnatomyData:
    """Parse ``mouse.owl``, ``human.owl`` and ``reference.rdf`` (equivalences only) from ``directory``."""
    d = Path(directory)
    for name in ANATOMY_FILES:
        if not (d / name).exists():
            raise FileNotFoundError(f"{d / name} missing - see tests/fixtures/oaei/anatomy/README.md")
    ref = [a for a in OAEIAlignmentParser.parse(d / "reference.rdf") if a.relation == "="]
    return AnatomyData(d, OWLParser.parse(d / "mouse.owl"), OWLParser.parse(d / "human.owl"), ref)


def anatomy_candidate_scores(encoder: Any, data: AnatomyData, k: int, *, csls_k: int = 10) -> Scores:
    """Sparse top-``k`` CSLS-calibrated embedding scores between mouse and human classes.

    Same text and calibration as the Conference embedding matcher (mean of the label and
    full-description cosines, CSLS with ``csls_k`` neighbours); the lexical matcher is
    skipped here because its all-pairs Python loop is far too slow on 9 M pairs.
    """
    ms = [t for t in build_entity_texts(data.mouse) if t.kind == "class"]
    hs = [t for t in build_entity_texts(data.human) if t.kind == "class"]
    ml = encoder.encode([t.label for t in ms])
    mf = encoder.encode([t.full for t in ms])
    hl = encoder.encode([t.label for t in hs])
    hf = encoder.encode([t.full for t in hs])
    sim = 0.5 * (ml @ hl.T + mf @ hf.T)
    sim = np.clip(csls_calibrate(sim, csls_k), 0.0, 1.0)
    return topk_from_matrix(sim, [t.name for t in ms], [t.name for t in hs], k)


# ---------------------------------------------------------------------------
# Cross-encoder
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    base_model: str = DEFAULT_BASE_MODEL
    revision: str = DEFAULT_BASE_REVISION
    seed: int = 0
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    max_length: int = 96
    time_budget_s: float = 900.0  # hard stop: 15 min of CPU
    threads: int | None = None
    log_every: int = 20


@dataclass
class TrainingStats:
    steps: int = 0
    epochs_completed: float = 0.0
    elapsed_s: float = 0.0
    stopped_by_budget: bool = False
    loss_curve: list[dict[str, float]] = field(default_factory=list)  # {"step", "loss"} every log_every
    epoch_losses: list[float] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)  # per epoch (and at the end)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrossEncoderReranker:
    """``AutoModelForSequenceClassification`` with one logit; ``score`` returns sigmoid probabilities."""

    def __init__(
        self,
        model: str | Path = DEFAULT_BASE_MODEL,
        *,
        revision: str = DEFAULT_BASE_REVISION,
        max_length: int = 96,
        batch_size: int = 64,
        threads: int | None = None,
        seed: int = 0,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        torch.manual_seed(seed)
        if threads:
            torch.set_num_threads(threads)
        self.name = str(model)
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self.seed = seed
        kw = {} if Path(str(model)).exists() else {"revision": revision}
        self.tokenizer = AutoTokenizer.from_pretrained(str(model), **kw)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model), num_labels=1, **kw)
        self.model.eval()
        self.manifest: dict[str, Any] = {}
        mpath = Path(str(model)) / "manifest.json"
        if mpath.exists():
            self.manifest = json.loads(mpath.read_text())

    # -- inference -----------------------------------------------------------
    def _encode(self, batch: list[tuple[str, str]]) -> dict[str, Any]:
        return self.tokenizer(
            [a for a, _ in batch],
            [b for _, b in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def score(self, texts: Sequence[tuple[str, str]]) -> np.ndarray:
        """Probability that each ``(text A, text B)`` pair is a correspondence."""
        torch = self._torch
        out = np.zeros(len(texts), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i : i + self.batch_size])
                logits = self.model(**self._encode(batch)).logits.squeeze(-1)
                out[i : i + len(batch)] = torch.sigmoid(logits).float().numpy()
        return out

    def score_pairs(
        self, pairs: Iterable[tuple[str, str]], source_texts: dict[str, EntityText], target_texts: dict[str, EntityText]
    ) -> Scores:
        pairs = sorted(pairs)
        probs = self.score(pair_texts(pairs, source_texts, target_texts))
        return {p: float(v) for p, v in zip(pairs, probs, strict=True)}

    # -- training ------------------------------------------------------------
    def train(
        self,
        pairs: Sequence[Pair],
        texts: Sequence[tuple[str, str]],
        config: TrainingConfig,
        *,
        validation: tuple[Sequence[Pair], Sequence[tuple[str, str]]] | None = None,
    ) -> TrainingStats:
        """Fine-tune with binary cross-entropy on ``(pair, label)`` examples under a time budget.

        ``texts[i]`` is the cross-encoder input of ``pairs[i]``.  Deterministic for a fixed
        seed and thread count (CPU).  Validation (AUC / hits@1) is run after every epoch
        and once more at the end when the budget stopped an epoch early.
        """
        torch = self._torch
        torch.manual_seed(config.seed)
        rng = random.Random(config.seed)
        self.model.train()
        n = len(pairs)
        steps_per_epoch = max(1, math.ceil(n / config.batch_size))
        total_steps = steps_per_epoch * config.epochs
        warmup = int(config.warmup_fraction * total_steps)
        opt = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        def lr_lambda(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        stats = TrainingStats()
        t0 = time.perf_counter()
        step = 0
        stopped = False
        for epoch in range(config.epochs):
            order = list(range(n))
            rng.shuffle(order)
            epoch_loss, epoch_steps = 0.0, 0
            for b in range(0, n, config.batch_size):
                if time.perf_counter() - t0 > config.time_budget_s:
                    stopped = True
                    break
                idx = order[b : b + config.batch_size]
                enc = self._encode([texts[i] for i in idx])
                y = torch.tensor([float(pairs[i].label) for i in idx])
                logits = self.model(**enc).logits.squeeze(-1)
                loss = loss_fn(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                epoch_steps += 1
                epoch_loss += float(loss.detach())
                if step % config.log_every == 0:
                    stats.loss_curve.append({"step": step, "loss": round(epoch_loss / epoch_steps, 5)})
            if epoch_steps:
                stats.epoch_losses.append(round(epoch_loss / epoch_steps, 5))
            stats.epochs_completed = epoch + (epoch_steps / steps_per_epoch if stopped else 1.0)
            if validation is not None and (not stopped or epoch_steps):
                vp, vt = validation
                rep = validation_report(vp, self.score(vt).tolist())
                rep["epoch"] = round(stats.epochs_completed, 3)
                stats.validation.append(rep)
                self.model.train()
            if stopped:
                break
        stats.steps = step
        stats.elapsed_s = round(time.perf_counter() - t0, 1)
        stats.stopped_by_budget = stopped
        self.model.eval()
        return stats

    def save(self, directory: str | Path, manifest: dict[str, Any]) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(d)
        self.tokenizer.save_pretrained(d)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        self.manifest = manifest
        return d


# ---------------------------------------------------------------------------
# Conference evaluation
# ---------------------------------------------------------------------------


@dataclass
class PairCandidates:
    pair: str
    source: str
    target: str
    base: Scores
    candidates: Scores  # base scores restricted to the candidate set
    ce: Scores  # cross-encoder probabilities of the candidates
    candidate_recall: float  # fraction of reference pairs inside the candidate set (the reranker's ceiling)
    seconds: float


class RerankBenchmark:
    """Rerank a base matcher's candidates on the Conference pairs and score every ``(w, threshold)``."""

    def __init__(self, bench: OAEIConferenceBenchmark, reranker: CrossEncoderReranker):
        self.bench = bench
        self.reranker = reranker
        self._texts: dict[str, dict[str, EntityText]] = {}
        self._cands: dict[tuple[str, str, str, int], PairCandidates] = {}

    def texts(self, name: str) -> dict[str, EntityText]:
        key = name.lower()
        if key not in self._texts:
            self._texts[key] = entity_texts(self.bench.ontology(name))
        return self._texts[key]

    def candidates(self, matcher: Matcher, source: str, target: str, k: int) -> PairCandidates:
        key = (matcher.name, source.lower(), target.lower(), k)
        if key not in self._cands:
            t0 = time.perf_counter()
            base = {(a.entity1, a.entity2): a.measure for a in self.bench.scored_pairs(matcher, source, target)}
            cset = candidate_pairs(base, k)
            cands = {p: base[p] for p in cset}
            ce = self.reranker.score_pairs(cset, self.texts(source), self.texts(target))
            ref = self.bench.reference(source, target)
            inside = sum(1 for a in ref if (a.entity1, a.entity2) in cset)
            self._cands[key] = PairCandidates(
                pair=f"{source}-{target}",
                source=source,
                target=target,
                base=base,
                candidates=cands,
                ce=ce,
                candidate_recall=round(inside / len(ref), 4) if ref else 0.0,
                seconds=round(time.perf_counter() - t0, 2),
            )
        return self._cands[key]

    def evaluate_pair(
        self, matcher: Matcher, source: str, target: str, k: int, weight: float, threshold: float
    ) -> PairEvaluation:
        pc = self.candidates(matcher, source, target, k)
        system = select(fuse_scores(pc.candidates, pc.ce, weight), threshold)
        src, tgt = self.bench.ontology(source), self.bench.ontology(target)
        ref = self.bench.reference(source, target)
        return PairEvaluation(pc.pair, source, target, len(ref), len(system), evaluate_alignment(system, ref, src, tgt))

    def run(
        self, matcher: Matcher, pairs: Iterable[tuple[str, str]], k: int, weight: float, threshold: float
    ) -> dict[str, Any]:
        evals = [self.evaluate_pair(matcher, s, t, k, weight, threshold) for s, t in pairs]
        return {
            "matcher": matcher.name,
            "k": k,
            "weight": weight,
            "threshold": threshold,
            "aggregates": aggregate(evals),
            "pairs": [e.to_dict() for e in evals],
        }

    def grid(
        self,
        matchers: dict[str, Matcher],
        pairs: Iterable[tuple[str, str]],
        ks: Iterable[int],
        weights: Iterable[float],
        thresholds: Iterable[float],
    ) -> GridResults:
        """``{config_label: {pair: {modality: PRF}}}`` for every grid point (reuses the base CV helpers)."""
        pairs = list(pairs)
        out: GridResults = {}
        for mname, matcher in matchers.items():
            for k in ks:
                for w in weights:
                    for th in thresholds:
                        label = config_label(mname, k, w, th)
                        out[label] = {f"{s}-{t}": self.evaluate_pair(matcher, s, t, k, w, th).results for s, t in pairs}
        return out


def micro(grid: GridResults, label: str, pairs: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Micro P/R/F1 per modality of one grid point over ``pairs``."""
    out = {}
    for m in MODALITIES:
        total = PRF()
        for p in pairs:
            total = total + grid[label][p][m]
        out[m] = total.to_dict()
    return out
