"""Error analysis and alternative decoders for the OAEI Conference benchmark.

Two things live here:

1. **Error analysis** (:func:`analyse_pair`, :func:`analyse_run`, :func:`categorise`):
   for a matcher and a threshold, every reference correspondence that the system
   did not produce (false negative) and every produced correspondence that is not
   in the reference (false positive) is listed with the entity labels, comments,
   parents / domains / ranges, the matcher's score, the rank of the gold target among
   the candidates of its row and column and the *stage* at which the gold pair was
   lost: ``cascade`` (one entity was consumed by the lexical stage with another
   partner), ``scoring`` (the gold target is not the entity's top-1 candidate),
   ``mutual-best`` (it is the top-1 of its row but another source entity wins the
   column) or ``threshold`` (it is mutual-best but scores below the cut-off).  False
   negatives are then put into lexical categories (synonym / paraphrase, compound or
   head mismatch, abbreviation, property naming, identical label ...).  The
   categorisation is *descriptive* and uses only the two entity labels; it never
   feeds back into the matcher.

2. **Decoders**: the alternatives to the mutual-best + global threshold rule
   (:func:`olaya.benchmarks.oaei.select_hungarian`, :func:`select_mutual_best_margin`,
   :func:`apply_adaptive_threshold`) are re-exported here and measured under
   cross-validation by ``scripts/analyze_oaei_errors.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from olaya.benchmarks.oaei import (
    CONFERENCE_PAIRS,
    HybridMatcher,
    Matcher,
    OAEIAlignment,
    OAEIConferenceBenchmark,
    _display_name,
    apply_adaptive_threshold,
    apply_threshold,
    correspondence_kind,
    hungarian,
    normalize_string,
    select_hungarian,
    select_mutual_best,
    select_mutual_best_margin,
    tokenize_identifier,
)
from olaya.ontology import Ontology

# ---------------------------------------------------------------------------
# Entity descriptions
# ---------------------------------------------------------------------------


@dataclass
class EntityView:
    """What a human needs to see about one entity when judging an error."""

    name: str
    kind: str
    label: str
    comment: str
    parents: list[str]  # parent labels (classes) or "domain -> range" labels (properties)

    def short(self) -> str:
        ctx = ", ".join(self.parents) if self.parents else "-"
        return f"{self.label} [{ctx}]"


def entity_view(ontology: Ontology, name: str) -> EntityView:
    label = _display_name(ontology, name)
    c = ontology.concepts.get(name)
    if c is not None:
        return EntityView(
            name, "class", label, (c.description or "").strip(), [_display_name(ontology, p) for p in sorted(c.parents)]
        )
    p = ontology.properties[name]
    dom = ", ".join(_display_name(ontology, d) for d in sorted(p.domains)) or "?"
    rng = ", ".join(_display_name(ontology, r) for r in sorted(p.ranges)) or str(p.metadata.get("datatype") or "?")
    return EntityView(name, "property", label, (p.description or "").strip(), [f"{dom} -> {rng}"])


# ---------------------------------------------------------------------------
# Error records
# ---------------------------------------------------------------------------

STAGES = ("cascade", "scoring", "mutual-best", "threshold")


@dataclass
class ErrorRecord:
    pair: str
    status: str  # "tp", "fn" or "fp"
    source: EntityView
    target: EntityView
    score: float  # the matcher's score of the pair (0 when not scored)
    lexical: float = 0.0  # lexical-stage score (hybrid only)
    embedding: float = 0.0  # calibrated embedding score (hybrid only)
    rank_row: int | None = None  # rank of the target among the source entity's candidates (1 = best)
    rank_col: int | None = None  # rank of the source among the target entity's candidates
    stage: str = ""  # where the gold pair was lost (fn only)
    row_best: str = ""  # the source entity's best candidate (label) and score
    col_best: str = ""  # the target entity's best candidate (label) and score
    category: str = ""  # lexical category (fn only)
    steals: str = ""  # fp only: which gold correspondences this pair blocks
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.source.kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": self.pair,
            "status": self.status,
            "kind": self.kind,
            "source": self.source.name,
            "target": self.target.name,
            "source_label": self.source.label,
            "target_label": self.target.label,
            "source_comment": self.source.comment,
            "target_comment": self.target.comment,
            "source_parents": self.source.parents,
            "target_parents": self.target.parents,
            "score": round(self.score, 4),
            "lexical": round(self.lexical, 4),
            "embedding": round(self.embedding, 4),
            "rank_row": self.rank_row,
            "rank_col": self.rank_col,
            "stage": self.stage,
            "row_best": self.row_best,
            "col_best": self.col_best,
            "category": self.category,
            "steals": self.steals,
            **self.extra,
        }


def _rank(scores: dict[tuple[str, str], float], axis_entity: str, other: str, *, row: bool) -> int | None:
    """1-based rank of ``other`` among the candidates of ``axis_entity`` (ties: optimistic)."""
    if row:
        cands = {t: v for (s, t), v in scores.items() if s == axis_entity}
    else:
        cands = {s: v for (s, t), v in scores.items() if t == axis_entity}
    if other not in cands:
        return None
    return 1 + sum(1 for v in cands.values() if v > cands[other])


def _best(scores: dict[tuple[str, str], float], axis_entity: str, *, row: bool) -> tuple[str, float] | None:
    if row:
        cands = [(t, v) for (s, t), v in scores.items() if s == axis_entity]
    else:
        cands = [(s, v) for (s, t), v in scores.items() if t == axis_entity]
    if not cands:
        return None
    return max(cands, key=lambda x: (x[1], x[0]))


def analyse_pair(
    bench: OAEIConferenceBenchmark,
    matcher: Matcher,
    source: str,
    target: str,
    threshold: float,
    *,
    selection: str = "mutual",
) -> list[ErrorRecord]:
    """TP / FN / FP records of one ontology pair under the matcher's own decoding."""
    src, tgt = bench.ontology(source), bench.ontology(target)
    pair = f"{source}-{target}"
    scored = bench.scored_pairs(matcher, source, target)
    scores = {(a.entity1, a.entity2): a.measure for a in scored}
    mutual = select_mutual_best(scored) if selection == "mutual" else select_hungarian(scored)
    mutual_keys = {(a.entity1, a.entity2) for a in mutual}
    system = apply_threshold(mutual, threshold)
    sys_keys = {(a.entity1, a.entity2): a.measure for a in system}
    reference = bench.reference(source, target)
    ref_keys = {(a.entity1, a.entity2) for a in reference}
    ref_by_src = {a.entity1: a.entity2 for a in reference}
    ref_by_tgt = {a.entity2: a.entity1 for a in reference}

    lex_scores: dict[tuple[str, str], float] = {}
    emb_scores: dict[tuple[str, str], float] = {}
    taken_src: dict[str, str] = {}
    taken_tgt: dict[str, str] = {}
    if isinstance(matcher, HybridMatcher):
        lex_all = matcher.lexical.score_all(src, tgt)
        lex_scores = {(a.entity1, a.entity2): a.measure for a in lex_all}
        st, tt, sim = matcher.embedding.similarity_matrix(src, tgt)
        for i, s in enumerate(st):
            for j, t in enumerate(tt):
                if s.kind == t.kind:
                    emb_scores[(s.name, t.name)] = float(sim[i, j])
        if matcher.config.combine == "cascade":
            for a in apply_threshold(select_mutual_best(lex_all), matcher.config.cascade_threshold):
                taken_src[a.entity1] = a.entity2
                taken_tgt[a.entity2] = a.entity1

    views: dict[tuple[str, str], EntityView] = {}

    def view(onto: Ontology, name: str, side: str) -> EntityView:
        key = (side, name)
        if key not in views:
            views[key] = entity_view(onto, name)
        return views[key]

    def describe(name: str, side: str, hit: tuple[str, float] | None) -> str:
        if hit is None:
            return "-"
        onto = tgt if side == "row" else src
        return f"{_display_name(onto, hit[0])} ({hit[1]:.2f})"

    out: list[ErrorRecord] = []
    for e1, e2 in sorted(ref_keys):
        if correspondence_kind(OAEIAlignment(e1, e2), src, tgt) == "other":
            continue
        rec = ErrorRecord(
            pair,
            "tp" if (e1, e2) in sys_keys else "fn",
            view(src, e1, "s"),
            view(tgt, e2, "t"),
            scores.get((e1, e2), 0.0),
            lex_scores.get((e1, e2), 0.0),
            emb_scores.get((e1, e2), 0.0),
            _rank(scores, e1, e2, row=True),
            _rank(scores, e2, e1, row=False),
        )
        rb, cb = _best(scores, e1, row=True), _best(scores, e2, row=False)
        rec.row_best, rec.col_best = describe(e1, "row", rb), describe(e2, "col", cb)
        if rec.status == "fn":
            if (e1, e2) not in scores and (e1 in taken_src or e2 in taken_tgt):
                rec.stage = "cascade"
                who = []
                if e1 in taken_src:
                    who.append(f"{rec.source.label} -> {_display_name(tgt, taken_src[e1])}")
                if e2 in taken_tgt:
                    who.append(f"{_display_name(src, taken_tgt[e2])} -> {rec.target.label}")
                rec.extra["consumed_by"] = "; ".join(who)
            elif (e1, e2) in mutual_keys:
                rec.stage = "threshold"
            elif rec.rank_row is not None and rec.rank_row > 1:
                rec.stage = "scoring"
            elif rec.rank_row == 1:
                rec.stage = "mutual-best"
            else:
                rec.stage = "scoring"  # not scored at all (no candidate score)
            rec.category = categorise(rec)
        out.append(rec)
    for (e1, e2), v in sorted(sys_keys.items()):
        if (e1, e2) in ref_keys:
            continue
        rec = ErrorRecord(
            pair,
            "fp",
            view(src, e1, "s"),
            view(tgt, e2, "t"),
            v,
            lex_scores.get((e1, e2), 0.0),
            emb_scores.get((e1, e2), 0.0),
            _rank(scores, e1, e2, row=True),
            _rank(scores, e2, e1, row=False),
        )
        blocked = []
        if e1 in ref_by_src:
            blocked.append(f"{rec.source.label} = {_display_name(tgt, ref_by_src[e1])}")
        if e2 in ref_by_tgt:
            blocked.append(f"{_display_name(src, ref_by_tgt[e2])} = {rec.target.label}")
        rec.steals = "; ".join(blocked) if blocked else "none (both entities unmatched in ra1)"
        rec.category = categorise(rec)
        out.append(rec)
    return out


def analyse_run(
    bench: OAEIConferenceBenchmark,
    matcher: Matcher,
    threshold: float,
    *,
    pairs: Iterable[tuple[str, str]] = CONFERENCE_PAIRS,
    selection: str = "mutual",
) -> list[ErrorRecord]:
    out: list[ErrorRecord] = []
    for s, t in pairs:
        out.extend(analyse_pair(bench, matcher, s, t, threshold, selection=selection))
    return out


# ---------------------------------------------------------------------------
# Lexical categories of an error (labels only, descriptive)
# ---------------------------------------------------------------------------

CATEGORIES = (
    "identical label",
    "inflection / spelling",
    "head noun shared, modifier differs",
    "compound / partial overlap",
    "abbreviation",
    "property naming",
    "synonym / paraphrase",
)


def _head(tokens: Sequence[str]) -> str:
    return tokens[-1] if tokens else ""


def _is_abbreviation(a: str, b: str) -> bool:
    """``a`` is an abbreviation of ``b`` (initials or a subsequence at most 60% as long)."""
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 2 or len(short) > 0.6 * len(long_):
        return False
    it = iter(long_)
    return all(ch in it for ch in short)


def categorise(rec: ErrorRecord) -> str:
    """Descriptive lexical category of an error, from the two labels alone."""
    a, b = rec.source.label, rec.target.label
    na, nb = normalize_string(a), normalize_string(b)
    ta, tb = tokenize_identifier(a), tokenize_identifier(b)
    if na == nb:
        return "identical label"
    if na and nb and (na in nb or nb in na) and min(len(na), len(nb)) >= 0.75 * max(len(na), len(nb)):
        return "inflection / spelling"
    shared = set(ta) & set(tb)
    if rec.kind == "property":
        if shared:
            return "compound / partial overlap"
        return "property naming"
    if shared:
        if _head(ta) == _head(tb):
            return "head noun shared, modifier differs"
        return "compound / partial overlap"
    if _is_abbreviation(na, nb):
        return "abbreviation"
    return "synonym / paraphrase"


def category_counts(records: Iterable[ErrorRecord], *, status: str = "fn") -> dict[str, dict[str, int]]:
    """``{category: {"class": n, "property": n, "all": n}}`` plus a per-stage breakdown."""
    out: dict[str, dict[str, int]] = {}
    for r in records:
        if r.status != status:
            continue
        row = out.setdefault(r.category, {"class": 0, "property": 0, "all": 0, **{s: 0 for s in STAGES}})
        row[r.kind] += 1
        row["all"] += 1
        if r.stage:
            row[r.stage] += 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["all"]))


def stage_counts(records: Iterable[ErrorRecord]) -> dict[str, int]:
    out = dict.fromkeys(STAGES, 0)
    for r in records:
        if r.status == "fn":
            out[r.stage] = out.get(r.stage, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------


__all__ = [
    "CATEGORIES",
    "STAGES",
    "EntityView",
    "ErrorRecord",
    "analyse_pair",
    "analyse_run",
    "apply_adaptive_threshold",
    "categorise",
    "category_counts",
    "entity_view",
    "hungarian",
    "select_hungarian",
    "select_mutual_best_margin",
    "stage_counts",
]
