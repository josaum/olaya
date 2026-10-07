"""Ensemble of cached LLM judges for the OAEI Conference (ra1) judge stage.

A single judge (``olaya.benchmarks.llm_judge``) is recall-strong and precision-weak on the
candidate set (qwen3.8-max, M1: P 0.70 / R 0.87).  Several judges that were run over the *same*
candidate set with the *same* prompt version can be combined per candidate pair, and agreement
across judges is the precision lever: a pair that only one model calls equivalent is more often a
false positive than one every model agrees on.

This module works purely from the committed judgment caches
(``benchmarks/results/llm-judgments/<provider>__<model>.jsonl``); it never calls an API.

* :func:`load_judge_cache` reads one cache keyed by the *pair identity* (source ontology, target
  ontology, source entity, target entity): the ``key`` digest of a record encodes the model name,
  so it differs between judges and cannot be used to align them.
* :func:`align_verdicts` lines the judges up per candidate of a :class:`CandidateSet`; a judge
  with no verdict for a pair *abstains* (``None``).
* The combiners (:data:`COMBINERS`) turn one row of verdicts into a plain
  :class:`~olaya.benchmarks.llm_judge.Judgment`, so :func:`~olaya.benchmarks.llm_judge.decode`,
  :func:`~olaya.benchmarks.llm_judge.judge_quality` and the cross-validation code of
  ``olaya.benchmarks.oaei`` run unchanged on the ensemble verdicts:

  ``majority``
      equivalent iff a strict majority of the non-abstaining judges say so; confidence = mean
      confidence of the majority side (a 1-1 tie is *not* equivalent).
  ``unanimous``
      equivalent iff every non-abstaining judge says so; confidence = min of their confidences
      (when not equivalent: mean confidence of the dissenting judges).
  ``weighted``
      weighted mean of each judge's P(equivalent) (confidence when equivalent, else
      ``1 - confidence``); equivalent iff the mean is >= 0.5.  The weights are the judges' own
      M3 F1 on the candidate set of the *training* pairs (:func:`judge_weights`), so inside
      cross-validation they never see the held-out pair (:func:`cross_validate_nested`).
  ``anchor``
      the first judge (``max``) says equivalent AND at least one other judge agrees; confidence =
      mean confidence of the agreeing judges.

* :func:`cross_validate_nested` is :func:`~olaya.benchmarks.oaei.cross_validate` for a grid
  that may depend on the training pairs (needed by ``weighted``); for a constant grid both give
  the same result.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olaya.benchmarks.llm_judge import PROMPT_VERSION, CandidateSet, Judgment
from olaya.benchmarks.oaei import MODALITIES, PRF, OAEIAlignment, select_config

#: ``(source ontology name, target ontology name, source entity, target entity)`` as the cache stores them.
PairKey = tuple[str, str, str, str]

ABSTAIN_RATIONALE = "ensemble: no verdict"


# ---------------------------------------------------------------------------
# Loading and aligning caches
# ---------------------------------------------------------------------------


@dataclass
class JudgeCache:
    """One judge's cached verdicts keyed by pair identity (not by the model-specific digest)."""

    name: str  # cache stem, e.g. "qwen__qwen3.8-max"
    model: str
    prompt_version: str
    verdicts: dict[PairKey, Judgment]
    path: Path | None = None
    records: int = 0

    @property
    def short(self) -> str:
        return self.name.split("__", 1)[1] if "__" in self.name else self.name

    def get(self, key: PairKey) -> Judgment | None:
        return self.verdicts.get(key)


def pair_key(source_ontology: str, target_ontology: str, source: str, target: str) -> PairKey:
    return (source_ontology, target_ontology, source, target)


def load_judge_cache(path: Path, *, name: str | None = None, prompt_version: str | None = PROMPT_VERSION) -> JudgeCache:
    """Read one JSONL judgment cache.

    Every record must carry the same ``model`` and ``prompt_version``; with ``prompt_version``
    given, records of another prompt version are rejected (verdicts from different prompts must not
    be mixed into one ensemble).  A pair recorded twice keeps the last record.
    """
    path = Path(path)
    models: set[str] = set()
    versions: set[str] = set()
    verdicts: dict[PairKey, Judgment] = {}
    n = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            models.add(str(rec.get("model", "")))
            versions.add(str(rec.get("prompt_version", "")))
            key = pair_key(rec["source_ontology"], rec["target_ontology"], rec["source"], rec["target"])
            verdicts[key] = Judgment.from_dict(rec["verdict"])
    if len(models) > 1:
        raise ValueError(f"{path}: several models in one cache: {sorted(models)}")
    if len(versions) > 1:
        raise ValueError(f"{path}: several prompt versions in one cache: {sorted(versions)}")
    version = next(iter(versions)) if versions else (prompt_version or "")
    if prompt_version is not None and versions and version != prompt_version:
        raise ValueError(f"{path}: prompt version {version!r} != {prompt_version!r}")
    return JudgeCache(
        name=name or path.stem,
        model=next(iter(models)) if models else (name or path.stem),
        prompt_version=version,
        verdicts=verdicts,
        path=path,
        records=n,
    )


def load_judge_caches(
    judgments_dir: Path, names: Iterable[str], *, prompt_version: str | None = PROMPT_VERSION
) -> list[JudgeCache]:
    """``names`` are cache stems (``provider__model``); each must exist and be non-empty."""
    out = []
    for name in names:
        path = Path(judgments_dir) / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"judgment cache {path} not found")
        cache = load_judge_cache(path, name=name, prompt_version=prompt_version)
        if not cache.verdicts:
            raise ValueError(f"judgment cache {path} is empty")
        out.append(cache)
    return out


#: Per candidate pair, one verdict per judge (``None`` = the judge has no verdict: abstain).
AlignedVerdicts = dict[tuple[str, str], list[Judgment | None]]


def align_verdicts(
    caches: Sequence[JudgeCache], cands: CandidateSet, source_ontology: str, target_ontology: str
) -> AlignedVerdicts:
    """One row per candidate of ``cands`` with the verdict of every judge in ``caches`` order.

    ``source_ontology`` / ``target_ontology`` are the ontology *names* as the caches store them
    (``Ontology.name``, e.g. ``Conference``), which can differ in case from the benchmark pair names.
    """
    return {
        c.pair: [cache.get(pair_key(source_ontology, target_ontology, c.source, c.target)) for cache in caches]
        for c in cands.candidates
    }


def coverage(aligned: AlignedVerdicts, n_judges: int) -> list[dict[str, Any]]:
    """Per judge: how many candidates it judged, and the fraction (1.0 = complete)."""
    total = len(aligned)
    out = []
    for j in range(n_judges):
        n = sum(1 for row in aligned.values() if row[j] is not None)
        out.append({"judged": n, "candidates": total, "fraction": round(n / total, 4) if total else 1.0})
    return out


# ---------------------------------------------------------------------------
# Combiners
# ---------------------------------------------------------------------------


def abstain() -> Judgment:
    """The ensemble verdict when no judge has judged a pair: not equivalent, confidence 0.

    With confidence 0 the ``repair`` decoding keeps the hybrid decision (a rejection needs
    confidence >= threshold) and the ``judge`` / ``fused`` decodings score the judge part as 0.
    """
    return Judgment(False, 0.0, "unrelated", ABSTAIN_RATIONALE)


def _present(verdicts: Sequence[Judgment | None]) -> list[Judgment]:
    return [v for v in verdicts if v is not None]


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _relation(votes: Sequence[Judgment], equivalent: bool) -> str:
    if equivalent:
        return "equivalent"
    rels = [v.relation for v in votes if v.relation != "equivalent"]
    if not rels:
        return "unrelated"
    # most common; ties broken by the order of the judges
    return max(dict.fromkeys(rels), key=rels.count)


def _p_equivalent(v: Judgment) -> float:
    c = min(max(v.confidence, 0.0), 1.0)
    return c if v.equivalent else 1.0 - c


def combine_majority(verdicts: Sequence[Judgment | None], **_: Any) -> Judgment:
    present = _present(verdicts)
    if not present:
        return abstain()
    yes = [v for v in present if v.equivalent]
    no = [v for v in present if not v.equivalent]
    equivalent = len(yes) > len(present) / 2
    side = yes if equivalent else no
    conf = _mean([v.confidence for v in side]) if side else 0.0
    return Judgment(
        equivalent,
        round(conf, 6),
        _relation(side, equivalent),
        f"majority {len(yes)}/{len(present)} equivalent",
    )


def combine_unanimous(verdicts: Sequence[Judgment | None], **_: Any) -> Judgment:
    present = _present(verdicts)
    if not present:
        return abstain()
    yes = [v for v in present if v.equivalent]
    if len(yes) == len(present):
        return Judgment(True, round(min(v.confidence for v in yes), 6), "equivalent", "unanimous equivalent")
    no = [v for v in present if not v.equivalent]
    return Judgment(
        False, round(_mean([v.confidence for v in no]), 6), _relation(no, False), f"unanimous: {len(no)} dissent"
    )


def combine_weighted(
    verdicts: Sequence[Judgment | None], *, weights: Sequence[float] | None = None, **_: Any
) -> Judgment:
    pairs = [(v, (1.0 if weights is None else float(weights[i]))) for i, v in enumerate(verdicts) if v is not None]
    pairs = [(v, w) for v, w in pairs if w > 0.0]
    if not pairs:
        return abstain()
    total = sum(w for _, w in pairs)
    p = sum(w * _p_equivalent(v) for v, w in pairs) / total
    equivalent = p >= 0.5
    side = [v for v, _ in pairs if v.equivalent == equivalent]
    return Judgment(
        equivalent, round(p if equivalent else 1.0 - p, 6), _relation(side, equivalent), f"weighted P(eq)={p:.3f}"
    )


def combine_anchor(verdicts: Sequence[Judgment | None], *, anchor: int = 0, **_: Any) -> Judgment:
    """``verdicts[anchor]`` says equivalent and at least one other judge agrees."""
    if anchor >= len(verdicts) or verdicts[anchor] is None:
        return abstain()
    lead = verdicts[anchor]
    others = [v for i, v in enumerate(verdicts) if i != anchor and v is not None]
    if not lead.equivalent:
        return Judgment(False, round(lead.confidence, 6), lead.relation, "anchor: not equivalent")
    agreeing = [v for v in others if v.equivalent]
    if agreeing:
        conf = _mean([lead.confidence, *[v.confidence for v in agreeing]])
        return Judgment(True, round(conf, 6), "equivalent", f"anchor + {len(agreeing)} agree")
    no = [v for v in others if not v.equivalent]
    if not no:  # the anchor alone: nobody can agree
        return abstain()
    return Judgment(False, round(_mean([v.confidence for v in no]), 6), _relation(no, False), "anchor without support")


Combiner = Callable[..., Judgment]

COMBINERS: dict[str, Combiner] = {
    "majority": combine_majority,
    "unanimous": combine_unanimous,
    "weighted": combine_weighted,
    "anchor": combine_anchor,
}


def ensemble_verdicts(
    aligned: AlignedVerdicts,
    combiner: str,
    *,
    weights: Sequence[float] | None = None,
    anchor: int = 0,
) -> dict[tuple[str, str], Judgment]:
    """The :class:`Judgment`-compatible view of the ensemble: ``decode`` / ``judge_quality`` take it as is."""
    fn = COMBINERS[combiner]
    return {pair: fn(row, weights=weights, anchor=anchor) for pair, row in aligned.items()}


def single_judge_verdicts(aligned: AlignedVerdicts, index: int) -> dict[tuple[str, str], Judgment]:
    """One judge alone (abstains where it has no verdict), for the per-judge reference rows."""
    return {pair: (row[index] if row[index] is not None else abstain()) for pair, row in aligned.items()}


# ---------------------------------------------------------------------------
# Weights (judge-only F1 on training pairs) and nested cross-validation
# ---------------------------------------------------------------------------


def judge_counts(
    cands: CandidateSet,
    aligned: AlignedVerdicts,
    index: int,
    reference: Iterable[OAEIAlignment],
    *,
    threshold: float = 0.5,
) -> PRF:
    """TP / FP / FN of judge ``index`` on the candidates it judged (abstentions are skipped)."""
    gold = {(r.entity1, r.entity2) for r in reference}
    prf = PRF()
    for c in cands.candidates:
        v = aligned[c.pair][index]
        if v is None:
            continue
        accept = v.equivalent and v.confidence >= threshold
        if accept and c.pair in gold:
            prf = prf + PRF(1, 0, 0)
        elif accept:
            prf = prf + PRF(0, 1, 0)
        elif c.pair in gold:
            prf = prf + PRF(0, 0, 1)
    return prf


def judge_weights(
    per_pair: Mapping[str, tuple[CandidateSet, AlignedVerdicts, Sequence[OAEIAlignment]]],
    n_judges: int,
    train_pairs: Iterable[str],
    *,
    threshold: float = 0.5,
) -> list[float]:
    """Each judge's micro M3 F1 on the candidates of ``train_pairs`` (its weight in ``weighted``).

    ``per_pair`` maps a pair name to ``(candidates, aligned verdicts, reference)``.  A judge
    without a single verdict on the training pairs gets weight 0 (it abstains everywhere).
    """
    out = []
    for j in range(n_judges):
        total = PRF()
        for p in train_pairs:
            cands, aligned, ref = per_pair[p]
            total = total + judge_counts(cands, aligned, j, ref, threshold=threshold)
        out.append(round(total.f1, 6))
    return out


GridResults = dict[str, dict[str, dict[str, PRF]]]


def cross_validate_nested(
    grid_for: Callable[[Sequence[str]], GridResults],
    folds: Mapping[str, tuple[list[str], list[str]]],
    *,
    modality: str = "M3",
) -> dict[str, Any]:
    """:func:`~olaya.benchmarks.oaei.cross_validate` for a grid that may depend on the training pairs.

    For every fold ``grid_for(train)`` is built from the training pairs only (the ``weighted``
    combiner estimates its judge weights there), the config is selected on the training pairs and
    its test pairs are scored.  Per-fold test counts are summed into micro P / R / F1.
    """
    totals = {m: PRF() for m in MODALITIES}
    chosen: dict[str, str] = {}
    for fold, (train, test) in folds.items():
        grid = grid_for(list(train))
        label = select_config(grid, train, modality)
        chosen[fold] = label
        for p in test:
            for m in MODALITIES:
                totals[m] = totals[m] + grid[label][p][m]
    return {
        "modality_selected": modality,
        "folds": len(folds),
        "chosen": chosen,
        "distinct_configs_chosen": sorted(set(chosen.values())),
        **{m: totals[m].to_dict() for m in MODALITIES},
    }


def ensemble_label(combiner: str, mode_label: str) -> str:
    return f"combiner={combiner},{mode_label}"


@dataclass
class EnsemblePair:
    """Everything the benchmark keeps per ontology pair."""

    pair: str
    candidates: CandidateSet
    aligned: AlignedVerdicts
    reference: list[OAEIAlignment]
    coverage: list[dict[str, Any]] = field(default_factory=list)


__all__ = [
    "ABSTAIN_RATIONALE",
    "COMBINERS",
    "AlignedVerdicts",
    "Combiner",
    "EnsemblePair",
    "GridResults",
    "JudgeCache",
    "PairKey",
    "abstain",
    "align_verdicts",
    "combine_anchor",
    "combine_majority",
    "combine_unanimous",
    "combine_weighted",
    "coverage",
    "cross_validate_nested",
    "ensemble_label",
    "ensemble_verdicts",
    "judge_counts",
    "judge_weights",
    "load_judge_cache",
    "load_judge_caches",
    "pair_key",
    "single_judge_verdicts",
]
