"""Ensemble of cached LLM judges on the OAEI Conference track (ra1): combiners x decoding x CV.

Usage::

    .venv/bin/python scripts/benchmark_oaei_llm_ensemble.py --judgments qwen__qwen3.8-max,qwen__qwen3.8-flash
    .venv/bin/python scripts/benchmark_oaei_llm_ensemble.py \
        --judgments qwen__qwen3.8-max,qwen__qwen3.8-flash,qwen__glm-5.3 --combiner majority,unanimous

No language model is called: the verdicts come from the committed judgment caches
``benchmarks/results/llm-judgments/<provider>__<model>.jsonl`` written by
``scripts/benchmark_oaei_llm.py`` (same candidate set, same prompt version).  The first judge
in ``--judgments`` is the *anchor* of the ``anchor`` combiner.

Steps: (1) the same candidate set as the single-judge script (same ``CandidateConfig`` defaults
and encoder), (2) alignment of the caches per candidate pair with the per-judge coverage
(ontology pairs where a judge covers less than ``--min-coverage`` of the candidates are left
out of the evaluation; a missing verdict elsewhere is an abstention), (3) per-judge reference
rows, (4) for every combiner the judge-only quality and the decoding grid
(judge / repair / fused x threshold x fusion weight), scored at the a-priori configuration
(``combiner=majority,mode=repair,th=0.50``, fixed before the grid was run), at the best grid
point on ra1 (optimistic) and by leave-one-pair-out / leave-one-ontology-out cross-validation
over the union grid (the combiner, the mode and the threshold are all chosen on the training
folds; the ``weighted`` combiner's judge weights are estimated on the training folds too).

Writes ``benchmarks/results/oaei-llm-ensemble.json`` / ``.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from olaya.benchmarks.llm_ensemble import (  # noqa: E402
    COMBINERS,
    EnsemblePair,
    GridResults,
    JudgeCache,
    align_verdicts,
    coverage,
    cross_validate_nested,
    ensemble_label,
    ensemble_verdicts,
    judge_weights,
    load_judge_caches,
    single_judge_verdicts,
)
from olaya.benchmarks.llm_judge import (  # noqa: E402
    DECODE_MODES,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    CandidateConfig,
    CandidateGenerator,
    CandidateSet,
    Judgment,
    config_label,
    decode,
    judge_quality,
)
from olaya.benchmarks.oaei import (  # noqa: E402
    CONFERENCE_ONTOLOGIES,
    CONFERENCE_PAIRS,
    MODALITIES,
    PRF,
    OAEIConferenceBenchmark,
    PairEvaluation,
    aggregate,
    evaluate_alignment,
    leave_one_pair_out_folds,
    select_config,
)


def _load_single_judge_script() -> Any:
    """The single-judge benchmark script, for its constants and report helpers (shared, not copied)."""
    spec = importlib.util.spec_from_file_location("benchmark_oaei_llm", REPO / "scripts" / "benchmark_oaei_llm.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_single = _load_single_judge_script()
FIXTURES: Path = _single.FIXTURES
RESULTS_DIR: Path = _single.RESULTS_DIR
JUDGMENTS_DIR: Path = _single.JUDGMENTS_DIR
ENCODER: str = _single.ENCODER
GRID_THRESHOLDS: list[float] = _single.GRID_THRESHOLDS
GRID_WEIGHTS: list[float] = _single.GRID_WEIGHTS
git_commit = _single.git_commit
prf = _single.prf
micro = _single.micro

RESULT_STEM = "oaei-llm-ensemble"

# A-priori configuration, fixed before any ensemble grid was run: majority vote of the judges, precision +
# recall repair of the hybrid alignment at confidence 0.50 (the single-judge a-priori decoding).
APRIORI_COMBINER = "majority"
APRIORI_MODE: str = _single.APRIORI_MODE
APRIORI_THRESHOLD: float = _single.APRIORI_THRESHOLD

#: OAEI 2023 Conference track, ra1, F1 (M1 classes, M3 classes + properties).
PUBLISHED = {
    "GraphMatcher": {"M1": 0.83, "M3": 0.78},
    "SORBETMtch": {"M1": 0.76, "M3": 0.70},
    "LogMap": {"M1": 0.72, "M3": 0.68},
    "Matcha": {"M1": 0.72, "M3": 0.67},
    "edna": {"M1": 0.67, "M3": 0.59},
}
OURS = "olaya (this run)"


def leave_one_ontology_out_folds_for(pairs: Sequence[str]) -> dict[str, tuple[list[str], list[str]]]:
    """LOOO folds restricted to ``pairs`` (folds without train or test pairs are dropped)."""
    folds = {}
    for o in CONFERENCE_ONTOLOGIES:
        test = [p for p in pairs if o in p.split("-")]
        train = [p for p in pairs if p not in test]
        if test and train:
            folds[o] = (train, test)
    return folds


def _verdict_digest(verdicts: dict[str, dict[tuple[str, str], Judgment]]) -> str:
    h = hashlib.sha256()
    for p in sorted(verdicts):
        for pair in sorted(verdicts[p]):
            v = verdicts[p][pair]
            h.update(f"{p}|{pair[0]}|{pair[1]}|{int(v.equivalent)}|{v.confidence:.6f}\n".encode())
    return h.hexdigest()


class Runner:
    """Decoding grid over the per-pair candidates for any verdict mapping (memoised by its content)."""

    def __init__(self, bench: OAEIConferenceBenchmark, per_pair: dict[str, EnsemblePair], pairs: Sequence[str]):
        self.bench = bench
        self.per_pair = per_pair
        self.pairs = list(pairs)
        self._grids: dict[str, GridResults] = {}
        self.decodes = 0

    def evaluate(self, verdicts: dict[str, dict[tuple[str, str], Judgment]], mode: str, th: float, w: float = 0.5):
        evals = []
        for p in self.pairs:
            ep = self.per_pair[p]
            s, t = ep.candidates.source, ep.candidates.target
            system = decode(ep.candidates, verdicts[p], mode=mode, threshold=th, weight=w)
            self.decodes += 1
            evals.append(
                PairEvaluation(
                    p,
                    s,
                    t,
                    len(ep.reference),
                    len(system),
                    evaluate_alignment(system, ep.reference, self.bench.ontology(s), self.bench.ontology(t)),
                )
            )
        return evals

    def grid(self, verdicts: dict[str, dict[tuple[str, str], Judgment]], prefix: str = "") -> GridResults:
        key = prefix + "|" + _verdict_digest(verdicts)
        if key in self._grids:
            return self._grids[key]
        grid: GridResults = {}
        for mode in DECODE_MODES:
            for th in GRID_THRESHOLDS:
                for w in GRID_WEIGHTS if mode == "fused" else [0.5]:
                    label = prefix + config_label(mode, th, w)
                    grid[label] = {e.pair: e.results for e in self.evaluate(verdicts, mode, th, w)}
        self._grids[key] = grid
        return grid


def judge_quality_rows(
    bench: OAEIConferenceBenchmark,
    per_pair: dict[str, EnsemblePair],
    verdicts: dict[str, dict[tuple[str, str], Judgment]],
) -> dict[str, dict[str, Any]]:
    """The judge's own P / R / F1 on the candidate set (FN includes reference pairs outside the set)."""
    jq = {m: PRF() for m in MODALITIES}
    for p, ep in per_pair.items():
        cs = ep.candidates
        src = bench.ontology(cs.source)
        for kind, mods in (("class", ("M1", "M3")), ("property", ("M2", "M3"))):
            sub = CandidateSet(cs.source, cs.target, [c for c in cs.candidates if c.kind == kind])
            counts = judge_quality(sub, verdicts[p], ep.reference)
            ref_kind = [r for r in ep.reference if (r.entity1 in src.concepts) == (kind == "class")]
            counts["fn"] += len([r for r in ref_kind if (r.entity1, r.entity2) not in sub.pairs])
            for m in mods:
                jq[m] = jq[m] + PRF(**counts)
    return {m: jq[m].to_dict() for m in MODALITIES}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--judgments",
        required=True,
        help="comma-separated cache stems under --judgments-dir (provider__model); the first is the anchor",
    )
    ap.add_argument(
        "--combiner",
        default="all",
        help=f"comma-separated subset of {', '.join(COMBINERS)} (default all)",
    )
    ap.add_argument("--anchor", default=None, help="cache stem used as the anchor (default: the first of --judgments)")
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=1.0,
        help="ontology pairs where any judge judged less than this fraction of the candidates are left out (default 1.0)",
    )
    ap.add_argument("--encoder", default=ENCODER, help="sentence encoder for the candidates ('hash' = offline stub)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=0.15)
    ap.add_argument("--rich-k", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=0, help="only the first N ontology pairs (0 = all 21)")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--judgments-dir", type=Path, default=JUDGMENTS_DIR)
    ap.add_argument("--result-stem", default=RESULT_STEM)
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.judgments.split(",") if n.strip()]
    if len(names) < 2:
        print("--judgments needs at least two caches", file=sys.stderr)
        return 2
    combiners = list(COMBINERS) if args.combiner == "all" else [c.strip() for c in args.combiner.split(",")]
    unknown = [c for c in combiners if c not in COMBINERS]
    if unknown:
        print(f"unknown combiner(s) {unknown}; choose from {list(COMBINERS)}", file=sys.stderr)
        return 2
    anchor_name = args.anchor or names[0]
    if anchor_name not in names:
        print(f"--anchor {anchor_name} is not in --judgments", file=sys.stderr)
        return 2
    anchor = names.index(anchor_name)
    try:
        caches: list[JudgeCache] = load_judge_caches(args.judgments_dir, names)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    t_start = time.perf_counter()
    bench = OAEIConferenceBenchmark(FIXTURES / "ontologies", FIXTURES / "reference_alignments")
    pairs = list(CONFERENCE_PAIRS[: args.pairs] if args.pairs else CONFERENCE_PAIRS)
    if args.encoder == "hash":
        from olaya.embeddings import HashEncoder

        encoder: Any = HashEncoder()
    else:
        from olaya.embeddings import load_encoder

        encoder = load_encoder(args.encoder, threads=args.threads)
    gen = CandidateGenerator(encoder, CandidateConfig(k=args.k, min_score=args.min_score, rich_k=args.rich_k))

    # -- 1. candidates (same as the single-judge script) and 2. alignment of the caches ----------
    all_pairs: dict[str, EnsemblePair] = {}
    for s, t in pairs:
        src, tgt = bench.ontology(s), bench.ontology(t)
        cs = gen.build(src, tgt)
        aligned = align_verdicts(caches, cs, src.name, tgt.name)
        name = f"{s}-{t}"
        all_pairs[name] = EnsemblePair(name, cs, aligned, bench.reference(s, t), coverage(aligned, len(caches)))
    t_cands = time.perf_counter() - t_start
    used = [p for p, ep in all_pairs.items() if all(c["fraction"] >= args.min_coverage for c in ep.coverage)]
    dropped = [p for p in all_pairs if p not in used]
    if len(used) < 2:
        print(
            f"only {len(used)} ontology pair(s) meet --min-coverage {args.min_coverage}; "
            f"coverage per pair: {[(p, [c['fraction'] for c in ep.coverage]) for p, ep in all_pairs.items()]}",
            file=sys.stderr,
        )
        return 1
    per_pair = {p: all_pairs[p] for p in used}
    n_cands = sum(len(ep.candidates.candidates) for ep in per_pair.values())
    cov_total = [
        {
            "judge": names[j],
            "model": caches[j].model,
            "records_in_cache": caches[j].records,
            "judged": sum(ep.coverage[j]["judged"] for ep in all_pairs.values()),
            "candidates": sum(len(ep.candidates.candidates) for ep in all_pairs.values()),
            "judged_on_used_pairs": sum(ep.coverage[j]["judged"] for ep in per_pair.values()),
            "candidates_on_used_pairs": n_cands,
        }
        for j in range(len(caches))
    ]
    for c in cov_total:
        c["fraction"] = round(c["judged"] / c["candidates"], 4) if c["candidates"] else 1.0
        c["fraction_on_used_pairs"] = round(c["judged_on_used_pairs"] / n_cands, 4) if n_cands else 1.0
    print(
        f"candidates: {sum(c['candidates'] for c in cov_total[:1])} over {len(all_pairs)} ontology pairs; "
        + "; ".join(f"{c['judge']}: {c['judged']} judged ({c['fraction']:.3f})" for c in cov_total)
        + f"; {len(used)} pairs used, {len(dropped)} dropped by --min-coverage {args.min_coverage}"
    )
    hits = {m: [0, 0] for m in MODALITIES}
    for p, ep in per_pair.items():
        s = p.split("-")[0]
        for e1, e2 in sorted({(r.entity1, r.entity2) for r in ep.reference}):
            kind = "class" if e1 in bench.ontology(s).concepts else "property"
            for m in ("M1",) if kind == "class" else ("M2",):
                hits[m][1] += 1
                hits[m][0] += int((e1, e2) in ep.candidates.pairs)
            hits["M3"][1] += 1
            hits["M3"][0] += int((e1, e2) in ep.candidates.pairs)
    ceiling = {m: round(h / n, 4) if n else 1.0 for m, (h, n) in hits.items()}

    runner = Runner(bench, per_pair, used)
    lopo = leave_one_pair_out_folds(used)
    looo = leave_one_ontology_out_folds_for(used)
    weight_input = {p: (ep.candidates, ep.aligned, ep.reference) for p, ep in per_pair.items()}
    weights_all = judge_weights(weight_input, len(caches), used)

    # -- 3. per-judge reference rows --------------------------------------------------------------
    per_judge: dict[str, Any] = {}
    for j, cache in enumerate(caches):
        verdicts = {p: single_judge_verdicts(ep.aligned, j) for p, ep in per_pair.items()}
        grid = runner.grid(verdicts, prefix=f"judge={cache.short},")
        apriori = aggregate(runner.evaluate(verdicts, APRIORI_MODE, APRIORI_THRESHOLD))
        best = select_config(grid, used)
        per_judge[cache.name] = {
            "model": cache.model,
            "weight_all_pairs": weights_all[j],
            "judge_quality_on_candidates": judge_quality_rows(bench, per_pair, verdicts),
            "apriori": {"label": config_label(APRIORI_MODE, APRIORI_THRESHOLD), "aggregates": apriori},
            "best_on_ra1": {"label": best, **micro(grid, best, used)},
            "lopo_cv": cross_validate_nested(lambda _train, g=grid: g, lopo),
            "looo_cv": cross_validate_nested(lambda _train, g=grid: g, looo),
        }

    # -- 4. combiners: judge-only quality, grid, a-priori, best, CV --------------------------------
    def verdicts_for(combiner: str, train: Sequence[str]) -> dict[str, dict[tuple[str, str], Judgment]]:
        w = judge_weights(weight_input, len(caches), train) if combiner == "weighted" else None
        return {p: ensemble_verdicts(ep.aligned, combiner, weights=w, anchor=anchor) for p, ep in per_pair.items()}

    def grid_for(combiner: str, train: Sequence[str]) -> GridResults:
        return runner.grid(verdicts_for(combiner, train), prefix=f"combiner={combiner},")

    def union_grid(train: Sequence[str]) -> GridResults:
        out: GridResults = {}
        for c in combiners:
            out.update(grid_for(c, train))
        return out

    per_combiner: dict[str, Any] = {}
    for combiner in combiners:
        verdicts = verdicts_for(combiner, used)  # weighted: weights from all used pairs (optimistic rows only)
        grid = grid_for(combiner, used)
        apriori = aggregate(runner.evaluate(verdicts, APRIORI_MODE, APRIORI_THRESHOLD))
        best = select_config(grid, used)
        per_combiner[combiner] = {
            "judge_quality_on_candidates": judge_quality_rows(bench, per_pair, verdicts),
            "apriori": {
                "label": ensemble_label(combiner, config_label(APRIORI_MODE, APRIORI_THRESHOLD)),
                "aggregates": apriori,
            },
            "best_on_ra1": {"label": best, **micro(grid, best, used)},
            "lopo_cv": cross_validate_nested(lambda train, c=combiner: grid_for(c, train), lopo),
            "looo_cv": cross_validate_nested(lambda train, c=combiner: grid_for(c, train), looo),
        }
    apriori_label = ensemble_label(APRIORI_COMBINER, config_label(APRIORI_MODE, APRIORI_THRESHOLD))
    if APRIORI_COMBINER in per_combiner:
        apriori_row = per_combiner[APRIORI_COMBINER]["apriori"]
    else:  # the a-priori combiner was excluded on the command line: compute it anyway, it is the fixed row
        apriori_row = {
            "label": apriori_label,
            "aggregates": aggregate(
                runner.evaluate(verdicts_for(APRIORI_COMBINER, used), APRIORI_MODE, APRIORI_THRESHOLD)
            ),
        }
    full = union_grid(used)
    best_label = select_config(full, used)
    lopo_cv = cross_validate_nested(union_grid, lopo)
    looo_cv = cross_validate_nested(union_grid, looo)
    hybrid_only = aggregate(
        [
            PairEvaluation(
                p,
                ep.candidates.source,
                ep.candidates.target,
                len(ep.reference),
                0,
                evaluate_alignment(
                    gen.accepted(bench.ontology(ep.candidates.source), bench.ontology(ep.candidates.target)),
                    ep.reference,
                    bench.ontology(ep.candidates.source),
                    bench.ontology(ep.candidates.target),
                ),
            )
            for p, ep in per_pair.items()
        ]
    )

    standings = sorted(
        [*((k, v["M1"], v["M3"]) for k, v in PUBLISHED.items()), (OURS, lopo_cv["M1"]["f1"], lopo_cv["M3"]["f1"])],
        key=lambda x: (-x[2], -x[1]),
    )
    result: dict[str, Any] = {
        "benchmark": "OAEI Conference track, reference alignment ra1",
        "category": "LLM-based ensemble (hybrid lexical + embedding candidates, several cached language-model judges)",
        "date_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git_commit(),
        "judges": names,
        "judge_models": [c.model for c in caches],
        "judgment_files": [
            str(c.path.relative_to(REPO)) if c.path and c.path.is_relative_to(REPO) else str(c.path) for c in caches
        ],
        "anchor": anchor_name,
        "combiners": combiners,
        "prompt_version": PROMPT_VERSION,
        "cache_prompt_versions": sorted({c.prompt_version for c in caches}),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "encoder": getattr(encoder, "name", args.encoder),
        "candidate_generator": gen.name,
        "candidate_config": {"k": args.k, "min_score": args.min_score, "rich_k": args.rich_k},
        "pairs_all": list(all_pairs),
        "pairs": used,
        "pairs_dropped": dropped,
        "min_coverage": args.min_coverage,
        "coverage": cov_total,
        "coverage_per_pair": {p: [c["fraction"] for c in ep.coverage] for p, ep in all_pairs.items()},
        "candidates": {"total": n_cands, "recall_ceiling_micro": ceiling},
        "weights_all_pairs": dict(zip(names, weights_all, strict=True)),
        "hybrid_alignment_apriori": hybrid_only,
        "per_judge": per_judge,
        "per_combiner": per_combiner,
        "apriori": apriori_row,
        "best_on_ra1": {"label": best_label, **micro(full, best_label, used)},
        "lopo_cv": lopo_cv,
        "looo_cv": looo_cv,
        "grid_points": len(full),
        "decodes": runner.decodes,
        "published": PUBLISHED,
        "standings": [{"system": k, "M1": m1, "M3": m3} for k, m1, m3 in standings],
        "rank_M3": next(i for i, (k, _, _) in enumerate(standings, 1) if k == OURS),
        "dataset_hashes": bench.dataset_hashes(),
        "timing": {"candidates_s": round(t_cands, 1)},
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / f"{args.result_stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    write_markdown(result, args.results_dir / f"{args.result_stem}.md")
    print(
        f"a-priori {apriori_row['label']}: M1 {prf(apriori_row['aggregates']['M1']['micro'])} "
        f"M3 {prf(apriori_row['aggregates']['M3']['micro'])}; "
        f"LOPO-CV: M1 {prf(lopo_cv['M1'])} M3 {prf(lopo_cv['M3'])}; LOOO-CV: M3 {prf(looo_cv['M3'])}; "
        f"best {best_label}; rank {result['rank_M3']}/{len(standings)} by M3 F1"
    )
    return 0


def _row(label: str, cells: dict[str, Any]) -> str:
    return f"| {label} | " + " | ".join(prf(cells[m]) for m in MODALITIES) + " |"


def write_markdown(res: dict[str, Any], path: Path) -> None:
    cv, looo = res["lopo_cv"], res["looo_cv"]
    judges = ", ".join(f"`{j}` ({m})" for j, m in zip(res["judges"], res["judge_models"], strict=True))
    lines = [
        "# OAEI Conference track (ra1) - ensemble of cached LLM judges",
        "",
        f"Date: {res['date_utc']}  ·  Commit: `{res['commit']}`  ·  Runtime: {res['runtime_s']} s "
        f"(candidates {res['timing']['candidates_s']} s; {res['decodes']} decodings)",
        "",
        f"**Category: LLM-based ensemble.**  Judges: {judges}; anchor `{res['anchor']}`.  **All verdicts are cached**",
        f"(no model was called for this run): prompt version `{res['prompt_version']}` (cache versions "
        f"{', '.join(res['cache_prompt_versions'])}), system prompt sha256 `{res['system_prompt_sha256'][:12]}`, files "
        + ", ".join(f"`{f}`" for f in res["judgment_files"])
        + f".  Candidate generator: `{res['candidate_generator']}` on `{res['encoder']}` (k = {res['candidate_config']['k']}, "
        f"min = {res['candidate_config']['min_score']}, rich_k = {res['candidate_config']['rich_k']}), the same set the "
        "single-judge runs were judged on.",
        "",
        "## Design",
        "",
        "A single judge is recall-strong and precision-weak on the candidate set; the ensemble turns agreement across",
        "judges into precision.  Each judge's cache is read by *pair identity* (source / target ontology and entity:",
        "the record `key` digest encodes the model name and differs between judges), lined up per candidate, and one",
        "combiner turns the row of verdicts into an ordinary `Judgment`, so the single-judge decodings and the",
        "cross-validation run unchanged.  A judge without a verdict for a pair abstains; a pair nobody judged is",
        "`not equivalent, confidence 0` (the `repair` decoding then keeps the hybrid decision).",
        "",
        "Combiners (`olaya.benchmarks.llm_ensemble`):",
        "",
        "- `majority`: equivalent iff a strict majority of the non-abstaining judges say so; confidence = mean of the",
        "  majority side (with two judges this is unanimity).",
        "- `unanimous`: equivalent iff every non-abstaining judge says so; confidence = min.",
        "- `weighted`: weighted mean of each judge's P(equivalent) (confidence if equivalent, else 1 - confidence),",
        "  equivalent iff >= 0.5; each weight is the judge's own M3 F1 on the candidates of the **training** pairs",
        "  (nested inside the cross-validation; the a-priori / best rows below use weights from all pairs).",
        f"- `anchor`: `{res['anchor']}` says equivalent AND at least one other judge agrees; confidence = mean of the",
        "  agreeing judges.",
        "",
        "Decoding and selection are those of `scripts/benchmark_oaei_llm.py`: `judge`, `repair` and `fused` decodings,",
        "threshold 0.30-0.80 (step 0.01), fusion weight in {0.25, 0.5, 0.75}; the headline is cross-validated over the",
        f"union grid of every combiner ({res['grid_points']} points): on each fold the combiner, the mode and the",
        "threshold (and the `weighted` judge weights) are chosen on the training pairs only, and the held-out pair(s)",
        f"are scored under that choice.  The a-priori row `{res['apriori']['label']}` was fixed before the grid was run.",
        "",
        "## Coverage of the caches",
        "",
        f"{res['coverage'][0]['candidates']} candidate pairs over {len(res['pairs_all'])} ontology pairs (ra1 recall ceiling of the set on the "
        f"pairs used: M1 {res['candidates']['recall_ceiling_micro']['M1']:.3f}, M3 {res['candidates']['recall_ceiling_micro']['M3']:.3f}).",
        "",
        "| judge | records in cache | candidates judged | fraction | on the pairs used |",
        "|---|---|---|---|---|",
    ]
    for c in res["coverage"]:
        lines.append(
            f"| `{c['judge']}` | {c['records_in_cache']} | {c['judged']} / {c['candidates']} | {c['fraction']:.3f} | "
            f"{c['judged_on_used_pairs']} / {c['candidates_on_used_pairs']} ({c['fraction_on_used_pairs']:.3f}) |"
        )
    if res["pairs_dropped"]:
        lines += [
            "",
            f"**{len(res['pairs_dropped'])} of {len(res['pairs_all'])} ontology pairs left out** (a judge covers less than "
            f"{res['min_coverage']:.2f} of their candidates): "
            + ", ".join(f"`{p}`" for p in res["pairs_dropped"])
            + ".  "
            f"Every number below is over the {len(res['pairs'])} remaining pairs and is **not** comparable to the full-track "
            "numbers until the caches are complete; rerun when they are.",
        ]
    else:
        lines += ["", f"All {len(res['pairs'])} ontology pairs are covered by every judge."]
    lines += [
        "",
        "## Results (micro P / R / F1; M1 classes, M2 properties, M3 all)",
        "",
        "### Per judge (reference)",
        "",
        "| Run | M1 | M2 | M3 |",
        "|---|---|---|---|",
        _row(
            "hybrid alignment alone (cascade, threshold 0.50)",
            {m: res["hybrid_alignment_apriori"][m]["micro"] for m in MODALITIES},
        ),
    ]
    for j, v in res["per_judge"].items():
        lines += [
            _row(
                f"`{j}` own verdicts on the candidate set (equivalent, confidence >= 0.5)",
                v["judge_quality_on_candidates"],
            ),
            _row(
                f"`{j}` a-priori `{v['apriori']['label']}`",
                {m: v["apriori"]["aggregates"][m]["micro"] for m in MODALITIES},
            ),
            _row(f"`{j}` LOPO-CV over its own grid", v["lopo_cv"]),
            _row(f"`{j}` LOOO-CV over its own grid", v["looo_cv"]),
        ]
    lines += [
        "",
        "### Per combiner",
        "",
        "| Run | M1 | M2 | M3 |",
        "|---|---|---|---|",
    ]
    for c, v in res["per_combiner"].items():
        lines += [
            _row(
                f"`{c}` ensemble verdicts on the candidate set (equivalent, confidence >= 0.5)",
                v["judge_quality_on_candidates"],
            ),
            _row(
                f"`{c}` a-priori `{v['apriori']['label']}`",
                {m: v["apriori"]["aggregates"][m]["micro"] for m in MODALITIES},
            ),
            _row(f"`{c}` best grid point on ra1 (optimistic, `{v['best_on_ra1']['label']}`)", v["best_on_ra1"]),
            _row(f"`{c}` LOPO-CV over its grid", v["lopo_cv"]),
            _row(f"`{c}` LOOO-CV over its grid", v["looo_cv"]),
        ]
    lines += [
        "",
        "### Headline",
        "",
        "| Run | M1 | M2 | M3 |",
        "|---|---|---|---|",
        _row(
            f"a-priori `{res['apriori']['label']}` (fixed before the grid)",
            {m: res["apriori"]["aggregates"][m]["micro"] for m in MODALITIES},
        ),
        _row(f"best grid point on ra1 (optimistic, `{res['best_on_ra1']['label']}`)", res["best_on_ra1"]),
        _row(f"**leave-one-pair-out CV over the union grid ({res['grid_points']} points)**", cv),
        _row(f"leave-one-ontology-out CV over the union grid ({looo['folds']} folds)", looo),
        "",
        f"Configs chosen across the {cv['folds']} LOPO folds: "
        + ", ".join(f"`{x}`" for x in cv["distinct_configs_chosen"])
        + ".",
        f"Configs chosen across the {looo['folds']} LOOO folds: "
        + ", ".join(f"`{x}`" for x in looo["distinct_configs_chosen"])
        + ".",
        "",
        "## Where we land",
        "",
        "Published OAEI 2023 Conference ra1 F1 against this run's LOPO-CV row (ranked by M3 F1, then M1):",
        "",
        "| rank | system | M1 F1 | M3 F1 |",
        "|---|---|---|---|",
    ]
    for i, s in enumerate(res["standings"], 1):
        name = f"**{s['system']}**" if s["system"] == OURS else s["system"]
        lines.append(f"| {i} | {name} | {s['M1']:.2f} | {s['M3']:.2f} |")
    above = [s["system"] for s in res["standings"] if s["M3"] > cv["M3"]["f1"]]
    below = [s["system"] for s in res["standings"] if s["M3"] < cv["M3"]["f1"] and s["system"] != OURS]
    lines += [
        "",
        f"Rank {res['rank_M3']} of {len(res['standings'])} by M3 F1"
        + (f", behind {', '.join(above)}" if above else ", ahead of every published system listed")
        + (f" and ahead of {', '.join(below)}" if below else "")
        + ".  "
        "These are LLM-ensemble numbers: the matcher is in the LLM-based category, every judge was run once over the same",
        "candidate set with the same prompt and its verdicts committed, the judges never saw the reference alignment, and",
        "the combiner / decoding mode / threshold (and the `weighted` weights) are chosen by leave-one-pair-out",
        "cross-validation.  The best-on-ra1 row is optimistic (selected on the test pairs) and is not the claim.  The",
        "published numbers are the OAEI 2023 campaign results; our evaluation uses the same ra1 reference and the same",
        "micro P / R / F1 but the local scorer, so small differences in the evaluation script are possible.",
        "",
    ]
    # Does the ensemble beat its best single judge on the same pairs?  Said in numbers, not left to the reader.
    best_judge = max(res["per_judge"].items(), key=lambda kv: kv[1]["lopo_cv"]["M3"]["f1"])
    bj, bm = best_judge[0], best_judge[1]["lopo_cv"]
    best_comb = max(res["per_combiner"].items(), key=lambda kv: kv[1]["lopo_cv"]["M3"]["f1"])
    bc, bcm = best_comb[0], best_comb[1]["lopo_cv"]
    d_union = cv["M3"]["f1"] - bm["M3"]["f1"]
    d_comb = bcm["M3"]["f1"] - bm["M3"]["f1"]
    verdict = "helps" if d_union > 0.005 else ("is a wash" if abs(d_union) <= 0.005 else "does not help yet")
    lines += [
        "**Ensemble versus the best single judge, same pairs and protocol:** "
        f"`{bj}` alone reaches LOPO-CV M1 {bm['M1']['f1']:.3f} / M3 {bm['M3']['f1']:.3f}; the union-grid ensemble "
        f"reaches M1 {cv['M1']['f1']:.3f} / M3 {cv['M3']['f1']:.3f} ({d_union:+.3f} M3), and the best single combiner "
        f"`{bc}` reaches M1 {bcm['M1']['f1']:.3f} / M3 {bcm['M3']['f1']:.3f} ({d_comb:+.3f} M3).  On this evidence the "
        f"ensemble {verdict}: the agreement step raises the judge-only precision (see the per-combiner table) but a "
        "weaker judge also vetoes true pairs, and a larger union grid costs selection noise in cross-validation."
        + (
            "  With two judges `majority`, `unanimous` and `anchor` are the same rule (both must agree), so their rows coincide."
            if len(res["judges"]) == 2
            else ""
        ),
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
