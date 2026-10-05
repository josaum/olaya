"""LLM-judge stage on the OAEI Conference track (ra1): candidates, judging, decoding, CV.

Usage::

    .venv/bin/python scripts/benchmark_oaei_llm.py --judge stub            # offline sanity run
    .venv/bin/python scripts/benchmark_oaei_llm.py --judge anthropic       # needs ANTHROPIC_API_KEY
    .venv/bin/python scripts/benchmark_oaei_llm.py --judge anthropic --model claude-sonnet-5-5

Writes ``benchmarks/results/oaei-llm.json`` / ``.md`` and the judgment cache
``benchmarks/results/llm-judgments/<model>.jsonl`` (committed; a rerun only sends the
pairs that are not cached, so the benchmark is reproducible offline once judged).

Steps: (1) candidate generation from the hybrid matcher and its reference-recall
ceiling, (2) cost estimate per model, (3) judging (cached), (4) decoding grid
(judge-only / repair / fused x threshold) scored at the a-priori configuration, at the
best grid point (optimistic) and under leave-one-pair-out cross-validation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from olaya.benchmarks.llm_judge import (  # noqa: E402
    DECODE_MODES,
    DEFAULT_MODEL,
    PROMPT_VERSION,
    SUPPORTED_MODELS,
    SYSTEM_PROMPT,
    AnthropicJudge,
    CandidateConfig,
    CandidateGenerator,
    JudgeRequest,
    JudgmentCache,
    NullJudge,
    StubJudge,
    Usage,
    batched,
    config_label,
    cost_estimate,
    decode,
    entity_cards,
    has_api_key,
    judge_candidates,
    judge_quality,
)
from olaya.benchmarks.oaei import (  # noqa: E402
    CONFERENCE_PAIRS,
    MODALITIES,
    PRF,
    OAEIConferenceBenchmark,
    PairEvaluation,
    aggregate,
    cross_validate,
    evaluate_alignment,
    leave_one_pair_out_folds,
    select_config,
)

FIXTURES = REPO / "tests" / "fixtures" / "oaei"
RESULTS_DIR = REPO / "benchmarks" / "results"
JUDGMENTS_DIR = RESULTS_DIR / "llm-judgments"
RESULT_STEM = "oaei-llm"
ENCODER = "intfloat/e5-small-v2"

# A-priori decoding configuration, fixed before any grid was run: precision + recall repair of the
# hybrid alignment at the judge confidence 0.50 (the same threshold the other benchmarks use).
APRIORI_MODE = "repair"
APRIORI_THRESHOLD = 0.5
GRID_THRESHOLDS = [round(0.30 + i / 100, 2) for i in range(51)]
GRID_WEIGHTS = [0.25, 0.5, 0.75]
PUBLISHED = {"GraphMatcher": {"M1": 0.83, "M3": 0.78}, "SORBETMtch": {"M1": 0.76, "M3": 0.70}}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def prf(d: dict[str, Any]) -> str:
    return f"{d['precision']:.2f} / {d['recall']:.2f} / **{d['f1']:.2f}**"


def micro(grid: dict[str, dict[str, dict[str, PRF]]], label: str, pairs: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for m in MODALITIES:
        total = PRF()
        for p in pairs:
            total = total + grid[label][p][m]
        out[m] = total.to_dict()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", choices=("stub", "anthropic", "null"), default="stub")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"judge model id, one of {', '.join(SUPPORTED_MODELS)}")
    ap.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--encoder", default=ENCODER, help="sentence encoder for the candidates ('hash' = offline stub)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=0.15)
    ap.add_argument("--rich-k", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--stub-noise", type=float, default=0.0)
    ap.add_argument("--pairs", type=int, default=0, help="only the first N ontology pairs (0 = all 21)")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--judgments-dir", type=Path, default=JUDGMENTS_DIR)
    ap.add_argument("--no-cache", action="store_true", help="do not read or write the judgment cache")
    ap.add_argument(
        "--cache-stub",
        action="store_true",
        help="also write the stub / null judge's verdicts to the judgment cache (off by default: they are derived from ra1)",
    )
    ap.add_argument("--dry-run", action="store_true", help="candidates and cost estimate only, no judging")
    args = ap.parse_args()

    if args.judge == "anthropic" and args.model not in SUPPORTED_MODELS:
        print(f"warning: {args.model} is not in {SUPPORTED_MODELS}; prices will be reported as 0", file=sys.stderr)
    if args.judge == "anthropic" and not has_api_key():
        print("ANTHROPIC_API_KEY is not set; use --judge stub for the offline run", file=sys.stderr)
        return 2

    t_start = time.perf_counter()
    bench = OAEIConferenceBenchmark(FIXTURES / "ontologies", FIXTURES / "reference_alignments")
    pairs = list(CONFERENCE_PAIRS[: args.pairs] if args.pairs else CONFERENCE_PAIRS)
    pair_names = [f"{s}-{t}" for s, t in pairs]

    if args.encoder == "hash":
        from olaya.embeddings import HashEncoder

        encoder: Any = HashEncoder()
    else:
        from olaya.embeddings import load_encoder

        encoder = load_encoder(args.encoder, threads=args.threads)
    gen = CandidateGenerator(encoder, CandidateConfig(k=args.k, min_score=args.min_score, rich_k=args.rich_k))

    # -- 1. candidates and their recall ceiling ---------------------------------------
    cards = {name: entity_cards(bench.ontology(name)) for name in sorted({x for p in pairs for x in p})}
    cand_sets = {}
    recall_rows = {}
    hits = {m: [0, 0] for m in MODALITIES}
    n_cands = 0
    n_by_origin: dict[str, int] = {}
    for s, t in pairs:
        cs = gen.build(bench.ontology(s), bench.ontology(t))
        cand_sets[(s, t)] = cs
        ref = bench.reference(s, t)
        recall_rows[f"{s}-{t}"] = {
            "candidates": len(cs.candidates),
            **{m: round(v, 4) for m, v in cs.recall(ref).items()},
        }
        n_cands += len(cs.candidates)
        for c in cs.candidates:
            for o in c.origin:
                n_by_origin[o] = n_by_origin.get(o, 0) + 1
        for e1, e2 in sorted({(r.entity1, r.entity2) for r in ref}):
            kind = "class" if e1 in bench.ontology(s).concepts else "property"
            for m in ("M1",) if kind == "class" else ("M2",):
                hits[m][1] += 1
                hits[m][0] += int((e1, e2) in cs.pairs)
            hits["M3"][1] += 1
            hits["M3"][0] += int((e1, e2) in cs.pairs)
    ceiling = {m: round(h / n, 4) if n else 1.0 for m, (h, n) in hits.items()}
    t_cands = time.perf_counter() - t_start
    print(f"candidates: {n_cands} pairs over {len(pairs)} ontology pairs; reference recall ceiling {ceiling}")

    # -- 2. cost estimate ------------------------------------------------------------
    all_requests = {
        (s, t): [JudgeRequest(cards[s][c.source], cards[t][c.target]) for c in cs.candidates]
        for (s, t), cs in cand_sets.items()
    }
    batches = [b for reqs in all_requests.values() for b in batched(reqs, args.batch_size)]
    counted = None
    if args.judge == "anthropic":
        try:
            probe = AnthropicJudge(args.model, effort=args.effort)
            counted = [probe.count_tokens(b) for b in batches]
        except Exception as exc:  # noqa: BLE001
            print(f"count_tokens failed ({exc}); falling back to the chars/4 estimate", file=sys.stderr)
    cost = cost_estimate(batches, counted_input=counted)
    print("cost estimate:", {m: v["usd"] for m, v in cost.items()})

    result: dict[str, Any] = {
        "benchmark": "OAEI Conference track, reference alignment ra1",
        "category": "LLM-based matcher (hybrid lexical + embedding candidates, language-model judge)",
        "date_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git_commit(),
        "judge": args.judge,
        "model": args.model if args.judge == "anthropic" else args.judge,
        "effort": args.effort if args.judge == "anthropic" else None,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": __import__("hashlib").sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "encoder": getattr(encoder, "name", args.encoder),
        "candidate_generator": gen.name,
        "candidate_config": {"k": args.k, "min_score": args.min_score, "rich_k": args.rich_k},
        "pairs": pair_names,
        "candidates": {
            "total": n_cands,
            "by_origin": dict(sorted(n_by_origin.items())),
            "recall_ceiling_micro": ceiling,
            "per_pair": recall_rows,
        },
        "cost_estimate": cost,
        "batch_size": args.batch_size,
        "requests": len(batches),
        "dataset_hashes": bench.dataset_hashes(),
    }
    if args.dry_run:
        result["runtime_s"] = round(time.perf_counter() - t_start, 1)
        (args.results_dir / f"{RESULT_STEM}.json").write_text(json.dumps(result, indent=2) + "\n")
        print("dry run: wrote candidates and cost estimate")
        return 0

    # -- 3. judging (cached) ---------------------------------------------------------
    if args.judge == "stub":
        # keyed by the ontology *names* (the OWL file stems, e.g. "Conference"), which the cards carry
        reference = {
            (bench.ontology(s).name, bench.ontology(t).name): {(r.entity1, r.entity2) for r in bench.reference(s, t)}
            for s, t in pairs
        }
        judge: Any = StubJudge(reference, noise=args.stub_noise)
    elif args.judge == "null":
        judge = NullJudge()
    else:
        judge = AnthropicJudge(args.model, effort=args.effort)
    cache_name = judge.model if args.judge == "anthropic" else args.judge
    use_cache = not args.no_cache and (args.judge == "anthropic" or args.cache_stub)
    cache = JudgmentCache(args.judgments_dir / f"{cache_name}.jsonl") if use_cache else None
    cached_before = len(cache) if cache is not None else 0
    ledger: list[Usage] = []
    verdicts = {}
    t0 = time.perf_counter()
    for s, t in pairs:
        verdicts[(s, t)] = judge_candidates(
            cand_sets[(s, t)], cards[s], cards[t], judge, cache, batch_size=args.batch_size, ledger=ledger
        )
    t_judge = time.perf_counter() - t0
    if cache is not None and ledger:
        cache.rewrite()
    sent = sum(u.n_pairs for u in ledger)
    print(f"judged {n_cands} candidates ({sent} sent, {n_cands - sent} from cache) in {t_judge:.1f} s")

    # -- 4. judge quality and decoding grid -------------------------------------------
    jq = {m: PRF() for m in MODALITIES}
    for s, t in pairs:
        cs = cand_sets[(s, t)]
        src = bench.ontology(s)
        for kind, mods in (("class", ("M1", "M3")), ("property", ("M2", "M3"))):
            sub = type(cs)(cs.source, cs.target, [c for c in cs.candidates if c.kind == kind])
            counts = judge_quality(sub, verdicts[(s, t)], bench.reference(s, t))
            ref_kind = [r for r in bench.reference(s, t) if (r.entity1 in src.concepts) == (kind == "class")]
            # reference pairs outside the candidate set are misses of the judge stage as a whole
            counts["fn"] += len([r for r in ref_kind if (r.entity1, r.entity2) not in sub.pairs])
            for m in mods:
                jq[m] = jq[m] + PRF(**counts)
    judge_metrics = {m: jq[m].to_dict() for m in MODALITIES}

    def run(mode: str, th: float, w: float = 0.5) -> list[PairEvaluation]:
        evals = []
        for s, t in pairs:
            system = decode(cand_sets[(s, t)], verdicts[(s, t)], mode=mode, threshold=th, weight=w)
            ref = bench.reference(s, t)
            evals.append(
                PairEvaluation(
                    f"{s}-{t}",
                    s,
                    t,
                    len(ref),
                    len(system),
                    evaluate_alignment(system, ref, bench.ontology(s), bench.ontology(t)),
                )
            )
        return evals

    grid: dict[str, dict[str, dict[str, PRF]]] = {}
    for mode in DECODE_MODES:
        for th in GRID_THRESHOLDS:
            for w in GRID_WEIGHTS if mode == "fused" else [0.5]:
                label = config_label(mode, th, w)
                grid[label] = {e.pair: e.results for e in run(mode, th, w)}
    apriori_evals = run(APRIORI_MODE, APRIORI_THRESHOLD)
    apriori = aggregate(apriori_evals)
    best = select_config(grid, pair_names)
    folds = leave_one_pair_out_folds(pair_names)
    cv = cross_validate(grid, folds)
    cv_by_mode = {
        mode: cross_validate({lab: g for lab, g in grid.items() if lab.startswith(f"mode={mode},")}, folds)
        for mode in DECODE_MODES
    }
    hybrid_only = aggregate(
        [
            PairEvaluation(
                f"{s}-{t}",
                s,
                t,
                len(bench.reference(s, t)),
                0,
                evaluate_alignment(
                    gen.accepted(bench.ontology(s), bench.ontology(t)),
                    bench.reference(s, t),
                    bench.ontology(s),
                    bench.ontology(t),
                ),
            )
            for s, t in pairs
        ]
    )

    result.update(
        {
            "judge_name": judge.name,
            "judgments_file": None
            if cache is None
            else (str(cache.path.relative_to(REPO)) if cache.path.is_relative_to(REPO) else str(cache.path)),
            "judgments_cached_before": cached_before,
            "judgments_sent": sent,
            "usage_ledger": [u.__dict__ for u in ledger],
            "usage_total": {
                "input_tokens": sum(u.input_tokens for u in ledger),
                "output_tokens": sum(u.output_tokens for u in ledger),
                "cache_read_input_tokens": sum(u.cache_read_input_tokens for u in ledger),
                "cache_creation_input_tokens": sum(u.cache_creation_input_tokens for u in ledger),
                "usd": round(sum(u.cost_usd() for u in ledger), 4),
            },
            "judge_quality_on_candidates": judge_metrics,
            "hybrid_alignment_apriori": hybrid_only,
            "apriori": {"label": config_label(APRIORI_MODE, APRIORI_THRESHOLD), "aggregates": apriori},
            "best_on_ra1": {"label": best, **micro(grid, best, pair_names)},
            "lopo_cv": cv,
            "lopo_cv_by_mode": cv_by_mode,
            "grid_points": len(grid),
            "published": PUBLISHED,
            "timing": {"candidates_s": round(t_cands, 1), "judge_s": round(t_judge, 1)},
            "runtime_s": round(time.perf_counter() - t_start, 1),
        }
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / f"{RESULT_STEM}.json").write_text(json.dumps(result, indent=2) + "\n")
    write_markdown(result, args.results_dir / f"{RESULT_STEM}.md")
    print(
        f"a-priori {result['apriori']['label']}: M3 {prf(apriori['M3']['micro'])}; "
        f"LOPO-CV: M1 {prf(cv['M1'])} M3 {prf(cv['M3'])}; best {best}"
    )
    return 0


def write_markdown(res: dict[str, Any], path: Path) -> None:
    c = res["candidates"]
    real = res["judge"] == "anthropic"
    cv = res["lopo_cv"]
    lines = [
        "# OAEI Conference track (ra1) - LLM judge stage",
        "",
        f"Date: {res['date_utc']}  ·  Commit: `{res['commit']}`  ·  Runtime: {res['runtime_s']} s "
        f"(candidates {res['timing']['candidates_s']} s, judging {res['timing']['judge_s']} s)",
        "",
        f"**Category: LLM-based matcher.**  Judge: `{res['judge_name']}` (model `{res['model']}`, prompt "
        f"`{res['prompt_version']}`, system prompt sha256 `{res['system_prompt_sha256'][:12]}`), judgments in "
        f"`{res['judgments_file']}`.  Candidate generator: `{res['candidate_generator']}` on `{res['encoder']}`.",
        "",
    ]
    if not real:
        lines += [
            "> **No language model was used for this run.**  The judge is the `"
            + res["judge_name"]
            + "` stand-in, which answers from the reference alignment; the numbers below validate the"
            " candidate ceiling, the cache and the decoding, not a matcher.  Real-judge numbers are pending"
            " an `ANTHROPIC_API_KEY` (`scripts/benchmark_oaei_llm.py --judge anthropic`).",
            "",
        ]
    lines += [
        "## Design",
        "",
        "The hybrid matcher misses 112 of the 305 ra1 correspondences (`benchmarks/results/oaei-error-analysis.md`):",
        "49 are mutual-best pairs scoring 0.30-0.49 (under the threshold), 39 are synonyms the encoder scores near",
        "zero (Paper / Contribution, Event / Activity, Attendee / Delegate), 15 are entities the lexical stage",
        "consumed with the wrong partner; and 34 of its 80 false positives are identical-label pairs ra1 rejects.",
        "All of these are decisions a reader of the two ontologies makes easily, so a language-model *judge* is",
        "asked, with the ontology context of both entities, whether each candidate pair is equivalent.",
        "",
        "1. **Candidates** (`olaya.benchmarks.llm_judge.CandidateGenerator`): the a-priori hybrid alignment",
        "   (cascade, mutual-best, threshold 0.50; to be *verified*) plus, per entity, the top-k same-kind pairs by",
        "   the dense hybrid score (`0.5 lexical + 0.5 CSLS cosine`) above a minimum, the top-k by the path-text",
        "   CSLS cosine, and the top-`rich_k` by the *raw* cosine of the rich text (the CSLS correction zeroes the",
        "   synonym pairs because the encoder puts them in the hub region; the raw cosine keeps them near the top).",
        f"   Config: k = {res['candidate_config']['k']}, min = {res['candidate_config']['min_score']}, "
        f"rich_k = {res['candidate_config']['rich_k']}.",
        "2. **Prompt**: both entities with label (camelCase split), kind, path from the root, parents / children /",
        "   siblings, properties with domain -> range (classes), domain / range (properties), comment and ontology",
        f"   name; {res['batch_size']} pairs per request, deterministic order; structured JSON verdicts",
        "   (`equivalent`, `confidence`, `relation` in {equivalent, subsumes, subsumed_by, related, unrelated},",
        "   `rationale`) through `output_config.format`; system prompt cached (`cache_control`).  Model",
        "   `claude-opus-5-5` by default (`--model claude-sonnet-5-5` / `claude-haiku-4-5`), adaptive thinking,",
        "   `output_config.effort` = `--effort` (default medium), `max_retries=5`, 300 s timeout, key from the",
        "   environment only.",
        "3. **Cache**: one JSON line per verdict keyed by `sha256(model, prompt version, rendered pair)` under",
        "   `benchmarks/results/llm-judgments/<model>.jsonl`; a rerun sends only unjudged pairs; the per-request",
        "   token usage is kept in the results JSON (`usage_ledger`).",
        "4. **Decoding**: `judge` (judge score = confidence when equivalent, else 0; mutual-best + threshold),",
        "   `repair` (hybrid alignment minus judge-rejected pairs, plus judge-accepted candidates; mutual-best) and",
        "   `fused` (`(1 - w) hybrid + w judge`; mutual-best + threshold).  Threshold 0.30-0.80 (step 0.01) and",
        "   w in {0.25, 0.5, 0.75}; the headline is leave-one-pair-out cross-validated over the grid; the a-priori",
        f"   row is `{res['apriori']['label']}`, fixed before the grid was run.",
        "",
        "## Candidate recall ceiling (no judge involved)",
        "",
        f"{c['total']} candidate pairs over {len(res['pairs'])} ontology pairs "
        f"({', '.join(f'{k}: {v}' for k, v in c['by_origin'].items())}; a pair can have several origins).  "
        "Fraction of ra1 correspondences inside the candidate set (micro): "
        f"M1 **{c['recall_ceiling_micro']['M1']:.3f}**, M2 **{c['recall_ceiling_micro']['M2']:.3f}**, "
        f"M3 **{c['recall_ceiling_micro']['M3']:.3f}** (target >= 0.95).  No decoding can exceed this recall.",
        "",
        "How the sources were chosen (measured on the 21 pairs with `intfloat/e5-small-v2` before any judging; the",
        "candidate count is the cost driver): hybrid top-3 >= 0.15 plus path top-3 alone reaches M3 recall 0.898",
        "with 2,489 candidates and does not improve with k = 5 (the synonym pairs score under the minimum, not",
        "below the rank cut); lowering the minimum to 0.05 with k = 10 reaches 0.954 at 9,385 candidates; adding",
        "the raw rich-text cosine top-1 / top-2 / top-3 per entity reaches 0.938 / 0.961 / 0.971 at 5,845 / 9,560 /",
        "13,311 candidates.  `rich_k = 2` is the default.  The 12 reference pairs still outside the set at that",
        "setting are mostly property paraphrases (`hasBeenAssigned` / `reviewes`, `assignedTo` / `isReviewedBy`).",
        "",
        "| pair | candidates | M1 | M2 | M3 |",
        "|---|---|---|---|---|",
    ]
    for p, r in c["per_pair"].items():
        lines.append(f"| {p} | {r['candidates']} | {r['M1']:.3f} | {r['M2']:.3f} | {r['M3']:.3f} |")
    ce = res["cost_estimate"]
    first = next(iter(ce.values()))
    lines += [
        "",
        "## Expected cost of judging every candidate once",
        "",
        f"{first['requests']} requests of {res['batch_size']} pairs; input tokens {first['input_tokens']:,} "
        f"({first['method']}), output assumed {first['output_tokens_assumed']:,} (60 tokens per verdict).  "
        "Prices: Anthropic first-party rates per million tokens, no prompt-cache discount applied.",
        "",
        "| model | input $/M | output $/M | estimated USD |",
        "|---|---|---|---|",
    ]
    from olaya.benchmarks.llm_judge import MODEL_PRICES_USD_PER_MTOK

    for model, v in ce.items():
        pin, pout = MODEL_PRICES_USD_PER_MTOK[model]
        lines.append(f"| `{model}` | {pin:.2f} | {pout:.2f} | {v['usd']:.2f} |")
    ut = res["usage_total"]
    lines += [
        "",
        f"This run: {res['judgments_sent']} verdicts requested, {res['judgments_cached_before']} already cached; "
        f"actual usage {ut['input_tokens']:,} input / {ut['output_tokens']:,} output tokens "
        f"({ut['cache_read_input_tokens']:,} cache reads), USD {ut['usd']:.4f}.",
        "",
        "## Results (micro P / R / F1; M1 classes, M2 properties, M3 all)",
        "",
        "| Run | M1 | M2 | M3 |",
        "|---|---|---|---|",
        "| hybrid alignment alone (cascade, threshold 0.50; the candidates' `accepted` set) | "
        + " | ".join(prf(res["hybrid_alignment_apriori"][m]["micro"]) for m in MODALITIES)
        + " |",
        "| judge's own verdicts on the candidate set (equivalent, confidence >= 0.5; FN includes pairs outside the set) | "
        + " | ".join(prf(res["judge_quality_on_candidates"][m]) for m in MODALITIES)
        + " |",
        f"| a-priori decoding `{res['apriori']['label']}` | "
        + " | ".join(prf(res["apriori"]["aggregates"][m]["micro"]) for m in MODALITIES)
        + " |",
        f"| best grid point on ra1 (optimistic, `{res['best_on_ra1']['label']}`) | "
        + " | ".join(prf(res["best_on_ra1"][m]) for m in MODALITIES)
        + " |",
        f"| **leave-one-pair-out CV over the decoding grid ({res['grid_points']} points)** | "
        + " | ".join(prf(cv[m]) for m in MODALITIES)
        + " |",
    ]
    for mode, v in res["lopo_cv_by_mode"].items():
        lines.append(
            f"| leave-one-pair-out CV, `{mode}` decoding only | " + " | ".join(prf(v[m]) for m in MODALITIES) + " |"
        )
    lines += [
        "",
        f"Configs chosen across the {cv['folds']} folds: "
        + ", ".join(f"`{x}`" for x in cv["distinct_configs_chosen"])
        + ".",
        "",
        "Published (OAEI 2023, ra1 F1): "
        + ", ".join(f"{k} M1 {v['M1']:.2f} / M3 {v['M3']:.2f}" for k, v in res["published"].items())
        + ".",
        "",
    ]
    if real:
        lines += [
            "## Reading",
            "",
            "These are LLM-judge numbers: the matcher is in the LLM-based category (as OLaLa 2023 and the other",
            "language-model matchers of the Conference track), the model has no access to the reference alignment,",
            "the verdicts are committed so the run is reproducible offline, and the threshold / fusion knobs are",
            "chosen by leave-one-pair-out cross-validation.  The candidate recall ceiling bounds the recall.",
            "",
        ]
    else:
        lines += [
            "## Reading (stub sanity check)",
            "",
            "With a perfect stub the `judge` decoding reproduces ra1 up to the candidate ceiling and the 1:1",
            "selection (a reference entity that appears in two correspondences keeps one), the `repair` decoding",
            "removes every hybrid false positive and adds every reference pair in the candidate set, and the",
            "grid / CV machinery picks a config that does the same.  That proves the plumbing (candidates ->",
            "prompt -> cache -> decoding -> evaluation), not matching quality: **real-judge numbers are pending",
            "an API key.**  The stub's verdicts are not written to the judgment cache (they are ra1 itself);",
            "`--cache-stub` writes them for cache round-trip checks.",
            "",
        ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
