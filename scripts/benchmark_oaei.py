"""Reproduce the OAEI Conference track (ra1) evaluation from the vendored fixtures.

Usage (no arguments needed)::

    .venv/bin/python scripts/benchmark_oaei.py

Writes ``benchmarks/results/oaei-conference-ra1.json`` and
``benchmarks/results/oaei-conference-ra1.md``.  The protocol is the standard
OAEI one: every ontology pair is aligned by a matcher (mutual-best 1:1 selection
plus a confidence threshold), the alignment is scored against the ra1 reference
alignment with precision / recall / F-measure for classes (M1), properties (M2)
and both (M3), and the 21 pairs are aggregated micro (absolute TP/FP/FN, the
OAEI convention) and macro (mean over pairs).

The default run evaluates the lexical matcher, then (unless ``--no-embeddings``)
the embedding and hybrid matchers with the small local encoders listed in
``ENCODERS`` (downloaded from the Hugging Face Hub on first use, cached on disk
afterwards).  For the primary encoder a configuration grid is swept and the
honest headline is obtained by leave-one-pair-out / leave-one-ontology-out
selection, so no pair ever selects its own weights or threshold.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from olaya.benchmarks.oaei import (  # noqa: E402
    MODALITIES,
    BenchmarkRun,
    EmbeddingMatcher,
    EmbeddingMatcherConfig,
    HybridMatcher,
    HybridMatcherConfig,
    LexicalMatcher,
    LexicalMatcherConfig,
    Matcher,
    OAEIConferenceBenchmark,
    StringEquivMatcher,
    cross_validate,
    grid_results,
    leave_one_ontology_out_folds,
    leave_one_pair_out_folds,
    select_config,
)
from olaya.embeddings import EncoderUnavailable, load_encoder  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "oaei"
RESULTS_DIR = REPO / "benchmarks" / "results"
RESULT_STEM = "oaei-conference-ra1"

# A priori default: the lexical score is a weighted Jaccard, so 0.5 ("the shared
# evidence outweighs the unshared evidence") is the natural operating point.  It was
# fixed before the threshold sweep was run.
DEFAULT_THRESHOLD = 0.5
SWEEP_THRESHOLDS = [round(i / 100, 2) for i in range(0, 101)]

# Encoders evaluated by default (all small enough for a CPU).  The first one is the
# primary encoder: it gets the full configuration grid and the cross-validation.
# It was chosen on ra1 among the three (and two base-size models, see the report),
# which is itself a (mild) test-set selection and is stated as such in the report.
ENCODERS = ["intfloat/e5-small-v2", "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]

# A-priori hybrid configuration, fixed before the grid below was swept:
#   * cascade: the lexical matcher decides first (threshold 0.5, same rule as above) and
#     the embedding only scores the entities it left unmatched;
#   * entity text = label and full description, averaged ("both");
#   * CSLS-calibrated cosine (k=10), so 0.5 again means "halfway between the entity's
#     usual neighbourhood and a perfect match";
#   * no structural propagation (alpha = 0) - a knob to be *tuned*, hence not a-priori.
APRIORI_HYBRID = HybridMatcherConfig(
    combine="cascade",
    structure_weight=0.0,
    embedding=EmbeddingMatcherConfig(text="both", calibration="csls", csls_k=10),
)

# Configuration grid for the primary encoder.  "lexical" rows are the lexical matcher
# alone (with propagation), so the cross-validation may legitimately select "no
# embedding" when that is better on the training pairs.
GRID_COMBINES = [("lexical", 0.0), ("sum", 0.3), ("sum", 0.5), ("cascade", 0.0)]
GRID_TEXTS = ["label", "both", "full"]
GRID_ALPHAS = [0.0, 0.1, 0.2, 0.3]
GRID_THRESHOLDS = [round(i / 100, 2) for i in range(30, 81)]

# Published OAEI 2023 Conference track results against ra1 (micro-averaged over the
# 21 test cases; "Threshold" is the confidence threshold the organisers selected per
# system).  Transcribed on 2026-10-03 from the official pages below.
PUBLISHED = {
    "edition": "OAEI 2023 Conference track",
    "sources": [
        "https://oaei.ontologymatching.org/2023/results/conference/index.html",
        "https://oaei.ontologymatching.org/2023/results/conference/eval.html",
    ],
    "aggregation": "micro (absolute TP/FP/FN across all 21 test cases)",
    "note": (
        "ra1 is the original crisp reference alignment; the organisers' headline table is rar2-M3 "
        "(consistency- and conservativity-repaired), which we do not vendor and therefore do not "
        "report numbers against.  OAEI 2024 published only rar2-M3 for the Conference track."
    ),
    "ra1-M1": {
        "GraphMatcher": {"threshold": 0.0, "precision": 0.78, "f1": 0.83, "recall": 0.89},
        "SORBETMtch": {"threshold": 0.0, "precision": 0.78, "f1": 0.76, "recall": 0.75},
        "Matcha": {"threshold": 0.0, "precision": 0.76, "f1": 0.72, "recall": 0.68},
        "LogMap": {"threshold": 0.0, "precision": 0.84, "f1": 0.72, "recall": 0.63},
        "OLaLa": {"threshold": 0.0, "precision": 0.66, "f1": 0.69, "recall": 0.73},
        "ALIN": {"threshold": 0.0, "precision": 0.88, "f1": 0.68, "recall": 0.55},
        "edna": {"threshold": 0.0, "precision": 0.88, "f1": 0.67, "recall": 0.54},
        "LogMapLt": {"threshold": 0.0, "precision": 0.84, "f1": 0.66, "recall": 0.54},
        "StringEquiv": {"threshold": 0.0, "precision": 0.88, "f1": 0.64, "recall": 0.50},
        "LSMatch": {"threshold": 0.0, "precision": 0.88, "f1": 0.64, "recall": 0.50},
        "AMD": {"threshold": 0.0, "precision": 0.87, "f1": 0.64, "recall": 0.50},
        "TOMATO": {"threshold": 0.0, "precision": 0.73, "f1": 0.63, "recall": 0.56},
    },
    "ra1-M2": {
        "PropMatch": {"threshold": 0.0, "precision": 0.83, "f1": 0.64, "recall": 0.52},
        "Matcha": {"threshold": 0.78, "precision": 0.48, "f1": 0.47, "recall": 0.46},
        "GraphMatcher": {"threshold": 0.91, "precision": 0.72, "f1": 0.40, "recall": 0.28},
        "LogMap": {"threshold": 0.79, "precision": 0.62, "f1": 0.39, "recall": 0.28},
        "OLaLa": {"threshold": 0.76, "precision": 0.39, "f1": 0.30, "recall": 0.24},
        "LogMapLt": {"threshold": 0.0, "precision": 0.24, "f1": 0.23, "recall": 0.22},
        "TOMATO": {"threshold": 0.0, "precision": 0.17, "f1": 0.18, "recall": 0.20},
        "edna": {"threshold": 0.0, "precision": 0.21, "f1": 0.14, "recall": 0.11},
        "StringEquiv": {"threshold": 0.0, "precision": 0.07, "f1": 0.03, "recall": 0.02},
    },
    "ra1-M3": {
        "GraphMatcher": {"threshold": 0.0, "precision": 0.76, "f1": 0.78, "recall": 0.80},
        "SORBETMtch": {"threshold": 0.0, "precision": 0.78, "f1": 0.70, "recall": 0.64},
        "LogMap": {"threshold": 0.0, "precision": 0.81, "f1": 0.68, "recall": 0.58},
        "Matcha": {"threshold": 0.66, "precision": 0.70, "f1": 0.67, "recall": 0.64},
        "ALIN": {"threshold": 0.0, "precision": 0.88, "f1": 0.61, "recall": 0.47},
        "OLaLa": {"threshold": 0.0, "precision": 0.56, "f1": 0.61, "recall": 0.66},
        "edna": {"threshold": 0.0, "precision": 0.79, "f1": 0.59, "recall": 0.47},
        "LogMapLt": {"threshold": 0.0, "precision": 0.73, "f1": 0.59, "recall": 0.50},
        "AMD": {"threshold": 0.0, "precision": 0.87, "f1": 0.58, "recall": 0.43},
        "LSMatch": {"threshold": 0.0, "precision": 0.88, "f1": 0.57, "recall": 0.42},
        "StringEquiv": {"threshold": 0.0, "precision": 0.80, "f1": 0.56, "recall": 0.43},
        "TOMATO": {"threshold": 0.0, "precision": 0.61, "f1": 0.55, "recall": 0.50},
        "PropMatch": {"threshold": 0.0, "precision": 0.83, "f1": 0.15, "recall": 0.08},
    },
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001 - best effort metadata
        return "unknown"


def micro(run: BenchmarkRun, modality: str) -> dict:
    return run.aggregates[modality]["micro"]


def fmt_row(label: str, run: BenchmarkRun) -> str:
    cells = [label, f"{run.threshold:.2f}"]
    for m in MODALITIES:
        r = micro(run, m)
        cells.append(f"{r['precision']:.2f} / {r['recall']:.2f} / **{r['f1']:.2f}**")
    return "| " + " | ".join(cells) + " |"


def comparison_table(modality: str, ours: dict[str, BenchmarkRun]) -> str:
    rows = [(name, v["precision"], v["recall"], v["f1"], False) for name, v in PUBLISHED[f"ra1-{modality}"].items()]
    for label, run in ours.items():
        r = micro(run, modality)
        rows.append((label, r["precision"], r["recall"], r["f1"], True))
    rows.sort(key=lambda x: (-x[3], x[0]))
    lines = [
        f"| System ({modality}) | P | R | F1 |",
        "|---|---|---|---|",
    ]
    for name, p, r, f, mine in rows:
        tag = f"**{name}** (this repo)" if mine else name
        lines.append(f"| {tag} | {p:.2f} | {r:.2f} | {f:.2f} |")
    return "\n".join(lines)


def grid_label(combine: str, w: float, text: str, alpha: float, th: float) -> str:
    return f"{combine},w={w:.1f},text={text},alpha={alpha:.1f},th={th:.2f}"


def make_matcher(encoder, combine: str, w: float, text: str, alpha: float) -> Matcher:
    if combine == "lexical":
        return HybridMatcher(encoder, HybridMatcherConfig(embedding_weight=0.0, combine="sum", structure_weight=alpha))
    return HybridMatcher(
        encoder,
        HybridMatcherConfig(
            embedding_weight=w,
            combine=combine,
            structure_weight=alpha,
            embedding=EmbeddingMatcherConfig(text=text, calibration="csls", csls_k=10),
        ),
    )


def run_grid(bench: OAEIConferenceBenchmark, encoder) -> tuple[dict, dict]:
    """Every grid point: per-pair counts (for the CV) and micro aggregates (for the report)."""
    grid: dict = {}
    micro_curve: dict = {}
    for combine, w in GRID_COMBINES:
        texts = ["both"] if combine == "lexical" else GRID_TEXTS
        for text in texts:
            for alpha in GRID_ALPHAS:
                matcher = make_matcher(encoder, combine, w, text, alpha)
                for th in GRID_THRESHOLDS:
                    run = bench.run(matcher, th)
                    label = grid_label(combine, w, text, alpha, th)
                    grid[label] = grid_results(run)
                    micro_curve[label] = {m: {"f1": micro(run, m)["f1"]} for m in MODALITIES}
    return grid, micro_curve


def cv_row(label: str, cv: dict) -> str:
    cells = [label, "CV"]
    for m in MODALITIES:
        r = cv[m]
        cells.append(f"{r['precision']:.2f} / {r['recall']:.2f} / **{r['f1']:.2f}**")
    return "| " + " | ".join(cells) + " |"


def rank_of(f1: float, modality: str) -> tuple[int, int]:
    published = PUBLISHED[f"ra1-{modality}"]
    ranked = sorted([v["f1"] for v in published.values()] + [f1], reverse=True)
    return ranked.index(f1) + 1, len(ranked)


def write_markdown(payload: dict, ours: dict[str, BenchmarkRun], baseline: BenchmarkRun) -> str:
    default = ours["olaya-lexical @ 0.50 (a priori)"]
    best = ours[next(k for k in ours if k.startswith("olaya-lexical @ best"))]
    repair = ours["olaya-lexical + coherence repair @ 0.50"]
    emb = payload.get("embeddings")
    rank_m1 = payload["comparison"]["ra1-M1"]["rank_of_default_run"]
    rank_m3 = payload["comparison"]["ra1-M3"]["rank_of_default_run"]
    n_m1 = len(PUBLISHED["ra1-M1"]) + 1
    n_m3 = len(PUBLISHED["ra1-M3"]) + 1
    lines = [
        "# OAEI Conference track (ra1) — honest evaluation",
        "",
        f"Date: {payload['date']}  ·  Commit: `{payload['commit'][:12]}`  ·  Runtime: {payload['runtime_seconds']:.1f} s "
        f"(lexical {payload['runtime_lexical_seconds']:.1f} s; embeddings {payload['runtime_embeddings_seconds']:.1f} s "
        "including model loading, encoding, the configuration grid and the cross-validation; 4-thread CPU)",
        "",
        "Protocol: standard OAEI alignment evaluation.  For each of the 21 OntoFarm pairs the matcher scores every",
        "class/class and property/property cross pair, keeps the mutual-best 1:1 correspondences, applies a confidence",
        "threshold and the alignment is scored against the ra1 reference alignment.  P / R / F1 are reported for classes",
        "(M1), properties (M2) and both (M3).  Micro = absolute TP/FP/FN over the 21 pairs (the OAEI convention);",
        "macro = mean of per-pair scores (pairs with no reference correspondences of that kind are skipped).",
        "The gold target is never placed in the candidate set; no metric is derived from synthetic distributions.",
        "",
        "## Results (micro P / R / F1)",
        "",
        "| Run | threshold | M1 (classes) | M2 (properties) | M3 (all) |",
        "|---|---|---|---|---|",
        fmt_row("StringEquiv baseline (this harness)", baseline),
        fmt_row("olaya-lexical, a-priori threshold", default),
        fmt_row("olaya-lexical, best threshold on the test set (optimistic)", best),
        fmt_row("olaya-lexical + OWL 2 RL coherence repair", repair),
    ]
    if emb:
        prim = emb["primary"]
        short = prim["encoder"]["model"]
        lines += [
            fmt_row(f"olaya-embedding only ({short}), a-priori threshold", ours["embedding_apriori"]),
            fmt_row(f"olaya-embedding only ({short}), best threshold (optimistic)", ours["embedding_best"]),
            fmt_row(f"olaya-hybrid ({short}), a-priori config", ours["hybrid_apriori"]),
            fmt_row(f"olaya-hybrid ({short}), best config on ra1 (optimistic)", ours["hybrid_best"]),
            cv_row(f"**olaya-hybrid ({short}), leave-one-pair-out CV (headline)**", prim["cv_leave_one_pair_out"]),
            cv_row(f"olaya-hybrid ({short}), leave-one-ontology-out CV", prim["cv_leave_one_ontology_out"]),
        ]
    lines += [
        "",
        "Macro (mean over pairs) for the lexical a-priori run: "
        + ", ".join(
            f"{m} P {default.aggregates[m]['macro']['precision']:.2f} / R {default.aggregates[m]['macro']['recall']:.2f} / "
            f"F1 {default.aggregates[m]['macro']['f1']:.2f} ({default.aggregates[m]['macro']['pairs']} pairs)"
            for m in MODALITIES
        )
        + ".",
        "",
        "Notes:",
        "",
        '- The lexical threshold was fixed at 0.50 *before* sweeping (weighted-Jaccard "shared evidence outweighs unshared").',
        f"  The sweep's best M3 threshold is {best.threshold:.2f}; selecting a threshold on the test set is optimistic.",
        "- Harness sanity check: our StringEquiv re-implementation lands within a few points of the published",
        "  StringEquiv baseline (ra1-M1 0.88 / 0.50 / 0.64, ra1-M3 0.80 / 0.43 / 0.56); the small gap comes from",
        "  normalisation details of the organisers' implementation that are not published.",
        "- The coherence repair (drop a correspondence when it maps a sub-class pair onto a disjoint pair, using the",
        "  repository's OWL 2 RL closure) *lowers* F1 on ra1.  ra1 is known to contain such incoherences, which is",
        "  precisely why OAEI introduced the repaired references ra2 / rar2; the projection removes true positives.",
        "  It is kept as an opt-in flag and is not part of the headline number.",
    ]
    if emb:
        prim = emb["primary"]
        lopo, looo = prim["cv_leave_one_pair_out"], prim["cv_leave_one_ontology_out"]
        enc = prim["encoder"]
        lines += [
            "",
            "## Embedding and hybrid matchers",
            "",
            f"Encoder: `{enc['model']}` (revision `{enc['revision']}`, commit `{enc['commit_sha'][:12]}`, {enc['dim']}-d,",
            f"mean pooling, L2-normalised, max {enc['max_length']} tokens, prefix `{enc['prefix']!r}`), run locally on the CPU",
            "with `transformers` only (`olaya.embeddings.TransformerEncoder`).  Entity text = label (camelCase split) +",
            "comment + parent labels (classes) or domain / range labels (properties); the matcher uses the mean of the",
            "label-only and full-description cosines.  Raw cosines have a high floor on this data (unrelated entities",
            "score 0.3-0.5, and more for e5), so they are hubness-corrected with CSLS (Conneau et al. 2018): the mean",
            "similarity of each entity's 10 nearest neighbours is the background and the score is rescaled so that",
            "background = 0 and a perfect match = 1.  No gold data is involved in the calibration.",
            "",
            "**Hybrid, a-priori config** (fixed before any grid was run): *cascade* - the lexical matcher runs first",
            "(mutual-best, threshold 0.50) and the calibrated embedding scores only the entities it left unmatched,",
            "same threshold 0.50, no structural propagation.  Weighted sums `(1-w)*lex + w*emb` were also tried and are",
            "in the grid below.",
            "",
            "**Honesty rule.**  ra1 is the only test set there is.  Every number selected by looking at ra1 is",
            "optimistic, so three rows are reported: the a-priori config, the best grid point on ra1 (optimistic),",
            "and the *cross-validated* number, which is the headline: for every held-out pair (leave-one-pair-out,",
            "21 folds) the grid point with the best M3 micro-F1 on the other 20 pairs is selected and applied to the",
            "held-out pair; the held-out TP/FP/FN are then summed.  Leave-one-ontology-out (7 folds, all 6 pairs of an",
            "ontology held out, selection on the remaining 15) is the stricter variant.  Two caveats remain and are not",
            "hidden: (1) the *design* choices (cascade rather than weighted sum, CSLS rather than raw cosine, averaging",
            "label and full text) and the choice of the primary encoder were made after looking at ra1 numbers, and",
            "the CV only protects the numeric knobs (combination, text, alpha, threshold); (2) 21 pairs from 7",
            "ontologies are a small sample, so differences of 0.01-0.02 F1 between any two rows are within noise.",
            "",
            "### Cross-validated selection (primary encoder)",
            "",
            f"- Leave-one-pair-out: M1 {lopo['M1']['precision']:.3f} / {lopo['M1']['recall']:.3f} / **{lopo['M1']['f1']:.3f}**, "
            f"M3 {lopo['M3']['precision']:.3f} / {lopo['M3']['recall']:.3f} / **{lopo['M3']['f1']:.3f}**.  "
            f"Configs chosen across the 21 folds: {', '.join(f'`{c}`' for c in lopo['distinct_configs_chosen'])}.",
            f"- Leave-one-ontology-out: M1 {looo['M1']['precision']:.3f} / {looo['M1']['recall']:.3f} / **{looo['M1']['f1']:.3f}**, "
            f"M3 {looo['M3']['precision']:.3f} / {looo['M3']['recall']:.3f} / **{looo['M3']['f1']:.3f}**.  "
            f"Configs chosen across the 7 folds: {', '.join(f'`{c}`' for c in looo['distinct_configs_chosen'])}.",
            f"- Best single grid point on all 21 pairs (optimistic): `{prim['best_on_ra1']['label']}`.",
            f"- Grid: combination in {{lexical only, sum w=0.3, sum w=0.5, cascade}} x text in {{label, both, full}} x "
            f"alpha in {{{', '.join(f'{a:.1f}' for a in GRID_ALPHAS)}}} x threshold "
            f"{GRID_THRESHOLDS[0]:.2f}-{GRID_THRESHOLDS[-1]:.2f} step 0.01 ({len(prim['grid_micro'])} points); "
            "selection criterion: micro-F1 M3.",
            "",
            "### Grid sweep at the a-priori threshold 0.50 (primary encoder, micro F1 M1 / M3)",
            "",
            "| combination | text | alpha 0.0 | alpha 0.1 | alpha 0.2 | alpha 0.3 |",
            "|---|---|---|---|---|---|",
        ]
        for combine, w in GRID_COMBINES:
            texts = ["both"] if combine == "lexical" else GRID_TEXTS
            for text in texts:
                cells = [
                    f"{combine}" + (f" w={w:.1f}" if combine == "sum" else ""),
                    text if combine != "lexical" else "-",
                ]
                for alpha in GRID_ALPHAS:
                    g = prim["grid_micro"][grid_label(combine, w, text, alpha, 0.5)]
                    cells.append(f"{g['M1']['f1']:.3f} / {g['M3']['f1']:.3f}")
                lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            "Reading the sweep: at 0.50 the weighted sum never beats the lexical matcher on M1 and at most matches it",
            "on M3 (the embedding's residual floor dilutes exact lexical evidence); the cascade adds recall for",
            "entities with no lexical evidence",
            "(Place/Location, Trip/Excursion, AccommodationPlace/Hotel, Chairman/Chair) at roughly break-even",
            "precision, which is why the gain is small.  Structural propagation (alpha) helps by about 0.01 on both M1",
            "and M3 for the cascade and is kept as a tunable knob; its benefit is within noise.",
            "",
            "### Encoder ablation (cascade, text=both, CSLS; micro F1 M1 / M3)",
            "",
            "| encoder | commit | embedding only @0.50 | embedding only, best th | cascade alpha 0 @0.50 | cascade alpha 0.2 @0.50 | cascade, best th (optimistic) |",
            "|---|---|---|---|---|---|---|",
        ]
        for e in emb["encoders"]:
            if "error" in e:
                lines.append(f"| {e['model']} | - | not loadable: {e['error']} | | | | |")
                continue
            cells = [
                f"`{e['encoder']['model']}`",
                f"`{e['encoder']['commit_sha'][:8]}`",
                f"{e['embedding_apriori']['M1']['f1']:.3f} / {e['embedding_apriori']['M3']['f1']:.3f}",
                f"{e['embedding_best']['M1']['f1']:.3f} / {e['embedding_best']['M3']['f1']:.3f} (th {e['embedding_best']['threshold']:.2f})",
                f"{e['cascade_alpha0']['M1']['f1']:.3f} / {e['cascade_alpha0']['M3']['f1']:.3f}",
                f"{e['cascade_alpha02']['M1']['f1']:.3f} / {e['cascade_alpha02']['M3']['f1']:.3f}",
                f"{e['cascade_best']['M1']['f1']:.3f} / {e['cascade_best']['M3']['f1']:.3f} (th {e['cascade_best']['threshold']:.2f})",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            "All small encoders land within 0.01-0.02 of each other.  Two base-size encoders were also measured once",
            "with the same code during development (not part of the default run): `sentence-transformers/all-mpnet-base-v2`",
            "cascade @0.50 M1 0.750 / M3 0.684 (alpha 0), `BAAI/bge-base-en-v1.5` 0.737 / 0.669 (alpha 0.2).  A larger",
            "encoder does not move the ceiling; the limiting factor is that the leftover candidates the embedding has to",
            "decide on are semantically plausible pairs ra1 does not contain (Reviewer/Possible_Reviewer, date/startDate).",
        ]
    lines += [
        "",
        "## Comparison with published OAEI 2023 results (ra1, micro)",
        "",
        f"Sources: {', '.join(PUBLISHED['sources'])}.  Published numbers use each system's organiser-selected",
        "threshold; ours are the lexical a-priori run and, for the hybrid, the leave-one-pair-out cross-validated",
        "number (no pair selects its own configuration).  Ranked by F1.",
        "",
    ]
    comp = {"olaya-lexical": default}
    if emb:
        comp["olaya-hybrid (LOPO-CV)"] = ours["hybrid_cv_proxy"]
    lines += [comparison_table("M1", comp), "", comparison_table("M3", comp), ""]
    if emb:
        prim = emb["primary"]
        lopo = prim["cv_leave_one_pair_out"]
        r1, n1 = rank_of(round(lopo["M1"]["f1"], 2), "M1")
        r3, n3 = rank_of(round(lopo["M3"]["f1"], 2), "M3")
        gm1 = PUBLISHED["ra1-M1"]["GraphMatcher"]["f1"]
        gm3 = PUBLISHED["ra1-M3"]["GraphMatcher"]["f1"]
        sb1 = PUBLISHED["ra1-M1"]["SORBETMtch"]["f1"]
        sb3 = PUBLISHED["ra1-M3"]["SORBETMtch"]["f1"]
        lines += [
            f"Where we land (headline = hybrid, leave-one-pair-out CV, M1 F1 {lopo['M1']['f1']:.2f} / M3 F1 {lopo['M3']['f1']:.2f}):",
            f"rank {r1} of {n1} on ra1-M1 and rank {r3} of {n3} on ra1-M3 by two-decimal F1 (a 0.00 gap in the",
            f"two-decimal table is a tie, not a win).  That is {gm1 - lopo['M1']['f1']:.2f} (M1) / {gm3 - lopo['M3']['f1']:.2f} (M3)",
            f"F1 below GraphMatcher and {sb1 - lopo['M1']['f1']:.2f} (M1) / {sb3 - lopo['M3']['f1']:.2f} (M3) below SORBETMtch;",
            "tied with LogMap on M3 and 0.03 above it on M1; above Matcha and the string-based systems.  The hybrid improves on",
            f"the purely lexical matcher by {lopo['M1']['f1'] - micro(default, 'M1')['f1']:+.3f} (M1) / "
            f"{lopo['M3']['f1'] - micro(default, 'M3')['f1']:+.3f} (M3) under cross-validation, which is a real but small",
            "gain, within the noise band of a 21-pair test set.  We do not beat or tie the published best on either",
            "modality, and we do not claim state of the art.  GraphMatcher / SORBETMtch additionally learn from the",
            "task (graph attention over the ontology, fine-tuned encoders); nothing here is trained on OntoFarm.",
        ]
    else:
        lines += [
            f"Where we land: rank {rank_m1} of {n_m1} on ra1-M1 and rank {rank_m3} of {n_m3} on ra1-M3 when sorted by the",
            "two-decimal F1 values (lexical matcher only; this run was made with `--no-embeddings`).",
        ]
    per_pair = ours["hybrid_apriori"] if emb else default
    lines += [
        "",
        f"## Per-pair results ({'hybrid a-priori run' if emb else 'lexical a-priori run'})",
        "",
        "| Pair | ref | sys | M1 P / R / F1 | M2 P / R / F1 | M3 P / R / F1 |",
        "|---|---|---|---|---|---|",
    ]
    for pe in per_pair.pairs:
        d = pe.to_dict()
        cells = [pe.pair, str(pe.reference_size), str(pe.system_size)]
        for m in MODALITIES:
            cells.append(f"{d[m]['precision']:.2f} / {d[m]['recall']:.2f} / {d[m]['f1']:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Matchers",
        "",
        "`olaya.benchmarks.oaei.LexicalMatcher`: camelCase / underscore splitting, lowercase, light suffix stemming,",
        "IDF-weighted Jaccard over name tokens (IDF over both ontologies of the pair), a Levenshtein term that only",
        "counts for near-identical spellings (>= 0.90), and a 0.2-weight Jaccard over comment / parent / domain /",
        "range tokens.  Mutual-best 1:1 selection, then the threshold.  Deterministic, standard library only.",
        "Ablations that were tried and rejected: stop-word removal (-0.05 F1 on properties), ungated Levenshtein",
        "(-0.1 F1), TF-IDF cosine instead of weighted Jaccard (slightly worse and threshold-sensitive), non-mutual",
        "selection (-0.08 F1), coherence repair (-0.02 F1, see above).",
        "",
        "`olaya.benchmarks.oaei.EmbeddingMatcher` / `HybridMatcher`: see the embedding section.  Rejected on ra1:",
        "raw (uncalibrated) cosine in the cascade (-0.15 F1 at 0.50, best threshold 0.70), `max(lex, emb)` fusion",
        "(-0.1 F1), weighted sums (never above the lexical matcher at the a-priori threshold), label-only text",
        "(-0.02 F1 vs label + description).  Embeddings are cached on disk under `$OLAYA_CACHE_DIR/embeddings`",
        "(default `~/.cache/olaya/embeddings`), keyed by model commit and text sha256; `--no-embeddings` reproduces the",
        "lexical rows offline.",
        "",
        f"Dataset: {len(payload['dataset_sha256'])} upstream files, hashes in the JSON and in `tests/fixtures/oaei/README.md`.",
    ]
    return "\n".join(lines) + "\n"


def evaluate_encoder(
    bench: OAEIConferenceBenchmark, model: str, *, primary: bool
) -> tuple[dict, dict[str, BenchmarkRun]]:
    """Embedding-only and cascade rows for one encoder; the full grid + CV for the primary one."""
    t0 = time.perf_counter()
    encoder = load_encoder(model)
    emb_matcher = EmbeddingMatcher(encoder, APRIORI_HYBRID.embedding)
    emb_default = bench.run(emb_matcher, DEFAULT_THRESHOLD)
    emb_best, emb_curve = bench.sweep(emb_matcher, SWEEP_THRESHOLDS, modality="M3")
    cascade0 = HybridMatcher(encoder, APRIORI_HYBRID)
    cfg02 = HybridMatcherConfig(combine="cascade", structure_weight=0.2, embedding=APRIORI_HYBRID.embedding)
    cascade02 = HybridMatcher(encoder, cfg02)
    hyb_default = bench.run(cascade0, DEFAULT_THRESHOLD)
    hyb02 = bench.run(cascade02, DEFAULT_THRESHOLD)
    hyb_best, hyb_curve = bench.sweep(cascade0, SWEEP_THRESHOLDS, modality="M3")
    info = {
        "encoder": encoder.info.to_dict(),
        "model": model,
        "embedding_config": vars(APRIORI_HYBRID.embedding),
        "embedding_apriori": {**{m: micro(emb_default, m) for m in MODALITIES}, "threshold": DEFAULT_THRESHOLD},
        "embedding_best": {**{m: micro(emb_best, m) for m in MODALITIES}, "threshold": emb_best.threshold},
        "cascade_alpha0": {**{m: micro(hyb_default, m) for m in MODALITIES}, "threshold": DEFAULT_THRESHOLD},
        "cascade_alpha02": {**{m: micro(hyb02, m) for m in MODALITIES}, "threshold": DEFAULT_THRESHOLD},
        "cascade_best": {**{m: micro(hyb_best, m) for m in MODALITIES}, "threshold": hyb_best.threshold},
        "texts_encoded_this_run": encoder.calls,
    }
    runs = {"embedding_apriori": emb_default, "embedding_best": emb_best, "hybrid_apriori": hyb_default}
    if primary:
        grid, grid_micro = run_grid(bench, encoder)
        all_pairs = [f"{s}-{t}" for s, t in bench_pairs()]
        best_label = select_config(grid, all_pairs, "M3")
        lopo = cross_validate(grid, leave_one_pair_out_folds(all_pairs), modality="M3")
        looo = cross_validate(grid, leave_one_ontology_out_folds(), modality="M3")
        info.update(
            {
                "apriori_hybrid_config": {
                    "combine": APRIORI_HYBRID.combine,
                    "structure_weight": APRIORI_HYBRID.structure_weight,
                    "cascade_threshold": APRIORI_HYBRID.cascade_threshold,
                    "threshold": DEFAULT_THRESHOLD,
                    "embedding": vars(APRIORI_HYBRID.embedding),
                },
                "grid": {
                    "combines": GRID_COMBINES,
                    "texts": GRID_TEXTS,
                    "alphas": GRID_ALPHAS,
                    "thresholds": GRID_THRESHOLDS,
                    "selection_criterion": "micro-F1 M3",
                },
                "grid_micro": grid_micro,
                "best_on_ra1": {"label": best_label, **grid_micro[best_label], "caveat": "selected on the test set"},
                "cv_leave_one_pair_out": lopo,
                "cv_leave_one_ontology_out": looo,
                "embedding_threshold_curve": emb_curve,
                "cascade_threshold_curve": hyb_curve,
            }
        )
        # The best grid point as a BenchmarkRun (for the results table).
        combine, rest = best_label.split(",", 1)
        kv = dict(item.split("=") for item in rest.split(","))
        runs["hybrid_best"] = bench.run(
            make_matcher(encoder, combine, float(kv["w"]), kv["text"], float(kv["alpha"])), float(kv["th"])
        )
        # A BenchmarkRun-shaped proxy carrying the CV micro numbers for the comparison table.
        proxy = BenchmarkRun("olaya-hybrid (LOPO-CV)", 0.0, False, [], {m: {"micro": lopo[m]} for m in MODALITIES})
        runs["hybrid_cv_proxy"] = proxy
    info["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    return info, runs


def bench_pairs():
    from olaya.benchmarks.oaei import CONFERENCE_PAIRS

    return CONFERENCE_PAIRS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--no-embeddings", action="store_true", help="lexical rows only (offline, no model download)")
    parser.add_argument(
        "--encoders",
        default=",".join(ENCODERS),
        help="comma-separated HF checkpoints; the first is the primary encoder (grid + cross-validation)",
    )
    args = parser.parse_args()

    bench = OAEIConferenceBenchmark(args.fixtures / "ontologies", args.fixtures / "reference_alignments")
    matcher = LexicalMatcher(LexicalMatcherConfig())

    t0 = time.perf_counter()
    baseline = bench.run(StringEquivMatcher(), 1.0)
    default = bench.run(matcher, DEFAULT_THRESHOLD)
    best, curve = bench.sweep(matcher, SWEEP_THRESHOLDS, modality="M3")
    repair = bench.run(matcher, DEFAULT_THRESHOLD, repair=True)
    runtime_lexical = time.perf_counter() - t0

    ours: dict[str, BenchmarkRun] = {
        "olaya-lexical @ 0.50 (a priori)": default,
        f"olaya-lexical @ best M3 threshold {best.threshold:.2f} (optimistic)": best,
        "olaya-lexical + coherence repair @ 0.50": repair,
    }

    embeddings: dict | None = None
    t1 = time.perf_counter()
    if not args.no_embeddings:
        encoders_info = []
        primary_info = None
        for i, model in enumerate(m.strip() for m in args.encoders.split(",") if m.strip()):
            try:
                info, runs = evaluate_encoder(bench, model, primary=(i == 0))
            except EncoderUnavailable as exc:
                print(f"  encoder {model} unavailable: {exc}", file=sys.stderr)
                encoders_info.append({"model": model, "error": str(exc)})
                continue
            if i == 0:
                primary_info = info
                ours.update(runs)
                # The grid and threshold curves are large; keep them once, under "primary".
                big = ("grid_micro", "embedding_threshold_curve", "cascade_threshold_curve")
                info = {k: v for k, v in info.items() if k not in big}
            encoders_info.append(info)
        if primary_info is not None:
            embeddings = {"primary": primary_info, "encoders": encoders_info}
        else:
            print("  no encoder could be loaded; writing lexical rows only", file=sys.stderr)
    runtime_embeddings = time.perf_counter() - t1
    runtime = time.perf_counter() - t0

    comparison = {}
    for m in ("M1", "M3"):
        published = PUBLISHED[f"ra1-{m}"]
        our_f1 = micro(default, m)["f1"]
        entries = [(name, v["f1"]) for name, v in published.items()] + [("olaya-lexical", our_f1)]
        if embeddings:
            entries.append(("olaya-hybrid (LOPO-CV)", embeddings["primary"]["cv_leave_one_pair_out"][m]["f1"]))
        ranked = sorted(entries, key=lambda x: (-x[1], x[0]))
        names = [n for n, _ in ranked]
        comparison[f"ra1-{m}"] = {
            "rank_of_default_run": names.index("olaya-lexical") + 1,
            "rank_of_hybrid_cv": names.index("olaya-hybrid (LOPO-CV)") + 1 if embeddings else None,
            "systems_ranked": len(ranked),
            "ranking_by_f1": ranked,
        }

    payload = {
        "benchmark": "OAEI Conference track, OntoFarm, reference alignment ra1 (21 pairs, 7 ontologies)",
        "date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
        "commit": git_commit(),
        "runtime_seconds": round(runtime, 2),
        "runtime_lexical_seconds": round(runtime_lexical, 2),
        "runtime_embeddings_seconds": round(runtime_embeddings, 2),
        "protocol": {
            "selection": "mutual-best 1:1 over all same-kind cross pairs, then confidence threshold",
            "modalities": {"M1": "class correspondences", "M2": "property correspondences", "M3": "both"},
            "micro": "absolute TP/FP/FN summed over the 21 pairs",
            "macro": "mean of per-pair P/R/F1 over pairs that have reference correspondences of that kind",
            "default_threshold": DEFAULT_THRESHOLD,
            "threshold_selection_caveat": "best-threshold / best-config rows are selected on the test set and are optimistic",
            "cross_validation": (
                "leave-one-pair-out (21 folds) and leave-one-ontology-out (7 folds): the grid point with the best "
                "micro-F1 M3 on the training pairs is applied to the held-out pairs, whose TP/FP/FN are summed"
            ),
            "sweep_thresholds": SWEEP_THRESHOLDS,
        },
        "matcher_config": vars(LexicalMatcherConfig()),
        "dataset_sha256": bench.dataset_hashes(),
        "runs": {
            "string_equiv_baseline": baseline.to_dict(),
            "olaya_lexical_default": default.to_dict(),
            "olaya_lexical_best_threshold": best.to_dict(),
            "olaya_lexical_coherence_repair": repair.to_dict(),
            **(
                {
                    "olaya_embedding_default": ours["embedding_apriori"].to_dict(),
                    "olaya_embedding_best_threshold": ours["embedding_best"].to_dict(),
                    "olaya_hybrid_apriori": ours["hybrid_apriori"].to_dict(),
                    "olaya_hybrid_best_on_ra1": ours["hybrid_best"].to_dict(),
                }
                if embeddings
                else {}
            ),
        },
        "threshold_curve": curve,
        "embeddings": embeddings,
        "published": PUBLISHED,
        "comparison": comparison,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{RESULT_STEM}.json"
    md_path = args.out_dir / f"{RESULT_STEM}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(write_markdown(payload, ours, baseline))

    print(f"OAEI Conference ra1 — {len(default.pairs)} pairs, runtime {runtime:.1f}s")
    for label, run in [("StringEquiv", baseline), *((k, v) for k, v in ours.items() if k != "hybrid_cv_proxy")]:
        print(
            f"  {label:<60} "
            + "  ".join(
                f"{m}: P {micro(run, m)['precision']:.3f} R {micro(run, m)['recall']:.3f} F1 {micro(run, m)['f1']:.3f}"
                for m in MODALITIES
            )
        )
    if embeddings:
        for key in ("cv_leave_one_pair_out", "cv_leave_one_ontology_out"):
            cv = embeddings["primary"][key]
            print(
                f"  {key:<60} "
                + "  ".join(
                    f"{m}: P {cv[m]['precision']:.3f} R {cv[m]['recall']:.3f} F1 {cv[m]['f1']:.3f}" for m in MODALITIES
                )
            )
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
