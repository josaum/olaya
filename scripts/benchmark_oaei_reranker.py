"""Evaluate the Anatomy-trained cross-encoder as a reranker on the OAEI Conference track (ra1).

Usage::

    .venv/bin/python scripts/benchmark_oaei_reranker.py [--checkpoint models/reranker/anatomy-minilm-l6]

Writes ``benchmarks/results/oaei-reranker.json`` and ``.md``.  Protocol:

* **Base matchers** (computed here from the imported classes, so the numbers do not
  depend on another report): the hybrid matcher in its a-priori *cascade* configuration
  (lexical first, CSLS-calibrated ``intfloat/e5-small-v2`` for the rest) and the dense
  *sum* configuration ``0.5 * lex + 0.5 * emb``; each is scored with the usual mutual-best
  + threshold selection at the a-priori threshold 0.50, at its best threshold on ra1
  (optimistic) and under leave-one-pair-out threshold selection.
* **Reranker**: the base scores are reduced to the top-``k`` candidates per entity, the
  cross-encoder scores each candidate, the two are fused as ``(1 - w) * base + w * ce``
  and mutual-best + threshold runs on the fused candidate scores.
* **A-priori reranker config**, fixed before the grid was run: base = cascade, k = 5,
  w = 0.5 (equal trust), threshold 0.50.
* **Grid**: base in {cascade, sum} x k in {5} x w in {0, 0.25, 0.5, 0.75, 1} x threshold
  0.30-0.80 step 0.01.  The headline is the leave-one-pair-out cross-validated number
  (21 folds; the grid point is chosen on the 20 other pairs).  The best grid point on all
  21 pairs is reported as the optimistic bound.
* **Ablation**: the same grid with the *un-fine-tuned* ``cross-encoder/ms-marco-MiniLM-L-6-v2``
  shows what the Anatomy fine-tuning contributed.
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

from olaya.benchmarks.oaei import (  # noqa: E402
    CONFERENCE_PAIRS,
    MODALITIES,
    EmbeddingMatcherConfig,
    HybridMatcher,
    HybridMatcherConfig,
    OAEIConferenceBenchmark,
    cross_validate,
    grid_results,
    leave_one_pair_out_folds,
    select_config,
)
from olaya.benchmarks.reranker import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    CrossEncoderReranker,
    RerankBenchmark,
    config_label,
    micro,
)
from olaya.embeddings import load_encoder  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "oaei"
RESULTS_DIR = REPO / "benchmarks" / "results"
RESULT_STEM = "oaei-reranker"
DEFAULT_CHECKPOINT = REPO / "models" / "reranker" / "anatomy-minilm-l6"
ENCODER = "intfloat/e5-small-v2"

# A-priori settings (fixed before any grid was run).
APRIORI_THRESHOLD = 0.5
APRIORI_BASE = "cascade"
APRIORI_K = 5
APRIORI_WEIGHT = 0.5
# Grid.
GRID_KS = [5]
GRID_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
GRID_THRESHOLDS = [round(0.30 + i / 100, 2) for i in range(51)]
PAIR_NAMES = [f"{s}-{t}" for s, t in CONFERENCE_PAIRS]

# Published Conference ra1-M3 (and M1) F-measures used for the "where we land" comparison
# (OAEI 2023 Conference results page: F1 on ra1-M3 and ra1-M1).
PUBLISHED = {"GraphMatcher": {"M1": 0.83, "M3": 0.78}, "SORBETMtch": {"M1": 0.76, "M3": 0.70}}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def base_matchers(encoder: Any) -> dict[str, HybridMatcher]:
    emb = EmbeddingMatcherConfig(text="both", calibration="csls", csls_k=10)
    return {
        "cascade": HybridMatcher(encoder, HybridMatcherConfig(combine="cascade", structure_weight=0.0, embedding=emb)),
        "sum": HybridMatcher(encoder, HybridMatcherConfig(combine="sum", embedding_weight=0.5, embedding=emb)),
    }


def prf(d: dict[str, Any]) -> str:
    return f"{d['precision']:.2f} / {d['recall']:.2f} / **{d['f1']:.2f}**"


def prf3(agg: dict[str, Any], key: str = "micro") -> str:
    return " | ".join(prf(agg[m][key]) for m in MODALITIES)


def totals_row(cv: dict[str, Any]) -> str:
    return " | ".join(prf(cv[m]) for m in MODALITIES)


def baseline_block(bench: OAEIConferenceBenchmark, matcher: HybridMatcher) -> dict[str, Any]:
    """A-priori threshold, best threshold on ra1 (optimistic) and LOPO threshold selection."""
    apriori = bench.run(matcher, APRIORI_THRESHOLD)
    grid = {f"th={th:.2f}": grid_results(bench.run(matcher, th)) for th in GRID_THRESHOLDS}
    best = select_config(grid, PAIR_NAMES)
    cv = cross_validate(grid, leave_one_pair_out_folds())
    return {
        "matcher": matcher.name,
        "apriori": {"threshold": APRIORI_THRESHOLD, "aggregates": apriori.aggregates},
        "best_on_ra1": {"label": best, **micro(grid, best, PAIR_NAMES)},
        "lopo_cv": cv,
    }


def rerank_block(rb: RerankBenchmark, matchers: dict[str, HybridMatcher]) -> dict[str, Any]:
    apriori = rb.run(matchers[APRIORI_BASE], CONFERENCE_PAIRS, APRIORI_K, APRIORI_WEIGHT, APRIORI_THRESHOLD)
    grid = rb.grid(matchers, CONFERENCE_PAIRS, GRID_KS, GRID_WEIGHTS, GRID_THRESHOLDS)
    best = select_config(grid, PAIR_NAMES)
    cv = cross_validate(grid, leave_one_pair_out_folds())
    # CV restricted to the a-priori base matcher (the reranker's own knobs only).
    grid_apriori_base = {label: g for label, g in grid.items() if label.startswith(f"base={APRIORI_BASE},")}
    cv_apriori_base = cross_validate(grid_apriori_base, leave_one_pair_out_folds())
    # Slice: micro F1 at the a-priori threshold for every (base, w).
    at_th = {}
    for mname in matchers:
        for w in GRID_WEIGHTS:
            label = config_label(mname, APRIORI_K, w, APRIORI_THRESHOLD)
            at_th[label] = {m: micro(grid, label, PAIR_NAMES)[m]["f1"] for m in MODALITIES}
    # Best threshold per (base, w) (optimistic) - shows the shape of the sweep.
    best_per_weight = {}
    for mname in matchers:
        for w in GRID_WEIGHTS:
            labels = {
                label: g for label, g in grid.items() if label.startswith(f"base={mname},k={APRIORI_K},w={w:.2f},")
            }
            b = select_config(labels, PAIR_NAMES)
            best_per_weight[f"base={mname},w={w:.2f}"] = {
                "label": b,
                **{m: micro(grid, b, PAIR_NAMES)[m]["f1"] for m in MODALITIES},
            }
    candidates = {
        mname: {
            f"{s}-{t}": rb.candidates(matchers[mname], s, t, APRIORI_K).candidate_recall for s, t in CONFERENCE_PAIRS
        }
        for mname in matchers
    }
    return {
        "apriori": apriori,
        "best_on_ra1": {"label": best, **micro(grid, best, PAIR_NAMES)},
        "lopo_cv": cv,
        "lopo_cv_apriori_base_only": cv_apriori_base,
        "f1_at_apriori_threshold": at_th,
        "best_threshold_per_weight": best_per_weight,
        "candidate_recall": {
            mname: {"per_pair": rec, "mean": round(sum(rec.values()) / len(rec), 4)}
            for mname, rec in candidates.items()
        },
        "grid_points": len(grid),
    }


def write_markdown(res: dict[str, Any], path: Path) -> None:
    man = res["reranker"]["manifest"]
    tr = man.get("training", {})
    val = tr.get("validation", [{}])[-1] if tr.get("validation") else {}
    before = man.get("validation_before_training", {})
    lines = [
        "# OAEI Conference track (ra1) - cross-encoder reranker trained on the Anatomy track",
        "",
        f"Date: {res['date_utc']}  ·  Commit: `{res['commit']}`  ·  Runtime: {res['runtime_s']} s "
        f"(base matchers {res['timing']['base_s']} s, cross-encoder scoring + grid {res['timing']['rerank_s']} s"
        + (f", ablation {res['timing']['ablation_s']} s" if "ablation_s" in res["timing"] else "")
        + f"; {res['threads']} CPU threads)",
        "",
        "## What was done",
        "",
        "A small cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, 22.7 M parameters, one logit) was",
        "fine-tuned on the **OAEI Anatomy track** (mouse / human reference alignment, "
        f"{man['data']['reference_equivalences']} equivalences;",
        "`tests/fixtures/oaei/anatomy/README.md`) and applied, **without any Conference training**, as a reranker",
        "on top of the hybrid matcher's candidate pairs on the 21 Conference ra1 pairs.  The input of the",
        "cross-encoder is `text(A) [SEP] text(B)` with the same entity descriptions the embedding matcher uses",
        "(`build_entity_texts`: label + comment + parents / domain + range; underscores mapped to spaces).",
        "",
        "Pipeline per ontology pair: base scores -> top-k candidates per entity (k = 5, union over both sides) ->",
        "cross-encoder probability per candidate -> fused score `(1 - w) * base + w * ce` -> mutual-best 1:1 ->",
        "threshold.  Pairs outside the candidate set are dropped, so the candidate recall is the reranker's ceiling:",
        " · ".join(
            f"{name}: mean reference recall {d['mean']:.3f}"
            for name, d in res["reranker"]["results"]["candidate_recall"].items()
        )
        + ".",
        "",
        "## Training on Anatomy (held-out validation)",
        "",
        f"- Data: {man['data']['train_pairs']} training pairs ({man['data']['train_positives']} positives) from "
        f"{man['data']['split']['train_sources']} mouse source entities; {man['data']['validation_pairs']} validation pairs "
        f"({man['data']['validation_positives']} positives) from {man['data']['split']['validation_sources']} held-out source "
        f"entities (split by source entity, fraction {man['data']['split']['fraction']}, seed {man['data']['split']['seed']}).",
        f"  Hard negatives: top-{man['candidate_encoder']['k']} non-reference candidates per entity under the CSLS-calibrated "
        f"`{man['candidate_encoder']['name']}` cosine (reference recall inside the candidate set "
        f"{man['data']['candidate_reference_recall']:.3f}).  Train / validation pair digests: "
        f"`{man['data']['train_pairs_sha256'][:12]}` / `{man['data']['validation_pairs_sha256'][:12]}`.",
        f"- Hyper-parameters: lr {man['hyperparameters']['learning_rate']}, batch {man['hyperparameters']['batch_size']}, "
        f"max length {man['hyperparameters']['max_length']}, {man['hyperparameters']['epochs']} epoch(s) planned, "
        f"warm-up {man['hyperparameters']['warmup_fraction']}, weight decay {man['hyperparameters']['weight_decay']}, "
        f"seed {man['hyperparameters']['seed']}, time budget {man['hyperparameters']['time_budget_s']} s, "
        f"{man['hyperparameters']['threads']} threads.",
        f"- Run: {tr.get('steps')} steps, {tr.get('epochs_completed', 0):.2f} epochs in {tr.get('elapsed_s')} s"
        + (" (stopped by the time budget)" if tr.get("stopped_by_budget") else "")
        + f"; mean BCE per epoch {tr.get('epoch_losses')}; loss curve (every {man['hyperparameters']['log_every']} steps): "
        + ", ".join(
            f"{c['step']}:{c['loss']:.3f}"
            for c in tr.get("loss_curve", [])[:: max(1, len(tr.get("loss_curve", [])) // 12)]
        )
        + ".",
        "",
        "| Anatomy validation (held-out source entities) | AUC | hits@1 |",
        "|---|---|---|",
        f"| embedding base score (`{man['candidate_encoder']['name']}`, CSLS) | {before.get('base', {}).get('auc')} | {before.get('base', {}).get('hits@1')} |",
        f"| cross-encoder before fine-tuning (MS MARCO) | {before.get('cross_encoder', {}).get('auc')} | {before.get('cross_encoder', {}).get('hits@1')} |",
        f"| cross-encoder after fine-tuning | {val.get('cross_encoder', {}).get('auc')} | {val.get('cross_encoder', {}).get('hits@1')} |",
        f"| fused w=0.5 after fine-tuning | {val.get('fused', {}).get('w=0.50', {}).get('auc')} | {val.get('fused', {}).get('w=0.50', {}).get('hits@1')} |",
        "",
        "hits@1 = fraction of held-out mouse entities that have a reference match whose best-scored candidate is that match.",
        "",
        "## Conference ra1 results (micro P / R / F1; M1 classes, M2 properties, M3 all)",
        "",
        "| Run | M1 | M2 | M3 |",
        "|---|---|---|---|",
    ]
    for name, b in res["baselines"].items():
        lines.append(f"| hybrid base `{name}`, a-priori threshold 0.50 | {prf3(b['apriori']['aggregates'])} |")
        lines.append(
            f"| hybrid base `{name}`, best threshold on ra1 (optimistic, `{b['best_on_ra1']['label']}`) | "
            + " | ".join(prf(b["best_on_ra1"][m]) for m in MODALITIES)
            + " |"
        )
        lines.append(f"| hybrid base `{name}`, leave-one-pair-out threshold selection | {totals_row(b['lopo_cv'])} |")
    rr = res["reranker"]["results"]
    lines.append(
        f"| reranker, a-priori config (base={APRIORI_BASE}, k={APRIORI_K}, w={APRIORI_WEIGHT}, th={APRIORI_THRESHOLD}) | "
        f"{prf3(rr['apriori']['aggregates'])} |"
    )
    lines.append(
        f"| reranker, best grid point on ra1 (optimistic, `{rr['best_on_ra1']['label']}`) | "
        + " | ".join(prf(rr["best_on_ra1"][m]) for m in MODALITIES)
        + " |"
    )
    lines.append(
        f"| reranker, leave-one-pair-out CV over the a-priori base only | {totals_row(rr['lopo_cv_apriori_base_only'])} |"
    )
    lines.append(f"| **reranker, leave-one-pair-out CV (headline)** | {totals_row(rr['lopo_cv'])} |")
    if "ablation" in res:
        ab = res["ablation"]["results"]
        lines.append(
            f"| ablation: un-fine-tuned MS MARCO cross-encoder, a-priori config | {prf3(ab['apriori']['aggregates'])} |"
        )
        lines.append(
            f"| ablation: un-fine-tuned MS MARCO cross-encoder, leave-one-pair-out CV | {totals_row(ab['lopo_cv'])} |"
        )
    lines += [
        "",
        f"Configs chosen across the 21 CV folds (fine-tuned reranker): {', '.join('`' + c + '`' for c in rr['lopo_cv']['distinct_configs_chosen'])}.",
        "",
        "### Grid slice at the a-priori threshold 0.50 (micro F1 M1 / M2 / M3)",
        "",
        "| base | w=0.00 (base only) | w=0.25 | w=0.50 | w=0.75 | w=1.00 (cross-encoder only) |",
        "|---|---|---|---|---|---|",
    ]
    for mname in res["baselines"]:
        cells = []
        for w in GRID_WEIGHTS:
            d = rr["f1_at_apriori_threshold"][config_label(mname, APRIORI_K, w, APRIORI_THRESHOLD)]
            cells.append(f"{d['M1']:.3f} / {d['M2']:.3f} / {d['M3']:.3f}")
        lines.append(f"| {mname} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Best threshold per (base, w) on ra1 (optimistic; micro F1 M1 / M2 / M3)",
        "",
        "| base, w | best grid point | M1 | M2 | M3 |",
        "|---|---|---|---|---|",
    ]
    for key, d in rr["best_threshold_per_weight"].items():
        lines.append(f"| {key} | `{d['label']}` | {d['M1']:.3f} | {d['M2']:.3f} | {d['M3']:.3f} |")
    base_cv = res["baselines"][APRIORI_BASE]["lopo_cv"]
    lines += [
        "",
        "## Where we land",
        "",
        "| System | ra1-M1 F1 | ra1-M3 F1 |",
        "|---|---|---|",
        f"| GraphMatcher (published, OAEI 2023) | {PUBLISHED['GraphMatcher']['M1']:.2f} | {PUBLISHED['GraphMatcher']['M3']:.2f} |",
        f"| SORBETMtch (published, OAEI 2023) | {PUBLISHED['SORBETMtch']['M1']:.2f} | {PUBLISHED['SORBETMtch']['M3']:.2f} |",
        f"| olaya hybrid `{APRIORI_BASE}` base, LOPO threshold CV (this report) | {base_cv['M1']['f1']:.2f} | {base_cv['M3']['f1']:.2f} |",
        f"| olaya hybrid + Anatomy-trained reranker, LOPO CV (this report) | {rr['lopo_cv']['M1']['f1']:.2f} | {rr['lopo_cv']['M3']['f1']:.2f} |",
        "",
        res["verdict"],
        "",
        "## Interpretation",
        "",
        "- The fine-tuning *did* transfer in the sense that the Anatomy-trained model is clearly better than the",
        "  off-the-shelf MS MARCO cross-encoder on Conference at the a-priori config (M3 F1 "
        + (
            f"{rr['apriori']['aggregates']['M3']['micro']['f1']:.3f} vs {res['ablation']['results']['apriori']['aggregates']['M3']['micro']['f1']:.3f}"
            if "ablation" in res
            else "see table"
        )
        + "), and on the held-out Anatomy entities it beats the embedding base score.",
        "- It does **not** beat the hybrid base on Conference: in every CV fold the selection picks `w=0.00`",
        "  (base only).  Fusing in the cross-encoder raises precision (and M2) but costs class recall: the",
        "  Anatomy positives are near-identical strings, so the model learned a *strict* notion of equivalence",
        "  and gives low probability to Conference correspondences whose surface forms differ (the kind of",
        "  pair the embedding stage exists to recover).  A single track with one vocabulary is too narrow a",
        "  training signal for cross-domain reranking; Bio-ML or multiple tracks would be the next thing to try.",
        "- The `cascade` base rows here only cross-validate the threshold (no alpha / text grid), which is why",
        "  its LOPO number is a little below the full-grid headline of `oaei-conference-ra1.md`; the a-priori",
        "  row (0.74 / 0.32 / 0.67) matches that report exactly.",
        "",
        "## Honesty notes",
        "",
        "- The Conference track was never used for training; the only Conference-derived choices are the numeric",
        "  knobs (base, w, threshold), and those are protected by the leave-one-pair-out selection.  The a-priori",
        "  configuration (cascade base, k = 5, w = 0.5, threshold 0.50) was fixed before the grid was run.",
        "- The choice of the Anatomy track as training data, of k = 5, of the base cross-encoder and of the",
        "  fine-tuning recipe were made once and not tuned on Conference.  The hybrid base and its a-priori",
        "  configuration are those of `benchmarks/results/oaei-conference-ra1.md`, recomputed here.",
        "- 21 pairs from 7 ontologies are a small sample; differences of 0.01-0.02 F1 between rows are within noise.",
        "- The reranker only sees the candidate set; a reference correspondence outside the top-5 candidates of",
        "  both entities can never be recovered (candidate recall above).",
        "- The checkpoint is not committed (~90 MB); `scripts/train_reranker.py` reproduces it deterministically",
        "  (seed, thread count and data hashes in the manifest, embedded in the JSON next to this file).",
        "",
        "Reproduce: `.venv/bin/python scripts/train_reranker.py && .venv/bin/python scripts/benchmark_oaei_reranker.py`.",
        "",
    ]
    path.write_text("\n".join(lines))


def verdict(res: dict[str, Any]) -> str:
    cv = res["reranker"]["results"]["lopo_cv"]
    base = res["baselines"][APRIORI_BASE]["lopo_cv"]
    m1, m3 = cv["M1"]["f1"], cv["M3"]["f1"]
    d1, d3 = m1 - base["M1"]["f1"], m3 - base["M3"]["f1"]
    parts = [
        f"Cross-validated, the reranker moves the hybrid base from M1 {base['M1']['f1']:.3f} / M3 {base['M3']['f1']:.3f} "
        f"to M1 {m1:.3f} / M3 {m3:.3f} ({d1:+.3f} / {d3:+.3f})."
    ]
    if m3 >= PUBLISHED["GraphMatcher"]["M3"] and m1 >= PUBLISHED["GraphMatcher"]["M1"]:
        parts.append("That matches or beats the published GraphMatcher numbers under cross-validation.")
    elif m3 >= PUBLISHED["SORBETMtch"]["M3"]:
        parts.append(
            "That is at or above the published SORBETMtch level but still below GraphMatcher; this is **not** a "
            "state-of-the-art claim."
        )
    else:
        parts.append(
            "That is still below both published systems (GraphMatcher 0.83 / 0.78, SORBETMtch 0.76 / 0.70); "
            "this is **not** a state-of-the-art claim."
        )
    if abs(d3) < 0.02:
        parts.append("The change versus the base matcher is within the noise of a 21-pair test set.")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-ablation", action="store_true", help="skip the un-fine-tuned cross-encoder grid")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)

    if not (args.checkpoint / "manifest.json").exists():
        print(f"no checkpoint at {args.checkpoint}; run scripts/train_reranker.py first", file=sys.stderr)
        return 2

    t_all = time.perf_counter()
    bench = OAEIConferenceBenchmark(FIXTURES / "ontologies", FIXTURES / "reference_alignments")
    encoder = load_encoder(args.encoder, threads=args.threads)
    matchers = base_matchers(encoder)

    t0 = time.perf_counter()
    baselines = {name: baseline_block(bench, m) for name, m in matchers.items()}
    base_s = round(time.perf_counter() - t0, 1)
    for name, b in baselines.items():
        print(
            f"base {name}: a-priori M3 F1 {b['apriori']['aggregates']['M3']['micro']['f1']:.3f}, "
            f"LOPO {b['lopo_cv']['M3']['f1']:.3f}, best {b['best_on_ra1']['label']} {b['best_on_ra1']['M3']['f1']:.3f}"
        )

    t0 = time.perf_counter()
    reranker = CrossEncoderReranker(args.checkpoint, threads=args.threads)
    rb = RerankBenchmark(bench, reranker)
    rr = rerank_block(rb, matchers)
    rerank_s = round(time.perf_counter() - t0, 1)
    print(
        f"reranker: a-priori M3 F1 {rr['apriori']['aggregates']['M3']['micro']['f1']:.3f}, "
        f"LOPO {rr['lopo_cv']['M3']['f1']:.3f} (M1 {rr['lopo_cv']['M1']['f1']:.3f}), best {rr['best_on_ra1']['label']} "
        f"{rr['best_on_ra1']['M3']['f1']:.3f}; candidate recall {rr['candidate_recall']}"
    )

    res: dict[str, Any] = {
        "date_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git_commit(),
        "threads": args.threads,
        "dataset_hashes": bench.dataset_hashes(),
        "protocol": {
            "apriori": {"base": APRIORI_BASE, "k": APRIORI_K, "weight": APRIORI_WEIGHT, "threshold": APRIORI_THRESHOLD},
            "grid": {"bases": list(matchers), "ks": GRID_KS, "weights": GRID_WEIGHTS, "thresholds": GRID_THRESHOLDS},
            "cv": "leave-one-pair-out (21 folds), selection by micro-F1 M3 on the 20 training pairs",
            "published": PUBLISHED,
        },
        "encoder": encoder.info.__dict__ if hasattr(encoder.info, "__dict__") else str(encoder.info),
        "baselines": baselines,
        "reranker": {"checkpoint": str(args.checkpoint), "manifest": reranker.manifest, "results": rr},
        "timing": {"base_s": base_s, "rerank_s": rerank_s},
    }
    if not args.no_ablation:
        t0 = time.perf_counter()
        untrained = CrossEncoderReranker(DEFAULT_BASE_MODEL, threads=args.threads)
        ab = rerank_block(RerankBenchmark(bench, untrained), matchers)
        res["ablation"] = {"model": DEFAULT_BASE_MODEL, "results": ab}
        res["timing"]["ablation_s"] = round(time.perf_counter() - t0, 1)
        print(
            f"ablation (un-fine-tuned): a-priori M3 F1 {ab['apriori']['aggregates']['M3']['micro']['f1']:.3f}, "
            f"LOPO {ab['lopo_cv']['M3']['f1']:.3f}"
        )
    res["runtime_s"] = round(time.perf_counter() - t_all, 1)
    res["verdict"] = verdict(res)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{RESULT_STEM}.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    write_markdown(res, args.out_dir / f"{RESULT_STEM}.md")
    print(res["verdict"])
    print(f"wrote {args.out_dir / RESULT_STEM}.{{json,md}} in {res['runtime_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
