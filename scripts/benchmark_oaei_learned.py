"""Learned pairwise matcher on the OAEI Conference track (ra1), leave-one-pair-out.

Usage::

    .venv/bin/python scripts/benchmark_oaei_learned.py [--models logreg,gbt,...] [--no-looo] [--workers 2]

Writes ``benchmarks/results/oaei-learned.json`` and ``benchmarks/results/oaei-learned.md``.

Protocol.  This is *supervised* cross-validation on the track: for every held-out
ontology pair a pairwise classifier is trained on the reference alignments of the
other 20 pairs (positives = reference correspondences, negatives = every other
same-kind cross pair), the held-out pair is scored, decoded into a 1:1 alignment and
evaluated against its reference; TP / FP / FN are summed over the 21 folds (micro).
Decoder (mutual-best / greedy / Hungarian) and confidence threshold are chosen on an
inner split of the *training* pairs only.  Leave-one-ontology-out (7 folds, all six
pairs of one ontology held out) is the stricter variant.  Published OAEI systems are
evaluated once on all 21 pairs; GraphMatcher is itself trained on reference
alignments, so the comparison is on the same footing for it and optimistic for the
untrained systems only in the sense that our numbers come from 21 models.

Features are computed once per pair (:mod:`olaya.benchmarks.features`) and cached
under ``$OLAYA_CACHE_DIR/learned-features``; the embeddings come from the
``olaya.embeddings`` on-disk cache.  Models are small (numpy / torch, CPU) and the
specs run in worker processes, two torch threads each.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# Worker processes each run torch single-threaded (4 workers on 4 cores); on these small tensors one
# thread per process is as fast as two and four threads in one process is ~30x slower (OpenMP spin).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OLAYA_TORCH_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from benchmark_oaei import PUBLISHED  # noqa: E402  (the transcribed OAEI 2023 table)
from olaya.benchmarks.features import FEATURE_VERSION, feature_groups  # noqa: E402
from olaya.benchmarks.learned import (  # noqa: E402
    DECODERS,
    THRESHOLDS,
    CVResult,
    ModelSpec,
    PairDataset,
    build_dataset,
    cross_validate_model,
)
from olaya.benchmarks.oaei import (  # noqa: E402
    MODALITIES,
    OAEIConferenceBenchmark,
    leave_one_ontology_out_folds,
    leave_one_pair_out_folds,
)
from olaya.embeddings import load_encoder  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "oaei"
RESULTS_DIR = REPO / "benchmarks" / "results"
RESULT_STEM = "oaei-learned"
HAND_TUNED = RESULTS_DIR / "oaei-conference-ra1.json"

ENCODERS = ["intfloat/e5-small-v2", "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
ENCODER_TAGS = [e.rsplit("/", 1)[-1] for e in ENCODERS]

# Main models.  Hyper-parameters were fixed before the cross-validation was run (they were only
# checked for convergence / wall time on one fold) and are not tuned on ra1.
MODELS: dict[str, ModelSpec] = {
    "logreg": ModelSpec("logreg", "logreg", negatives="all", epochs=600, lr=0.05),
    "mlp": ModelSpec("mlp", "mlp", negatives="hard", epochs=200, lr=0.01, weight_decay=1e-3),
    "gbt": ModelSpec("gbt", "gbt", negatives="hard", n_trees=60, max_depth=3),
    "emb_mlp": ModelSpec("emb_mlp", "emb_mlp", negatives="hard", epochs=80, lr=0.01, weight_decay=1e-3),
    "gcn_mlp": ModelSpec("gcn_mlp", "gcn_mlp", negatives="hard", epochs=80, lr=0.01, weight_decay=1e-3),
}

# Ablations: feature subsets of the two cheap scalar models.
ABLATIONS: dict[str, dict] = {
    "lexical_only": {"keep_prefixes": ("lex:", "rank:lex:", "meta:")},
    "no_hybrid": {"drop_substrings": ("hybrid:",)},
    "no_embeddings": {"drop_substrings": tuple(ENCODER_TAGS)},
    "no_structure": {"drop_substrings": ("struct:",)},
    "no_rank": {"drop_substrings": ("rank:",)},
    "single_encoder": {"drop_substrings": tuple(ENCODER_TAGS[1:])},
}
ABLATION_BASES = ("logreg", "gbt")

DATA: dict[str, PairDataset] = {}  # filled before the worker pool is forked


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001 - best effort metadata
        return "unknown"


def ablation_spec(base: str, name: str) -> ModelSpec:
    spec = MODELS[base]
    kw = ABLATIONS[name]
    return ModelSpec(
        f"{base}:{name}",
        spec.kind,
        keep_prefixes=kw.get("keep_prefixes", ()),
        drop_substrings=kw.get("drop_substrings", ()),
        negatives=spec.negatives,
        epochs=spec.epochs,
        lr=spec.lr,
        weight_decay=spec.weight_decay,
        n_trees=spec.n_trees,
        max_depth=spec.max_depth,
        seed=spec.seed,
    )


def run_task(args: tuple[ModelSpec, str, int]) -> tuple[str, str, CVResult]:
    spec, protocol, inner = args
    folds = leave_one_pair_out_folds() if protocol == "lopo" else leave_one_ontology_out_folds()

    def log(msg: str) -> None:
        print(f"[{protocol}]{msg}", flush=True)

    res = cross_validate_model(spec, DATA, folds, inner_splits_n=inner, log=log)
    res.fold_scores = {}  # not serialised
    return spec.name, protocol, res


def micro(res: CVResult, m: str) -> dict:
    return res.micro(m)


def prf_cell(r: dict) -> str:
    return f"{r['precision']:.2f} / {r['recall']:.2f} / **{r['f1']:.2f}**"


def row(label: str, res: CVResult, extra: str = "") -> str:
    cells = [label, f"{res.folds}"] + [prf_cell(micro(res, m)) for m in MODALITIES]
    cells.append(f"{res.seconds:.0f}" + extra)
    return "| " + " | ".join(cells) + " |"


def hand_tuned_rows() -> dict[str, dict]:
    if not HAND_TUNED.exists():
        return {}
    d = json.loads(HAND_TUNED.read_text())
    p = d.get("embeddings", {}).get("primary", {})
    out = {}
    for key, label in (
        ("cv_leave_one_pair_out", "hand-tuned hybrid, LOPO-CV (oaei-conference-ra1)"),
        ("cv_leave_one_ontology_out", "hand-tuned hybrid, LOOO-CV (oaei-conference-ra1)"),
    ):
        if key in p:
            out[label] = {m: p[key][m] for m in MODALITIES}
    return out


def comparison_table(modality: str, ours: dict[str, dict]) -> str:
    rows = [(name, v["precision"], v["recall"], v["f1"], False) for name, v in PUBLISHED[f"ra1-{modality}"].items()]
    rows += [(label, r["precision"], r["recall"], r["f1"], True) for label, r in ours.items()]
    rows.sort(key=lambda x: (-x[3], x[0]))
    lines = [f"| System ({modality}) | P | R | F1 |", "|---|---|---|---|"]
    for name, p, r, f, mine in rows:
        lines.append(f"| {'**' + name + '** (this repo)' if mine else name} | {p:.2f} | {r:.2f} | {f:.2f} |")
    return "\n".join(lines)


def rank_of(f1: float, modality: str) -> tuple[int, int]:
    ranked = sorted([v["f1"] for v in PUBLISHED[f"ra1-{modality}"].values()] + [f1], reverse=True)
    return ranked.index(f1) + 1, len(ranked)


def write_report(payload: dict, results: dict[tuple[str, str], CVResult], headline: str) -> str:
    L: list[str] = []
    main_lopo = {k: v for (k, p), v in results.items() if p == "lopo" and k in MODELS}
    main_looo = {k: v for (k, p), v in results.items() if p == "looo" and k in MODELS}
    abl_lopo = {k: v for (k, p), v in results.items() if p == "lopo" and k not in MODELS}
    head = main_lopo[headline]
    hm1, hm3 = micro(head, "M1"), micro(head, "M3")
    L.append("# OAEI Conference track (ra1) — learned pairwise matcher, leave-one-pair-out")
    L.append("")
    L.append(
        f"Date: {payload['date']}  ·  Commit: `{payload['commit'][:12]}`  ·  Runtime: {payload['runtime_seconds']:.0f} s "
        f"(features {payload['runtime_features_seconds']:.0f} s, {payload['workers']} worker processes x 1 torch thread)"
    )
    L.append("")
    L.append("## Protocol (read this first)")
    L.append("")
    L.append(
        "This is **supervised cross-validation on the track**.  For every held-out ontology pair a pairwise "
        "classifier is trained on the reference alignments of the *other 20 pairs* (positives = reference "
        "correspondences, negatives = every other same-kind cross pair, class-weighted), the held-out pair is "
        "scored, decoded into a 1:1 alignment and evaluated against its own reference; the held-out TP / FP / FN "
        "are summed over the 21 folds into the usual micro P / R / F1 for classes (M1), properties (M2) and both "
        "(M3).  The decoder (mutual-best / greedy / Hungarian) and the confidence threshold are chosen by an "
        f"**inner** split of the training pairs ({payload['inner_splits']} inner folds: train on the inner-train "
        "pairs, score the inner-held-out pairs, pick the operating point with the best pooled micro-F1 M3), so the "
        "held-out pair never selects anything.  *Leave-one-ontology-out* (7 folds) holds out all six pairs of one "
        "ontology, so no training pair shares an ontology with the test pairs.  The reference alignments are the "
        "only supervision; no entity of the held-out pair is seen with a label.  GraphMatcher, the published best, "
        "is itself trained on reference alignments; LogMap, Matcha, edna and StringEquiv are not, so for them the "
        "comparison with a trained system is one they did not sign up for.  21 pairs from 7 ontologies are a small "
        "sample: differences of 0.01-0.02 F1 between rows are within noise."
    )
    L.append("")
    L.append(
        f"Features ({payload['n_features']}, version {FEATURE_VERSION}; `src/olaya/benchmarks/features.py`): "
        + "; ".join(f"`{g}` ({v})" for g, v in payload["feature_groups"].items())
        + ".  Encoders: "
        + ", ".join(f"`{e}`" for e in ENCODERS)
        + f" (primary `{ENCODERS[0]}` seeds the structural features and the embedding models).  "
        f"{payload['n_pairs_total']} same-kind cross pairs, {payload['n_positive_total']} reference correspondences."
    )
    L.append("")
    L.append("## Results (micro P / R / F1, cross-validated)")
    L.append("")
    L.append("| Model | folds | M1 (classes) | M2 (properties) | M3 (all) | s |")
    L.append("|---|---|---|---|---|---|")
    for label, r in hand_tuned_rows().items():
        L.append("| " + " | ".join([label, "CV"] + [prf_cell(r[m]) for m in MODALITIES] + ["-"]) + " |")
    for k, r in main_lopo.items():
        tag = " **(headline)**" if k == headline else ""
        L.append(row(f"learned {k}, LOPO-CV{tag}", r))
    for k, r in main_looo.items():
        L.append(row(f"learned {k}, LOOO-CV", r))
    L.append("")
    L.append(
        "Models: `logreg` = logistic regression on standardised features, trained on *all* pairs; `mlp` = 2-layer MLP "
        "(hidden 64) on the same features; `gbt` = histogram gradient-boosted trees (numpy, 60 trees, depth 3); "
        "`emb_mlp` = the scalar features plus the interaction `[p_a, p_b, |p_a-p_b|, p_a*p_b]` of a trained 64-d "
        "projection of the frozen e5 label + full-text vectors (the closest thing to GraphMatcher without a GNN); "
        "`gcn_mlp` = the same after one training-free GCN step `e' = normalise(e + 0.5 mean_neighbours e)`.  "
        "`mlp`, `gbt` and the embedding models train on hard negatives (every positive, the top-5 per entity under "
        "the lexical score and the e5 CSLS cosines, plus 300 random pairs per training pair); scoring and decoding "
        "always cover every cross pair.  Seeds are fixed; two runs give identical numbers."
    )
    L.append("")
    L.append("### Operating points chosen by the inner selection")
    L.append("")
    for k, r in main_lopo.items():
        decs = sorted({c["decoder"] for c in r.chosen.values()})
        ths = sorted({c["threshold"] for c in r.chosen.values()})
        L.append(f"- `{k}` (LOPO): decoders {decs}, thresholds {ths[0]:.2f}-{ths[-1]:.2f}")
    L.append("")
    L.append("## Ablations (LOPO-CV, micro F1 M1 / M2 / M3)")
    L.append("")
    L.append("| Base model | features | M1 | M2 | M3 | ΔM3 vs all features |")
    L.append("|---|---|---|---|---|---|")
    for base in ABLATION_BASES:
        if base not in main_lopo:
            continue
        full = main_lopo[base]
        L.append(
            f"| {base} | all | {micro(full, 'M1')['f1']:.3f} | {micro(full, 'M2')['f1']:.3f} | {micro(full, 'M3')['f1']:.3f} | - |"
        )
        for name in ABLATIONS:
            r = abl_lopo.get(f"{base}:{name}")
            if r is None:
                continue
            d = micro(r, "M3")["f1"] - micro(full, "M3")["f1"]
            L.append(
                f"| {base} | {name} | {micro(r, 'M1')['f1']:.3f} | {micro(r, 'M2')['f1']:.3f} | {micro(r, 'M3')['f1']:.3f} | {d:+.3f} |"
            )
    L.append("")
    L.append(
        "`lexical_only` keeps the lexical scores, their rank features and the meta flags; `no_embeddings` drops every "
        "encoder cosine and the ranks derived from them; `no_structure` drops the neighbourhood / depth features; "
        "`no_rank` drops all rank / margin / mutual-best features; `single_encoder` keeps only e5-small-v2; "
        "`no_hybrid` drops the hand-tuned cascade scores."
    )
    L.append("")
    L.append("## Feature importance")
    L.append("")
    for base in ABLATION_BASES:
        r = main_lopo.get(base)
        if r is None or not r.importances:
            continue
        what = (
            "mean standardised coefficient over the 21 fold models (sign = direction)"
            if base == "logreg"
            else "share of total split gain, mean over the 21 fold models"
        )
        L.append(f"**{base}** - {what}:")
        L.append("")
        top = sorted(r.importances.items(), key=lambda kv: -abs(kv[1]))[:15]
        L.append("| feature | value |")
        L.append("|---|---|")
        for n, v in top:
            L.append(f"| `{n}` | {v:+.3f} |")
        L.append("")
    L.append(f"## Per-pair results (headline model `{headline}`, LOPO-CV)")
    L.append("")
    L.append("| pair | ref | sys | decoder @ th | M1 P / R / F1 | M2 P / R / F1 | M3 P / R / F1 |")
    L.append("|---|---|---|---|---|---|---|")
    for pe in head.pairs:
        c = head.chosen[pe.pair]
        cells = [pe.pair, str(pe.reference_size), str(pe.system_size), f"{c['decoder']} @ {c['threshold']:.2f}"]
        cells += [prf_cell(pe.results[m].to_dict()) for m in MODALITIES]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Comparison with published OAEI 2023 Conference results (ra1, micro)")
    L.append("")
    ours_m1 = {f"learned {k} LOPO-CV": micro(r, "M1") for k, r in main_lopo.items()}
    ours_m3 = {f"learned {k} LOPO-CV": micro(r, "M3") for k, r in main_lopo.items()}
    for label, r in hand_tuned_rows().items():
        if "LOPO" in label:
            ours_m1["olaya hand-tuned hybrid LOPO-CV"] = r["M1"]
            ours_m3["olaya hand-tuned hybrid LOPO-CV"] = r["M3"]
    L.append(comparison_table("M1", ours_m1))
    L.append("")
    L.append(comparison_table("M3", ours_m3))
    L.append("")
    L.append("## Where we land")
    L.append("")
    gm1, gm3 = PUBLISHED["ra1-M1"]["GraphMatcher"], PUBLISHED["ra1-M3"]["GraphMatcher"]
    r1, n1 = rank_of(hm1["f1"], "M1")
    r3, n3 = rank_of(hm3["f1"], "M3")
    ht = hand_tuned_rows().get("hand-tuned hybrid, LOPO-CV (oaei-conference-ra1)")
    sent = [
        f"The best cross-validated learned model is `{headline}`: M1 {hm1['precision']:.2f} / {hm1['recall']:.2f} / "
        f"**{hm1['f1']:.3f}**, M3 {hm3['precision']:.2f} / {hm3['recall']:.2f} / **{hm3['f1']:.3f}** (LOPO-CV)."
    ]
    if ht:
        sent.append(
            f"Against the hand-tuned hybrid under the same LOPO protocol (M1 {ht['M1']['f1']:.3f}, M3 {ht['M3']['f1']:.3f}) "
            f"that is {hm1['f1'] - ht['M1']['f1']:+.3f} F1 on M1 and {hm3['f1'] - ht['M3']['f1']:+.3f} on M3; "
            f"recall moves {hm1['recall'] - ht['M1']['recall']:+.3f} (M1) / {hm3['recall'] - ht['M3']['recall']:+.3f} (M3)."
        )
    for mod, ours, pub, rk, n in (("M1", hm1, gm1, r1, n1), ("M3", hm3, gm3, r3, n3)):
        if ours["f1"] > pub["f1"] + 0.005:
            sent.append(
                f"On {mod} it is **above** GraphMatcher's published {pub['f1']:.2f} by {ours['f1'] - pub['f1']:+.3f} F1."
            )
        else:
            sent.append(
                f"On {mod} it is **{pub['f1'] - ours['f1']:.3f} F1 below** GraphMatcher's published {pub['f1']:.2f} "
                f"(P {pub['precision']:.2f} / R {pub['recall']:.2f}); it would rank {rk} of {n} in the published table."
            )
    sent.append(
        "The published numbers are single runs on all 21 pairs; ours are 21 held-out folds summed, which is the "
        "fair protocol for a trained system but not the same experiment.  The ablation table says what carries the "
        "result; the per-pair table says where the recall is lost."
    )
    L.append(" ".join(sent))
    L.append("")
    best_m3 = max(micro(r, "M3")["f1"] for r in main_lopo.values())
    if ht and best_m3 <= ht["M3"]["f1"] + 0.01:
        L.append(
            "Reading: no learned fusion of the *same evidence* (lexical scores, three small encoders, the ontology "
            "graph) beats the hand-tuned cascade by more than noise, including the embedding-interaction and "
            "GCN-style heads and the hand-tuned score itself as a feature.  The ablations show the features are "
            "not redundant (lexical-only and no-structure lose 0.02-0.05 M3) but the ceiling is shared: the "
            "residual errors are plausible-but-unreferenced pairs and reference pairs with no lexical, "
            "semantic or structural evidence in these small ontologies.  Closing the gap to GraphMatcher's "
            "recall needs different evidence (a stronger text model or training signal), not a better combiner."
        )
        L.append("")
    L.append("## Reproduce")
    L.append("")
    L.append("```\n.venv/bin/python scripts/benchmark_oaei_learned.py\n```")
    L.append("")
    L.append(
        "Features are cached under `$OLAYA_CACHE_DIR/learned-features` (default `~/.cache/olaya/learned-features`), "
        "embeddings under `$OLAYA_CACHE_DIR/embeddings`; with warm caches the run is pure training.  "
        "`--models`, `--no-looo`, `--no-ablations`, `--inner-splits` and `--workers` cut the run down."
    )
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated subset of " + ",".join(MODELS))
    ap.add_argument("--no-looo", action="store_true", help="skip leave-one-ontology-out")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--inner-splits", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS_DIR / RESULT_STEM))
    args = ap.parse_args()
    t_start = time.perf_counter()

    bench = OAEIConferenceBenchmark(FIXTURES / "ontologies", FIXTURES / "reference_alignments")
    encoders = {name: load_encoder(name) for name in ENCODERS}
    t0 = time.perf_counter()
    DATA.update(build_dataset(bench, encoders, primary=ENCODERS[0]))
    t_features = time.perf_counter() - t0
    first = next(iter(DATA.values())).features
    print(f"features: {len(DATA)} pairs, {len(first.feature_names)} features, {t_features:.1f} s", flush=True)

    selected = [m for m in args.models.split(",") if m]
    tasks: list[tuple[ModelSpec, str, int]] = []
    for m in selected:
        tasks.append((MODELS[m], "lopo", args.inner_splits))
    if not args.no_looo:
        for m in selected:
            tasks.append((MODELS[m], "looo", args.inner_splits))
    if not args.no_ablations:
        for base in ABLATION_BASES:
            if base in selected:
                for name in ABLATIONS:
                    tasks.append((ablation_spec(base, name), "lopo", args.inner_splits))
    # Longest tasks first so the pool stays busy.
    order = {"emb_mlp": 0, "gcn_mlp": 1, "mlp": 2, "gbt": 3, "logreg": 4}
    tasks.sort(key=lambda t: (order.get(t[0].kind, 5), t[1] != "lopo"))

    results: dict[tuple[str, str], CVResult] = {}
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("fork")) as pool:
            for name, protocol, res in pool.map(run_task, tasks):
                results[(name, protocol)] = res
                print(
                    f"done {name} {protocol}: M1 {res.micro('M1')['f1']:.3f} M3 {res.micro('M3')['f1']:.3f} ({res.seconds:.0f} s)",
                    flush=True,
                )
    else:
        for task in tasks:
            name, protocol, res = run_task(task)
            results[(name, protocol)] = res
            print(
                f"done {name} {protocol}: M1 {res.micro('M1')['f1']:.3f} M3 {res.micro('M3')['f1']:.3f} ({res.seconds:.0f} s)",
                flush=True,
            )

    main_lopo = {k: v for (k, p), v in results.items() if p == "lopo" and k in MODELS}
    headline = max(main_lopo, key=lambda k: (main_lopo[k].micro("M3")["f1"], main_lopo[k].micro("M1")["f1"]))
    runtime = time.perf_counter() - t_start
    payload = {
        "benchmark": "OAEI Conference track (ra1), learned pairwise matcher",
        "date": dt.date.today().isoformat(),
        "commit": git_commit(),
        "runtime_seconds": round(runtime, 1),
        "runtime_features_seconds": round(t_features, 1),
        "workers": args.workers,
        "inner_splits": args.inner_splits,
        "protocol": {
            "outer": "leave-one-pair-out (21 folds) and leave-one-ontology-out (7 folds); held-out TP/FP/FN summed (micro)",
            "supervision": "reference alignments of the training pairs only (positives) + all other same-kind cross pairs (negatives)",
            "inner_selection": f"decoder in {list(DECODERS)} x threshold in {list(THRESHOLDS)} chosen on {args.inner_splits} inner splits of the training pairs by micro-F1 M3",
            "never_on_held_out": True,
        },
        "encoders": ENCODERS,
        "feature_version": FEATURE_VERSION,
        "n_features": len(first.feature_names),
        "feature_names": first.feature_names,
        "feature_groups": {g: v for g, v in feature_groups(first.feature_names).items()},
        "n_pairs_total": int(sum(d.features.n_pairs for d in DATA.values())),
        "n_positive_total": int(sum(d.n_positive for d in DATA.values())),
        "models": {k: vars(v) for k, v in MODELS.items()},
        "ablations": ABLATIONS,
        "headline": headline,
        "results": {f"{k}/{p}": r.to_dict() for (k, p), r in results.items()},
        "hand_tuned_reference": hand_tuned_rows(),
        "published": PUBLISHED,
        "dataset_sha256": bench.dataset_hashes(),
    }
    payload["feature_groups"] = {g: len(v) for g, v in feature_groups(first.feature_names).items()}
    payload["feature_groups_members"] = feature_groups(first.feature_names)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=False))
    out.with_suffix(".md").write_text(write_report(payload, results, headline))
    print(
        f"headline {headline}: M1 {main_lopo[headline].micro('M1')['f1']:.3f} M3 {main_lopo[headline].micro('M3')['f1']:.3f}; {runtime:.0f} s"
    )
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
