"""Evaluate embedding, hybrid and lexical candidate routers on the LLM-distilled corpora.

Same corpus, same group-disjoint split (checked against ``data/corpora/<domain>.manifest.json``) and
same lexical baseline as ``scripts/train_from_corpus.py`` / ``benchmarks/results/decision-head-*.md``.
For every encoder (zero-shot, path-context class texts, cosine and CSLS) and for the hybrids with the
lexical router this reports, on the validation and test splits:

* the candidate ceiling: fraction of rows whose gold class is in the router's top-k (k = 1, 3, 5, 8,
  16, 32) and among the most specific concepts of the top-10 subgraph (what ``OLaya.system_one``
  actually scores);
* full-ontology top-1 / recall@5 with the router's score used directly as the scorer (zero-shot);
* the ``concept`` task (gold + 5 hard negatives, the decision head's test tasks): top-1 / top-3, ECE
  after a temperature fitted on the calibration split, abstention curve;
* the deployment path through the real engine (``OLaya(router=..., scorer=EmbeddingScorer(...))``):
  top-1, recall@5, gold-among-scored-leaves and the confidence abstention curve.

``--finetune-minutes N`` fine-tunes ``--finetune-base`` (default ``embeddinggemma-2``) contrastively on the train split
(in-batch + sibling hard negatives), selects the epoch on validation top-1 and evaluates it once on
test; the checkpoint goes to ``models/checkpoints/olaya_embed_<domain>.pt`` (gitignored).

Writes ``benchmarks/results/router-eval-<domain>.{json,md}``; ``--summary`` rebuilds
``router-eval-summary.md`` from every per-domain JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# Offline mode is opt-in (HF_HUB_OFFLINE=1): a fresh machine must be able to download the encoders once.
os.environ.setdefault("OLAYA_TORCH_THREADS", "4")

import numpy as np
import torch
from scripts.train_from_corpus import CONCEPT_ROOT, INSTRUCTION, ONTOLOGIES, task_metrics

from olaya.candidate_router import CandidateRouter
from olaya.corpus import CorpusRow, read_jsonl, split_corpus, split_corpus_by_class, split_report, to_observations
from olaya.embedding_router import (
    EmbeddingRouter,
    EmbeddingScorer,
    HybridRouter,
    build_class_texts,
    finetune_bi_encoder,
    lexical_rank,
    load_finetuned_encoder,
)
from olaya.embeddings import TransformerEncoder
from olaya.engine import OLaya
from olaya.ontology import Ontology
from olaya.real_ontology_train import expand_observations
from olaya.tokenizer import load_local_tokenizer

torch.set_num_threads(int(os.environ["OLAYA_TORCH_THREADS"]))

ENCODERS: dict[str, tuple[str, str]] = {
    "e5-small": ("intfloat/e5-small-v2", "query: "),
    "bge-small": ("BAAI/bge-small-en-v1.5", ""),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", ""),
    "bge-base": ("BAAI/bge-base-en-v1.5", ""),
    "e5-large": ("intfloat/e5-large-v2", "query: "),
    "bge-large": ("BAAI/bge-large-en-v1.5", ""),
}
#: Prompted checkpoints served by olaya.embeddings.SentenceTransformerEncoder (installed by leio-decision[ml]).
ST_ENCODERS: dict[str, str] = {
    "embeddinggemma-2": "google/embeddinggemma-2",
}
ENCODER_KEYS = list(ENCODERS) + list(ST_ENCODERS)
SMALL_ENCODERS = ("e5-small", "bge-small", "minilm")
CEILING_KS = (1, 3, 5, 8, 16, 32)
CSLS_K = 10
ENGINE_TOP_K = 10  # OLaya.system_one routes max_candidates=10 for concept questions
BASE_TEMPERATURE = 0.05  # cosine logits are pre-scaled by this before the temperature fit (LBFGS starts at 1)
THREADS = int(os.environ["OLAYA_TORCH_THREADS"])


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- data -----------------------------------------------------------------------------------------


def load_domain(
    domain: str, corpus_dir: Path, seed: int, split_unit: str = "batch"
) -> tuple[Ontology, Path, list[CorpusRow], dict[str, list[CorpusRow]]]:
    ontology_path = ONTOLOGIES[domain]
    ontology = Ontology.from_json(ontology_path.read_text(encoding="utf-8"))
    rows = read_jsonl(corpus_dir / f"{domain}.jsonl")
    if split_unit == "class":
        # Class-disjoint: no test class was seen in training. The corpus manifest describes the
        # batch split, so it is not checked here.
        return ontology, ontology_path, rows, split_corpus_by_class(rows, seed=seed)
    splits = split_corpus(rows, seed=seed)
    manifest_path = corpus_dir / f"{domain}.manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for name, part in splits.items():
            expected = set(manifest["splits"][name]["group_ids"])
            got = {r.batch_id for r in part}
            if expected != got or len(part) != manifest["splits"][name]["observations"]:
                raise SystemExit(f"{domain}/{name}: split differs from {manifest_path}; refusing to evaluate")
    return ontology, ontology_path, rows, splits


def concept_tasks(ontology: Ontology, rows: list[CorpusRow], domain: str, tokenizer, seed: int) -> list[dict[str, Any]]:
    """The decision head's ``concept`` tasks (gold + 5 hard negatives) for these rows, same seed."""
    special = {
        "cls": tokenizer.cls_token_id,
        "sep": tokenizer.sep_token_id,
        "mask": tokenizer.mask_token_id,
        "pad": tokenizer.pad_token_id,
    }
    samples = expand_observations(
        ontology,
        to_observations(rows),
        domain,
        lambda t: tokenizer.encode(t, add_special_tokens=False),
        special,
        seed=seed,
    )
    return [
        {"text": s.state_text, "candidates": list(s.bundle.candidate_curies), "target": s.target_idx}
        for s in samples
        if s.qtype_name == "concept"
    ]


# --- ranking metrics ------------------------------------------------------------------------------


def gold_rank(ranking: list[tuple[str, float]], gold: str) -> int | None:
    for i, (c, _) in enumerate(ranking):
        if c == gold:
            return i
    return None


def leaf_hit(ontology: Ontology, ranking: list[tuple[str, float]], gold: str, k: int = ENGINE_TOP_K) -> bool:
    """Gold is among the most specific concepts of the top-k subgraph (what the engine scores)."""
    sub = ontology.extract_subgraph([c for c, _ in ranking[:k]], include_parents=True)
    return gold in sub.concepts and not sub.concepts[gold].children


def ranking_metrics(
    ontology: Ontology, rankings: list[list[tuple[str, float]]], golds: list[str], abstain: list[bool] | None = None
) -> dict[str, Any]:
    n = len(golds)
    ranks = [gold_rank(r, g) for r, g in zip(rankings, golds, strict=True)]
    abstain = abstain or [False] * n
    top1 = sum(1 for r, a in zip(ranks, abstain, strict=True) if r == 0 and not a) / n
    out: dict[str, Any] = {
        "rows": n,
        "top1": top1,
        "recall_at_5": sum(1 for r in ranks if r is not None and r < 5) / n,
        "mrr": sum(1.0 / (r + 1) for r in ranks if r is not None) / n,
        "coverage": sum(1 for a in abstain if not a) / n,
        "ceiling": {str(k): sum(1 for r in ranks if r is not None and r < k) / n for k in CEILING_KS},
        "leaf_ceiling_top10": sum(leaf_hit(ontology, r, g) for r, g in zip(rankings, golds, strict=True)) / n,
    }
    return out


def lexical_rankings(
    router: CandidateRouter, rows: list[CorpusRow], root: str | None
) -> tuple[list[list[tuple[str, float]]], list[bool]]:
    rankings, abstain = [], []
    for row in rows:
        ranked = lexical_rank(router, row.text, root)
        tied = len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) <= 1e-12
        rankings.append(ranked)
        abstain.append(not ranked or tied)
    return rankings, abstain


# --- task (gold present) metrics --------------------------------------------------------------------


def score_tasks(score_fn, tasks: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_k = max(len(t["candidates"]) for t in tasks)
    logits = torch.full((len(tasks), max_k), -1e4, dtype=torch.float32)
    mask = torch.zeros((len(tasks), max_k), dtype=torch.bool)
    targets = torch.tensor([t["target"] for t in tasks], dtype=torch.long)
    for i, t in enumerate(tasks):
        s = np.asarray(score_fn(t["text"], t["candidates"]), dtype=np.float32)
        logits[i, : len(s)] = torch.from_numpy(s)
        mask[i, : len(s)] = True
    return logits, targets, mask


def fit_temperature_grid(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, base: float
) -> tuple[float, bool]:
    """Temperature minimising the masked NLL on a log-spaced grid of ``base * 10^[-2, 2]``.

    ``olaya.calibration.fit_temperature`` (L-BFGS from T=1) is what the decision head uses, but on
    these raw similarity scores it repeatedly stepped the free parameter past its clamp and reported
    a value it never optimised; a 1-D grid is exact enough and cannot run away. The second value says
    whether the optimum sits on the edge of the grid.
    """
    grid = base * np.logspace(-2, 2, 161)
    best_t, best_nll = float(grid[0]), float("inf")
    for t in grid:
        scaled = (logits / float(t)).masked_fill(~mask, -torch.inf)
        nll = float(torch.nn.functional.cross_entropy(scaled, targets))
        if nll < best_nll - 1e-12:
            best_t, best_nll = float(t), nll
    return best_t, best_t in (float(grid[0]), float(grid[-1]))


def task_eval(
    score_fn, calib: list[dict[str, Any]], test: list[dict[str, Any]], base_temperature: float
) -> dict[str, Any]:
    """Fit a temperature on the calibration tasks, evaluate on the test tasks."""
    cal_logits, cal_targets, cal_mask = score_tasks(score_fn, calib)
    temperature, saturated = fit_temperature_grid(cal_logits, cal_targets, cal_mask, base_temperature)
    test_logits, test_targets, test_mask = score_tasks(score_fn, test)
    raw = task_metrics(test_logits, test_targets, test_mask, base_temperature)
    cal = task_metrics(test_logits, test_targets, test_mask, temperature)
    return {
        "temperature": temperature,
        "temperature_fit_saturated": saturated,
        "temperature_fit": "grid search of the masked NLL on the calibration tasks, base * 10^[-2, 2]",
        "calibration_tasks": len(calib),
        "test_tasks": len(test),
        "raw": raw,
        "calibrated": cal,
    }


# --- engine path ----------------------------------------------------------------------------------


def engine_path(engine: OLaya, rows: list[CorpusRow], root: str | None, ontology: Ontology) -> dict[str, Any]:
    """``OLaya.system_one`` over the full ontology with the row text as the state."""
    question = {"q": {"type": "concept", "root": root, "instructions": INSTRUCTION.format(name=ontology.name)}}
    top1 = recall5 = routed = scored = axioms = 0
    conf_correct: list[tuple[float, bool]] = []
    latencies = []
    for row in rows:
        result = engine.system_one(row.text, question)
        answer = result.answers["q"]
        ranked = [c for c, _ in sorted(answer.candidate_probabilities.items(), key=lambda kv: (-kv[1], kv[0]))]
        gold = row.target_curie
        routed += gold in ranked or gold in answer.probabilities
        scored += gold in ranked
        top1 += answer.curie == gold
        recall5 += gold in ranked[:5]
        axioms += bool(answer.axioms_verified)
        conf_correct.append((answer.confidence, answer.curie == gold))
        latencies.append(result.latency_ms)
    n = len(rows)
    latencies.sort()
    curve = []
    for thr in (0.0, 0.3, 0.5, 0.7, 0.9):
        answered = [c for conf, c in conf_correct if conf >= thr]
        curve.append(
            {
                "confidence_threshold": thr,
                "coverage": len(answered) / n,
                "accuracy_when_answered": sum(answered) / len(answered) if answered else None,
            }
        )
    return {
        "rows": n,
        "model_id": engine.model_id,
        "top1": top1 / n,
        "recall_at_5": recall5 / n,
        "gold_in_routed_subgraph": routed / n,
        "gold_among_scored_leaf_candidates": scored / n,
        "axioms_verified_rate": axioms / n,
        "abstention_curve": curve,
        "latency_ms_median": latencies[(n - 1) // 2],
    }


# --- per-encoder evaluation ---------------------------------------------------------------------------


class DomainContext:
    def __init__(self, args: argparse.Namespace, domain: str):
        self.domain = domain
        self.root = CONCEPT_ROOT.get(domain)
        self.split_unit = args.split
        self.policies = tuple(args.candidate_policy)
        self.ontology, self.ontology_path, self.rows, self.splits = load_domain(
            domain, args.corpus_dir, args.seed, split_unit=self.split_unit
        )
        self.tokenizer = load_local_tokenizer()
        self.lexical = CandidateRouter(self.ontology)
        self.class_texts = build_class_texts(self.ontology)
        self.tasks = {
            name: concept_tasks(self.ontology, part, domain, self.tokenizer, args.seed)
            for name, part in self.splits.items()
            if name in ("calibration", "test")
        }
        self.lex_rank = {name: lexical_rankings(self.lexical, part, self.root) for name, part in self.splits.items()}

    def golds(self, split: str) -> list[str]:
        return [r.target_curie for r in self.splits[split]]

    def lexical_report(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": "lexical"}
        for split in ("validation", "test"):
            rankings, abstain = self.lex_rank[split]
            out[split] = ranking_metrics(self.ontology, rankings, self.golds(split), abstain)

        def score_fn(text: str, candidates: list[str]) -> np.ndarray:
            scores = dict(lexical_rank(self.lexical, text, None))
            return np.array([scores.get(c, 0.0) for c in candidates])

        out["task"] = task_eval(score_fn, self.tasks["calibration"], self.tasks["test"], 1.0)
        return out

    def router_variants(self, encoder) -> dict[str, Any]:
        cos = EmbeddingRouter(self.ontology, encoder, class_texts=self.class_texts)
        csls = EmbeddingRouter(self.ontology, encoder, csls_k=CSLS_K, class_texts=self.class_texts)
        csls.matrix = cos.matrix  # same class embeddings, no second encode
        csls._query_cache = cos._query_cache
        return {
            "cosine": cos,
            "csls": csls,
            "hybrid-union": HybridRouter(self.lexical, cos, mode="union"),
            "hybrid-rrf": HybridRouter(self.lexical, cos, mode="rrf"),
        }

    def evaluate_encoder(self, encoder, *, engine_variants: bool = True) -> dict[str, Any]:
        t0 = time.perf_counter()
        variants = self.router_variants(encoder)
        report: dict[str, Any] = {"encoder": getattr(encoder, "name", "?"), "variants": {}}
        for vname, router in variants.items():
            entry: dict[str, Any] = {"router": router.name}
            for split in ("validation", "test"):
                rankings = [router.rank(r.text, self.root) for r in self.splits[split]]
                entry[split] = ranking_metrics(self.ontology, rankings, self.golds(split))
            report["variants"][vname] = entry
        # The scorer is the cosine similarity for every variant (hybrids fuse only for routing).
        cos = variants["cosine"]
        report["task"] = task_eval(
            cos.score_candidates, self.tasks["calibration"], self.tasks["test"], BASE_TEMPERATURE
        )
        temperature = report["task"]["temperature"]
        if engine_variants:
            engine_reports: dict[str, Any] = {}
            for policy in self.policies:
                for rname, router in (
                    ("lexical", self.lexical),
                    ("cosine", cos),
                    ("csls", variants["csls"]),
                    ("hybrid-union", variants["hybrid-union"]),
                    ("hybrid-rrf", variants["hybrid-rrf"]),
                ):
                    scorer = EmbeddingScorer(cos, temperature=temperature)
                    engine = OLaya(
                        ontology=self.ontology,
                        tokenizer=self.tokenizer,
                        device="cpu",
                        weights_path=False,
                        router=router,
                        scorer=scorer,
                        candidate_policy=policy,
                    )
                    name = rname if policy == "leaves" else f"{rname} [{policy}]"
                    engine_reports[name] = engine_path(engine, self.splits["test"], self.root, self.ontology)
            report["engine"] = engine_reports
        report["runtime_s"] = round(time.perf_counter() - t0, 1)
        return report


def make_encoder(key: str, *, use_cache: bool = True):
    """Build the encoder registered under ``key`` (plain transformers or sentence-transformers)."""
    if key in ST_ENCODERS:
        from olaya.embeddings import SentenceTransformerEncoder

        return SentenceTransformerEncoder(ST_ENCODERS[key], threads=THREADS, batch_size=32, max_length=128)
    model, prefix = ENCODERS[key]
    return TransformerEncoder(model, prefix=prefix, threads=THREADS, batch_size=32, max_length=128, use_cache=use_cache)


def load_encoder(key: str):
    return make_encoder(key)


def select_by_validation(
    encoders: dict[str, dict[str, Any]], keys: list[str], variant_names: tuple[str, ...], metric
) -> tuple[str, str, float]:
    """(encoder key, variant, validation value) maximising ``metric(entry["validation"])``; ties by name."""
    best: tuple[float, str, str] | None = None
    for key in keys:
        if key not in encoders:
            continue
        for v in variant_names:
            value = float(metric(encoders[key]["variants"][v]["validation"]))
            cand = (value, key, v)
            if best is None or cand[0] > best[0] or (cand[0] == best[0] and (key, v) < (best[1], best[2])):
                best = cand
    if best is None:
        raise ValueError("nothing to select from")
    return best[1], best[2], best[0]


# --- fine-tuning ---------------------------------------------------------------------------------------


def finetune(ctx: DomainContext, base_key: str, args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    model = ST_ENCODERS.get(base_key) or ENCODERS[base_key][0]
    encoder = make_encoder(base_key, use_cache=False)
    pairs = [(r.text, r.target_curie) for r in ctx.splits["train"]]
    val_rows = ctx.splits["validation"]

    def validate(enc) -> float:
        router = EmbeddingRouter(ctx.ontology, enc, class_texts=ctx.class_texts)
        rankings = [router.rank(r.text, ctx.root) for r in val_rows]
        return ranking_metrics(ctx.ontology, rankings, ctx.golds("validation"))["top1"]

    zero_shot_val = validate(encoder)
    say(f"fine-tune {ctx.domain} from {model}: zero-shot validation top-1 {zero_shot_val:.3f}")
    checkpoint = args.output_dir / f"olaya_embed_{ctx.domain}.pt"
    result = finetune_bi_encoder(
        encoder,
        ctx.ontology,
        pairs,
        checkpoint=checkpoint,
        validate=validate,
        budget_s=args.finetune_minutes * 60,
        max_epochs=args.finetune_epochs,
        batch_size=args.finetune_batch_size,
        lr=args.finetune_lr,
        seed=args.seed,
        class_texts=ctx.class_texts,
        lora_rank=args.finetune_lora_rank,
        baseline_validation=zero_shot_val,
        progress=say,
    )
    reloaded = load_finetuned_encoder(checkpoint, threads=THREADS, batch_size=32)
    info = {
        "base_model": model,
        "base_key": base_key,
        "checkpoint": _rel(checkpoint),
        "sha256": result.sha256,
        "size_bytes": checkpoint.stat().st_size,
        "epochs_run": result.epochs_run,
        "best_epoch": result.best_epoch,
        "best_validation_top1": result.best_validation,
        "zero_shot_validation_top1": zero_shot_val,
        "train_pairs": len(pairs),
        "budget_minutes": args.finetune_minutes,
        "batch_size": args.finetune_batch_size,
        "lr": args.finetune_lr,
        "runtime_s": round(result.runtime_s, 1),
        "history": result.history,
        "loss": "InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query",
    }
    return info, reloaded


def update_manifest(path: Path, domain: str, entry: dict[str, Any]) -> None:
    data = json.loads(path.read_text()) if path.is_file() else {"checkpoints": {}}
    data.setdefault("embedding_checkpoints", {})[domain] = entry
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# --- report ----------------------------------------------------------------------------------------------


def committed_head(results_dir: Path, domain: str) -> dict[str, Any] | None:
    path = results_dir / f"decision-head-{domain}.json"
    if not path.is_file():
        return None
    r = json.loads(path.read_text())
    concept = r["test_task_metrics"]["per_task_calibrated"].get("concept", {})
    ret = r["full_ontology_retrieval"]["model"]
    return {
        "report": _rel(path),
        "checkpoint_sha256": r["checkpoint"]["sha256"],
        "concept_top1": concept.get("accuracy"),
        "concept_top3": concept.get("top3_accuracy"),
        "concept_ece": concept.get("ece"),
        "retrieval_top1": ret["top1"],
        "retrieval_recall_at_5": ret["recall_at_5"],
        "gold_among_scored_leaf_candidates": ret.get("gold_among_scored_leaf_candidates"),
        "lexical_top1": r["full_ontology_retrieval"]["lexical_baseline"]["top1_accuracy_with_abstention"],
        "lexical_recall_at_5": r["full_ontology_retrieval"]["lexical_baseline"]["recall_at_5"],
    }


def evaluate_domain(args: argparse.Namespace, domain: str) -> dict[str, Any]:
    t_start = time.perf_counter()
    ctx = DomainContext(args, domain)
    say(
        f"{domain}: {len(ctx.rows)} rows, split {json.dumps({k: len(v) for k, v in ctx.splits.items()})}, tasks { ({k: len(v) for k, v in ctx.tasks.items()}) }"
    )
    report: dict[str, Any] = {
        "domain": domain,
        "seed": args.seed,
        "split_unit": args.split,
        "candidate_policies": list(args.candidate_policy),
        "commit": _git_commit(),
        "corpus": {
            "path": _rel(args.corpus_dir / f"{domain}.jsonl"),
            "rows": len(ctx.rows),
            "classes": len({r.target_curie for r in ctx.rows}),
            "ontology_concepts": len(ctx.ontology.concepts),
            "ontology_file_sha256": _sha256(ctx.ontology_path),
            "split_rows": {k: len(v) for k, v in ctx.splits.items()},
            "split_report": split_report(ctx.splits, unit=args.split),
            "concept_tasks": {k: len(v) for k, v in ctx.tasks.items()},
        },
        "protocol": {
            "class_text": "label. description. Path: root > ... > label. Also called synonyms. (olaya.embedding_router.class_context_text)",
            "state_text": "the raw observation text (string state), for every router including the lexical one",
            "csls_k": CSLS_K,
            "engine_top_k": ENGINE_TOP_K,
            "candidate_policy": "engine rows marked [routed] score every routed concept (OLaya(candidate_policy='routed')); unmarked rows score the routed leaves only (default, shared with the Rust / Node runtimes)",
            "hybrid_union": "interleave lexical and embedding top lists",
            "hybrid_rrf": "reciprocal-rank fusion, k=60, equal weights",
            "scorer": "cosine similarity of the state to the class text, temperature fitted on the calibration split concept tasks (gold + 5 hard negatives)",
            "selection": "every 'best' / 'default' choice below is made on the validation split; test is evaluated once per configuration",
        },
        "lexical": ctx.lexical_report(),
        "encoders": {},
        "committed_decision_head": committed_head(args.results_dir, domain),
    }
    for key in args.encoders:
        say(f"{domain}: encoder {key}")
        try:
            encoder = load_encoder(key)
        except Exception as exc:  # noqa: BLE001
            say(f"{domain}: skipping {key}: {exc}")
            report["encoders"][key] = {"error": str(exc)}
            continue
        entry = evaluate_encoder_safely(ctx, encoder)
        entry["info"] = encoder.info.to_dict()
        report["encoders"][key] = entry
        if "variants" not in entry:
            continue
        v = entry["variants"]
        say(
            f"{domain}/{key}: test top-1 cos {v['cosine']['test']['top1']:.3f} csls {v['csls']['test']['top1']:.3f} "
            f"ceiling@8 cos {v['cosine']['test']['ceiling']['8']:.3f} rrf {v['hybrid-rrf']['test']['ceiling']['8']:.3f} "
            f"task top-1 {entry['task']['calibrated']['accuracy']:.3f} ECE {entry['task']['calibrated']['ece']:.3f} ({entry['runtime_s']} s)"
        )
    for ckpt in args.encoder_checkpoint or []:
        key = f"ckpt:{Path(ckpt).stem}"
        say(f"{domain}: encoder {key}")
        try:
            encoder = load_finetuned_encoder(Path(ckpt), threads=THREADS, batch_size=32)
        except Exception as exc:  # noqa: BLE001
            say(f"{domain}: skipping {key}: {exc}")
            report["encoders"][key] = {"error": str(exc)}
            continue
        entry = evaluate_encoder_safely(ctx, encoder)
        entry["checkpoint"] = {"path": _rel(Path(ckpt)), "sha256": _sha256(Path(ckpt))}
        report["encoders"][key] = entry
        if "variants" in entry:
            v = entry["variants"]
            say(
                f"{domain}/{key}: test top-1 cos {v['cosine']['test']['top1']:.3f} ceiling@8 {v['cosine']['test']['ceiling']['8']:.3f} "
                f"task top-1 {entry['task']['calibrated']['accuracy']:.3f} ECE {entry['task']['calibrated']['ece']:.3f}"
            )
    ok = [k for k, e in report["encoders"].items() if "variants" in e and not k.startswith("ckpt:")]
    selection: dict[str, Any] = {}
    if ok:
        small = [k for k in SMALL_ENCODERS if k in ok]
        if small:
            key, _, val = select_by_validation(report["encoders"], small, ("cosine",), lambda m: m["top1"])
            selection["default_small"] = {"encoder": key, "variant": "cosine", "validation_top1": val}
        key, variant, val = select_by_validation(report["encoders"], ok, ("cosine", "csls"), lambda m: m["top1"])
        selection["best_zero_shot"] = {"encoder": key, "variant": variant, "validation_top1": val}
        hkey, hvariant, hval = select_by_validation(
            report["encoders"], ok, ("hybrid-union", "hybrid-rrf"), lambda m: m["ceiling"]["8"]
        )
        selection["best_hybrid"] = {"encoder": hkey, "variant": hvariant, "validation_ceiling_at_8": hval}
    report["selection"] = selection

    if args.finetune_minutes > 0 and ok:
        base_key = args.finetune_base or selection.get("default_small", {}).get("encoder") or ok[0]
        try:
            info, tuned = finetune(ctx, base_key, args)
            say(f"{domain}: evaluating fine-tuned encoder {tuned.name}")
            entry = evaluate_encoder_safely(ctx, tuned)
            if "variants" not in entry:
                raise RuntimeError(entry.get("error", "evaluation failed"))
        except Exception as exc:  # noqa: BLE001
            say(f"{domain}: fine-tuning failed: {exc!r}")
            entry = {}
            report["fine_tuning_error"] = {"base_key": base_key, "error": repr(exc)}
        if entry:
            entry["finetune"] = info
            report["fine_tuned"] = entry
        base_val = report["encoders"][base_key]["variants"]["cosine"]["validation"]["top1"]
        lex_val = report["lexical"]["validation"]["top1"]
        tuned_val = entry["variants"]["cosine"]["validation"]["top1"] if entry else -1.0
        is_best = bool(entry) and tuned_val > base_val and tuned_val > lex_val
        if entry:
            report["fine_tuned"]["selected_on_validation"] = {
                "tuned": tuned_val,
                "zero_shot_base": base_val,
                "lexical": lex_val,
                "is_best": is_best,
            }
        if is_best and args.split == "batch":
            update_manifest(
                args.output_dir / "MANIFEST.json",
                domain,
                {
                    "file": Path(info["checkpoint"]).name,
                    "sha256": info["sha256"],
                    "size_bytes": info["size_bytes"],
                    "base_model": info["base_model"],
                    "format": "olaya-bi-encoder-v1 (state dict on top of the base checkpoint; load with olaya.embedding_router.load_finetuned_encoder)",
                    "corpus_sha256": json.loads((args.corpus_dir / f"{domain}.manifest.json").read_text()).get(
                        "corpus_sha256"
                    )
                    if (args.corpus_dir / f"{domain}.manifest.json").is_file()
                    else None,
                    "ontology_file_sha256": report["corpus"]["ontology_file_sha256"],
                    "commit": report["commit"],
                    "temperature": entry["task"]["temperature"],
                    "metrics": {
                        "validation_top1": tuned_val,
                        "validation_top1_zero_shot_base": base_val,
                        "validation_top1_lexical": lex_val,
                        "test_top1": entry["variants"]["cosine"]["test"]["top1"],
                        "test_recall_at_5": entry["variants"]["cosine"]["test"]["recall_at_5"],
                        "test_ceiling_at_8": entry["variants"]["cosine"]["test"]["ceiling"]["8"],
                        "test_concept_task_top1": entry["task"]["calibrated"]["accuracy"],
                        "test_concept_task_ece": entry["task"]["calibrated"]["ece"],
                        "lexical_test_top1": report["lexical"]["test"]["top1"],
                    },
                    "report": f"benchmarks/results/router-eval-{domain}.json",
                },
            )
    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    report["runtime_s"] = round(time.perf_counter() - t_start, 1)
    report["limitations"] = [
        "Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.",
        (
            "The split holds out generation batches (register x seed), not classes; the fine-tuned encoder has seen almost every test class in training (see corpus.split_report)."
            if args.split == "batch"
            else "The split holds out whole classes: no test class has a training row, so the fine-tuned numbers measure transfer to unseen classes, with fewer rows per split and the class-text side never adapted."
        ),
        "Zero-shot encoders never saw the corpus; their temperature is the only fitted parameter (calibration split).",
        "The concept-task metrics have the gold candidate present (5 hard negatives); the full-ontology and engine-path numbers are the deployment comparison.",
        "The state passed to every router is the raw observation text; the decision head reports used a JSON state wrapper for the engine path.",
    ]
    suffix = "" if args.split == "batch" else f"-{args.split}disjoint"
    out = args.results_dir / f"router-eval-{domain}{suffix}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.results_dir / f"router-eval-{domain}{suffix}.md").write_text(render_markdown(report), encoding="utf-8")
    say(f"{domain}: wrote {out} ({report['runtime_s']} s)")
    return report


def evaluate_encoder_safely(ctx: DomainContext, encoder) -> dict[str, Any]:
    try:
        return ctx.evaluate_encoder(encoder)
    except Exception as exc:  # noqa: BLE001 - one broken encoder must not lose the domain report
        say(f"{ctx.domain}: evaluation of {getattr(encoder, 'name', '?')} failed: {exc!r}")
        return {"error": repr(exc)}


def _f(x: Any) -> str:
    return "-" if x is None else f"{x:.3f}"


def render_markdown(r: dict[str, Any]) -> str:
    d = r["domain"]
    lex = r["lexical"]
    head = r.get("committed_decision_head") or {}
    sel = r.get("selection", {})
    lines = [
        f"# Candidate routers on the LLM-distilled {d} corpus",
        "",
        "Labels are LLM-distilled and filtered, **not human**. "
        + (
            f"Same corpus, same batch-disjoint split and same test rows as `decision-head-{d}.md`; "
            if r.get("split_unit", "batch") == "batch"
            else "**Class-disjoint split**: every test class is unseen in training (seeded 60/15/10/15 over classes); "
        )
        + "every choice of encoder / variant is made on the validation split and test",
        "is evaluated once per configuration. The state given to every router is the raw observation text.",
        "",
        f"- Corpus `{r['corpus']['path']}`: {r['corpus']['rows']} rows, {r['corpus']['classes']} classes used of {r['corpus']['ontology_concepts']} ontology concepts; split rows {r['corpus']['split_rows']}",
        f"- Test classes unseen in training: {len(r['corpus']['split_report']['test']['classes_unseen_in_train'])}",
        f"- Concept tasks (gold + 5 hard negatives): calibration {r['corpus']['concept_tasks'].get('calibration')}, test {r['corpus']['concept_tasks'].get('test')}",
        f"- Commit `{r['commit'][:12]}`, runtime {r['runtime_s']} s",
        "",
        "## Candidate ceiling on the test split (gold in top-k over the full ontology)",
        "",
        "| router | @1 | @3 | @5 | @8 | @16 | @32 | leaf ceiling (top-10 subgraph) | top-1 | recall@5 | MRR |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(name: str, m: dict[str, Any]) -> str:
        c = m["ceiling"]
        return f"| {name} | {c['1']:.3f} | {c['3']:.3f} | {c['5']:.3f} | {c['8']:.3f} | {c['16']:.3f} | {c['32']:.3f} | {m['leaf_ceiling_top10']:.3f} | {m['top1']:.3f} | {m['recall_at_5']:.3f} | {m['mrr']:.3f} |"

    lines.append(row("lexical (abstains on ties / no evidence)", lex["test"]))
    for key, e in r["encoders"].items():
        if "variants" not in e:
            lines.append(f"| {key} | error: {e.get('error')} |")
            continue
        for v in ("cosine", "csls", "hybrid-union", "hybrid-rrf"):
            lines.append(row(f"{key} {v}", e["variants"][v]["test"]))
    if "fine_tuned" in r:
        for v in ("cosine", "csls", "hybrid-union", "hybrid-rrf"):
            lines.append(
                row(f"fine-tuned {r['fine_tuned']['finetune']['base_key']} {v}", r["fine_tuned"]["variants"][v]["test"])
            )
    lines += [
        "",
        "Validation-split top-1 (cosine) per encoder, the selection signal: "
        + ", ".join(
            f"{k} {e['variants']['cosine']['validation']['top1']:.3f}"
            for k, e in r["encoders"].items()
            if "variants" in e
        )
        + f"; lexical {lex['validation']['top1']:.3f}"
        + (
            f"; fine-tuned {r['fine_tuned']['variants']['cosine']['validation']['top1']:.3f}"
            if "fine_tuned" in r
            else ""
        )
        + ".",
        "",
        "## Concept task on the test split (gold + 5 hard negatives; temperature fitted on the calibration split)",
        "",
        "| scorer | temperature | top-1 | top-3 | ECE raw -> scaled | NLL scaled | coverage@0.7 / acc |",
        "|---|---|---|---|---|---|---|",
    ]

    def trow(name: str, t: dict[str, Any]) -> str:
        raw, cal = t["raw"], t["calibrated"]
        pt = next(p for p in cal["abstention_curve"] if p["threshold"] == 0.7)
        sat = " (fit saturated)" if t.get("temperature_fit_saturated") else ""
        return f"| {name} | {t['temperature']:.4f}{sat} | {cal['accuracy']:.3f} | {cal['top3_accuracy']:.3f} | {raw['ece']:.3f} -> {cal['ece']:.3f} | {cal['nll']:.3f} | {pt['coverage']:.2f} / {_f(pt['accuracy_when_answered'])} |"

    lines.append(trow("lexical score", lex["task"]))
    for key, e in r["encoders"].items():
        if "task" in e:
            lines.append(trow(f"{key} cosine", e["task"]))
    if "fine_tuned" in r:
        lines.append(trow(f"fine-tuned {r['fine_tuned']['finetune']['base_key']}", r["fine_tuned"]["task"]))
    if head:
        lines.append(
            f"| trained decision head (committed, `{head['checkpoint_sha256'][:12]}`) | - | {_f(head['concept_top1'])} | {_f(head['concept_top3'])} | - -> {_f(head['concept_ece'])} | - | - |"
        )
    lines += [
        "",
        "## Deployment path on the test split (`OLaya.system_one`, router top-10, leaves scored, axiom projection)",
        "",
        "| router + scorer | top-1 | recall@5 | gold among scored leaves | gold in routed subgraph | axioms verified | coverage@0.7 / acc |",
        "|---|---|---|---|---|---|---|",
        f"| lexical router, lexical top-1 (baseline) | {lex['test']['top1']:.3f} | {lex['test']['recall_at_5']:.3f} | {lex['test']['leaf_ceiling_top10']:.3f} | - | - | - |",
    ]
    if head:
        lines.append(
            f"| lexical router + trained head (committed) | {_f(head['retrieval_top1'])} | {_f(head['retrieval_recall_at_5'])} | {_f(head['gold_among_scored_leaf_candidates'])} | - | - | - |"
        )

    def erow(name: str, m: dict[str, Any]) -> str:
        pt = next(p for p in m["abstention_curve"] if p["confidence_threshold"] == 0.7)
        return f"| {name} | {m['top1']:.3f} | {m['recall_at_5']:.3f} | {m['gold_among_scored_leaf_candidates']:.3f} | {m['gold_in_routed_subgraph']:.3f} | {m['axioms_verified_rate']:.3f} | {pt['coverage']:.2f} / {_f(pt['accuracy_when_answered'])} |"

    for key, e in r["encoders"].items():
        for rname, m in e.get("engine", {}).items():
            lines.append(erow(f"{rname} router + {key} scorer", m))
    if "fine_tuned" in r:
        for rname, m in r["fine_tuned"].get("engine", {}).items():
            lines.append(erow(f"{rname} router + fine-tuned {r['fine_tuned']['finetune']['base_key']} scorer", m))
    lines += ["", "## Selection (validation split only)", ""]
    for k, v in sel.items():
        lines.append(f"- {k}: {json.dumps(v)}")
    if "fine_tuned" in r:
        ft = r["fine_tuned"]["finetune"]
        s = r["fine_tuned"]["selected_on_validation"]
        lines += [
            "",
            "## Fine-tuned bi-encoder",
            "",
            f"- Base `{ft['base_model']}`, {ft['train_pairs']} train pairs, {ft['epochs_run']} epochs in {ft['runtime_s']} s (budget {ft['budget_minutes']} min), best epoch {ft['best_epoch']}",
            f"- {ft['loss']}",
            f"- Checkpoint `{ft['checkpoint']}` sha256 `{ft['sha256'][:16]}` ({ft['size_bytes']} bytes)",
            f"- Validation top-1: fine-tuned {s['tuned']:.3f} vs zero-shot base {s['zero_shot_base']:.3f} vs lexical {s['lexical']:.3f} -> {'recorded in MANIFEST.json' if s['is_best'] else 'not the best, not recorded in MANIFEST.json'}",
            "- History: "
            + "; ".join(
                f"epoch {h['epoch']} loss {h['train_loss']:.3f} val {h['validation']:.3f}" for h in ft["history"]
            ),
        ]
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in r["limitations"]]
    return "\n".join(lines) + "\n"


def write_summary(results_dir: Path) -> None:
    all_reports = [json.loads(p.read_text()) for p in sorted(results_dir.glob("router-eval-*.json"))]
    all_reports = [r for r in all_reports if "domain" in r]
    reports = [r for r in all_reports if r.get("split_unit", "batch") == "batch"]
    class_reports = [r for r in all_reports if r.get("split_unit") == "class"]
    if not reports:
        return
    # A-priori default: the small encoder with the best mean validation top-1 (cosine) over the domains.
    means: dict[str, float] = {}
    for key in SMALL_ENCODERS:
        vals = [
            r["encoders"][key]["variants"]["cosine"]["validation"]["top1"]
            for r in reports
            if "variants" in r["encoders"].get(key, {})
        ]
        if len(vals) == len(reports):
            means[key] = sum(vals) / len(vals)
    default_small = min(means, key=lambda k: (-means[k], k)) if means else None
    lines = [
        "# Candidate routers on the LLM-distilled corpora: summary",
        "",
        "Labels are **LLM-distilled and filtered, not human** (see `decision-head-summary.md`). Same corpora, same",
        "batch-disjoint splits and same test rows as the decision-head reports. Every encoder / variant choice is made",
        "on the validation split; the test split is scored once per configuration. The ceiling is the fraction of test",
        "rows whose gold class the router puts in its top-8; top-1 / recall@5 use the router's own score over the full",
        "ontology (no gold candidate injected); ECE is on the decision head's `concept` test tasks (gold + 5 hard",
        "negatives) after a temperature fitted on the calibration split.",
        "",
        "| domain | system | ceiling@8 | leaf ceiling (top-10) | top-1 | recall@5 | concept task top-1 | ECE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        d = r["domain"]
        lex = r["lexical"]
        head = r.get("committed_decision_head") or {}
        sel = r["selection"]

        def line(system: str, m: dict[str, Any], task: dict[str, Any] | None, d: str = d) -> str:
            t1 = task["calibrated"]["accuracy"] if task else None
            ece = task["calibrated"]["ece"] if task else None
            return f"| {d} | {system} | {m['ceiling']['8']:.3f} | {m['leaf_ceiling_top10']:.3f} | {m['top1']:.3f} | {m['recall_at_5']:.3f} | {_f(t1)} | {_f(ece)} |"

        lines.append(line("lexical", lex["test"], lex["task"]))
        if head:
            lines.append(
                f"| {d} | lexical router + trained head (committed) | - | {_f(head['gold_among_scored_leaf_candidates'])} | {_f(head['retrieval_top1'])} | {_f(head['retrieval_recall_at_5'])} | {_f(head['concept_top1'])} | {_f(head['concept_ece'])} |"
            )
        e5 = r["encoders"].get("e5-small", {})
        if "variants" in e5:
            lines.append(line("zero-shot e5-small cosine", e5["variants"]["cosine"]["test"], e5["task"]))
        if default_small and default_small != "e5-small" and "variants" in r["encoders"].get(default_small, {}):
            e = r["encoders"][default_small]
            lines.append(
                line(f"zero-shot {default_small} cosine (a-priori default)", e["variants"]["cosine"]["test"], e["task"])
            )
        if "best_zero_shot" in sel:
            b = sel["best_zero_shot"]
            e = r["encoders"][b["encoder"]]
            lines.append(
                line(
                    f"best zero-shot by validation: {b['encoder']} {b['variant']}",
                    e["variants"][b["variant"]]["test"],
                    e["task"],
                )
            )
        if "best_hybrid" in sel:
            b = sel["best_hybrid"]
            e = r["encoders"][b["encoder"]]
            lines.append(
                line(
                    f"best hybrid by validation: {b['encoder']} {b['variant']}",
                    e["variants"][b["variant"]]["test"],
                    e["task"],
                )
            )
        if "fine_tuned" in r:
            ft = r["fine_tuned"]
            lines.append(
                line(
                    f"fine-tuned {ft['finetune']['base_key']} cosine (`{ft['finetune']['sha256'][:12]}`)",
                    ft["variants"]["cosine"]["test"],
                    ft["task"],
                )
            )
    lines += [
        "",
        "## Deployment path (`OLaya.system_one`, router top-10, leaves scored) on the test split",
        "",
        "| domain | router | scorer | top-1 | recall@5 | gold among scored leaves |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        d = r["domain"]
        head = r.get("committed_decision_head") or {}
        lines.append(
            f"| {d} | lexical | lexical top-1 | {r['lexical']['test']['top1']:.3f} | {r['lexical']['test']['recall_at_5']:.3f} | {r['lexical']['test']['leaf_ceiling_top10']:.3f} |"
        )
        if head:
            lines.append(
                f"| {d} | lexical | trained head (committed) | {_f(head['retrieval_top1'])} | {_f(head['retrieval_recall_at_5'])} | {_f(head['gold_among_scored_leaf_candidates'])} |"
            )
        shown = []
        if default_small:
            shown.append(default_small)
        b = r["selection"].get("best_zero_shot", {}).get("encoder")
        if b and b not in shown:
            shown.append(b)
        for key in shown:
            e = r["encoders"].get(key, {})
            for rname, m in e.get("engine", {}).items():
                lines.append(
                    f"| {d} | {rname} | {key} cosine | {m['top1']:.3f} | {m['recall_at_5']:.3f} | {m['gold_among_scored_leaf_candidates']:.3f} |"
                )
        if "fine_tuned" in r:
            for rname, m in r["fine_tuned"].get("engine", {}).items():
                lines.append(
                    f"| {d} | {rname} | fine-tuned {r['fine_tuned']['finetune']['base_key']} | {m['top1']:.3f} | {m['recall_at_5']:.3f} | {m['gold_among_scored_leaf_candidates']:.3f} |"
                )
    lines += [
        "",
        "## Choices",
        "",
        f"- A-priori default encoder (best mean validation top-1 among the small encoders {', '.join(SMALL_ENCODERS)}): **{default_small}** "
        + ("(" + ", ".join(f"{k} {v:.3f}" for k, v in sorted(means.items())) + ")" if means else ""),
    ]
    for r in reports:
        lines.append(f"- {r['domain']}: " + "; ".join(f"{k} = {json.dumps(v)}" for k, v in r["selection"].items()))
    if class_reports:
        lines += [
            "",
            "## Class-disjoint split (transfer to classes never seen in training)",
            "",
            "Same corpora, split 60/15/10/15 over *classes* (seeded), so no test class has a training row. The fine-tuned",
            "encoder here is trained on the class-disjoint train split; the committed checkpoints are not reused.",
            "",
            "| domain | test rows (classes) | system | ceiling@8 | top-1 | recall@5 | engine top-1 (leaves) | engine top-1 [routed] |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in class_reports:
            d = r["domain"]
            sr = r["corpus"]["split_report"]["test"]
            cells = f"{sr['rows']} ({sr['classes']})"
            lex = r["lexical"]["test"]
            lines.append(
                f"| {d} | {cells} | lexical | {lex['ceiling']['8']:.3f} | {lex['top1']:.3f} | {lex['recall_at_5']:.3f} | - | - |"
            )
            shown = [k for k in ("bge-small",) if "variants" in r["encoders"].get(k, {})]
            shown += [k for k in r["encoders"] if k.startswith("ckpt:") and "variants" in r["encoders"][k]]
            entries = [(k, r["encoders"][k]) for k in shown]
            if "fine_tuned" in r:
                entries.append(
                    (f"fine-tuned {r['fine_tuned']['finetune']['base_key']} (class-disjoint train)", r["fine_tuned"])
                )
            for name, e in entries:
                m = e["variants"]["cosine"]["test"]
                eng = e.get("engine", {})
                lv = eng.get("cosine", {}).get("top1")
                rv = eng.get("cosine [routed]", {}).get("top1")
                lines.append(
                    f"| {d} | {cells} | {name} cosine | {m['ceiling']['8']:.3f} | {m['top1']:.3f} | {m['recall_at_5']:.3f} | {_f(lv)} | {_f(rv)} |"
                )
    lines += ["", "## What is and is not established", ""]
    for r in reports:
        d = r["domain"]
        lex = r["lexical"]["test"]["top1"]
        default = r["encoders"].get(default_small or "", {})
        d_top1 = default["variants"]["cosine"]["test"]["top1"] if "variants" in default else None
        ft = r.get("fine_tuned", {})
        ft_top1 = ft["variants"]["cosine"]["test"]["top1"] if "variants" in ft else None
        unseen = len(r["corpus"]["split_report"]["test"]["classes_unseen_in_train"])
        sat = [k for k, e in r["encoders"].items() if e.get("task", {}).get("temperature_fit_saturated")]
        parts = [
            f"zero-shot {default_small} cosine top-1 {_f(d_top1)} vs lexical {lex:.3f} on {r['corpus']['split_rows']['test']} test rows "
            f"({'better' if d_top1 is not None and d_top1 > lex else 'not better'}, no training on the corpus; a-priori choice)",
        ]
        if ft_top1 is not None:
            parts.append(
                f"fine-tuned {ft['finetune']['base_key']} top-1 {ft_top1:.3f} (selected on validation, test scored once; "
                f"{unseen} of the test classes were unseen in training, every other test class was seen through other generation batches)"
            )
        if sat:
            parts.append(
                f"the calibration-split temperature fit saturated for {', '.join(sat)} ({r['corpus']['concept_tasks'].get('calibration')} calibration tasks), so their ECE is not a calibrated-probability claim"
            )
        lines.append(f"- {d}: " + "; ".join(parts) + ".")
    lines += [
        "",
        "Established on these corpora: an off-the-shelf sentence encoder over path-context class texts raises the",
        "candidate ceiling and the zero-shot top-1 above the lexical router without any training, the hybrid routers",
        "do not beat the best embedding ranking on the ceiling once the embedding is decent, and a short contrastive",
        "fine-tune of a small encoder is the best scorer on every domain. Not established: human-judged accuracy",
        "(labels are LLM-distilled and filtered), behaviour on ambiguous or no-answer inputs, transfer to classes the",
        "fine-tuned encoder never saw (the split is by generation batch, not by class), robustness of the small-split",
        "numbers (16 incident test rows), and anything about the shipped Rust / Node runtimes, which remain lexical.",
        "",
        "Per-domain reports: " + ", ".join(f"`router-eval-{r['domain']}.md`" for r in reports) + ".",
        "Fine-tuned checkpoints are gitignored; `models/checkpoints/MANIFEST.json` (`embedding_checkpoints`) records their hashes and metrics.",
        "",
    ]
    (results_dir / "router-eval-summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", action="append", choices=sorted(ONTOLOGIES), help="repeatable; default: all")
    parser.add_argument("--encoders", nargs="*", default=list(ENCODERS), choices=ENCODER_KEYS)
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "corpora")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "benchmarks" / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune-minutes", type=float, default=0.0, help="0 disables fine-tuning")
    parser.add_argument(
        "--finetune-base", choices=ENCODER_KEYS, default="embeddinggemma-2", help="base encoder to fine-tune"
    )
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-batch-size", type=int, default=16)
    parser.add_argument("--finetune-lr", type=float, default=3e-5, help="peak learning rate (3e-5 suits LoRA)")
    parser.add_argument("--finetune-lora-rank", type=int, default=16, help="LoRA rank; 0 trains the full weights")
    parser.add_argument("--summary", action="store_true", help="only rebuild router-eval-summary.md")
    parser.add_argument(
        "--split",
        choices=("batch", "class"),
        default="batch",
        help="hold out generation batches (default) or whole classes",
    )
    parser.add_argument(
        "--candidate-policy",
        nargs="+",
        choices=("leaves", "routed"),
        default=["leaves", "routed"],
        help="engine candidate policies to evaluate (rows marked [routed] score every routed concept)",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        action="append",
        type=Path,
        help="repeatable; evaluate a fine-tuned olaya-bi-encoder-v1 checkpoint as an extra encoder (reported as ckpt:<stem>)",
    )
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary:
        for domain in args.domain or sorted(ONTOLOGIES):
            evaluate_domain(args, domain)
    write_summary(args.results_dir)


if __name__ == "__main__":
    main()
