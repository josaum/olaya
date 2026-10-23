#!/usr/bin/env python3
"""RevOps bowtie routing benchmark — ontology-grounded classification of
pt-BR vertical utterances against revops.ttl, end-to-end through the real
ingestion path (Rust converter → synonyms → class texts → router → scorer).

Motivation: the mycelia GLiNER NER scores 0.000 F1 on semantic classification
(event type, metric intent) in every quantization profile — measured by
zero-shot-gateaway's gliner_bench. This benchmark measures whether an
ontology-grounded router (this repo) closes that gap, and which encoder
carries it.

Usage:
    python benchmarks/bench_revops_routing.py \
        --ttl /path/to/revops.ttl \
        --encoder intfloat/multilingual-e5-small [--prefix "query: "] \
        [--csls-k 3] [--json benchmarks/results/revops-routing-<enc>.json]

Cases are hand-labeled pt-BR utterances from the fitness-collections and
revops-analytics verticals. Two are deliberately ambiguous boundary cases
(check-in via turnstile; per-member frequency) — the honest ceiling is 10/10
but 8/10 with the two known-boundary misses is already a working system.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from olaya.embedding_router import EmbeddingRouter, EmbeddingScorer  # noqa: E402
from olaya.embeddings import TransformerEncoder  # noqa: E402
from olaya.engine import OLaya  # noqa: E402
from olaya.real_ontology_train import load_ontology_file  # noqa: E402

CASES = [
    ("Já paguei a mensalidade no PIX dia 02/04, segue o comprovante", "PaymentConfirmedEvent", "RevOpsEvent"),
    ("O membro entrou na academia às 7h e fez check-in na catraca", "MemberCheckinEvent", "RevOpsEvent"),
    ("Quero mudar meu plano do mensal para o anual", "ContractChangedEvent", "RevOpsEvent"),
    ("A catraca negou o acesso do aluno hoje", "AcessoEvent", "RevOpsEvent"),
    ("Recebi uma cobrança indevida da mensalidade", "PagamentoEvent", "RevOpsEvent"),
    ("Quero cancelar meu contrato", "ContratoEvent", "RevOpsEvent"),
    ("Qual a taxa de conversão de trial para matrícula este mês?", "ConversionMetric", "BowtieMetricSpec"),
    ("Quanto tempo leva do primeiro contato até o fechamento?", "TimeMetric", "BowtieMetricSpec"),
    ("Quantos leads entraram na etapa de qualificação?", "VolumeMetric", "BowtieMetricSpec"),
    ("Qual a frequência média semanal dos alunos?", "PerformanceMetric", "BowtieMetricSpec"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl", required=True, help="Path to revops.ttl")
    parser.add_argument("--encoder", default="intfloat/multilingual-e5-small")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--csls-k", type=int, default=None)
    parser.add_argument("--json", default=None, help="Write results JSON here")
    args = parser.parse_args()

    # The expected CURIEs below are local names, as produced by the Rust converter.
    onto = load_ontology_file(args.ttl, "revops", curie_style="local")
    encoder = TransformerEncoder(args.encoder, prefix=args.prefix)
    router = EmbeddingRouter(onto, encoder, max_candidates=12, csls_k=args.csls_k)
    engine = OLaya(onto, router=router, scorer=EmbeddingScorer(router), weights_path=False, device="cpu")

    rows = []
    hits = 0
    started = time.perf_counter()
    for text, expected, root in CASES:
        ans = engine.system_one(text, {"q": {"type": "concept", "root": root}}).answers["q"]
        top = sorted(ans.candidate_probabilities.items(), key=lambda kv: -kv[1])[:2]
        hit = ans.curie == expected
        hits += hit
        rows.append(
            {
                "text": text,
                "root": root,
                "expected": expected,
                "predicted": ans.curie,
                "hit": hit,
                "p_top": round(top[0][1], 3),
                "p_runner": round(top[1][1], 3),
                "confidence": round(ans.confidence, 3),
                "axioms_verified": ans.axioms_verified,
            }
        )
        print(
            f"{'PASS' if hit else 'FAIL'} {expected:24s} predicted={ans.curie:24s} "
            f"p={top[0][1]:.2f} runner={top[1][1]:.2f} conf={ans.confidence:.2f} axioms={ans.axioms_verified}"
        )

    elapsed = time.perf_counter() - started
    all_verified = all(r["axioms_verified"] for r in rows)
    print(f"\ntop-1 {hits}/{len(CASES)}  axioms_verified_all={all_verified}  elapsed={elapsed:.1f}s")

    if args.json:
        payload = {
            "encoder": args.encoder,
            "prefix": args.prefix,
            "csls_k": args.csls_k,
            "ttl": str(args.ttl),
            "top1": hits,
            "n": len(CASES),
            "axioms_verified_all": all_verified,
            "rows": rows,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return 0 if hits == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
