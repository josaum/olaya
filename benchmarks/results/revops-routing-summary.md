# RevOps bowtie routing: encoder comparison (pt-BR vertical utterances)

Ontology-grounded classification of 10 hand-labeled pt-BR utterances from the
fitness-collections and revops-analytics verticals against `revops.ttl` (the
bowtie operating model: 6 RevOpsEvent leaves + 4 BowtieMetricSpec leaves),
end-to-end through the real ingestion path: Rust `convert_ontology` →
`skos:altLabel` → `synonyms` → `class_context_text` → router → scorer →
axiom projection. Zero training; every answer carries lineage and a computed
`axioms_verified`.

Reproduce:

```
python benchmarks/bench_revops_routing.py \
    --ttl <revops repo>/engines/revops/core/ontologies/revops.ttl \
    --encoder unsloth/embeddinggemma-300m
```

## Results (2026-10-09, ontology revops.ttl @ eaf3996 with pt-BR altLabels)

| encoder | params | top-1 | notes |
|---|---|---|---|
| `unsloth/embeddinggemma-300m` (= `google/embeddinggemma-300m`) | 300M | **10/10** | p_top 0.50–0.86, confidence 0.12–0.61, resolves both boundary cases |
| `intfloat/multilingual-e5-small` (`prefix="query: "`) | 118M | 8/10 | misses the two boundary cases (one near-tie, one exact tie) |
| `unsloth/gemma-2-2b` (= `google/gemma-2-2b`) mean-pooled | 2.6B | 2/10 | hub collapse: everything routes to one class per root |
| gemma-2-2b + CSLS k=3 | 2.6B | 2/10 | correction amplifies the hub: p=0.93, conf=0.80, **wrong** |

`google/*` repos are gated; the unsloth mirrors carry the same weights.

## What this establishes

1. **A causal LM is not an embedding model.** gemma-2-2b mean-pooled is not
   merely worse — it is confidently wrong, the failure mode calibration exists
   to catch. Purpose-trained embedders win at 10× fewer parameters.
2. **The binding constraint was ontology authorship**, resolved upstream:
   adding pt-BR `skos:altLabel` surface forms to the 10 routing-target classes
   (revops@eaf3996) moved e5-small from coin-flip-flat to 8/10 before any
   encoder change. Lineage-note descriptions do not route; surface forms do.
3. **embeddinggemma-300m is the encoder for this slot**: perfect score on the
   seed set including both deliberately-ambiguous boundary cases, CPU-viable
   (300M), and confidence values that separate easy from borderline
   (0.12 on the two hardest vs 0.57–0.61 on the clearest).

## Honest limits

- 10 utterances is a seed set, hand-labeled by the same person who wrote the
  altLabels — leakage-by-authorship is real. Grow with production messages
  labeled by someone who did not write the ontology.
- Both subtrees are small (6 + 4 leaves); router recall is near-vacuous at
  this size and the scorer carries the discrimination, as with every small
  ontology. The discriminative test comes with bigger TBoxes.
- The `unsloth` mirror provenance is trusted-but-unverified; for production,
  accept the license on `google/embeddinggemma-300m` and pin a revision.
