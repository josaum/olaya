# Candidate routers on the LLM-distilled schemaorg corpus

Labels are LLM-distilled and filtered, **not human**. **Class-disjoint split**: every test class is unseen in training (seeded 60/15/10/15 over classes); every choice of encoder / variant is made on the validation split and test
is evaluated once per configuration. The state given to every router is the raw observation text.

- Corpus `data/corpora/schemaorg.jsonl`: 862 rows, 119 classes used of 660 ontology concepts; split rows {'train': 509, 'validation': 132, 'calibration': 94, 'test': 127}
- Test classes unseen in training: 18
- Concept tasks (gold + 5 hard negatives): calibration 92, test 124
- Commit `60c046659c83`, runtime 1280.7 s

## Candidate ceiling on the test split (gold in top-k over the full ontology)

| router | @1 | @3 | @5 | @8 | @16 | @32 | leaf ceiling (top-10 subgraph) | top-1 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| lexical (abstains on ties / no evidence) | 0.205 | 0.339 | 0.386 | 0.394 | 0.425 | 0.488 | 0.402 | 0.205 | 0.386 | 0.284 |
| bge-small cosine | 0.291 | 0.409 | 0.457 | 0.535 | 0.661 | 0.764 | 0.535 | 0.291 | 0.457 | 0.383 |
| bge-small csls | 0.268 | 0.394 | 0.457 | 0.504 | 0.622 | 0.701 | 0.520 | 0.268 | 0.457 | 0.362 |
| bge-small hybrid-union | 0.205 | 0.417 | 0.512 | 0.543 | 0.614 | 0.732 | 0.528 | 0.205 | 0.512 | 0.345 |
| bge-small hybrid-rrf | 0.307 | 0.402 | 0.441 | 0.472 | 0.535 | 0.693 | 0.488 | 0.307 | 0.441 | 0.380 |
| bge-base cosine | 0.276 | 0.465 | 0.551 | 0.661 | 0.764 | 0.858 | 0.661 | 0.276 | 0.551 | 0.406 |
| bge-base csls | 0.283 | 0.441 | 0.575 | 0.606 | 0.717 | 0.772 | 0.622 | 0.283 | 0.575 | 0.398 |
| bge-base hybrid-union | 0.205 | 0.394 | 0.535 | 0.614 | 0.732 | 0.803 | 0.614 | 0.205 | 0.535 | 0.353 |
| bge-base hybrid-rrf | 0.283 | 0.402 | 0.465 | 0.496 | 0.559 | 0.709 | 0.480 | 0.283 | 0.465 | 0.372 |
| fine-tuned bge-small cosine | 0.417 | 0.630 | 0.732 | 0.780 | 0.866 | 0.929 | 0.764 | 0.417 | 0.732 | 0.548 |
| fine-tuned bge-small csls | 0.417 | 0.646 | 0.748 | 0.787 | 0.874 | 0.953 | 0.748 | 0.417 | 0.748 | 0.551 |
| fine-tuned bge-small hybrid-union | 0.205 | 0.535 | 0.646 | 0.748 | 0.811 | 0.890 | 0.732 | 0.205 | 0.646 | 0.416 |
| fine-tuned bge-small hybrid-rrf | 0.370 | 0.472 | 0.504 | 0.543 | 0.685 | 0.843 | 0.543 | 0.370 | 0.504 | 0.450 |

Validation-split top-1 (cosine) per encoder, the selection signal: bge-small 0.326, bge-base 0.295; lexical 0.091; fine-tuned 0.568.

## Concept task on the test split (gold + 5 hard negatives; temperature fitted on the calibration split)

| scorer | temperature | top-1 | top-3 | ECE raw -> scaled | NLL scaled | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical score | 0.3350 | 0.548 | 0.815 | 0.200 -> 0.086 | 1.123 | 0.40 / 0.900 |
| bge-small cosine | 0.0177 | 0.806 | 0.968 | 0.194 -> 0.066 | 0.654 | 0.80 / 0.879 |
| bge-base cosine | 0.0177 | 0.855 | 0.968 | 0.283 -> 0.044 | 0.421 | 0.83 / 0.932 |
| fine-tuned bge-small | 0.0375 | 0.911 | 0.992 | 0.051 -> 0.027 | 0.219 | 0.90 / 0.964 |
| trained decision head (committed, `f2c8f4503f63`) | - | 0.179 | 0.538 | - -> 0.111 | - | - |

## Deployment path on the test split (`OLaya.system_one`, router top-10, leaves scored, axiom projection)

| router + scorer | top-1 | recall@5 | gold among scored leaves | gold in routed subgraph | axioms verified | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical router, lexical top-1 (baseline) | 0.205 | 0.386 | 0.402 | - | - | - |
| lexical router + trained head (committed) | 0.075 | 0.142 | - | - | - | - |
| lexical router + bge-small scorer | 0.307 | 0.378 | 0.378 | 0.465 | 1.000 | 0.46 / 0.517 |
| cosine router + bge-small scorer | 0.283 | 0.457 | 0.465 | 0.646 | 1.000 | 0.12 / 0.733 |
| csls router + bge-small scorer | 0.283 | 0.441 | 0.449 | 0.622 | 1.000 | 0.19 / 0.625 |
| hybrid-union router + bge-small scorer | 0.283 | 0.480 | 0.488 | 0.646 | 1.000 | 0.24 / 0.548 |
| hybrid-rrf router + bge-small scorer | 0.276 | 0.417 | 0.433 | 0.575 | 1.000 | 0.22 / 0.607 |
| lexical [routed] router + bge-small scorer | 0.315 | 0.386 | 0.386 | 0.465 | 1.000 | 0.44 / 0.536 |
| cosine [routed] router + bge-small scorer | 0.291 | 0.449 | 0.472 | 0.646 | 1.000 | 0.10 / 0.846 |
| csls [routed] router + bge-small scorer | 0.291 | 0.449 | 0.457 | 0.622 | 1.000 | 0.19 / 0.625 |
| hybrid-union [routed] router + bge-small scorer | 0.291 | 0.504 | 0.512 | 0.646 | 1.000 | 0.20 / 0.615 |
| hybrid-rrf [routed] router + bge-small scorer | 0.283 | 0.433 | 0.449 | 0.575 | 1.000 | 0.19 / 0.667 |
| lexical router + bge-base scorer | 0.283 | 0.378 | 0.378 | 0.465 | 1.000 | 0.39 / 0.531 |
| cosine router + bge-base scorer | 0.283 | 0.543 | 0.559 | 0.780 | 1.000 | 0.07 / 0.889 |
| csls router + bge-base scorer | 0.291 | 0.551 | 0.551 | 0.717 | 1.000 | 0.11 / 0.857 |
| hybrid-union router + bge-base scorer | 0.283 | 0.504 | 0.504 | 0.724 | 1.000 | 0.16 / 0.700 |
| hybrid-rrf router + bge-base scorer | 0.236 | 0.449 | 0.449 | 0.583 | 1.000 | 0.17 / 0.682 |
| lexical [routed] router + bge-base scorer | 0.283 | 0.386 | 0.386 | 0.465 | 1.000 | 0.37 / 0.553 |
| cosine [routed] router + bge-base scorer | 0.276 | 0.551 | 0.567 | 0.780 | 1.000 | 0.07 / 0.889 |
| csls [routed] router + bge-base scorer | 0.283 | 0.551 | 0.551 | 0.717 | 1.000 | 0.11 / 0.857 |
| hybrid-union [routed] router + bge-base scorer | 0.276 | 0.528 | 0.528 | 0.724 | 1.000 | 0.15 / 0.684 |
| hybrid-rrf [routed] router + bge-base scorer | 0.220 | 0.457 | 0.457 | 0.583 | 1.000 | 0.16 / 0.750 |
| lexical router + fine-tuned bge-small scorer | 0.362 | 0.378 | 0.378 | 0.465 | 1.000 | 0.63 / 0.475 |
| cosine router + fine-tuned bge-small scorer | 0.417 | 0.701 | 0.701 | 0.882 | 1.000 | 0.17 / 0.857 |
| csls router + fine-tuned bge-small scorer | 0.417 | 0.709 | 0.724 | 0.866 | 1.000 | 0.18 / 0.826 |
| hybrid-union router + fine-tuned bge-small scorer | 0.417 | 0.622 | 0.630 | 0.843 | 1.000 | 0.26 / 0.727 |
| hybrid-rrf router + fine-tuned bge-small scorer | 0.386 | 0.496 | 0.504 | 0.638 | 1.000 | 0.34 / 0.605 |
| lexical [routed] router + fine-tuned bge-small scorer | 0.370 | 0.386 | 0.386 | 0.465 | 1.000 | 0.59 / 0.480 |
| cosine [routed] router + fine-tuned bge-small scorer | 0.417 | 0.724 | 0.732 | 0.882 | 1.000 | 0.15 / 0.895 |
| csls [routed] router + fine-tuned bge-small scorer | 0.417 | 0.740 | 0.780 | 0.866 | 1.000 | 0.16 / 0.900 |
| hybrid-union [routed] router + fine-tuned bge-small scorer | 0.417 | 0.654 | 0.661 | 0.843 | 1.000 | 0.20 / 0.880 |
| hybrid-rrf [routed] router + fine-tuned bge-small scorer | 0.370 | 0.512 | 0.520 | 0.638 | 1.000 | 0.30 / 0.632 |

## Selection (validation split only)

- default_small: {"encoder": "bge-small", "variant": "cosine", "validation_top1": 0.32575757575757575}
- best_zero_shot: {"encoder": "bge-small", "variant": "cosine", "validation_top1": 0.32575757575757575}
- best_hybrid: {"encoder": "bge-base", "variant": "hybrid-union", "validation_ceiling_at_8": 0.6363636363636364}

## Fine-tuned bi-encoder

- Base `BAAI/bge-small-en-v1.5`, 509 train pairs, 3 epochs in 936.9 s (budget 15.0 min), best epoch 2
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query
- Checkpoint `models/checkpoints/olaya_embed_schemaorg.pt` sha256 `d85404c0400020f2` (133506663 bytes)
- Validation top-1: fine-tuned 0.568 vs zero-shot base 0.326 vs lexical 0.091 -> recorded in MANIFEST.json
- History: epoch 1 loss 0.946 val 0.545; epoch 2 loss 0.307 val 0.568; epoch 3 loss 0.135 val 0.553

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out whole classes: no test class has a training row, so the fine-tuned numbers measure transfer to unseen classes, with fewer rows per split and the class-text side never adapted.
- Zero-shot encoders never saw the corpus; their temperature is the only fitted parameter (calibration split).
- The concept-task metrics have the gold candidate present (5 hard negatives); the full-ontology and engine-path numbers are the deployment comparison.
- The state passed to every router is the raw observation text; the decision head reports used a JSON state wrapper for the engine path.
