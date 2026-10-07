# Candidate routers on the LLM-distilled gs1 corpus

Labels are LLM-distilled and filtered, **not human**. **Class-disjoint split**: every test class is unseen in training (seeded 60/15/10/15 over classes); every choice of encoder / variant is made on the validation split and test
is evaluated once per configuration. The state given to every router is the raw observation text.

- Corpus `data/corpora/gs1.jsonl`: 422 rows, 63 classes used of 63 ontology concepts; split rows {'train': 249, 'validation': 71, 'calibration': 41, 'test': 61}
- Test classes unseen in training: 9
- Concept tasks (gold + 5 hard negatives): calibration 41, test 60
- Commit `2e54b06e5343`, runtime 1055.3 s

## Candidate ceiling on the test split (gold in top-k over the full ontology)

| router | @1 | @3 | @5 | @8 | @16 | @32 | leaf ceiling (top-10 subgraph) | top-1 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| lexical (abstains on ties / no evidence) | 0.131 | 0.410 | 0.426 | 0.443 | 0.492 | 0.656 | 0.459 | 0.131 | 0.426 | 0.272 |
| bge-small cosine | 0.344 | 0.525 | 0.607 | 0.721 | 0.885 | 0.984 | 0.787 | 0.344 | 0.607 | 0.476 |
| bge-small csls | 0.393 | 0.574 | 0.656 | 0.787 | 0.885 | 0.951 | 0.820 | 0.393 | 0.656 | 0.514 |
| bge-small hybrid-union | 0.131 | 0.574 | 0.656 | 0.721 | 0.852 | 0.951 | 0.770 | 0.131 | 0.656 | 0.374 |
| bge-small hybrid-rrf | 0.377 | 0.525 | 0.607 | 0.672 | 0.803 | 0.918 | 0.721 | 0.377 | 0.607 | 0.486 |
| bge-base cosine | 0.279 | 0.393 | 0.541 | 0.590 | 0.820 | 0.951 | 0.639 | 0.279 | 0.541 | 0.404 |
| bge-base csls | 0.361 | 0.557 | 0.705 | 0.787 | 0.852 | 0.951 | 0.820 | 0.361 | 0.705 | 0.509 |
| bge-base hybrid-union | 0.148 | 0.557 | 0.607 | 0.656 | 0.754 | 0.934 | 0.705 | 0.148 | 0.607 | 0.367 |
| bge-base hybrid-rrf | 0.295 | 0.459 | 0.607 | 0.623 | 0.754 | 0.934 | 0.656 | 0.295 | 0.607 | 0.426 |
| fine-tuned bge-small cosine | 0.459 | 0.738 | 0.803 | 0.885 | 0.967 | 1.000 | 0.951 | 0.459 | 0.803 | 0.619 |
| fine-tuned bge-small csls | 0.508 | 0.705 | 0.803 | 0.820 | 0.967 | 1.000 | 0.885 | 0.508 | 0.803 | 0.637 |
| fine-tuned bge-small hybrid-union | 0.148 | 0.656 | 0.770 | 0.869 | 0.918 | 0.984 | 0.902 | 0.148 | 0.770 | 0.433 |
| fine-tuned bge-small hybrid-rrf | 0.361 | 0.557 | 0.623 | 0.754 | 0.836 | 0.967 | 0.787 | 0.361 | 0.623 | 0.499 |

Validation-split top-1 (cosine) per encoder, the selection signal: bge-small 0.310, bge-base 0.282; lexical 0.282; fine-tuned 0.465.

## Concept task on the test split (gold + 5 hard negatives; temperature fitted on the calibration split)

| scorer | temperature | top-1 | top-3 | ECE raw -> scaled | NLL scaled | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical score | 0.4217 | 0.483 | 0.750 | 0.155 -> 0.150 | 1.345 | 0.37 / 0.682 |
| bge-small cosine | 0.0141 | 0.717 | 0.967 | 0.225 -> 0.164 | 1.037 | 0.75 / 0.778 |
| bge-base cosine | 0.0100 | 0.633 | 0.933 | 0.226 -> 0.215 | 1.528 | 0.80 / 0.708 |
| fine-tuned bge-small | 0.0251 | 0.817 | 1.000 | 0.106 -> 0.084 | 0.437 | 0.82 / 0.857 |
| trained decision head (committed, `c1ec2e0aa46b`) | - | 0.190 | 0.571 | - -> 0.208 | - | - |

## Deployment path on the test split (`OLaya.system_one`, router top-10, leaves scored, axiom projection)

| router + scorer | top-1 | recall@5 | gold among scored leaves | gold in routed subgraph | axioms verified | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical router, lexical top-1 (baseline) | 0.131 | 0.426 | 0.459 | - | - | - |
| lexical router + trained head (committed) | 0.141 | 0.469 | - | - | - | - |
| lexical router + bge-small scorer | 0.344 | 0.443 | 0.443 | 0.459 | 1.000 | 0.51 / 0.323 |
| cosine router + bge-small scorer | 0.344 | 0.607 | 0.672 | 0.787 | 1.000 | 0.34 / 0.524 |
| csls router + bge-small scorer | 0.328 | 0.689 | 0.738 | 0.820 | 1.000 | 0.34 / 0.524 |
| hybrid-union router + bge-small scorer | 0.344 | 0.705 | 0.705 | 0.770 | 1.000 | 0.34 / 0.524 |
| hybrid-rrf router + bge-small scorer | 0.361 | 0.656 | 0.656 | 0.721 | 1.000 | 0.36 / 0.500 |
| lexical [routed] router + bge-small scorer | 0.344 | 0.443 | 0.443 | 0.459 | 1.000 | 0.52 / 0.344 |
| cosine [routed] router + bge-small scorer | 0.344 | 0.607 | 0.672 | 0.787 | 1.000 | 0.33 / 0.550 |
| csls [routed] router + bge-small scorer | 0.328 | 0.689 | 0.738 | 0.820 | 1.000 | 0.34 / 0.524 |
| hybrid-union [routed] router + bge-small scorer | 0.344 | 0.705 | 0.705 | 0.770 | 1.000 | 0.33 / 0.550 |
| hybrid-rrf [routed] router + bge-small scorer | 0.361 | 0.656 | 0.656 | 0.721 | 1.000 | 0.34 / 0.524 |
| lexical router + bge-base scorer | 0.230 | 0.426 | 0.443 | 0.459 | 1.000 | 0.57 / 0.286 |
| cosine router + bge-base scorer | 0.279 | 0.541 | 0.607 | 0.639 | 1.000 | 0.33 / 0.550 |
| csls router + bge-base scorer | 0.262 | 0.607 | 0.738 | 0.820 | 1.000 | 0.39 / 0.458 |
| hybrid-union router + bge-base scorer | 0.279 | 0.623 | 0.639 | 0.705 | 1.000 | 0.38 / 0.478 |
| hybrid-rrf router + bge-base scorer | 0.279 | 0.574 | 0.623 | 0.656 | 1.000 | 0.38 / 0.522 |
| lexical [routed] router + bge-base scorer | 0.230 | 0.443 | 0.443 | 0.459 | 1.000 | 0.56 / 0.294 |
| cosine [routed] router + bge-base scorer | 0.279 | 0.541 | 0.590 | 0.639 | 1.000 | 0.33 / 0.550 |
| csls [routed] router + bge-base scorer | 0.262 | 0.607 | 0.738 | 0.820 | 1.000 | 0.39 / 0.458 |
| hybrid-union [routed] router + bge-base scorer | 0.279 | 0.607 | 0.623 | 0.705 | 1.000 | 0.41 / 0.480 |
| hybrid-rrf [routed] router + bge-base scorer | 0.279 | 0.574 | 0.623 | 0.656 | 1.000 | 0.39 / 0.500 |
| lexical router + fine-tuned bge-small scorer | 0.328 | 0.443 | 0.443 | 0.459 | 1.000 | 0.48 / 0.448 |
| cosine router + fine-tuned bge-small scorer | 0.475 | 0.820 | 0.902 | 0.951 | 1.000 | 0.28 / 0.706 |
| csls router + fine-tuned bge-small scorer | 0.475 | 0.787 | 0.836 | 0.885 | 1.000 | 0.26 / 0.750 |
| hybrid-union router + fine-tuned bge-small scorer | 0.459 | 0.836 | 0.836 | 0.902 | 1.000 | 0.30 / 0.722 |
| hybrid-rrf router + fine-tuned bge-small scorer | 0.443 | 0.721 | 0.738 | 0.787 | 1.000 | 0.39 / 0.667 |
| lexical [routed] router + fine-tuned bge-small scorer | 0.311 | 0.443 | 0.443 | 0.459 | 1.000 | 0.46 / 0.464 |
| cosine [routed] router + fine-tuned bge-small scorer | 0.459 | 0.803 | 0.885 | 0.951 | 1.000 | 0.28 / 0.706 |
| csls [routed] router + fine-tuned bge-small scorer | 0.459 | 0.754 | 0.820 | 0.885 | 1.000 | 0.26 / 0.750 |
| hybrid-union [routed] router + fine-tuned bge-small scorer | 0.459 | 0.820 | 0.820 | 0.902 | 1.000 | 0.30 / 0.722 |
| hybrid-rrf [routed] router + fine-tuned bge-small scorer | 0.426 | 0.705 | 0.721 | 0.787 | 1.000 | 0.41 / 0.680 |

## Selection (validation split only)

- default_small: {"encoder": "bge-small", "variant": "cosine", "validation_top1": 0.30985915492957744}
- best_zero_shot: {"encoder": "bge-small", "variant": "cosine", "validation_top1": 0.30985915492957744}
- best_hybrid: {"encoder": "bge-base", "variant": "hybrid-union", "validation_ceiling_at_8": 0.8169014084507042}

## Fine-tuned bi-encoder

- Base `BAAI/bge-small-en-v1.5`, 249 train pairs, 19 epochs in 906.2 s (budget 15.0 min), best epoch 1
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query
- Checkpoint `models/checkpoints/olaya_embed_gs1.pt` sha256 `3ba0cbb76299adc9` (133505433 bytes)
- Validation top-1: fine-tuned 0.465 vs zero-shot base 0.310 vs lexical 0.282 -> recorded in MANIFEST.json
- History: epoch 1 loss 1.069 val 0.465; epoch 2 loss 0.420 val 0.423; epoch 3 loss 0.215 val 0.423; epoch 4 loss 0.133 val 0.423; epoch 5 loss 0.075 val 0.423; epoch 6 loss 0.066 val 0.408; epoch 7 loss 0.059 val 0.394; epoch 8 loss 0.029 val 0.394; epoch 9 loss 0.018 val 0.366; epoch 10 loss 0.017 val 0.380; epoch 11 loss 0.010 val 0.408; epoch 12 loss 0.004 val 0.366; epoch 13 loss 0.003 val 0.380; epoch 14 loss 0.004 val 0.352; epoch 15 loss 0.002 val 0.380; epoch 16 loss 0.002 val 0.380; epoch 17 loss 0.002 val 0.394; epoch 18 loss 0.002 val 0.380; epoch 19 loss 0.001 val 0.380

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out whole classes: no test class has a training row, so the fine-tuned numbers measure transfer to unseen classes, with fewer rows per split and the class-text side never adapted.
- Zero-shot encoders never saw the corpus; their temperature is the only fitted parameter (calibration split).
- The concept-task metrics have the gold candidate present (5 hard negatives); the full-ontology and engine-path numbers are the deployment comparison.
- The state passed to every router is the raw observation text; the decision head reports used a JSON state wrapper for the engine path.
