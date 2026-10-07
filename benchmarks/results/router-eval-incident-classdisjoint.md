# Candidate routers on the LLM-distilled incident corpus

Labels are LLM-distilled and filtered, **not human**. **Class-disjoint split**: every test class is unseen in training (seeded 60/15/10/15 over classes); every choice of encoder / variant is made on the validation split and test
is evaluated once per configuration. The state given to every router is the raw observation text.

- Corpus `data/corpora/incident.jsonl`: 109 rows, 13 classes used of 13 ontology concepts; split rows {'train': 61, 'validation': 14, 'calibration': 11, 'test': 23}
- Test classes unseen in training: 2
- Concept tasks (gold + 5 hard negatives): calibration 11, test 23
- Commit `5244e2d48933`, runtime 216.2 s

## Candidate ceiling on the test split (gold in top-k over the full ontology)

| router | @1 | @3 | @5 | @8 | @16 | @32 | leaf ceiling (top-10 subgraph) | top-1 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| lexical (abstains on ties / no evidence) | 0.478 | 0.739 | 0.826 | 0.870 | 0.870 | 0.870 | 0.870 | 0.478 | 0.826 | 0.623 |
| bge-small cosine | 0.609 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.609 | 1.000 | 0.783 |
| bge-small csls | 0.783 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.783 | 1.000 | 0.884 |
| bge-small hybrid-union | 0.478 | 0.870 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.478 | 1.000 | 0.692 |
| bge-small hybrid-rrf | 0.739 | 0.870 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.739 | 0.957 | 0.818 |
| bge-base cosine | 0.826 | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.826 | 1.000 | 0.891 |
| bge-base csls | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.913 | 1.000 | 0.949 |
| bge-base hybrid-union | 0.478 | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.478 | 1.000 | 0.710 |
| bge-base hybrid-rrf | 0.783 | 0.870 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.783 | 0.957 | 0.840 |
| fine-tuned bge-small cosine | 0.696 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.696 | 1.000 | 0.833 |
| fine-tuned bge-small csls | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.913 | 1.000 | 0.957 |
| fine-tuned bge-small hybrid-union | 0.478 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.478 | 1.000 | 0.714 |
| fine-tuned bge-small hybrid-rrf | 0.696 | 0.913 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.696 | 0.957 | 0.800 |

Validation-split top-1 (cosine) per encoder, the selection signal: bge-small 0.714, bge-base 0.857; lexical 0.714; fine-tuned 0.714.

## Concept task on the test split (gold + 5 hard negatives; temperature fitted on the calibration split)

| scorer | temperature | top-1 | top-3 | ECE raw -> scaled | NLL scaled | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical score | 0.5012 | 0.783 | 0.913 | 0.282 -> 0.217 | 0.615 | 0.61 / 1.000 |
| bge-small cosine | 0.0141 | 0.696 | 1.000 | 0.142 -> 0.203 | 0.481 | 0.83 / 0.789 |
| bge-base cosine | 0.0089 | 0.870 | 1.000 | 0.313 -> 0.109 | 0.221 | 0.87 / 1.000 |
| fine-tuned bge-small | 0.0265 | 0.957 | 1.000 | 0.323 -> 0.153 | 0.294 | 0.70 / 1.000 |
| trained decision head (committed, `f8271b8ce947`) | - | 0.438 | 0.750 | - -> 0.261 | - | - |

## Deployment path on the test split (`OLaya.system_one`, router top-10, leaves scored, axiom projection)

| router + scorer | top-1 | recall@5 | gold among scored leaves | gold in routed subgraph | axioms verified | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical router, lexical top-1 (baseline) | 0.478 | 0.826 | 0.870 | - | - | - |
| lexical router + trained head (committed) | 0.188 | 0.688 | - | - | - | - |
| lexical router + bge-small scorer | 0.652 | 0.870 | 0.870 | 0.870 | 1.000 | 0.70 / 0.938 |
| cosine router + bge-small scorer | 0.609 | 1.000 | 1.000 | 1.000 | 1.000 | 0.61 / 0.929 |
| csls router + bge-small scorer | 0.609 | 1.000 | 1.000 | 1.000 | 1.000 | 0.61 / 0.929 |
| hybrid-union router + bge-small scorer | 0.609 | 1.000 | 1.000 | 1.000 | 1.000 | 0.61 / 0.929 |
| hybrid-rrf router + bge-small scorer | 0.565 | 0.957 | 0.957 | 1.000 | 1.000 | 0.57 / 0.923 |
| lexical [routed] router + bge-small scorer | 0.652 | 0.870 | 0.870 | 0.870 | 1.000 | 0.61 / 1.000 |
| cosine [routed] router + bge-small scorer | 0.609 | 1.000 | 1.000 | 1.000 | 1.000 | 0.57 / 0.923 |
| csls [routed] router + bge-small scorer | 0.609 | 1.000 | 1.000 | 1.000 | 1.000 | 0.57 / 0.923 |
| hybrid-union [routed] router + bge-small scorer | 0.609 | 1.000 | 1.000 | 1.000 | 1.000 | 0.57 / 0.923 |
| hybrid-rrf [routed] router + bge-small scorer | 0.565 | 0.957 | 0.957 | 1.000 | 1.000 | 0.52 / 0.917 |
| lexical router + bge-base scorer | 0.783 | 0.870 | 0.870 | 0.870 | 1.000 | 0.87 / 0.900 |
| cosine router + bge-base scorer | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 0.87 / 0.950 |
| csls router + bge-base scorer | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 0.87 / 0.950 |
| hybrid-union router + bge-base scorer | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 0.87 / 0.950 |
| hybrid-rrf router + bge-base scorer | 0.783 | 0.957 | 0.957 | 1.000 | 1.000 | 0.87 / 0.900 |
| lexical [routed] router + bge-base scorer | 0.783 | 0.870 | 0.870 | 0.870 | 1.000 | 0.87 / 0.900 |
| cosine [routed] router + bge-base scorer | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 0.78 / 0.944 |
| csls [routed] router + bge-base scorer | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 0.78 / 0.944 |
| hybrid-union [routed] router + bge-base scorer | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 0.83 / 0.947 |
| hybrid-rrf [routed] router + bge-base scorer | 0.783 | 0.957 | 0.957 | 1.000 | 1.000 | 0.83 / 0.895 |
| lexical router + fine-tuned bge-small scorer | 0.826 | 0.870 | 0.870 | 0.870 | 1.000 | 0.65 / 1.000 |
| cosine router + fine-tuned bge-small scorer | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.57 / 1.000 |
| csls router + fine-tuned bge-small scorer | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.57 / 1.000 |
| hybrid-union router + fine-tuned bge-small scorer | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.57 / 1.000 |
| hybrid-rrf router + fine-tuned bge-small scorer | 0.870 | 0.957 | 0.957 | 1.000 | 1.000 | 0.52 / 1.000 |
| lexical [routed] router + fine-tuned bge-small scorer | 0.652 | 0.870 | 0.870 | 0.870 | 1.000 | 0.57 / 1.000 |
| cosine [routed] router + fine-tuned bge-small scorer | 0.696 | 1.000 | 1.000 | 1.000 | 1.000 | 0.35 / 1.000 |
| csls [routed] router + fine-tuned bge-small scorer | 0.696 | 1.000 | 1.000 | 1.000 | 1.000 | 0.35 / 1.000 |
| hybrid-union [routed] router + fine-tuned bge-small scorer | 0.696 | 1.000 | 1.000 | 1.000 | 1.000 | 0.35 / 1.000 |
| hybrid-rrf [routed] router + fine-tuned bge-small scorer | 0.652 | 0.957 | 0.957 | 1.000 | 1.000 | 0.35 / 1.000 |

## Selection (validation split only)

- default_small: {"encoder": "bge-small", "variant": "cosine", "validation_top1": 0.7142857142857143}
- best_zero_shot: {"encoder": "bge-base", "variant": "cosine", "validation_top1": 0.8571428571428571}
- best_hybrid: {"encoder": "bge-base", "variant": "hybrid-rrf", "validation_ceiling_at_8": 1.0}

## Fine-tuned bi-encoder

- Base `BAAI/bge-small-en-v1.5`, 61 train pairs, 20 epochs in 190.0 s (budget 15.0 min), best epoch 1
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query
- Checkpoint `models/checkpoints/olaya_embed_incident.pt` sha256 `875b0ab27738cecf` (133506458 bytes)
- Validation top-1: fine-tuned 0.714 vs zero-shot base 0.714 vs lexical 0.714 -> not the best, not recorded in MANIFEST.json
- History: epoch 1 loss 1.092 val 0.714; epoch 2 loss 0.551 val 0.500; epoch 3 loss 0.328 val 0.500; epoch 4 loss 0.155 val 0.500; epoch 5 loss 0.122 val 0.500; epoch 6 loss 0.037 val 0.500; epoch 7 loss 0.026 val 0.500; epoch 8 loss 0.011 val 0.500; epoch 9 loss 0.006 val 0.500; epoch 10 loss 0.003 val 0.500; epoch 11 loss 0.002 val 0.429; epoch 12 loss 0.001 val 0.429; epoch 13 loss 0.001 val 0.429; epoch 14 loss 0.001 val 0.429; epoch 15 loss 0.001 val 0.429; epoch 16 loss 0.001 val 0.429; epoch 17 loss 0.001 val 0.429; epoch 18 loss 0.001 val 0.429; epoch 19 loss 0.000 val 0.429; epoch 20 loss 0.000 val 0.429

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out whole classes: no test class has a training row, so the fine-tuned numbers measure transfer to unseen classes, with fewer rows per split and the class-text side never adapted.
- Zero-shot encoders never saw the corpus; their temperature is the only fitted parameter (calibration split).
- The concept-task metrics have the gold candidate present (5 hard negatives); the full-ontology and engine-path numbers are the deployment comparison.
- The state passed to every router is the raw observation text; the decision head reports used a JSON state wrapper for the engine path.
