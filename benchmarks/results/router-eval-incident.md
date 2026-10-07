# Candidate routers on the LLM-distilled incident corpus

Labels are LLM-distilled and filtered, **not human**. Same corpus, same batch-disjoint split and same test
rows as `decision-head-incident.md`; every choice of encoder / variant is made on the validation split and test
is evaluated once per configuration. The state given to every router is the raw observation text.

- Corpus `data/corpora/incident.jsonl`: 109 rows, 13 classes used of 13 ontology concepts; split rows {'train': 67, 'validation': 17, 'calibration': 9, 'test': 16}
- Test classes unseen in training: 1
- Concept tasks (gold + 5 hard negatives): calibration 9, test 16
- Commit `ed07ef4400a2`, runtime 215.5 s

## Candidate ceiling on the test split (gold in top-k over the full ontology)

| router | @1 | @3 | @5 | @8 | @16 | @32 | leaf ceiling (top-10 subgraph) | top-1 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| lexical (abstains on ties / no evidence) | 0.562 | 0.812 | 0.875 | 0.875 | 0.875 | 0.875 | 0.750 | 0.562 | 0.875 | 0.682 |
| e5-small cosine | 0.438 | 0.875 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.438 | 0.875 | 0.649 |
| e5-small csls | 0.438 | 0.750 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.438 | 0.875 | 0.617 |
| e5-small hybrid-union | 0.562 | 0.750 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.562 | 0.875 | 0.680 |
| e5-small hybrid-rrf | 0.625 | 0.875 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.625 | 0.875 | 0.743 |
| bge-small cosine | 0.562 | 0.875 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.562 | 0.938 | 0.716 |
| bge-small csls | 0.500 | 0.625 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.500 | 0.875 | 0.630 |
| bge-small hybrid-union | 0.562 | 0.812 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.562 | 0.938 | 0.707 |
| bge-small hybrid-rrf | 0.625 | 0.875 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.625 | 0.938 | 0.768 |
| minilm cosine | 0.562 | 0.875 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.562 | 0.938 | 0.726 |
| minilm csls | 0.562 | 0.875 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.562 | 0.875 | 0.724 |
| minilm hybrid-union | 0.562 | 0.812 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.562 | 0.938 | 0.700 |
| minilm hybrid-rrf | 0.562 | 0.875 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.562 | 0.938 | 0.736 |
| mpnet cosine | 0.625 | 0.812 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.625 | 0.938 | 0.735 |
| mpnet csls | 0.625 | 0.812 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.625 | 0.938 | 0.729 |
| mpnet hybrid-union | 0.562 | 0.688 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.562 | 0.938 | 0.691 |
| mpnet hybrid-rrf | 0.625 | 0.875 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.625 | 0.938 | 0.751 |
| bge-base cosine | 0.625 | 0.875 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.625 | 0.938 | 0.770 |
| bge-base csls | 0.562 | 0.875 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.562 | 0.938 | 0.726 |
| bge-base hybrid-union | 0.562 | 0.812 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.562 | 0.938 | 0.710 |
| bge-base hybrid-rrf | 0.688 | 0.875 | 0.938 | 0.938 | 1.000 | 1.000 | 0.812 | 0.688 | 0.938 | 0.799 |
| e5-large cosine | 0.625 | 0.875 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.625 | 0.875 | 0.743 |
| e5-large csls | 0.500 | 0.812 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.500 | 0.875 | 0.661 |
| e5-large hybrid-union | 0.562 | 0.750 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.562 | 0.875 | 0.701 |
| e5-large hybrid-rrf | 0.688 | 0.812 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.688 | 0.875 | 0.779 |
| bge-large cosine | 0.625 | 0.875 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.625 | 0.875 | 0.755 |
| bge-large csls | 0.625 | 0.875 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.625 | 0.875 | 0.754 |
| bge-large hybrid-union | 0.562 | 0.750 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.562 | 0.875 | 0.693 |
| bge-large hybrid-rrf | 0.625 | 0.875 | 0.875 | 0.938 | 1.000 | 1.000 | 0.812 | 0.625 | 0.875 | 0.755 |
| fine-tuned bge-small cosine | 0.688 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 0.812 | 0.688 | 1.000 | 0.806 |
| fine-tuned bge-small csls | 0.688 | 0.875 | 0.938 | 1.000 | 1.000 | 1.000 | 0.812 | 0.688 | 0.938 | 0.804 |
| fine-tuned bge-small hybrid-union | 0.562 | 0.875 | 0.938 | 1.000 | 1.000 | 1.000 | 0.812 | 0.562 | 0.938 | 0.745 |
| fine-tuned bge-small hybrid-rrf | 0.750 | 0.938 | 0.938 | 1.000 | 1.000 | 1.000 | 0.812 | 0.750 | 0.938 | 0.854 |

Validation-split top-1 (cosine) per encoder, the selection signal: e5-small 0.588, bge-small 0.824, minilm 0.647, mpnet 0.647, bge-base 0.941, e5-large 0.824, bge-large 0.765; lexical 0.588; fine-tuned 1.000.

## Concept task on the test split (gold + 5 hard negatives; temperature fitted on the calibration split)

| scorer | temperature | top-1 | top-3 | ECE raw -> scaled | NLL scaled | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical score | 0.0100 (fit saturated) | 0.688 | 0.875 | 0.281 -> 0.316 | 8.648 | 1.00 / 0.688 |
| e5-small cosine | 0.0089 | 0.688 | 1.000 | 0.351 -> 0.245 | 0.589 | 0.56 / 0.889 |
| bge-small cosine | 0.0005 (fit saturated) | 0.875 | 1.000 | 0.287 -> 0.125 | 3.264 | 1.00 / 0.875 |
| minilm cosine | 0.0005 (fit saturated) | 0.875 | 1.000 | 0.173 -> 0.125 | 1.157 | 1.00 / 0.875 |
| mpnet cosine | 0.0005 (fit saturated) | 0.750 | 0.938 | 0.209 -> 0.250 | 5.119 | 1.00 / 0.750 |
| bge-base cosine | 0.0005 (fit saturated) | 1.000 | 1.000 | 0.378 -> 0.000 | 0.000 | 1.00 / 1.000 |
| e5-large cosine | 0.0005 (fit saturated) | 0.750 | 1.000 | 0.396 -> 0.250 | 5.030 | 1.00 / 0.750 |
| bge-large cosine | 0.0005 (fit saturated) | 0.875 | 1.000 | 0.274 -> 0.125 | 2.095 | 1.00 / 0.875 |
| fine-tuned bge-small | 0.0005 (fit saturated) | 0.875 | 1.000 | 0.070 -> 0.125 | 2.777 | 1.00 / 0.875 |
| trained decision head (committed, `f8271b8ce947`) | - | 0.438 | 0.750 | - -> 0.261 | - | - |

## Deployment path on the test split (`OLaya.system_one`, router top-10, leaves scored, axiom projection)

| router + scorer | top-1 | recall@5 | gold among scored leaves | gold in routed subgraph | axioms verified | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical router, lexical top-1 (baseline) | 0.562 | 0.875 | 0.750 | - | - | - |
| lexical router + trained head (committed) | 0.188 | 0.688 | - | - | - | - |
| lexical router + e5-small scorer | 0.438 | 0.750 | 0.750 | 0.938 | 1.000 | 0.50 / 0.625 |
| cosine router + e5-small scorer | 0.438 | 0.750 | 0.812 | 1.000 | 1.000 | 0.38 / 0.667 |
| csls router + e5-small scorer | 0.438 | 0.750 | 0.750 | 1.000 | 1.000 | 0.38 / 0.667 |
| hybrid-union router + e5-small scorer | 0.438 | 0.750 | 0.812 | 1.000 | 1.000 | 0.38 / 0.667 |
| hybrid-rrf router + e5-small scorer | 0.438 | 0.750 | 0.812 | 1.000 | 1.000 | 0.38 / 0.667 |
| lexical router + bge-small scorer | 0.625 | 0.750 | 0.750 | 0.938 | 1.000 | 1.00 / 0.625 |
| cosine router + bge-small scorer | 0.562 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.562 |
| csls router + bge-small scorer | 0.562 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.562 |
| hybrid-union router + bge-small scorer | 0.562 | 0.750 | 0.812 | 1.000 | 1.000 | 1.00 / 0.562 |
| hybrid-rrf router + bge-small scorer | 0.562 | 0.750 | 0.812 | 1.000 | 1.000 | 1.00 / 0.562 |
| lexical router + minilm scorer | 0.562 | 0.750 | 0.750 | 0.938 | 1.000 | 1.00 / 0.562 |
| cosine router + minilm scorer | 0.562 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.562 |
| csls router + minilm scorer | 0.562 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.562 |
| hybrid-union router + minilm scorer | 0.562 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.562 |
| hybrid-rrf router + minilm scorer | 0.562 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.562 |
| lexical router + mpnet scorer | 0.625 | 0.750 | 0.750 | 0.938 | 1.000 | 1.00 / 0.625 |
| cosine router + mpnet scorer | 0.625 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.625 |
| csls router + mpnet scorer | 0.625 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.625 |
| hybrid-union router + mpnet scorer | 0.625 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.625 |
| hybrid-rrf router + mpnet scorer | 0.625 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.625 |
| lexical router + bge-base scorer | 0.688 | 0.750 | 0.750 | 0.938 | 1.000 | 1.00 / 0.688 |
| cosine router + bge-base scorer | 0.625 | 0.750 | 0.812 | 1.000 | 1.000 | 1.00 / 0.625 |
| csls router + bge-base scorer | 0.625 | 0.750 | 0.812 | 1.000 | 1.000 | 1.00 / 0.625 |
| hybrid-union router + bge-base scorer | 0.625 | 0.750 | 0.812 | 1.000 | 1.000 | 1.00 / 0.625 |
| hybrid-rrf router + bge-base scorer | 0.625 | 0.750 | 0.812 | 1.000 | 1.000 | 1.00 / 0.625 |
| lexical router + e5-large scorer | 0.562 | 0.750 | 0.750 | 0.938 | 1.000 | 1.00 / 0.562 |
| cosine router + e5-large scorer | 0.562 | 0.750 | 0.812 | 1.000 | 1.000 | 1.00 / 0.562 |
| csls router + e5-large scorer | 0.562 | 0.750 | 0.750 | 1.000 | 1.000 | 1.00 / 0.562 |
| hybrid-union router + e5-large scorer | 0.562 | 0.750 | 0.812 | 1.000 | 1.000 | 1.00 / 0.562 |
| hybrid-rrf router + e5-large scorer | 0.562 | 0.750 | 0.812 | 1.000 | 1.000 | 1.00 / 0.562 |
| lexical router + bge-large scorer | 0.625 | 0.750 | 0.750 | 0.938 | 1.000 | 1.00 / 0.625 |
| cosine router + bge-large scorer | 0.625 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.625 |
| csls router + bge-large scorer | 0.625 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.625 |
| hybrid-union router + bge-large scorer | 0.625 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.625 |
| hybrid-rrf router + bge-large scorer | 0.625 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.625 |
| lexical router + fine-tuned bge-small scorer | 0.688 | 0.750 | 0.750 | 0.938 | 1.000 | 1.00 / 0.688 |
| cosine router + fine-tuned bge-small scorer | 0.688 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.688 |
| csls router + fine-tuned bge-small scorer | 0.688 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.688 |
| hybrid-union router + fine-tuned bge-small scorer | 0.688 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.688 |
| hybrid-rrf router + fine-tuned bge-small scorer | 0.688 | 0.812 | 0.812 | 1.000 | 1.000 | 1.00 / 0.688 |

## Selection (validation split only)

- default_small: {"encoder": "bge-small", "variant": "cosine", "validation_top1": 0.8235294117647058}
- best_zero_shot: {"encoder": "bge-base", "variant": "cosine", "validation_top1": 0.9411764705882353}
- best_hybrid: {"encoder": "bge-base", "variant": "hybrid-rrf", "validation_ceiling_at_8": 1.0}

## Fine-tuned bi-encoder

- Base `BAAI/bge-small-en-v1.5`, 67 train pairs, 20 epochs in 188.5 s (budget 15.0 min), best epoch 5
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query
- Checkpoint `models/checkpoints/olaya_embed_incident.pt` sha256 `93b5027405178091` (133506458 bytes)
- Validation top-1: fine-tuned 1.000 vs zero-shot base 0.824 vs lexical 0.588 -> recorded in MANIFEST.json
- History: epoch 1 loss 0.912 val 0.941; epoch 2 loss 0.609 val 0.941; epoch 3 loss 0.314 val 0.941; epoch 4 loss 0.176 val 0.941; epoch 5 loss 0.088 val 1.000; epoch 6 loss 0.056 val 1.000; epoch 7 loss 0.033 val 1.000; epoch 8 loss 0.013 val 1.000; epoch 9 loss 0.006 val 1.000; epoch 10 loss 0.004 val 0.941; epoch 11 loss 0.003 val 0.941; epoch 12 loss 0.002 val 0.941; epoch 13 loss 0.002 val 0.941; epoch 14 loss 0.001 val 0.941; epoch 15 loss 0.001 val 0.941; epoch 16 loss 0.001 val 0.941; epoch 17 loss 0.001 val 0.941; epoch 18 loss 0.001 val 0.941; epoch 19 loss 0.001 val 0.941; epoch 20 loss 0.001 val 0.941

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out generation batches (register x seed), not classes; the fine-tuned encoder has seen almost every test class in training (see corpus.split_report).
- Zero-shot encoders never saw the corpus; their temperature is the only fitted parameter (calibration split).
- The concept-task metrics have the gold candidate present (5 hard negatives); the full-ontology and engine-path numbers are the deployment comparison.
- The state passed to every router is the raw observation text; the decision head reports used a JSON state wrapper for the engine path.
