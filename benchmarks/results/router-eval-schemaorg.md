# Candidate routers on the LLM-distilled schemaorg corpus

Labels are LLM-distilled and filtered, **not human**. Same corpus, same batch-disjoint split and same test rows as `decision-head-schemaorg.md`; every choice of encoder / variant is made on the validation split and test
is evaluated once per configuration. The state given to every router is the raw observation text.

- Corpus `data/corpora/schemaorg.jsonl`: 862 rows, 119 classes used of 660 ontology concepts; split rows {'train': 524, 'validation': 132, 'calibration': 86, 'test': 120}
- Test classes unseen in training: 9
- Concept tasks (gold + 5 hard negatives): calibration 85, test 117
- Commit `60c046659c83`, runtime 1816.4 s

## Candidate ceiling on the test split (gold in top-k over the full ontology)

| router | @1 | @3 | @5 | @8 | @16 | @32 | leaf ceiling (top-10 subgraph) | top-1 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| lexical (abstains on ties / no evidence) | 0.100 | 0.142 | 0.175 | 0.208 | 0.258 | 0.375 | 0.225 | 0.100 | 0.175 | 0.141 |
| e5-small cosine | 0.275 | 0.425 | 0.483 | 0.617 | 0.708 | 0.800 | 0.600 | 0.275 | 0.483 | 0.384 |
| e5-small csls | 0.167 | 0.275 | 0.350 | 0.433 | 0.592 | 0.700 | 0.433 | 0.167 | 0.350 | 0.261 |
| e5-small hybrid-union | 0.100 | 0.317 | 0.408 | 0.500 | 0.633 | 0.742 | 0.525 | 0.100 | 0.408 | 0.260 |
| e5-small hybrid-rrf | 0.167 | 0.242 | 0.308 | 0.367 | 0.492 | 0.567 | 0.375 | 0.167 | 0.308 | 0.235 |
| bge-small cosine | 0.308 | 0.375 | 0.475 | 0.525 | 0.633 | 0.733 | 0.550 | 0.308 | 0.475 | 0.384 |
| bge-small csls | 0.217 | 0.300 | 0.358 | 0.458 | 0.542 | 0.617 | 0.475 | 0.217 | 0.358 | 0.298 |
| bge-small hybrid-union | 0.100 | 0.350 | 0.400 | 0.492 | 0.600 | 0.692 | 0.533 | 0.100 | 0.400 | 0.268 |
| bge-small hybrid-rrf | 0.150 | 0.233 | 0.325 | 0.367 | 0.467 | 0.592 | 0.375 | 0.150 | 0.325 | 0.228 |
| minilm cosine | 0.208 | 0.392 | 0.425 | 0.475 | 0.575 | 0.717 | 0.450 | 0.208 | 0.425 | 0.319 |
| minilm csls | 0.200 | 0.292 | 0.375 | 0.425 | 0.575 | 0.642 | 0.425 | 0.200 | 0.375 | 0.287 |
| minilm hybrid-union | 0.100 | 0.258 | 0.383 | 0.475 | 0.558 | 0.625 | 0.458 | 0.100 | 0.383 | 0.240 |
| minilm hybrid-rrf | 0.133 | 0.233 | 0.325 | 0.350 | 0.467 | 0.600 | 0.358 | 0.133 | 0.325 | 0.215 |
| mpnet cosine | 0.267 | 0.392 | 0.483 | 0.550 | 0.675 | 0.775 | 0.542 | 0.267 | 0.483 | 0.370 |
| mpnet csls | 0.267 | 0.417 | 0.450 | 0.558 | 0.667 | 0.775 | 0.558 | 0.267 | 0.450 | 0.371 |
| mpnet hybrid-union | 0.100 | 0.308 | 0.408 | 0.517 | 0.617 | 0.717 | 0.525 | 0.100 | 0.408 | 0.257 |
| mpnet hybrid-rrf | 0.158 | 0.242 | 0.325 | 0.367 | 0.467 | 0.667 | 0.400 | 0.158 | 0.325 | 0.240 |
| bge-base cosine | 0.267 | 0.392 | 0.525 | 0.583 | 0.700 | 0.783 | 0.592 | 0.267 | 0.525 | 0.380 |
| bge-base csls | 0.158 | 0.317 | 0.408 | 0.483 | 0.575 | 0.683 | 0.492 | 0.158 | 0.408 | 0.273 |
| bge-base hybrid-union | 0.100 | 0.300 | 0.400 | 0.525 | 0.642 | 0.758 | 0.550 | 0.100 | 0.400 | 0.260 |
| bge-base hybrid-rrf | 0.150 | 0.192 | 0.283 | 0.333 | 0.492 | 0.633 | 0.342 | 0.150 | 0.283 | 0.220 |
| e5-large cosine | 0.467 | 0.592 | 0.642 | 0.717 | 0.775 | 0.842 | 0.683 | 0.467 | 0.642 | 0.554 |
| e5-large csls | 0.267 | 0.483 | 0.550 | 0.625 | 0.742 | 0.808 | 0.608 | 0.267 | 0.550 | 0.401 |
| e5-large hybrid-union | 0.100 | 0.525 | 0.558 | 0.633 | 0.717 | 0.808 | 0.625 | 0.100 | 0.558 | 0.338 |
| e5-large hybrid-rrf | 0.192 | 0.275 | 0.367 | 0.392 | 0.475 | 0.675 | 0.408 | 0.192 | 0.367 | 0.269 |
| bge-large cosine | 0.375 | 0.525 | 0.600 | 0.667 | 0.750 | 0.850 | 0.667 | 0.375 | 0.600 | 0.477 |
| bge-large csls | 0.267 | 0.417 | 0.525 | 0.625 | 0.700 | 0.750 | 0.642 | 0.267 | 0.525 | 0.387 |
| bge-large hybrid-union | 0.100 | 0.392 | 0.508 | 0.575 | 0.692 | 0.792 | 0.617 | 0.100 | 0.508 | 0.298 |
| bge-large hybrid-rrf | 0.150 | 0.242 | 0.342 | 0.375 | 0.483 | 0.675 | 0.375 | 0.150 | 0.342 | 0.235 |
| fine-tuned bge-small cosine | 0.558 | 0.792 | 0.833 | 0.867 | 0.908 | 0.975 | 0.758 | 0.558 | 0.833 | 0.684 |
| fine-tuned bge-small csls | 0.600 | 0.767 | 0.833 | 0.850 | 0.925 | 0.967 | 0.775 | 0.600 | 0.833 | 0.699 |
| fine-tuned bge-small hybrid-union | 0.100 | 0.567 | 0.725 | 0.817 | 0.875 | 0.925 | 0.733 | 0.100 | 0.725 | 0.395 |
| fine-tuned bge-small hybrid-rrf | 0.217 | 0.325 | 0.425 | 0.467 | 0.592 | 0.792 | 0.450 | 0.217 | 0.425 | 0.314 |

Validation-split top-1 (cosine) per encoder, the selection signal: e5-small 0.212, bge-small 0.326, minilm 0.220, mpnet 0.348, bge-base 0.311, e5-large 0.356, bge-large 0.409; lexical 0.098; fine-tuned 0.652.

## Concept task on the test split (gold + 5 hard negatives; temperature fitted on the calibration split)

| scorer | temperature | top-1 | top-3 | ECE raw -> scaled | NLL scaled | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical score | 0.4217 | 0.419 | 0.726 | 0.172 -> 0.109 | 1.354 | 0.19 / 0.909 |
| e5-small cosine | 0.0094 | 0.803 | 0.991 | 0.469 -> 0.046 | 0.511 | 0.70 / 0.915 |
| bge-small cosine | 0.0265 | 0.778 | 1.000 | 0.203 -> 0.050 | 0.557 | 0.61 / 0.930 |
| minilm cosine | 0.0472 | 0.726 | 0.991 | 0.066 -> 0.050 | 0.683 | 0.57 / 0.910 |
| mpnet cosine | 0.0354 | 0.821 | 0.991 | 0.094 -> 0.052 | 0.519 | 0.71 / 0.904 |
| bge-base cosine | 0.0237 | 0.803 | 0.983 | 0.283 -> 0.106 | 0.506 | 0.65 / 0.961 |
| e5-large cosine | 0.0063 | 0.889 | 0.983 | 0.546 -> 0.049 | 0.311 | 0.79 / 0.946 |
| bge-large cosine | 0.0188 | 0.863 | 0.991 | 0.301 -> 0.034 | 0.374 | 0.79 / 0.935 |
| fine-tuned bge-small | 0.0354 | 0.949 | 1.000 | 0.043 -> 0.049 | 0.098 | 0.96 / 0.973 |
| trained decision head (committed, `f2c8f4503f63`) | - | 0.179 | 0.538 | - -> 0.111 | - | - |

## Deployment path on the test split (`OLaya.system_one`, router top-10, leaves scored, axiom projection)

| router + scorer | top-1 | recall@5 | gold among scored leaves | gold in routed subgraph | axioms verified | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical router, lexical top-1 (baseline) | 0.100 | 0.175 | 0.225 | - | - | - |
| lexical router + trained head (committed) | 0.075 | 0.142 | - | - | - | - |
| lexical router + e5-small scorer | 0.142 | 0.158 | 0.158 | 0.333 | 1.000 | 0.29 / 0.229 |
| cosine router + e5-small scorer | 0.275 | 0.467 | 0.475 | 0.733 | 1.000 | 0.03 / 0.333 |
| csls router + e5-small scorer | 0.267 | 0.358 | 0.358 | 0.542 | 1.000 | 0.04 / 0.400 |
| hybrid-union router + e5-small scorer | 0.283 | 0.417 | 0.417 | 0.642 | 1.000 | 0.12 / 0.500 |
| hybrid-rrf router + e5-small scorer | 0.175 | 0.317 | 0.317 | 0.475 | 1.000 | 0.07 / 0.500 |
| lexical [routed] router + e5-small scorer | 0.142 | 0.167 | 0.167 | 0.333 | 1.000 | 0.32 / 0.263 |
| cosine [routed] router + e5-small scorer | 0.275 | 0.483 | 0.492 | 0.733 | 1.000 | 0.03 / 0.333 |
| csls [routed] router + e5-small scorer | 0.250 | 0.333 | 0.333 | 0.542 | 1.000 | 0.03 / 0.333 |
| hybrid-union [routed] router + e5-small scorer | 0.275 | 0.400 | 0.400 | 0.642 | 1.000 | 0.08 / 0.600 |
| hybrid-rrf [routed] router + e5-small scorer | 0.167 | 0.317 | 0.317 | 0.475 | 1.000 | 0.06 / 0.429 |
| lexical router + bge-small scorer | 0.133 | 0.158 | 0.158 | 0.333 | 1.000 | 0.28 / 0.303 |
| cosine router + bge-small scorer | 0.300 | 0.458 | 0.475 | 0.675 | 1.000 | 0.02 / 0.500 |
| csls router + bge-small scorer | 0.267 | 0.367 | 0.367 | 0.575 | 1.000 | 0.04 / 0.800 |
| hybrid-union router + bge-small scorer | 0.300 | 0.400 | 0.400 | 0.650 | 1.000 | 0.06 / 0.429 |
| hybrid-rrf router + bge-small scorer | 0.192 | 0.317 | 0.317 | 0.467 | 1.000 | 0.04 / 0.600 |
| lexical [routed] router + bge-small scorer | 0.133 | 0.167 | 0.167 | 0.333 | 1.000 | 0.26 / 0.290 |
| cosine [routed] router + bge-small scorer | 0.308 | 0.475 | 0.483 | 0.675 | 1.000 | 0.02 / 0.500 |
| csls [routed] router + bge-small scorer | 0.275 | 0.375 | 0.375 | 0.575 | 1.000 | 0.04 / 0.800 |
| hybrid-union [routed] router + bge-small scorer | 0.308 | 0.400 | 0.400 | 0.650 | 1.000 | 0.05 / 0.500 |
| hybrid-rrf [routed] router + bge-small scorer | 0.192 | 0.325 | 0.325 | 0.467 | 1.000 | 0.03 / 0.500 |
| lexical router + minilm scorer | 0.125 | 0.158 | 0.158 | 0.333 | 1.000 | 0.33 / 0.308 |
| cosine router + minilm scorer | 0.192 | 0.392 | 0.392 | 0.608 | 1.000 | 0.03 / 0.667 |
| csls router + minilm scorer | 0.200 | 0.317 | 0.342 | 0.600 | 1.000 | 0.04 / 0.800 |
| hybrid-union router + minilm scorer | 0.192 | 0.375 | 0.375 | 0.608 | 1.000 | 0.12 / 0.600 |
| hybrid-rrf router + minilm scorer | 0.167 | 0.283 | 0.283 | 0.475 | 1.000 | 0.07 / 0.500 |
| lexical [routed] router + minilm scorer | 0.142 | 0.167 | 0.167 | 0.333 | 1.000 | 0.28 / 0.324 |
| cosine [routed] router + minilm scorer | 0.208 | 0.425 | 0.425 | 0.608 | 1.000 | 0.02 / 0.500 |
| csls [routed] router + minilm scorer | 0.217 | 0.375 | 0.392 | 0.600 | 1.000 | 0.03 / 0.750 |
| hybrid-union [routed] router + minilm scorer | 0.208 | 0.383 | 0.383 | 0.608 | 1.000 | 0.10 / 0.500 |
| hybrid-rrf [routed] router + minilm scorer | 0.167 | 0.325 | 0.325 | 0.475 | 1.000 | 0.06 / 0.429 |
| lexical router + mpnet scorer | 0.125 | 0.158 | 0.158 | 0.333 | 1.000 | 0.38 / 0.239 |
| cosine router + mpnet scorer | 0.258 | 0.458 | 0.483 | 0.692 | 1.000 | 0.14 / 0.706 |
| csls router + mpnet scorer | 0.283 | 0.467 | 0.467 | 0.708 | 1.000 | 0.16 / 0.684 |
| hybrid-union router + mpnet scorer | 0.258 | 0.417 | 0.417 | 0.658 | 1.000 | 0.23 / 0.607 |
| hybrid-rrf router + mpnet scorer | 0.183 | 0.283 | 0.292 | 0.508 | 1.000 | 0.19 / 0.435 |
| lexical [routed] router + mpnet scorer | 0.142 | 0.167 | 0.167 | 0.333 | 1.000 | 0.36 / 0.256 |
| cosine [routed] router + mpnet scorer | 0.267 | 0.483 | 0.517 | 0.692 | 1.000 | 0.08 / 0.800 |
| csls [routed] router + mpnet scorer | 0.292 | 0.475 | 0.483 | 0.708 | 1.000 | 0.09 / 0.727 |
| hybrid-union [routed] router + mpnet scorer | 0.267 | 0.417 | 0.417 | 0.658 | 1.000 | 0.17 / 0.619 |
| hybrid-rrf [routed] router + mpnet scorer | 0.192 | 0.308 | 0.317 | 0.508 | 1.000 | 0.13 / 0.500 |
| lexical router + bge-base scorer | 0.133 | 0.158 | 0.158 | 0.333 | 1.000 | 0.21 / 0.240 |
| cosine router + bge-base scorer | 0.283 | 0.508 | 0.508 | 0.758 | 1.000 | 0.00 / - |
| csls router + bge-base scorer | 0.267 | 0.408 | 0.408 | 0.600 | 1.000 | 0.01 / 1.000 |
| hybrid-union router + bge-base scorer | 0.283 | 0.417 | 0.417 | 0.683 | 1.000 | 0.03 / 0.250 |
| hybrid-rrf router + bge-base scorer | 0.167 | 0.258 | 0.258 | 0.450 | 1.000 | 0.02 / 0.500 |
| lexical [routed] router + bge-base scorer | 0.133 | 0.167 | 0.167 | 0.333 | 1.000 | 0.20 / 0.292 |
| cosine [routed] router + bge-base scorer | 0.267 | 0.525 | 0.525 | 0.758 | 1.000 | 0.00 / - |
| csls [routed] router + bge-base scorer | 0.242 | 0.408 | 0.408 | 0.600 | 1.000 | 0.01 / 1.000 |
| hybrid-union [routed] router + bge-base scorer | 0.267 | 0.408 | 0.408 | 0.683 | 1.000 | 0.02 / 0.500 |
| hybrid-rrf [routed] router + bge-base scorer | 0.167 | 0.267 | 0.267 | 0.450 | 1.000 | 0.02 / 0.500 |
| lexical router + e5-large scorer | 0.142 | 0.158 | 0.158 | 0.333 | 1.000 | 0.47 / 0.286 |
| cosine router + e5-large scorer | 0.467 | 0.625 | 0.650 | 0.833 | 1.000 | 0.22 / 0.654 |
| csls router + e5-large scorer | 0.400 | 0.517 | 0.517 | 0.733 | 1.000 | 0.29 / 0.600 |
| hybrid-union router + e5-large scorer | 0.458 | 0.567 | 0.567 | 0.758 | 1.000 | 0.30 / 0.639 |
| hybrid-rrf router + e5-large scorer | 0.275 | 0.367 | 0.375 | 0.517 | 1.000 | 0.27 / 0.469 |
| lexical [routed] router + e5-large scorer | 0.150 | 0.167 | 0.167 | 0.333 | 1.000 | 0.46 / 0.291 |
| cosine [routed] router + e5-large scorer | 0.467 | 0.642 | 0.683 | 0.833 | 1.000 | 0.20 / 0.667 |
| csls [routed] router + e5-large scorer | 0.408 | 0.517 | 0.517 | 0.733 | 1.000 | 0.29 / 0.600 |
| hybrid-union [routed] router + e5-large scorer | 0.467 | 0.567 | 0.567 | 0.758 | 1.000 | 0.31 / 0.622 |
| hybrid-rrf [routed] router + e5-large scorer | 0.275 | 0.367 | 0.367 | 0.517 | 1.000 | 0.27 / 0.469 |
| lexical router + bge-large scorer | 0.142 | 0.158 | 0.158 | 0.333 | 1.000 | 0.36 / 0.326 |
| cosine router + bge-large scorer | 0.383 | 0.592 | 0.600 | 0.817 | 1.000 | 0.07 / 0.889 |
| csls router + bge-large scorer | 0.392 | 0.517 | 0.517 | 0.758 | 1.000 | 0.12 / 0.786 |
| hybrid-union router + bge-large scorer | 0.400 | 0.508 | 0.508 | 0.742 | 1.000 | 0.13 / 0.688 |
| hybrid-rrf router + bge-large scorer | 0.200 | 0.325 | 0.325 | 0.492 | 1.000 | 0.12 / 0.400 |
| lexical [routed] router + bge-large scorer | 0.142 | 0.167 | 0.167 | 0.333 | 1.000 | 0.36 / 0.279 |
| cosine [routed] router + bge-large scorer | 0.375 | 0.600 | 0.617 | 0.817 | 1.000 | 0.07 / 1.000 |
| csls [routed] router + bge-large scorer | 0.375 | 0.508 | 0.508 | 0.758 | 1.000 | 0.10 / 0.833 |
| hybrid-union [routed] router + bge-large scorer | 0.375 | 0.500 | 0.500 | 0.742 | 1.000 | 0.12 / 0.733 |
| hybrid-rrf [routed] router + bge-large scorer | 0.183 | 0.333 | 0.342 | 0.492 | 1.000 | 0.09 / 0.455 |
| lexical router + fine-tuned bge-small scorer | 0.158 | 0.158 | 0.158 | 0.333 | 1.000 | 0.67 / 0.237 |
| cosine router + fine-tuned bge-small scorer | 0.558 | 0.733 | 0.742 | 0.933 | 1.000 | 0.32 / 0.842 |
| csls router + fine-tuned bge-small scorer | 0.567 | 0.750 | 0.767 | 0.942 | 1.000 | 0.32 / 0.842 |
| hybrid-union router + fine-tuned bge-small scorer | 0.558 | 0.708 | 0.708 | 0.908 | 1.000 | 0.44 / 0.774 |
| hybrid-rrf router + fine-tuned bge-small scorer | 0.333 | 0.392 | 0.392 | 0.567 | 1.000 | 0.43 / 0.558 |
| lexical [routed] router + fine-tuned bge-small scorer | 0.167 | 0.167 | 0.167 | 0.333 | 1.000 | 0.65 / 0.231 |
| cosine [routed] router + fine-tuned bge-small scorer | 0.558 | 0.833 | 0.850 | 0.933 | 1.000 | 0.27 / 0.906 |
| csls [routed] router + fine-tuned bge-small scorer | 0.558 | 0.833 | 0.842 | 0.942 | 1.000 | 0.27 / 0.906 |
| hybrid-union [routed] router + fine-tuned bge-small scorer | 0.558 | 0.783 | 0.783 | 0.908 | 1.000 | 0.38 / 0.826 |
| hybrid-rrf [routed] router + fine-tuned bge-small scorer | 0.333 | 0.425 | 0.425 | 0.567 | 1.000 | 0.40 / 0.562 |

## Selection (validation split only)

- default_small: {"encoder": "bge-small", "variant": "cosine", "validation_top1": 0.32575757575757575}
- best_zero_shot: {"encoder": "bge-large", "variant": "cosine", "validation_top1": 0.4090909090909091}
- best_hybrid: {"encoder": "e5-large", "variant": "hybrid-union", "validation_ceiling_at_8": 0.7803030303030303}

## Fine-tuned bi-encoder

- Base `BAAI/bge-small-en-v1.5`, 524 train pairs, 4 epochs in 931.0 s (budget 15.0 min), best epoch 4
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query
- Checkpoint `models/checkpoints/olaya_embed_schemaorg.pt` sha256 `a804f232a05d4d91` (133506663 bytes)
- Validation top-1: fine-tuned 0.652 vs zero-shot base 0.326 vs lexical 0.098 -> recorded in MANIFEST.json
- History: epoch 1 loss 0.906 val 0.583; epoch 2 loss 0.343 val 0.591; epoch 3 loss 0.171 val 0.644; epoch 4 loss 0.117 val 0.652

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out generation batches (register x seed), not classes; the fine-tuned encoder has seen almost every test class in training (see corpus.split_report).
- Zero-shot encoders never saw the corpus; their temperature is the only fitted parameter (calibration split).
- The concept-task metrics have the gold candidate present (5 hard negatives); the full-ontology and engine-path numbers are the deployment comparison.
- The state passed to every router is the raw observation text; the decision head reports used a JSON state wrapper for the engine path.
