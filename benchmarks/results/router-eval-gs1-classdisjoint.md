# Candidate routers on the LLM-distilled gs1 corpus

Labels are LLM-distilled and filtered, **not human**. **Class-disjoint split**: every test class is unseen in training (seeded 60/15/10/15 over classes); every choice of encoder / variant is made on the validation split and test
is evaluated once per configuration. The state given to every router is the raw observation text.

- Corpus `data/corpora/gs1.jsonl`: 422 rows, 63 classes used of 63 ontology concepts; split rows {'train': 249, 'validation': 71, 'calibration': 41, 'test': 61}
- Test classes unseen in training: 9
- Concept tasks (gold + 5 hard negatives): calibration 41, test 60
- Commit `feaca1a5806b`, runtime 215.7 s

## Candidate ceiling on the test split (gold in top-k over the full ontology)

| router | @1 | @3 | @5 | @8 | @16 | @32 | leaf ceiling (top-10 subgraph) | top-1 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| lexical (abstains on ties / no evidence) | 0.131 | 0.410 | 0.426 | 0.443 | 0.492 | 0.656 | 0.459 | 0.131 | 0.426 | 0.272 |
| e5-small cosine | 0.246 | 0.361 | 0.459 | 0.541 | 0.689 | 0.836 | 0.590 | 0.246 | 0.459 | 0.358 |
| e5-small csls | 0.262 | 0.328 | 0.393 | 0.525 | 0.738 | 0.902 | 0.607 | 0.262 | 0.393 | 0.354 |
| e5-small hybrid-union | 0.131 | 0.492 | 0.574 | 0.607 | 0.738 | 0.852 | 0.623 | 0.131 | 0.574 | 0.328 |
| e5-small hybrid-rrf | 0.246 | 0.344 | 0.475 | 0.541 | 0.689 | 0.852 | 0.574 | 0.246 | 0.475 | 0.351 |
| bge-small cosine | 0.344 | 0.525 | 0.607 | 0.721 | 0.885 | 0.984 | 0.787 | 0.344 | 0.607 | 0.476 |
| bge-small csls | 0.393 | 0.574 | 0.656 | 0.787 | 0.885 | 0.951 | 0.820 | 0.393 | 0.656 | 0.514 |
| bge-small hybrid-union | 0.131 | 0.574 | 0.656 | 0.721 | 0.852 | 0.951 | 0.770 | 0.131 | 0.656 | 0.374 |
| bge-small hybrid-rrf | 0.377 | 0.525 | 0.607 | 0.672 | 0.803 | 0.918 | 0.721 | 0.377 | 0.607 | 0.486 |
| minilm cosine | 0.213 | 0.475 | 0.590 | 0.705 | 0.836 | 0.951 | 0.738 | 0.213 | 0.590 | 0.388 |
| minilm csls | 0.344 | 0.557 | 0.656 | 0.803 | 0.852 | 0.967 | 0.820 | 0.344 | 0.656 | 0.485 |
| minilm hybrid-union | 0.131 | 0.475 | 0.656 | 0.672 | 0.803 | 0.918 | 0.672 | 0.131 | 0.656 | 0.335 |
| minilm hybrid-rrf | 0.262 | 0.443 | 0.541 | 0.639 | 0.820 | 0.902 | 0.656 | 0.262 | 0.541 | 0.402 |
| mpnet cosine | 0.295 | 0.443 | 0.590 | 0.705 | 0.820 | 0.951 | 0.738 | 0.295 | 0.590 | 0.426 |
| mpnet csls | 0.361 | 0.557 | 0.607 | 0.721 | 0.902 | 0.984 | 0.770 | 0.361 | 0.607 | 0.493 |
| mpnet hybrid-union | 0.164 | 0.492 | 0.607 | 0.689 | 0.836 | 0.902 | 0.721 | 0.164 | 0.607 | 0.359 |
| mpnet hybrid-rrf | 0.344 | 0.492 | 0.541 | 0.607 | 0.787 | 0.902 | 0.672 | 0.344 | 0.541 | 0.457 |
| bge-base cosine | 0.279 | 0.393 | 0.541 | 0.590 | 0.820 | 0.951 | 0.639 | 0.279 | 0.541 | 0.404 |
| bge-base csls | 0.361 | 0.557 | 0.705 | 0.787 | 0.852 | 0.951 | 0.820 | 0.361 | 0.705 | 0.509 |
| bge-base hybrid-union | 0.148 | 0.557 | 0.607 | 0.656 | 0.754 | 0.934 | 0.705 | 0.148 | 0.607 | 0.367 |
| bge-base hybrid-rrf | 0.295 | 0.459 | 0.607 | 0.623 | 0.754 | 0.934 | 0.656 | 0.295 | 0.607 | 0.426 |
| e5-large cosine | 0.295 | 0.492 | 0.574 | 0.705 | 0.820 | 0.902 | 0.754 | 0.295 | 0.574 | 0.432 |
| e5-large csls | 0.328 | 0.574 | 0.689 | 0.721 | 0.836 | 0.984 | 0.770 | 0.328 | 0.689 | 0.485 |
| e5-large hybrid-union | 0.131 | 0.590 | 0.639 | 0.689 | 0.803 | 0.885 | 0.721 | 0.131 | 0.639 | 0.371 |
| e5-large hybrid-rrf | 0.246 | 0.475 | 0.574 | 0.607 | 0.770 | 0.918 | 0.639 | 0.246 | 0.574 | 0.400 |
| bge-large cosine | 0.344 | 0.557 | 0.623 | 0.754 | 0.869 | 1.000 | 0.820 | 0.344 | 0.623 | 0.482 |
| bge-large csls | 0.410 | 0.672 | 0.754 | 0.803 | 0.951 | 0.967 | 0.869 | 0.410 | 0.754 | 0.556 |
| bge-large hybrid-union | 0.148 | 0.541 | 0.672 | 0.721 | 0.869 | 0.967 | 0.738 | 0.148 | 0.672 | 0.377 |
| bge-large hybrid-rrf | 0.311 | 0.525 | 0.557 | 0.656 | 0.820 | 0.967 | 0.672 | 0.311 | 0.557 | 0.454 |
| embeddinggemma-2 cosine | 0.410 | 0.623 | 0.656 | 0.754 | 0.869 | 0.934 | 0.803 | 0.410 | 0.656 | 0.538 |
| embeddinggemma-2 csls | 0.475 | 0.705 | 0.754 | 0.820 | 0.902 | 1.000 | 0.820 | 0.475 | 0.754 | 0.611 |
| embeddinggemma-2 hybrid-union | 0.148 | 0.639 | 0.721 | 0.754 | 0.869 | 0.918 | 0.770 | 0.148 | 0.721 | 0.412 |
| embeddinggemma-2 hybrid-rrf | 0.295 | 0.475 | 0.623 | 0.672 | 0.787 | 0.934 | 0.705 | 0.295 | 0.623 | 0.437 |
| fine-tuned embeddinggemma-2 cosine | 0.557 | 0.738 | 0.820 | 0.869 | 0.918 | 1.000 | 0.885 | 0.557 | 0.820 | 0.676 |
| fine-tuned embeddinggemma-2 csls | 0.623 | 0.820 | 0.869 | 0.902 | 0.967 | 1.000 | 0.934 | 0.623 | 0.869 | 0.737 |
| fine-tuned embeddinggemma-2 hybrid-union | 0.164 | 0.721 | 0.770 | 0.852 | 0.918 | 0.951 | 0.869 | 0.164 | 0.770 | 0.458 |
| fine-tuned embeddinggemma-2 hybrid-rrf | 0.377 | 0.525 | 0.623 | 0.672 | 0.820 | 0.967 | 0.721 | 0.377 | 0.623 | 0.495 |

Validation-split top-1 (cosine) per encoder, the selection signal: e5-small 0.282, bge-small 0.310, minilm 0.366, mpnet 0.282, bge-base 0.282, e5-large 0.549, bge-large 0.451, embeddinggemma-2 0.549; lexical 0.282; fine-tuned 0.620.

## Concept task on the test split (gold + 5 hard negatives; temperature fitted on the calibration split)

| scorer | temperature | top-1 | top-3 | ECE raw -> scaled | NLL scaled | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical score | 0.4217 | 0.483 | 0.733 | 0.155 -> 0.150 | 1.345 | 0.37 / 0.682 |
| e5-small cosine | 0.0067 | 0.567 | 0.850 | 0.280 -> 0.251 | 1.798 | 0.65 / 0.564 |
| bge-small cosine | 0.0141 | 0.717 | 0.967 | 0.225 -> 0.164 | 1.037 | 0.75 / 0.778 |
| minilm cosine | 0.0265 | 0.650 | 0.967 | 0.098 -> 0.202 | 1.200 | 0.78 / 0.660 |
| mpnet cosine | 0.0315 | 0.633 | 0.933 | 0.113 -> 0.207 | 1.050 | 0.73 / 0.705 |
| bge-base cosine | 0.0100 | 0.633 | 0.933 | 0.226 -> 0.215 | 1.528 | 0.80 / 0.708 |
| e5-large cosine | 0.0047 | 0.633 | 0.883 | 0.357 -> 0.281 | 1.404 | 0.73 / 0.705 |
| bge-large cosine | 0.0126 | 0.750 | 0.933 | 0.268 -> 0.110 | 0.864 | 0.73 / 0.818 |
| embeddinggemma-2 cosine | 0.0089 | 0.800 | 0.950 | 0.371 -> 0.105 | 0.823 | 0.77 / 0.891 |
| fine-tuned embeddinggemma-2 | 0.0047 | 0.917 | 1.000 | 0.348 -> 0.081 | 0.781 | 0.98 / 0.915 |
| trained decision head (committed, `c1ec2e0aa46b`) | - | 0.190 | 0.571 | - -> 0.208 | - | - |

## Deployment path on the test split (`OLaya.system_one`, router top-10, leaves scored, axiom projection)

| router + scorer | top-1 | recall@5 | gold among scored leaves | gold in routed subgraph | axioms verified | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical router, lexical top-1 (baseline) | 0.131 | 0.426 | 0.459 | - | - | - |
| lexical router + trained head (committed) | 0.141 | 0.469 | - | - | - | - |
| lexical router + e5-small scorer | 0.246 | 0.426 | 0.443 | 0.459 | 1.000 | 0.34 / 0.429 |
| cosine router + e5-small scorer | 0.279 | 0.475 | 0.508 | 0.590 | 1.000 | 0.13 / 0.875 |
| csls router + e5-small scorer | 0.279 | 0.410 | 0.426 | 0.607 | 1.000 | 0.15 / 0.889 |
| hybrid-union router + e5-small scorer | 0.279 | 0.525 | 0.574 | 0.623 | 1.000 | 0.18 / 0.818 |
| hybrid-rrf router + e5-small scorer | 0.246 | 0.459 | 0.508 | 0.574 | 1.000 | 0.21 / 0.769 |
| lexical [routed] router + e5-small scorer | 0.230 | 0.426 | 0.443 | 0.459 | 1.000 | 0.31 / 0.421 |
| cosine [routed] router + e5-small scorer | 0.246 | 0.459 | 0.459 | 0.590 | 1.000 | 0.11 / 0.857 |
| csls [routed] router + e5-small scorer | 0.262 | 0.393 | 0.410 | 0.607 | 1.000 | 0.11 / 0.857 |
| hybrid-union [routed] router + e5-small scorer | 0.246 | 0.525 | 0.574 | 0.623 | 1.000 | 0.18 / 0.727 |
| hybrid-rrf [routed] router + e5-small scorer | 0.230 | 0.426 | 0.492 | 0.574 | 1.000 | 0.23 / 0.643 |
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
| lexical router + minilm scorer | 0.311 | 0.443 | 0.443 | 0.459 | 1.000 | 0.54 / 0.333 |
| cosine router + minilm scorer | 0.246 | 0.590 | 0.689 | 0.738 | 1.000 | 0.25 / 0.267 |
| csls router + minilm scorer | 0.213 | 0.590 | 0.787 | 0.820 | 1.000 | 0.25 / 0.267 |
| hybrid-union router + minilm scorer | 0.246 | 0.623 | 0.656 | 0.672 | 1.000 | 0.28 / 0.294 |
| hybrid-rrf router + minilm scorer | 0.262 | 0.541 | 0.574 | 0.656 | 1.000 | 0.36 / 0.364 |
| lexical [routed] router + minilm scorer | 0.295 | 0.443 | 0.443 | 0.459 | 1.000 | 0.51 / 0.290 |
| cosine [routed] router + minilm scorer | 0.213 | 0.590 | 0.672 | 0.738 | 1.000 | 0.26 / 0.250 |
| csls [routed] router + minilm scorer | 0.213 | 0.607 | 0.787 | 0.820 | 1.000 | 0.26 / 0.250 |
| hybrid-union [routed] router + minilm scorer | 0.213 | 0.623 | 0.656 | 0.672 | 1.000 | 0.28 / 0.235 |
| hybrid-rrf [routed] router + minilm scorer | 0.230 | 0.541 | 0.574 | 0.656 | 1.000 | 0.31 / 0.368 |
| lexical router + mpnet scorer | 0.295 | 0.443 | 0.443 | 0.459 | 1.000 | 0.46 / 0.357 |
| cosine router + mpnet scorer | 0.295 | 0.607 | 0.705 | 0.738 | 1.000 | 0.20 / 0.417 |
| csls router + mpnet scorer | 0.295 | 0.607 | 0.721 | 0.770 | 1.000 | 0.21 / 0.462 |
| hybrid-union router + mpnet scorer | 0.295 | 0.607 | 0.672 | 0.721 | 1.000 | 0.21 / 0.538 |
| hybrid-rrf router + mpnet scorer | 0.328 | 0.541 | 0.607 | 0.672 | 1.000 | 0.26 / 0.438 |
| lexical [routed] router + mpnet scorer | 0.279 | 0.443 | 0.443 | 0.459 | 1.000 | 0.44 / 0.333 |
| cosine [routed] router + mpnet scorer | 0.295 | 0.590 | 0.689 | 0.738 | 1.000 | 0.18 / 0.455 |
| csls [routed] router + mpnet scorer | 0.295 | 0.607 | 0.721 | 0.770 | 1.000 | 0.18 / 0.545 |
| hybrid-union [routed] router + mpnet scorer | 0.295 | 0.607 | 0.672 | 0.721 | 1.000 | 0.21 / 0.538 |
| hybrid-rrf [routed] router + mpnet scorer | 0.311 | 0.541 | 0.607 | 0.672 | 1.000 | 0.25 / 0.400 |
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
| lexical router + e5-large scorer | 0.230 | 0.426 | 0.443 | 0.459 | 1.000 | 0.48 / 0.310 |
| cosine router + e5-large scorer | 0.311 | 0.607 | 0.656 | 0.754 | 1.000 | 0.23 / 0.571 |
| csls router + e5-large scorer | 0.295 | 0.639 | 0.705 | 0.770 | 1.000 | 0.25 / 0.533 |
| hybrid-union router + e5-large scorer | 0.311 | 0.639 | 0.672 | 0.721 | 1.000 | 0.26 / 0.562 |
| hybrid-rrf router + e5-large scorer | 0.328 | 0.590 | 0.590 | 0.639 | 1.000 | 0.31 / 0.474 |
| lexical [routed] router + e5-large scorer | 0.213 | 0.426 | 0.443 | 0.459 | 1.000 | 0.43 / 0.308 |
| cosine [routed] router + e5-large scorer | 0.295 | 0.574 | 0.639 | 0.754 | 1.000 | 0.21 / 0.615 |
| csls [routed] router + e5-large scorer | 0.279 | 0.656 | 0.705 | 0.770 | 1.000 | 0.25 / 0.533 |
| hybrid-union [routed] router + e5-large scorer | 0.295 | 0.623 | 0.656 | 0.721 | 1.000 | 0.28 / 0.529 |
| hybrid-rrf [routed] router + e5-large scorer | 0.295 | 0.590 | 0.590 | 0.639 | 1.000 | 0.33 / 0.450 |
| lexical router + bge-large scorer | 0.279 | 0.443 | 0.443 | 0.459 | 1.000 | 0.54 / 0.424 |
| cosine router + bge-large scorer | 0.344 | 0.623 | 0.721 | 0.820 | 1.000 | 0.25 / 0.467 |
| csls router + bge-large scorer | 0.344 | 0.689 | 0.803 | 0.869 | 1.000 | 0.26 / 0.500 |
| hybrid-union router + bge-large scorer | 0.344 | 0.689 | 0.705 | 0.738 | 1.000 | 0.26 / 0.500 |
| hybrid-rrf router + bge-large scorer | 0.344 | 0.590 | 0.639 | 0.672 | 1.000 | 0.31 / 0.526 |
| lexical [routed] router + bge-large scorer | 0.279 | 0.443 | 0.443 | 0.459 | 1.000 | 0.54 / 0.455 |
| cosine [routed] router + bge-large scorer | 0.344 | 0.623 | 0.705 | 0.820 | 1.000 | 0.25 / 0.467 |
| csls [routed] router + bge-large scorer | 0.344 | 0.689 | 0.803 | 0.869 | 1.000 | 0.26 / 0.500 |
| hybrid-union [routed] router + bge-large scorer | 0.344 | 0.689 | 0.705 | 0.738 | 1.000 | 0.26 / 0.500 |
| hybrid-rrf [routed] router + bge-large scorer | 0.344 | 0.590 | 0.639 | 0.672 | 1.000 | 0.33 / 0.500 |
| lexical router + embeddinggemma-2 scorer | 0.262 | 0.443 | 0.443 | 0.459 | 1.000 | 0.52 / 0.375 |
| cosine router + embeddinggemma-2 scorer | 0.410 | 0.672 | 0.721 | 0.803 | 1.000 | 0.39 / 0.667 |
| csls router + embeddinggemma-2 scorer | 0.410 | 0.721 | 0.803 | 0.820 | 1.000 | 0.43 / 0.654 |
| hybrid-union router + embeddinggemma-2 scorer | 0.410 | 0.721 | 0.754 | 0.770 | 1.000 | 0.43 / 0.654 |
| hybrid-rrf router + embeddinggemma-2 scorer | 0.328 | 0.607 | 0.672 | 0.705 | 1.000 | 0.38 / 0.565 |
| lexical [routed] router + embeddinggemma-2 scorer | 0.262 | 0.443 | 0.443 | 0.459 | 1.000 | 0.52 / 0.375 |
| cosine [routed] router + embeddinggemma-2 scorer | 0.410 | 0.656 | 0.721 | 0.803 | 1.000 | 0.39 / 0.667 |
| csls [routed] router + embeddinggemma-2 scorer | 0.410 | 0.705 | 0.787 | 0.820 | 1.000 | 0.43 / 0.654 |
| hybrid-union [routed] router + embeddinggemma-2 scorer | 0.410 | 0.721 | 0.754 | 0.770 | 1.000 | 0.43 / 0.654 |
| hybrid-rrf [routed] router + embeddinggemma-2 scorer | 0.328 | 0.607 | 0.656 | 0.705 | 1.000 | 0.38 / 0.565 |
| lexical router + fine-tuned embeddinggemma-2 scorer | 0.344 | 0.443 | 0.443 | 0.459 | 1.000 | 0.89 / 0.370 |
| cosine router + fine-tuned embeddinggemma-2 scorer | 0.574 | 0.820 | 0.852 | 0.885 | 1.000 | 0.84 / 0.627 |
| csls router + fine-tuned embeddinggemma-2 scorer | 0.574 | 0.852 | 0.885 | 0.934 | 1.000 | 0.85 / 0.615 |
| hybrid-union router + fine-tuned embeddinggemma-2 scorer | 0.557 | 0.803 | 0.836 | 0.869 | 1.000 | 0.84 / 0.627 |
| hybrid-rrf router + fine-tuned embeddinggemma-2 scorer | 0.492 | 0.639 | 0.689 | 0.721 | 1.000 | 0.82 / 0.560 |
| lexical [routed] router + fine-tuned embeddinggemma-2 scorer | 0.361 | 0.443 | 0.443 | 0.459 | 1.000 | 0.89 / 0.389 |
| cosine [routed] router + fine-tuned embeddinggemma-2 scorer | 0.557 | 0.787 | 0.836 | 0.885 | 1.000 | 0.82 / 0.620 |
| csls [routed] router + fine-tuned embeddinggemma-2 scorer | 0.557 | 0.852 | 0.885 | 0.934 | 1.000 | 0.85 / 0.596 |
| hybrid-union [routed] router + fine-tuned embeddinggemma-2 scorer | 0.557 | 0.787 | 0.820 | 0.869 | 1.000 | 0.84 / 0.627 |
| hybrid-rrf [routed] router + fine-tuned embeddinggemma-2 scorer | 0.492 | 0.623 | 0.672 | 0.721 | 1.000 | 0.80 / 0.571 |

## Selection (validation split only)

- default_small: {"encoder": "minilm", "variant": "cosine", "validation_top1": 0.36619718309859156}
- best_zero_shot: {"encoder": "e5-large", "variant": "cosine", "validation_top1": 0.5492957746478874}
- best_hybrid: {"encoder": "embeddinggemma-2", "variant": "hybrid-union", "validation_ceiling_at_8": 0.8873239436619719}

## Fine-tuned bi-encoder

- Base `google/embeddinggemma-2`, 249 train pairs, 6 epochs in 114.0 s (budget 60.0 min), best epoch 3
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query
- Checkpoint `models/checkpoints/olaya_embed_gs1.pt` sha256 `41ee46730cd3f54a` (1084153629 bytes)
- Validation top-1: fine-tuned 0.620 vs zero-shot base 0.549 vs lexical 0.282 -> recorded in MANIFEST.json
- History: epoch 1 loss 1.625 val 0.563; epoch 2 loss 1.463 val 0.592; epoch 3 loss 1.160 val 0.620; epoch 4 loss 0.841 val 0.606; epoch 5 loss 0.549 val 0.620; epoch 6 loss 0.371 val 0.620

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out whole classes: no test class has a training row, so the fine-tuned numbers measure transfer to unseen classes, with fewer rows per split and the class-text side never adapted.
- Zero-shot encoders never saw the corpus; their temperature is the only fitted parameter (calibration split).
- The concept-task metrics have the gold candidate present (5 hard negatives); the full-ontology and engine-path numbers are the deployment comparison.
- The state passed to every router is the raw observation text; the decision head reports used a JSON state wrapper for the engine path.
