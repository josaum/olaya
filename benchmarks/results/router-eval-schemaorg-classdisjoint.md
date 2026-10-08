# Candidate routers on the LLM-distilled schemaorg corpus

Labels are LLM-distilled and filtered, **not human**. **Class-disjoint split**: every test class is unseen in training (seeded 60/15/10/15 over classes); every choice of encoder / variant is made on the validation split and test
is evaluated once per configuration. The state given to every router is the raw observation text.

- Corpus `data/corpora/schemaorg.jsonl`: 862 rows, 119 classes used of 660 ontology concepts; split rows {'train': 509, 'validation': 132, 'calibration': 94, 'test': 127}
- Test classes unseen in training: 18
- Concept tasks (gold + 5 hard negatives): calibration 92, test 124
- Commit `feaca1a5806b`, runtime 646.0 s

## Candidate ceiling on the test split (gold in top-k over the full ontology)

| router | @1 | @3 | @5 | @8 | @16 | @32 | leaf ceiling (top-10 subgraph) | top-1 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| lexical (abstains on ties / no evidence) | 0.205 | 0.339 | 0.386 | 0.394 | 0.425 | 0.488 | 0.402 | 0.205 | 0.386 | 0.284 |
| e5-small cosine | 0.283 | 0.457 | 0.535 | 0.606 | 0.732 | 0.803 | 0.614 | 0.283 | 0.535 | 0.402 |
| e5-small csls | 0.252 | 0.394 | 0.488 | 0.543 | 0.630 | 0.724 | 0.575 | 0.252 | 0.488 | 0.363 |
| e5-small hybrid-union | 0.205 | 0.441 | 0.535 | 0.606 | 0.677 | 0.780 | 0.614 | 0.205 | 0.535 | 0.355 |
| e5-small hybrid-rrf | 0.299 | 0.417 | 0.472 | 0.504 | 0.583 | 0.724 | 0.488 | 0.299 | 0.472 | 0.383 |
| bge-small cosine | 0.291 | 0.409 | 0.457 | 0.535 | 0.661 | 0.764 | 0.535 | 0.291 | 0.457 | 0.383 |
| bge-small csls | 0.268 | 0.394 | 0.457 | 0.504 | 0.622 | 0.701 | 0.520 | 0.268 | 0.457 | 0.362 |
| bge-small hybrid-union | 0.205 | 0.417 | 0.512 | 0.543 | 0.614 | 0.732 | 0.528 | 0.205 | 0.512 | 0.345 |
| bge-small hybrid-rrf | 0.307 | 0.402 | 0.441 | 0.472 | 0.535 | 0.693 | 0.488 | 0.307 | 0.441 | 0.380 |
| minilm cosine | 0.276 | 0.394 | 0.480 | 0.535 | 0.669 | 0.795 | 0.528 | 0.276 | 0.480 | 0.376 |
| minilm csls | 0.252 | 0.433 | 0.520 | 0.559 | 0.638 | 0.780 | 0.567 | 0.252 | 0.520 | 0.376 |
| minilm hybrid-union | 0.205 | 0.433 | 0.504 | 0.583 | 0.638 | 0.724 | 0.567 | 0.205 | 0.504 | 0.352 |
| minilm hybrid-rrf | 0.331 | 0.433 | 0.480 | 0.496 | 0.567 | 0.709 | 0.512 | 0.331 | 0.480 | 0.409 |
| mpnet cosine | 0.260 | 0.402 | 0.472 | 0.575 | 0.732 | 0.843 | 0.591 | 0.260 | 0.472 | 0.374 |
| mpnet csls | 0.283 | 0.472 | 0.567 | 0.614 | 0.748 | 0.827 | 0.638 | 0.283 | 0.567 | 0.416 |
| mpnet hybrid-union | 0.205 | 0.425 | 0.535 | 0.614 | 0.717 | 0.787 | 0.598 | 0.205 | 0.535 | 0.362 |
| mpnet hybrid-rrf | 0.299 | 0.433 | 0.480 | 0.504 | 0.606 | 0.772 | 0.528 | 0.299 | 0.480 | 0.392 |
| bge-base cosine | 0.276 | 0.465 | 0.551 | 0.661 | 0.764 | 0.858 | 0.661 | 0.276 | 0.551 | 0.406 |
| bge-base csls | 0.283 | 0.441 | 0.575 | 0.606 | 0.717 | 0.772 | 0.622 | 0.283 | 0.575 | 0.398 |
| bge-base hybrid-union | 0.205 | 0.394 | 0.535 | 0.614 | 0.732 | 0.803 | 0.614 | 0.205 | 0.535 | 0.353 |
| bge-base hybrid-rrf | 0.283 | 0.402 | 0.465 | 0.496 | 0.559 | 0.709 | 0.480 | 0.283 | 0.465 | 0.372 |
| e5-large cosine | 0.465 | 0.575 | 0.654 | 0.756 | 0.858 | 0.890 | 0.748 | 0.465 | 0.654 | 0.560 |
| e5-large csls | 0.417 | 0.543 | 0.606 | 0.693 | 0.772 | 0.835 | 0.693 | 0.417 | 0.606 | 0.506 |
| e5-large hybrid-union | 0.205 | 0.551 | 0.638 | 0.693 | 0.819 | 0.874 | 0.693 | 0.205 | 0.638 | 0.415 |
| e5-large hybrid-rrf | 0.331 | 0.441 | 0.488 | 0.528 | 0.630 | 0.780 | 0.535 | 0.331 | 0.488 | 0.417 |
| bge-large cosine | 0.386 | 0.496 | 0.583 | 0.654 | 0.732 | 0.811 | 0.646 | 0.386 | 0.583 | 0.481 |
| bge-large csls | 0.362 | 0.512 | 0.598 | 0.646 | 0.709 | 0.780 | 0.654 | 0.362 | 0.598 | 0.464 |
| bge-large hybrid-union | 0.205 | 0.457 | 0.575 | 0.638 | 0.717 | 0.795 | 0.654 | 0.205 | 0.575 | 0.375 |
| bge-large hybrid-rrf | 0.299 | 0.409 | 0.465 | 0.496 | 0.575 | 0.764 | 0.504 | 0.299 | 0.465 | 0.382 |
| embeddinggemma-2 cosine | 0.512 | 0.669 | 0.772 | 0.811 | 0.898 | 0.929 | 0.756 | 0.512 | 0.772 | 0.622 |
| embeddinggemma-2 csls | 0.488 | 0.654 | 0.732 | 0.795 | 0.874 | 0.929 | 0.780 | 0.488 | 0.732 | 0.595 |
| embeddinggemma-2 hybrid-union | 0.205 | 0.630 | 0.724 | 0.803 | 0.866 | 0.921 | 0.772 | 0.205 | 0.724 | 0.450 |
| embeddinggemma-2 hybrid-rrf | 0.331 | 0.465 | 0.488 | 0.528 | 0.685 | 0.843 | 0.575 | 0.331 | 0.488 | 0.424 |
| fine-tuned embeddinggemma-2 cosine | 0.598 | 0.780 | 0.835 | 0.858 | 0.906 | 0.969 | 0.811 | 0.598 | 0.835 | 0.703 |
| fine-tuned embeddinggemma-2 csls | 0.591 | 0.780 | 0.850 | 0.882 | 0.921 | 0.937 | 0.819 | 0.591 | 0.850 | 0.700 |
| fine-tuned embeddinggemma-2 hybrid-union | 0.205 | 0.685 | 0.795 | 0.858 | 0.898 | 0.945 | 0.803 | 0.205 | 0.795 | 0.483 |
| fine-tuned embeddinggemma-2 hybrid-rrf | 0.386 | 0.488 | 0.512 | 0.535 | 0.677 | 0.890 | 0.559 | 0.386 | 0.512 | 0.464 |

Validation-split top-1 (cosine) per encoder, the selection signal: e5-small 0.295, bge-small 0.326, minilm 0.242, mpnet 0.326, bge-base 0.295, e5-large 0.386, bge-large 0.356, embeddinggemma-2 0.356; lexical 0.091; fine-tuned 0.591.

## Concept task on the test split (gold + 5 hard negatives; temperature fitted on the calibration split)

| scorer | temperature | top-1 | top-3 | ECE raw -> scaled | NLL scaled | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical score | 0.3350 | 0.548 | 0.806 | 0.200 -> 0.086 | 1.123 | 0.40 / 0.900 |
| e5-small cosine | 0.0067 | 0.855 | 0.984 | 0.518 -> 0.032 | 0.397 | 0.79 / 0.939 |
| bge-small cosine | 0.0177 | 0.806 | 0.968 | 0.194 -> 0.066 | 0.654 | 0.80 / 0.879 |
| minilm cosine | 0.0315 | 0.758 | 0.960 | 0.036 -> 0.086 | 0.653 | 0.78 / 0.856 |
| mpnet cosine | 0.0334 | 0.879 | 0.984 | 0.114 -> 0.050 | 0.370 | 0.77 / 0.979 |
| bge-base cosine | 0.0177 | 0.855 | 0.968 | 0.283 -> 0.044 | 0.421 | 0.83 / 0.932 |
| e5-large cosine | 0.0059 | 0.919 | 0.992 | 0.558 -> 0.033 | 0.226 | 0.85 / 0.972 |
| bge-large cosine | 0.0158 | 0.895 | 0.992 | 0.310 -> 0.071 | 0.386 | 0.83 / 0.932 |
| embeddinggemma-2 cosine | 0.0075 | 0.927 | 0.992 | 0.370 -> 0.034 | 0.205 | 0.96 / 0.958 |
| fine-tuned embeddinggemma-2 | 0.0211 | 0.960 | 1.000 | 0.053 -> 0.024 | 0.167 | 0.98 / 0.975 |
| trained decision head (committed, `f2c8f4503f63`) | - | 0.179 | 0.538 | - -> 0.111 | - | - |

## Deployment path on the test split (`OLaya.system_one`, router top-10, leaves scored, axiom projection)

| router + scorer | top-1 | recall@5 | gold among scored leaves | gold in routed subgraph | axioms verified | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical router, lexical top-1 (baseline) | 0.205 | 0.386 | 0.402 | - | - | - |
| lexical router + trained head (committed) | 0.075 | 0.142 | - | - | - | - |
| lexical router + e5-small scorer | 0.283 | 0.378 | 0.378 | 0.465 | 1.000 | 0.53 / 0.373 |
| cosine router + e5-small scorer | 0.299 | 0.535 | 0.535 | 0.724 | 1.000 | 0.11 / 0.714 |
| csls router + e5-small scorer | 0.307 | 0.488 | 0.488 | 0.669 | 1.000 | 0.13 / 0.588 |
| hybrid-union router + e5-small scorer | 0.299 | 0.512 | 0.520 | 0.717 | 1.000 | 0.20 / 0.520 |
| hybrid-rrf router + e5-small scorer | 0.268 | 0.441 | 0.449 | 0.583 | 1.000 | 0.20 / 0.640 |
| lexical [routed] router + e5-small scorer | 0.291 | 0.386 | 0.386 | 0.465 | 1.000 | 0.50 / 0.375 |
| cosine [routed] router + e5-small scorer | 0.283 | 0.528 | 0.535 | 0.724 | 1.000 | 0.09 / 0.750 |
| csls [routed] router + e5-small scorer | 0.291 | 0.465 | 0.472 | 0.669 | 1.000 | 0.11 / 0.643 |
| hybrid-union [routed] router + e5-small scorer | 0.283 | 0.512 | 0.520 | 0.717 | 1.000 | 0.15 / 0.632 |
| hybrid-rrf [routed] router + e5-small scorer | 0.276 | 0.457 | 0.457 | 0.583 | 1.000 | 0.18 / 0.696 |
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
| lexical router + minilm scorer | 0.307 | 0.378 | 0.378 | 0.465 | 1.000 | 0.53 / 0.403 |
| cosine router + minilm scorer | 0.291 | 0.472 | 0.504 | 0.630 | 1.000 | 0.16 / 0.650 |
| csls router + minilm scorer | 0.307 | 0.488 | 0.528 | 0.669 | 1.000 | 0.18 / 0.652 |
| hybrid-union router + minilm scorer | 0.291 | 0.504 | 0.512 | 0.669 | 1.000 | 0.25 / 0.562 |
| hybrid-rrf router + minilm scorer | 0.291 | 0.472 | 0.472 | 0.606 | 1.000 | 0.27 / 0.500 |
| lexical [routed] router + minilm scorer | 0.307 | 0.386 | 0.386 | 0.465 | 1.000 | 0.54 / 0.397 |
| cosine [routed] router + minilm scorer | 0.276 | 0.480 | 0.512 | 0.630 | 1.000 | 0.14 / 0.611 |
| csls [routed] router + minilm scorer | 0.283 | 0.496 | 0.543 | 0.669 | 1.000 | 0.17 / 0.571 |
| hybrid-union [routed] router + minilm scorer | 0.276 | 0.512 | 0.520 | 0.669 | 1.000 | 0.25 / 0.531 |
| hybrid-rrf [routed] router + minilm scorer | 0.260 | 0.488 | 0.488 | 0.606 | 1.000 | 0.25 / 0.500 |
| lexical router + mpnet scorer | 0.268 | 0.378 | 0.378 | 0.465 | 1.000 | 0.46 / 0.362 |
| cosine router + mpnet scorer | 0.260 | 0.465 | 0.512 | 0.701 | 1.000 | 0.10 / 0.692 |
| csls router + mpnet scorer | 0.260 | 0.535 | 0.559 | 0.748 | 1.000 | 0.13 / 0.688 |
| hybrid-union router + mpnet scorer | 0.260 | 0.551 | 0.551 | 0.709 | 1.000 | 0.19 / 0.583 |
| hybrid-rrf router + mpnet scorer | 0.220 | 0.472 | 0.480 | 0.622 | 1.000 | 0.21 / 0.407 |
| lexical [routed] router + mpnet scorer | 0.268 | 0.386 | 0.386 | 0.465 | 1.000 | 0.44 / 0.375 |
| cosine [routed] router + mpnet scorer | 0.260 | 0.472 | 0.543 | 0.701 | 1.000 | 0.09 / 0.636 |
| csls [routed] router + mpnet scorer | 0.260 | 0.543 | 0.583 | 0.748 | 1.000 | 0.11 / 0.643 |
| hybrid-union [routed] router + mpnet scorer | 0.260 | 0.567 | 0.567 | 0.709 | 1.000 | 0.15 / 0.632 |
| hybrid-rrf [routed] router + mpnet scorer | 0.220 | 0.472 | 0.488 | 0.622 | 1.000 | 0.17 / 0.500 |
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
| lexical router + e5-large scorer | 0.315 | 0.378 | 0.378 | 0.465 | 1.000 | 0.60 / 0.434 |
| cosine router + e5-large scorer | 0.472 | 0.646 | 0.654 | 0.874 | 1.000 | 0.22 / 0.857 |
| csls router + e5-large scorer | 0.441 | 0.583 | 0.591 | 0.803 | 1.000 | 0.28 / 0.806 |
| hybrid-union router + e5-large scorer | 0.472 | 0.614 | 0.614 | 0.795 | 1.000 | 0.28 / 0.800 |
| hybrid-rrf router + e5-large scorer | 0.346 | 0.480 | 0.488 | 0.622 | 1.000 | 0.31 / 0.641 |
| lexical [routed] router + e5-large scorer | 0.323 | 0.386 | 0.386 | 0.465 | 1.000 | 0.60 / 0.421 |
| cosine [routed] router + e5-large scorer | 0.465 | 0.654 | 0.661 | 0.874 | 1.000 | 0.17 / 0.818 |
| csls [routed] router + e5-large scorer | 0.449 | 0.591 | 0.598 | 0.803 | 1.000 | 0.28 / 0.743 |
| hybrid-union [routed] router + e5-large scorer | 0.465 | 0.638 | 0.638 | 0.795 | 1.000 | 0.26 / 0.788 |
| hybrid-rrf [routed] router + e5-large scorer | 0.354 | 0.472 | 0.488 | 0.622 | 1.000 | 0.28 / 0.667 |
| lexical router + bge-large scorer | 0.307 | 0.378 | 0.378 | 0.465 | 1.000 | 0.46 / 0.508 |
| cosine router + bge-large scorer | 0.394 | 0.583 | 0.598 | 0.756 | 1.000 | 0.18 / 0.826 |
| csls router + bge-large scorer | 0.386 | 0.583 | 0.583 | 0.756 | 1.000 | 0.24 / 0.700 |
| hybrid-union router + bge-large scorer | 0.394 | 0.559 | 0.559 | 0.748 | 1.000 | 0.26 / 0.697 |
| hybrid-rrf router + bge-large scorer | 0.323 | 0.457 | 0.457 | 0.591 | 1.000 | 0.28 / 0.639 |
| lexical [routed] router + bge-large scorer | 0.315 | 0.386 | 0.386 | 0.465 | 1.000 | 0.45 / 0.526 |
| cosine [routed] router + bge-large scorer | 0.386 | 0.567 | 0.614 | 0.756 | 1.000 | 0.17 / 0.818 |
| csls [routed] router + bge-large scorer | 0.378 | 0.567 | 0.575 | 0.756 | 1.000 | 0.22 / 0.714 |
| hybrid-union [routed] router + bge-large scorer | 0.386 | 0.575 | 0.575 | 0.748 | 1.000 | 0.24 / 0.767 |
| hybrid-rrf [routed] router + bge-large scorer | 0.315 | 0.465 | 0.465 | 0.591 | 1.000 | 0.27 / 0.647 |
| lexical router + embeddinggemma-2 scorer | 0.339 | 0.378 | 0.378 | 0.465 | 1.000 | 0.75 / 0.389 |
| cosine router + embeddinggemma-2 scorer | 0.520 | 0.724 | 0.740 | 0.882 | 1.000 | 0.46 / 0.729 |
| csls router + embeddinggemma-2 scorer | 0.528 | 0.693 | 0.701 | 0.898 | 1.000 | 0.61 / 0.636 |
| hybrid-union router + embeddinggemma-2 scorer | 0.512 | 0.701 | 0.701 | 0.882 | 1.000 | 0.58 / 0.622 |
| hybrid-rrf router + embeddinggemma-2 scorer | 0.339 | 0.488 | 0.488 | 0.669 | 1.000 | 0.51 / 0.477 |
| lexical [routed] router + embeddinggemma-2 scorer | 0.339 | 0.386 | 0.386 | 0.465 | 1.000 | 0.75 / 0.389 |
| cosine [routed] router + embeddinggemma-2 scorer | 0.512 | 0.772 | 0.787 | 0.882 | 1.000 | 0.43 / 0.745 |
| csls [routed] router + embeddinggemma-2 scorer | 0.528 | 0.717 | 0.724 | 0.898 | 1.000 | 0.60 / 0.632 |
| hybrid-union [routed] router + embeddinggemma-2 scorer | 0.512 | 0.732 | 0.732 | 0.882 | 1.000 | 0.56 / 0.662 |
| hybrid-rrf [routed] router + embeddinggemma-2 scorer | 0.346 | 0.496 | 0.496 | 0.669 | 1.000 | 0.49 / 0.484 |
| lexical router + fine-tuned embeddinggemma-2 scorer | 0.370 | 0.378 | 0.378 | 0.465 | 1.000 | 0.76 / 0.464 |
| cosine router + fine-tuned embeddinggemma-2 scorer | 0.614 | 0.787 | 0.787 | 0.945 | 1.000 | 0.51 / 0.831 |
| csls router + fine-tuned embeddinggemma-2 scorer | 0.614 | 0.795 | 0.803 | 0.953 | 1.000 | 0.52 / 0.818 |
| hybrid-union router + fine-tuned embeddinggemma-2 scorer | 0.606 | 0.772 | 0.772 | 0.929 | 1.000 | 0.59 / 0.760 |
| hybrid-rrf router + fine-tuned embeddinggemma-2 scorer | 0.449 | 0.496 | 0.496 | 0.669 | 1.000 | 0.58 / 0.622 |
| lexical [routed] router + fine-tuned embeddinggemma-2 scorer | 0.370 | 0.386 | 0.386 | 0.465 | 1.000 | 0.77 / 0.459 |
| cosine [routed] router + fine-tuned embeddinggemma-2 scorer | 0.598 | 0.835 | 0.858 | 0.945 | 1.000 | 0.46 / 0.862 |
| csls [routed] router + fine-tuned embeddinggemma-2 scorer | 0.598 | 0.843 | 0.858 | 0.953 | 1.000 | 0.48 / 0.836 |
| hybrid-union [routed] router + fine-tuned embeddinggemma-2 scorer | 0.598 | 0.803 | 0.803 | 0.929 | 1.000 | 0.56 / 0.775 |
| hybrid-rrf [routed] router + fine-tuned embeddinggemma-2 scorer | 0.441 | 0.520 | 0.520 | 0.669 | 1.000 | 0.60 / 0.566 |

## Selection (validation split only)

- default_small: {"encoder": "bge-small", "variant": "cosine", "validation_top1": 0.32575757575757575}
- best_zero_shot: {"encoder": "e5-large", "variant": "cosine", "validation_top1": 0.38636363636363635}
- best_hybrid: {"encoder": "embeddinggemma-2", "variant": "hybrid-union", "validation_ceiling_at_8": 0.7045454545454546}

## Fine-tuned bi-encoder

- Base `google/embeddinggemma-2`, 509 train pairs, 7 epochs in 435.5 s (budget 60.0 min), best epoch 4
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query
- Checkpoint `models/checkpoints/olaya_embed_schemaorg.pt` sha256 `ef94516da39b061f` (1084156143 bytes)
- Validation top-1: fine-tuned 0.591 vs zero-shot base 0.356 vs lexical 0.091 -> recorded in MANIFEST.json
- History: epoch 1 loss 1.683 val 0.371; epoch 2 loss 1.315 val 0.500; epoch 3 loss 0.675 val 0.568; epoch 4 loss 0.284 val 0.591; epoch 5 loss 0.149 val 0.576; epoch 6 loss 0.070 val 0.576; epoch 7 loss 0.065 val 0.583

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out whole classes: no test class has a training row, so the fine-tuned numbers measure transfer to unseen classes, with fewer rows per split and the class-text side never adapted.
- Zero-shot encoders never saw the corpus; their temperature is the only fitted parameter (calibration split).
- The concept-task metrics have the gold candidate present (5 hard negatives); the full-ontology and engine-path numbers are the deployment comparison.
- The state passed to every router is the raw observation text; the decision head reports used a JSON state wrapper for the engine path.
