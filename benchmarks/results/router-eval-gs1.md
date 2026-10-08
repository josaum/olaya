# Candidate routers on the LLM-distilled gs1 corpus

Labels are LLM-distilled and filtered, **not human**. Same corpus, same batch-disjoint split and same test rows as `decision-head-gs1.md`; every choice of encoder / variant is made on the validation split and test
is evaluated once per configuration. The state given to every router is the raw observation text.

- Corpus `data/corpora/gs1.jsonl`: 422 rows, 63 classes used of 63 ontology concepts; split rows {'train': 258, 'validation': 59, 'calibration': 41, 'test': 64}
- Test classes unseen in training: 4
- Concept tasks (gold + 5 hard negatives): calibration 41, test 63
- Commit `feaca1a5806b`, runtime 525.2 s

## Candidate ceiling on the test split (gold in top-k over the full ontology)

| router | @1 | @3 | @5 | @8 | @16 | @32 | leaf ceiling (top-10 subgraph) | top-1 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| lexical (abstains on ties / no evidence) | 0.281 | 0.453 | 0.531 | 0.562 | 0.562 | 0.609 | 0.562 | 0.281 | 0.531 | 0.382 |
| e5-small cosine | 0.266 | 0.453 | 0.547 | 0.641 | 0.766 | 0.922 | 0.641 | 0.266 | 0.547 | 0.400 |
| e5-small csls | 0.188 | 0.359 | 0.438 | 0.562 | 0.656 | 0.938 | 0.609 | 0.188 | 0.438 | 0.319 |
| e5-small hybrid-union | 0.281 | 0.594 | 0.656 | 0.703 | 0.781 | 0.906 | 0.688 | 0.281 | 0.656 | 0.456 |
| e5-small hybrid-rrf | 0.375 | 0.516 | 0.625 | 0.641 | 0.719 | 0.844 | 0.594 | 0.375 | 0.625 | 0.483 |
| bge-small cosine | 0.375 | 0.609 | 0.688 | 0.766 | 0.906 | 0.984 | 0.797 | 0.375 | 0.688 | 0.517 |
| bge-small csls | 0.312 | 0.531 | 0.625 | 0.734 | 0.875 | 0.969 | 0.781 | 0.312 | 0.625 | 0.460 |
| bge-small hybrid-union | 0.281 | 0.609 | 0.703 | 0.812 | 0.875 | 0.969 | 0.781 | 0.281 | 0.703 | 0.475 |
| bge-small hybrid-rrf | 0.438 | 0.578 | 0.656 | 0.672 | 0.750 | 0.906 | 0.672 | 0.438 | 0.656 | 0.540 |
| minilm cosine | 0.297 | 0.578 | 0.641 | 0.750 | 0.875 | 0.984 | 0.766 | 0.297 | 0.641 | 0.469 |
| minilm csls | 0.297 | 0.578 | 0.688 | 0.703 | 0.859 | 1.000 | 0.719 | 0.297 | 0.688 | 0.468 |
| minilm hybrid-union | 0.281 | 0.578 | 0.672 | 0.734 | 0.859 | 0.953 | 0.734 | 0.281 | 0.672 | 0.457 |
| minilm hybrid-rrf | 0.391 | 0.516 | 0.609 | 0.656 | 0.766 | 0.891 | 0.641 | 0.391 | 0.609 | 0.490 |
| mpnet cosine | 0.344 | 0.609 | 0.766 | 0.859 | 0.938 | 0.953 | 0.859 | 0.344 | 0.766 | 0.520 |
| mpnet csls | 0.375 | 0.656 | 0.766 | 0.875 | 0.938 | 1.000 | 0.891 | 0.375 | 0.766 | 0.553 |
| mpnet hybrid-union | 0.281 | 0.594 | 0.750 | 0.859 | 0.969 | 0.984 | 0.859 | 0.281 | 0.750 | 0.485 |
| mpnet hybrid-rrf | 0.422 | 0.547 | 0.594 | 0.656 | 0.812 | 0.938 | 0.688 | 0.422 | 0.594 | 0.517 |
| bge-base cosine | 0.406 | 0.578 | 0.703 | 0.734 | 0.859 | 0.938 | 0.797 | 0.406 | 0.703 | 0.524 |
| bge-base csls | 0.344 | 0.578 | 0.688 | 0.781 | 0.859 | 0.984 | 0.781 | 0.344 | 0.688 | 0.495 |
| bge-base hybrid-union | 0.281 | 0.609 | 0.719 | 0.812 | 0.891 | 0.953 | 0.812 | 0.281 | 0.719 | 0.488 |
| bge-base hybrid-rrf | 0.375 | 0.531 | 0.656 | 0.672 | 0.797 | 0.891 | 0.688 | 0.375 | 0.656 | 0.495 |
| e5-large cosine | 0.453 | 0.703 | 0.766 | 0.828 | 0.922 | 0.953 | 0.844 | 0.453 | 0.766 | 0.601 |
| e5-large csls | 0.328 | 0.562 | 0.750 | 0.797 | 0.906 | 0.984 | 0.797 | 0.328 | 0.750 | 0.492 |
| e5-large hybrid-union | 0.281 | 0.719 | 0.812 | 0.859 | 0.906 | 0.969 | 0.812 | 0.281 | 0.812 | 0.518 |
| e5-large hybrid-rrf | 0.375 | 0.594 | 0.625 | 0.672 | 0.781 | 0.938 | 0.672 | 0.375 | 0.625 | 0.509 |
| bge-large cosine | 0.344 | 0.625 | 0.703 | 0.828 | 0.859 | 0.984 | 0.844 | 0.344 | 0.703 | 0.508 |
| bge-large csls | 0.375 | 0.578 | 0.672 | 0.734 | 0.891 | 0.953 | 0.750 | 0.375 | 0.672 | 0.509 |
| bge-large hybrid-union | 0.281 | 0.578 | 0.703 | 0.797 | 0.891 | 0.938 | 0.797 | 0.281 | 0.703 | 0.469 |
| bge-large hybrid-rrf | 0.391 | 0.578 | 0.625 | 0.688 | 0.781 | 0.906 | 0.688 | 0.391 | 0.625 | 0.509 |
| embeddinggemma-2 cosine | 0.516 | 0.703 | 0.797 | 0.906 | 0.953 | 0.969 | 0.875 | 0.516 | 0.797 | 0.646 |
| embeddinggemma-2 csls | 0.438 | 0.734 | 0.828 | 0.922 | 0.938 | 0.984 | 0.891 | 0.438 | 0.828 | 0.601 |
| embeddinggemma-2 hybrid-union | 0.281 | 0.734 | 0.828 | 0.844 | 0.969 | 0.984 | 0.828 | 0.281 | 0.828 | 0.526 |
| embeddinggemma-2 hybrid-rrf | 0.391 | 0.578 | 0.656 | 0.688 | 0.828 | 0.938 | 0.688 | 0.391 | 0.656 | 0.517 |
| fine-tuned embeddinggemma-2 cosine | 0.859 | 0.922 | 0.969 | 1.000 | 1.000 | 1.000 | 0.938 | 0.859 | 0.969 | 0.905 |
| fine-tuned embeddinggemma-2 csls | 0.828 | 0.938 | 0.969 | 1.000 | 1.000 | 1.000 | 0.938 | 0.828 | 0.969 | 0.892 |
| fine-tuned embeddinggemma-2 hybrid-union | 0.281 | 0.922 | 0.953 | 0.984 | 1.000 | 1.000 | 0.938 | 0.281 | 0.953 | 0.609 |
| fine-tuned embeddinggemma-2 hybrid-rrf | 0.500 | 0.641 | 0.688 | 0.734 | 0.844 | 0.953 | 0.719 | 0.500 | 0.688 | 0.592 |

Validation-split top-1 (cosine) per encoder, the selection signal: e5-small 0.424, bge-small 0.424, minilm 0.475, mpnet 0.441, bge-base 0.508, e5-large 0.475, bge-large 0.441, embeddinggemma-2 0.441; lexical 0.169; fine-tuned 0.695.

## Concept task on the test split (gold + 5 hard negatives; temperature fitted on the calibration split)

| scorer | temperature | top-1 | top-3 | ECE raw -> scaled | NLL scaled | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical score | 0.5309 | 0.524 | 0.698 | 0.179 -> 0.136 | 1.303 | 0.30 / 0.737 |
| e5-small cosine | 0.0089 | 0.651 | 0.921 | 0.376 -> 0.179 | 1.192 | 0.60 / 0.737 |
| bge-small cosine | 0.0265 | 0.698 | 0.952 | 0.145 -> 0.112 | 0.799 | 0.65 / 0.780 |
| minilm cosine | 0.0334 | 0.635 | 0.968 | 0.080 -> 0.183 | 0.984 | 0.63 / 0.750 |
| mpnet cosine | 0.0530 | 0.746 | 0.952 | 0.120 -> 0.104 | 0.753 | 0.56 / 0.857 |
| bge-base cosine | 0.0237 | 0.746 | 0.921 | 0.292 -> 0.122 | 0.875 | 0.46 / 0.862 |
| e5-large cosine | 0.0084 | 0.714 | 0.952 | 0.416 -> 0.127 | 0.750 | 0.68 / 0.814 |
| bge-large cosine | 0.0177 | 0.730 | 0.937 | 0.219 -> 0.127 | 0.893 | 0.63 / 0.850 |
| embeddinggemma-2 cosine | 0.0177 | 0.841 | 0.952 | 0.364 -> 0.078 | 0.555 | 0.70 / 0.932 |
| fine-tuned embeddinggemma-2 | 0.0298 | 0.984 | 1.000 | 0.054 -> 0.026 | 0.135 | 0.98 / 0.984 |
| trained decision head (committed, `c1ec2e0aa46b`) | - | 0.190 | 0.571 | - -> 0.208 | - | - |

## Deployment path on the test split (`OLaya.system_one`, router top-10, leaves scored, axiom projection)

| router + scorer | top-1 | recall@5 | gold among scored leaves | gold in routed subgraph | axioms verified | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical router, lexical top-1 (baseline) | 0.281 | 0.531 | 0.562 | - | - | - |
| lexical router + trained head (committed) | 0.141 | 0.469 | - | - | - | - |
| lexical router + e5-small scorer | 0.328 | 0.547 | 0.547 | 0.562 | 1.000 | 0.33 / 0.333 |
| cosine router + e5-small scorer | 0.266 | 0.578 | 0.594 | 0.719 | 1.000 | 0.09 / 0.167 |
| csls router + e5-small scorer | 0.266 | 0.453 | 0.453 | 0.688 | 1.000 | 0.14 / 0.333 |
| hybrid-union router + e5-small scorer | 0.266 | 0.641 | 0.656 | 0.766 | 1.000 | 0.19 / 0.333 |
| hybrid-rrf router + e5-small scorer | 0.250 | 0.562 | 0.594 | 0.656 | 1.000 | 0.22 / 0.429 |
| lexical [routed] router + e5-small scorer | 0.312 | 0.547 | 0.547 | 0.562 | 1.000 | 0.33 / 0.333 |
| cosine [routed] router + e5-small scorer | 0.266 | 0.531 | 0.594 | 0.719 | 1.000 | 0.09 / 0.167 |
| csls [routed] router + e5-small scorer | 0.266 | 0.453 | 0.453 | 0.688 | 1.000 | 0.14 / 0.333 |
| hybrid-union [routed] router + e5-small scorer | 0.266 | 0.656 | 0.672 | 0.766 | 1.000 | 0.16 / 0.400 |
| hybrid-rrf [routed] router + e5-small scorer | 0.234 | 0.562 | 0.625 | 0.656 | 1.000 | 0.20 / 0.462 |
| lexical router + bge-small scorer | 0.453 | 0.547 | 0.547 | 0.562 | 1.000 | 0.38 / 0.458 |
| cosine router + bge-small scorer | 0.375 | 0.688 | 0.719 | 0.859 | 1.000 | 0.12 / 0.625 |
| csls router + bge-small scorer | 0.359 | 0.672 | 0.672 | 0.859 | 1.000 | 0.14 / 0.556 |
| hybrid-union router + bge-small scorer | 0.375 | 0.734 | 0.734 | 0.844 | 1.000 | 0.17 / 0.455 |
| hybrid-rrf router + bge-small scorer | 0.406 | 0.625 | 0.625 | 0.719 | 1.000 | 0.19 / 0.583 |
| lexical [routed] router + bge-small scorer | 0.453 | 0.547 | 0.547 | 0.562 | 1.000 | 0.33 / 0.524 |
| cosine [routed] router + bge-small scorer | 0.375 | 0.688 | 0.719 | 0.859 | 1.000 | 0.12 / 0.625 |
| csls [routed] router + bge-small scorer | 0.359 | 0.672 | 0.672 | 0.859 | 1.000 | 0.14 / 0.556 |
| hybrid-union [routed] router + bge-small scorer | 0.375 | 0.750 | 0.766 | 0.844 | 1.000 | 0.19 / 0.500 |
| hybrid-rrf [routed] router + bge-small scorer | 0.406 | 0.641 | 0.656 | 0.719 | 1.000 | 0.19 / 0.583 |
| lexical router + minilm scorer | 0.391 | 0.547 | 0.547 | 0.562 | 1.000 | 0.52 / 0.485 |
| cosine router + minilm scorer | 0.297 | 0.641 | 0.703 | 0.828 | 1.000 | 0.14 / 0.667 |
| csls router + minilm scorer | 0.297 | 0.641 | 0.703 | 0.781 | 1.000 | 0.14 / 0.667 |
| hybrid-union router + minilm scorer | 0.297 | 0.609 | 0.656 | 0.797 | 1.000 | 0.14 / 0.667 |
| hybrid-rrf router + minilm scorer | 0.281 | 0.531 | 0.578 | 0.688 | 1.000 | 0.19 / 0.750 |
| lexical [routed] router + minilm scorer | 0.391 | 0.547 | 0.547 | 0.562 | 1.000 | 0.50 / 0.469 |
| cosine [routed] router + minilm scorer | 0.297 | 0.641 | 0.719 | 0.828 | 1.000 | 0.14 / 0.667 |
| csls [routed] router + minilm scorer | 0.297 | 0.656 | 0.719 | 0.781 | 1.000 | 0.14 / 0.667 |
| hybrid-union [routed] router + minilm scorer | 0.297 | 0.625 | 0.688 | 0.797 | 1.000 | 0.14 / 0.667 |
| hybrid-rrf [routed] router + minilm scorer | 0.281 | 0.547 | 0.609 | 0.688 | 1.000 | 0.20 / 0.692 |
| lexical router + mpnet scorer | 0.406 | 0.547 | 0.547 | 0.562 | 1.000 | 0.28 / 0.611 |
| cosine router + mpnet scorer | 0.328 | 0.750 | 0.859 | 0.906 | 1.000 | 0.05 / 0.667 |
| csls router + mpnet scorer | 0.328 | 0.750 | 0.859 | 0.938 | 1.000 | 0.05 / 0.667 |
| hybrid-union router + mpnet scorer | 0.328 | 0.719 | 0.750 | 0.906 | 1.000 | 0.08 / 0.600 |
| hybrid-rrf router + mpnet scorer | 0.312 | 0.578 | 0.594 | 0.734 | 1.000 | 0.11 / 0.571 |
| lexical [routed] router + mpnet scorer | 0.406 | 0.547 | 0.547 | 0.562 | 1.000 | 0.28 / 0.611 |
| cosine [routed] router + mpnet scorer | 0.344 | 0.766 | 0.859 | 0.906 | 1.000 | 0.03 / 1.000 |
| csls [routed] router + mpnet scorer | 0.344 | 0.766 | 0.859 | 0.938 | 1.000 | 0.03 / 1.000 |
| hybrid-union [routed] router + mpnet scorer | 0.344 | 0.719 | 0.781 | 0.906 | 1.000 | 0.06 / 0.500 |
| hybrid-rrf [routed] router + mpnet scorer | 0.328 | 0.594 | 0.625 | 0.734 | 1.000 | 0.11 / 0.571 |
| lexical router + bge-base scorer | 0.359 | 0.547 | 0.547 | 0.562 | 1.000 | 0.28 / 0.389 |
| cosine router + bge-base scorer | 0.406 | 0.703 | 0.719 | 0.859 | 1.000 | 0.03 / 0.500 |
| csls router + bge-base scorer | 0.406 | 0.703 | 0.734 | 0.859 | 1.000 | 0.11 / 0.571 |
| hybrid-union router + bge-base scorer | 0.406 | 0.734 | 0.750 | 0.875 | 1.000 | 0.11 / 0.429 |
| hybrid-rrf router + bge-base scorer | 0.359 | 0.562 | 0.625 | 0.750 | 1.000 | 0.12 / 0.500 |
| lexical [routed] router + bge-base scorer | 0.359 | 0.547 | 0.547 | 0.562 | 1.000 | 0.31 / 0.400 |
| cosine [routed] router + bge-base scorer | 0.406 | 0.703 | 0.719 | 0.859 | 1.000 | 0.03 / 0.500 |
| csls [routed] router + bge-base scorer | 0.406 | 0.703 | 0.734 | 0.859 | 1.000 | 0.11 / 0.571 |
| hybrid-union [routed] router + bge-base scorer | 0.406 | 0.766 | 0.781 | 0.875 | 1.000 | 0.14 / 0.556 |
| hybrid-rrf [routed] router + bge-base scorer | 0.359 | 0.578 | 0.656 | 0.750 | 1.000 | 0.12 / 0.500 |
| lexical router + e5-large scorer | 0.375 | 0.547 | 0.547 | 0.562 | 1.000 | 0.28 / 0.500 |
| cosine router + e5-large scorer | 0.469 | 0.766 | 0.781 | 0.922 | 1.000 | 0.09 / 0.500 |
| csls router + e5-large scorer | 0.469 | 0.719 | 0.750 | 0.875 | 1.000 | 0.11 / 0.571 |
| hybrid-union router + e5-large scorer | 0.469 | 0.797 | 0.797 | 0.891 | 1.000 | 0.11 / 0.571 |
| hybrid-rrf router + e5-large scorer | 0.453 | 0.594 | 0.594 | 0.734 | 1.000 | 0.14 / 0.444 |
| lexical [routed] router + e5-large scorer | 0.375 | 0.547 | 0.547 | 0.562 | 1.000 | 0.28 / 0.500 |
| cosine [routed] router + e5-large scorer | 0.453 | 0.766 | 0.781 | 0.922 | 1.000 | 0.09 / 0.500 |
| csls [routed] router + e5-large scorer | 0.453 | 0.734 | 0.766 | 0.875 | 1.000 | 0.09 / 0.500 |
| hybrid-union [routed] router + e5-large scorer | 0.453 | 0.797 | 0.812 | 0.891 | 1.000 | 0.09 / 0.500 |
| hybrid-rrf [routed] router + e5-large scorer | 0.438 | 0.609 | 0.625 | 0.734 | 1.000 | 0.11 / 0.429 |
| lexical router + bge-large scorer | 0.375 | 0.547 | 0.547 | 0.562 | 1.000 | 0.50 / 0.531 |
| cosine router + bge-large scorer | 0.344 | 0.703 | 0.781 | 0.906 | 1.000 | 0.16 / 0.600 |
| csls router + bge-large scorer | 0.344 | 0.688 | 0.703 | 0.812 | 1.000 | 0.20 / 0.615 |
| hybrid-union router + bge-large scorer | 0.344 | 0.719 | 0.750 | 0.828 | 1.000 | 0.22 / 0.571 |
| hybrid-rrf router + bge-large scorer | 0.375 | 0.609 | 0.625 | 0.719 | 1.000 | 0.30 / 0.474 |
| lexical [routed] router + bge-large scorer | 0.375 | 0.547 | 0.547 | 0.562 | 1.000 | 0.52 / 0.545 |
| cosine [routed] router + bge-large scorer | 0.344 | 0.703 | 0.781 | 0.906 | 1.000 | 0.16 / 0.600 |
| csls [routed] router + bge-large scorer | 0.344 | 0.703 | 0.719 | 0.812 | 1.000 | 0.20 / 0.615 |
| hybrid-union [routed] router + bge-large scorer | 0.344 | 0.719 | 0.781 | 0.828 | 1.000 | 0.22 / 0.571 |
| hybrid-rrf [routed] router + bge-large scorer | 0.375 | 0.609 | 0.656 | 0.719 | 1.000 | 0.30 / 0.474 |
| lexical router + embeddinggemma-2 scorer | 0.422 | 0.547 | 0.547 | 0.562 | 1.000 | 0.42 / 0.556 |
| cosine router + embeddinggemma-2 scorer | 0.516 | 0.766 | 0.797 | 0.953 | 1.000 | 0.14 / 0.444 |
| csls router + embeddinggemma-2 scorer | 0.500 | 0.766 | 0.797 | 0.969 | 1.000 | 0.16 / 0.500 |
| hybrid-union router + embeddinggemma-2 scorer | 0.500 | 0.781 | 0.781 | 0.906 | 1.000 | 0.16 / 0.500 |
| hybrid-rrf router + embeddinggemma-2 scorer | 0.359 | 0.609 | 0.609 | 0.750 | 1.000 | 0.19 / 0.500 |
| lexical [routed] router + embeddinggemma-2 scorer | 0.422 | 0.547 | 0.547 | 0.562 | 1.000 | 0.41 / 0.615 |
| cosine [routed] router + embeddinggemma-2 scorer | 0.516 | 0.797 | 0.844 | 0.953 | 1.000 | 0.14 / 0.444 |
| csls [routed] router + embeddinggemma-2 scorer | 0.484 | 0.797 | 0.828 | 0.969 | 1.000 | 0.16 / 0.500 |
| hybrid-union [routed] router + embeddinggemma-2 scorer | 0.516 | 0.812 | 0.828 | 0.906 | 1.000 | 0.17 / 0.545 |
| hybrid-rrf [routed] router + embeddinggemma-2 scorer | 0.375 | 0.641 | 0.656 | 0.750 | 1.000 | 0.19 / 0.500 |
| lexical router + fine-tuned embeddinggemma-2 scorer | 0.531 | 0.547 | 0.547 | 0.562 | 1.000 | 0.70 / 0.711 |
| cosine router + fine-tuned embeddinggemma-2 scorer | 0.859 | 0.938 | 0.938 | 1.000 | 1.000 | 0.66 / 0.881 |
| csls router + fine-tuned embeddinggemma-2 scorer | 0.859 | 0.938 | 0.938 | 1.000 | 1.000 | 0.66 / 0.881 |
| hybrid-union router + fine-tuned embeddinggemma-2 scorer | 0.859 | 0.906 | 0.906 | 1.000 | 1.000 | 0.67 / 0.884 |
| hybrid-rrf router + fine-tuned embeddinggemma-2 scorer | 0.625 | 0.641 | 0.641 | 0.766 | 1.000 | 0.66 / 0.690 |
| lexical [routed] router + fine-tuned embeddinggemma-2 scorer | 0.531 | 0.547 | 0.547 | 0.562 | 1.000 | 0.70 / 0.711 |
| cosine [routed] router + fine-tuned embeddinggemma-2 scorer | 0.859 | 0.969 | 0.984 | 1.000 | 1.000 | 0.66 / 0.881 |
| csls [routed] router + fine-tuned embeddinggemma-2 scorer | 0.859 | 0.969 | 1.000 | 1.000 | 1.000 | 0.66 / 0.881 |
| hybrid-union [routed] router + fine-tuned embeddinggemma-2 scorer | 0.859 | 0.938 | 0.953 | 1.000 | 1.000 | 0.66 / 0.881 |
| hybrid-rrf [routed] router + fine-tuned embeddinggemma-2 scorer | 0.641 | 0.672 | 0.688 | 0.766 | 1.000 | 0.64 / 0.707 |

## Selection (validation split only)

- default_small: {"encoder": "minilm", "variant": "cosine", "validation_top1": 0.4745762711864407}
- best_zero_shot: {"encoder": "bge-base", "variant": "cosine", "validation_top1": 0.5084745762711864}
- best_hybrid: {"encoder": "bge-base", "variant": "hybrid-union", "validation_ceiling_at_8": 0.7966101694915254}

## Fine-tuned bi-encoder

- Base `google/embeddinggemma-2`, 258 train pairs, 15 epochs in 290.2 s (budget 60.0 min), best epoch 12
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query
- Checkpoint `models/checkpoints/olaya_embed_gs1.pt` sha256 `3a74d3b0d903533c` (1084153629 bytes)
- Validation top-1: fine-tuned 0.695 vs zero-shot base 0.441 vs lexical 0.169 -> recorded in MANIFEST.json
- History: epoch 1 loss 1.531 val 0.441; epoch 2 loss 1.400 val 0.475; epoch 3 loss 1.151 val 0.542; epoch 4 loss 0.822 val 0.627; epoch 5 loss 0.529 val 0.644; epoch 6 loss 0.324 val 0.644; epoch 7 loss 0.191 val 0.661; epoch 8 loss 0.128 val 0.661; epoch 9 loss 0.106 val 0.678; epoch 10 loss 0.103 val 0.678; epoch 11 loss 0.069 val 0.661; epoch 12 loss 0.038 val 0.695; epoch 13 loss 0.055 val 0.695; epoch 14 loss 0.045 val 0.678; epoch 15 loss 0.049 val 0.678

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out generation batches (register x seed), not classes; the fine-tuned encoder has seen almost every test class in training (see corpus.split_report).
- Zero-shot encoders never saw the corpus; their temperature is the only fitted parameter (calibration split).
- The concept-task metrics have the gold candidate present (5 hard negatives); the full-ontology and engine-path numbers are the deployment comparison.
- The state passed to every router is the raw observation text; the decision head reports used a JSON state wrapper for the engine path.
