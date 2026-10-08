# Candidate routers on the LLM-distilled incident corpus

Labels are LLM-distilled and filtered, **not human**. **Class-disjoint split**: every test class is unseen in training (seeded 60/15/10/15 over classes); every choice of encoder / variant is made on the validation split and test
is evaluated once per configuration. The state given to every router is the raw observation text.

- Corpus `data/corpora/incident.jsonl`: 109 rows, 13 classes used of 13 ontology concepts; split rows {'train': 61, 'validation': 14, 'calibration': 11, 'test': 23}
- Test classes unseen in training: 2
- Concept tasks (gold + 5 hard negatives): calibration 11, test 23
- Commit `feaca1a5806b`, runtime 95.6 s

## Candidate ceiling on the test split (gold in top-k over the full ontology)

| router | @1 | @3 | @5 | @8 | @16 | @32 | leaf ceiling (top-10 subgraph) | top-1 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| lexical (abstains on ties / no evidence) | 0.478 | 0.739 | 0.826 | 0.870 | 0.870 | 0.870 | 0.870 | 0.478 | 0.826 | 0.623 |
| e5-small cosine | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.913 | 1.000 | 0.957 |
| e5-small csls | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.913 | 1.000 | 0.957 |
| e5-small hybrid-union | 0.478 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.478 | 1.000 | 0.728 |
| e5-small hybrid-rrf | 0.696 | 0.870 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.696 | 0.957 | 0.811 |
| bge-small cosine | 0.609 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.609 | 1.000 | 0.783 |
| bge-small csls | 0.783 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.783 | 1.000 | 0.884 |
| bge-small hybrid-union | 0.478 | 0.870 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.478 | 1.000 | 0.692 |
| bge-small hybrid-rrf | 0.739 | 0.870 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.739 | 0.957 | 0.818 |
| minilm cosine | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| minilm csls | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| minilm hybrid-union | 0.478 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.478 | 1.000 | 0.739 |
| minilm hybrid-rrf | 0.783 | 0.913 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.783 | 0.957 | 0.850 |
| mpnet cosine | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.826 | 1.000 | 0.913 |
| mpnet csls | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| mpnet hybrid-union | 0.478 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.478 | 1.000 | 0.728 |
| mpnet hybrid-rrf | 0.696 | 0.913 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.696 | 0.957 | 0.807 |
| bge-base cosine | 0.826 | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.826 | 1.000 | 0.891 |
| bge-base csls | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.913 | 1.000 | 0.949 |
| bge-base hybrid-union | 0.478 | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.478 | 1.000 | 0.710 |
| bge-base hybrid-rrf | 0.783 | 0.870 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.783 | 0.957 | 0.840 |
| e5-large cosine | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| e5-large csls | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| e5-large hybrid-union | 0.478 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.478 | 1.000 | 0.739 |
| e5-large hybrid-rrf | 0.783 | 0.913 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.783 | 0.957 | 0.850 |
| bge-large cosine | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.957 | 1.000 | 0.978 |
| bge-large csls | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.913 | 1.000 | 0.957 |
| bge-large hybrid-union | 0.478 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.478 | 1.000 | 0.728 |
| bge-large hybrid-rrf | 0.783 | 0.826 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.783 | 0.957 | 0.843 |
| embeddinggemma-2 cosine | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.826 | 1.000 | 0.913 |
| embeddinggemma-2 csls | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.957 | 1.000 | 0.978 |
| embeddinggemma-2 hybrid-union | 0.478 | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.478 | 1.000 | 0.717 |
| embeddinggemma-2 hybrid-rrf | 0.739 | 0.870 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.739 | 0.957 | 0.825 |
| fine-tuned embeddinggemma-2 cosine | 0.739 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.739 | 1.000 | 0.855 |
| fine-tuned embeddinggemma-2 csls | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.913 | 1.000 | 0.942 |
| fine-tuned embeddinggemma-2 hybrid-union | 0.478 | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.478 | 1.000 | 0.706 |
| fine-tuned embeddinggemma-2 hybrid-rrf | 0.696 | 0.870 | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.696 | 0.957 | 0.801 |

Validation-split top-1 (cosine) per encoder, the selection signal: e5-small 0.714, bge-small 0.714, minilm 0.929, mpnet 0.857, bge-base 0.857, e5-large 0.786, bge-large 1.000, embeddinggemma-2 0.786; lexical 0.714; fine-tuned 1.000.

## Concept task on the test split (gold + 5 hard negatives; temperature fitted on the calibration split)

| scorer | temperature | top-1 | top-3 | ECE raw -> scaled | NLL scaled | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical score | 0.5012 | 0.783 | 0.913 | 0.282 -> 0.217 | 0.615 | 0.61 / 1.000 |
| e5-small cosine | 0.0075 | 1.000 | 1.000 | 0.636 -> 0.062 | 0.077 | 0.96 / 1.000 |
| bge-small cosine | 0.0141 | 0.696 | 1.000 | 0.142 -> 0.203 | 0.481 | 0.83 / 0.789 |
| minilm cosine | 0.0375 | 1.000 | 1.000 | 0.076 -> 0.034 | 0.036 | 1.00 / 1.000 |
| mpnet cosine | 0.0500 | 1.000 | 1.000 | 0.112 -> 0.112 | 0.130 | 0.91 / 1.000 |
| bge-base cosine | 0.0089 | 0.870 | 1.000 | 0.313 -> 0.109 | 0.221 | 0.87 / 1.000 |
| e5-large cosine | 0.0094 | 1.000 | 1.000 | 0.639 -> 0.087 | 0.099 | 0.96 / 1.000 |
| bge-large cosine | 0.0158 | 1.000 | 1.000 | 0.358 -> 0.090 | 0.108 | 0.83 / 1.000 |
| embeddinggemma-2 cosine | 0.0149 | 1.000 | 1.000 | 0.470 -> 0.097 | 0.111 | 0.96 / 1.000 |
| fine-tuned embeddinggemma-2 | 0.0167 | 0.913 | 1.000 | 0.381 -> 0.088 | 0.195 | 0.87 / 1.000 |
| trained decision head (committed, `f8271b8ce947`) | - | 0.438 | 0.750 | - -> 0.261 | - | - |

## Deployment path on the test split (`OLaya.system_one`, router top-10, leaves scored, axiom projection)

| router + scorer | top-1 | recall@5 | gold among scored leaves | gold in routed subgraph | axioms verified | coverage@0.7 / acc |
|---|---|---|---|---|---|---|
| lexical router, lexical top-1 (baseline) | 0.478 | 0.826 | 0.870 | - | - | - |
| lexical router + trained head (committed) | 0.188 | 0.688 | - | - | - | - |
| lexical router + e5-small scorer | 0.870 | 0.870 | 0.870 | 0.870 | 1.000 | 0.83 / 1.000 |
| cosine router + e5-small scorer | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.87 / 0.950 |
| csls router + e5-small scorer | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.87 / 0.950 |
| hybrid-union router + e5-small scorer | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.87 / 0.950 |
| hybrid-rrf router + e5-small scorer | 0.870 | 0.957 | 0.957 | 1.000 | 1.000 | 0.83 / 0.947 |
| lexical [routed] router + e5-small scorer | 0.870 | 0.870 | 0.870 | 0.870 | 1.000 | 0.74 / 1.000 |
| cosine [routed] router + e5-small scorer | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.74 / 1.000 |
| csls [routed] router + e5-small scorer | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.74 / 1.000 |
| hybrid-union [routed] router + e5-small scorer | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.78 / 0.944 |
| hybrid-rrf [routed] router + e5-small scorer | 0.870 | 0.957 | 0.957 | 1.000 | 1.000 | 0.74 / 0.941 |
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
| lexical router + minilm scorer | 0.870 | 0.870 | 0.870 | 0.870 | 1.000 | 0.87 / 0.950 |
| cosine router + minilm scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.96 / 1.000 |
| csls router + minilm scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.96 / 1.000 |
| hybrid-union router + minilm scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.96 / 1.000 |
| hybrid-rrf router + minilm scorer | 0.957 | 0.957 | 0.957 | 1.000 | 1.000 | 0.91 / 1.000 |
| lexical [routed] router + minilm scorer | 0.870 | 0.870 | 0.870 | 0.870 | 1.000 | 0.87 / 0.950 |
| cosine [routed] router + minilm scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.91 / 1.000 |
| csls [routed] router + minilm scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.91 / 1.000 |
| hybrid-union [routed] router + minilm scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.96 / 1.000 |
| hybrid-rrf [routed] router + minilm scorer | 0.957 | 0.957 | 0.957 | 1.000 | 1.000 | 0.91 / 1.000 |
| lexical router + mpnet scorer | 0.870 | 0.870 | 0.870 | 0.870 | 1.000 | 0.70 / 1.000 |
| cosine router + mpnet scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.65 / 1.000 |
| csls router + mpnet scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.65 / 1.000 |
| hybrid-union router + mpnet scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.65 / 1.000 |
| hybrid-rrf router + mpnet scorer | 0.957 | 0.957 | 0.957 | 1.000 | 1.000 | 0.61 / 1.000 |
| lexical [routed] router + mpnet scorer | 0.826 | 0.870 | 0.870 | 0.870 | 1.000 | 0.61 / 1.000 |
| cosine [routed] router + mpnet scorer | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 0.52 / 1.000 |
| csls [routed] router + mpnet scorer | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 0.52 / 1.000 |
| hybrid-union [routed] router + mpnet scorer | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 0.57 / 1.000 |
| hybrid-rrf [routed] router + mpnet scorer | 0.783 | 0.957 | 0.957 | 1.000 | 1.000 | 0.57 / 1.000 |
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
| lexical router + e5-large scorer | 0.870 | 0.870 | 0.870 | 0.870 | 1.000 | 0.74 / 1.000 |
| cosine router + e5-large scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.74 / 1.000 |
| csls router + e5-large scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.74 / 1.000 |
| hybrid-union router + e5-large scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.78 / 1.000 |
| hybrid-rrf router + e5-large scorer | 0.957 | 0.957 | 0.957 | 1.000 | 1.000 | 0.78 / 1.000 |
| lexical [routed] router + e5-large scorer | 0.870 | 0.870 | 0.870 | 0.870 | 1.000 | 0.70 / 1.000 |
| cosine [routed] router + e5-large scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.70 / 1.000 |
| csls [routed] router + e5-large scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.70 / 1.000 |
| hybrid-union [routed] router + e5-large scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.70 / 1.000 |
| hybrid-rrf [routed] router + e5-large scorer | 0.957 | 0.957 | 0.957 | 1.000 | 1.000 | 0.70 / 1.000 |
| lexical router + bge-large scorer | 0.870 | 0.870 | 0.870 | 0.870 | 1.000 | 0.78 / 1.000 |
| cosine router + bge-large scorer | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.78 / 1.000 |
| csls router + bge-large scorer | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.78 / 1.000 |
| hybrid-union router + bge-large scorer | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.78 / 1.000 |
| hybrid-rrf router + bge-large scorer | 0.913 | 0.957 | 0.957 | 1.000 | 1.000 | 0.78 / 1.000 |
| lexical [routed] router + bge-large scorer | 0.870 | 0.870 | 0.870 | 0.870 | 1.000 | 0.78 / 1.000 |
| cosine [routed] router + bge-large scorer | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.74 / 1.000 |
| csls [routed] router + bge-large scorer | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.74 / 1.000 |
| hybrid-union [routed] router + bge-large scorer | 0.957 | 1.000 | 1.000 | 1.000 | 1.000 | 0.74 / 1.000 |
| hybrid-rrf [routed] router + bge-large scorer | 0.913 | 0.957 | 0.957 | 1.000 | 1.000 | 0.74 / 1.000 |
| lexical router + embeddinggemma-2 scorer | 0.870 | 0.870 | 0.870 | 0.870 | 1.000 | 0.70 / 1.000 |
| cosine router + embeddinggemma-2 scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.65 / 1.000 |
| csls router + embeddinggemma-2 scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.65 / 1.000 |
| hybrid-union router + embeddinggemma-2 scorer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.65 / 1.000 |
| hybrid-rrf router + embeddinggemma-2 scorer | 0.957 | 0.957 | 0.957 | 1.000 | 1.000 | 0.61 / 1.000 |
| lexical [routed] router + embeddinggemma-2 scorer | 0.826 | 0.870 | 0.870 | 0.870 | 1.000 | 0.65 / 1.000 |
| cosine [routed] router + embeddinggemma-2 scorer | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 0.57 / 1.000 |
| csls [routed] router + embeddinggemma-2 scorer | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 0.57 / 1.000 |
| hybrid-union [routed] router + embeddinggemma-2 scorer | 0.826 | 1.000 | 1.000 | 1.000 | 1.000 | 0.57 / 1.000 |
| hybrid-rrf [routed] router + embeddinggemma-2 scorer | 0.783 | 0.957 | 0.957 | 1.000 | 1.000 | 0.57 / 1.000 |
| lexical router + fine-tuned embeddinggemma-2 scorer | 0.870 | 0.870 | 0.870 | 0.870 | 1.000 | 0.74 / 1.000 |
| cosine router + fine-tuned embeddinggemma-2 scorer | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.70 / 1.000 |
| csls router + fine-tuned embeddinggemma-2 scorer | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.70 / 1.000 |
| hybrid-union router + fine-tuned embeddinggemma-2 scorer | 0.913 | 1.000 | 1.000 | 1.000 | 1.000 | 0.70 / 1.000 |
| hybrid-rrf router + fine-tuned embeddinggemma-2 scorer | 0.870 | 0.957 | 0.957 | 1.000 | 1.000 | 0.65 / 1.000 |
| lexical [routed] router + fine-tuned embeddinggemma-2 scorer | 0.739 | 0.870 | 0.870 | 0.870 | 1.000 | 0.65 / 1.000 |
| cosine [routed] router + fine-tuned embeddinggemma-2 scorer | 0.739 | 1.000 | 1.000 | 1.000 | 1.000 | 0.52 / 1.000 |
| csls [routed] router + fine-tuned embeddinggemma-2 scorer | 0.739 | 1.000 | 1.000 | 1.000 | 1.000 | 0.52 / 1.000 |
| hybrid-union [routed] router + fine-tuned embeddinggemma-2 scorer | 0.739 | 1.000 | 1.000 | 1.000 | 1.000 | 0.52 / 1.000 |
| hybrid-rrf [routed] router + fine-tuned embeddinggemma-2 scorer | 0.696 | 0.957 | 0.957 | 1.000 | 1.000 | 0.52 / 1.000 |

## Selection (validation split only)

- default_small: {"encoder": "minilm", "variant": "cosine", "validation_top1": 0.9285714285714286}
- best_zero_shot: {"encoder": "bge-large", "variant": "cosine", "validation_top1": 1.0}
- best_hybrid: {"encoder": "bge-base", "variant": "hybrid-rrf", "validation_ceiling_at_8": 1.0}

## Fine-tuned bi-encoder

- Base `google/embeddinggemma-2`, 61 train pairs, 6 epochs in 25.1 s (budget 60.0 min), best epoch 3
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query
- Checkpoint `models/checkpoints/olaya_embed_incident.pt` sha256 `c45661e54368f06e` (1084155724 bytes)
- Validation top-1: fine-tuned 1.000 vs zero-shot base 0.786 vs lexical 0.714 -> recorded in MANIFEST.json
- History: epoch 1 loss 1.364 val 0.857; epoch 2 loss 1.246 val 0.929; epoch 3 loss 1.113 val 1.000; epoch 4 loss 0.910 val 1.000; epoch 5 loss 0.830 val 1.000; epoch 6 loss 0.659 val 1.000

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out whole classes: no test class has a training row, so the fine-tuned numbers measure transfer to unseen classes, with fewer rows per split and the class-text side never adapted.
- Zero-shot encoders never saw the corpus; their temperature is the only fitted parameter (calibration split).
- The concept-task metrics have the gold candidate present (5 hard negatives); the full-ontology and engine-path numbers are the deployment comparison.
- The state passed to every router is the raw observation text; the decision head reports used a JSON state wrapper for the engine path.
