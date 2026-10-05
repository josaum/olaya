# OAEI Conference track (ra1) — learned pairwise matcher, leave-one-pair-out

Date: 2026-10-04  ·  Commit: `d275db4738c5`  ·  Runtime: 2128 s (features 30 s, 4 worker processes x 1 torch thread)

## Protocol (read this first)

This is **supervised cross-validation on the track**.  For every held-out ontology pair a pairwise classifier is trained on the reference alignments of the *other 20 pairs* (positives = reference correspondences, negatives = every other same-kind cross pair, class-weighted), the held-out pair is scored, decoded into a 1:1 alignment and evaluated against its own reference; the held-out TP / FP / FN are summed over the 21 folds into the usual micro P / R / F1 for classes (M1), properties (M2) and both (M3).  The decoder (mutual-best / greedy / Hungarian) and the confidence threshold are chosen by an **inner** split of the training pairs (2 inner folds: train on the inner-train pairs, score the inner-held-out pairs, pick the operating point with the best pooled micro-F1 M3), so the held-out pair never selects anything.  *Leave-one-ontology-out* (7 folds) holds out all six pairs of one ontology, so no training pair shares an ontology with the test pairs.  The reference alignments are the only supervision; no entity of the held-out pair is seen with a label.  GraphMatcher, the published best, is itself trained on reference alignments; LogMap, Matcha, edna and StringEquiv are not, so for them the comparison with a trained system is one they did not sign up for.  21 pairs from 7 ontologies are a small sample: differences of 0.01-0.02 F1 between rows are within noise.

Features (63, version 4; `src/olaya/benchmarks/features.py`): `lex` (8); `emb` (12); `hybrid` (2); `rank` (20); `struct` (18); `meta` (3).  Encoders: `intfloat/e5-small-v2`, `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5` (primary `intfloat/e5-small-v2` seeds the structural features and the embedding models).  139585 same-kind cross pairs, 305 reference correspondences.

## Results (micro P / R / F1, cross-validated)

| Model | folds | M1 (classes) | M2 (properties) | M3 (all) | s |
|---|---|---|---|---|---|
| hand-tuned hybrid, LOPO-CV (oaei-conference-ra1) | CV | 0.81 / 0.69 / **0.75** | 0.32 / 0.48 / **0.38** | 0.69 / 0.66 / **0.68** | - |
| hand-tuned hybrid, LOOO-CV (oaei-conference-ra1) | CV | 0.81 / 0.68 / **0.74** | 0.33 / 0.46 / **0.38** | 0.70 / 0.65 / **0.67** | - |
| learned emb_mlp, LOPO-CV | 21 | 0.72 / 0.72 / **0.72** | 0.38 / 0.41 / **0.40** | 0.66 / 0.68 / **0.67** | 1708 |
| learned gcn_mlp, LOPO-CV | 21 | 0.75 / 0.69 / **0.72** | 0.47 / 0.30 / **0.37** | 0.71 / 0.63 / **0.67** | 1692 |
| learned mlp, LOPO-CV | 21 | 0.76 / 0.69 / **0.72** | 0.31 / 0.33 / **0.32** | 0.68 / 0.64 / **0.66** | 474 |
| learned gbt, LOPO-CV | 21 | 0.77 / 0.70 / **0.73** | 0.28 / 0.11 / **0.16** | 0.73 / 0.61 / **0.67** | 545 |
| learned logreg, LOPO-CV **(headline)** | 21 | 0.77 / 0.71 / **0.73** | 0.34 / 0.46 / **0.39** | 0.68 / 0.67 / **0.67** | 182 |
| learned emb_mlp, LOOO-CV | 7 | 0.74 / 0.65 / **0.69** | 0.38 / 0.35 / **0.36** | 0.68 / 0.61 / **0.64** | 456 |
| learned gcn_mlp, LOOO-CV | 7 | 0.72 / 0.71 / **0.71** | 0.35 / 0.32 / **0.33** | 0.67 / 0.65 / **0.66** | 459 |
| learned mlp, LOOO-CV | 7 | 0.72 / 0.67 / **0.69** | 0.32 / 0.30 / **0.31** | 0.66 / 0.61 / **0.64** | 126 |
| learned gbt, LOOO-CV | 7 | 0.73 / 0.71 / **0.72** | 0.21 / 0.13 / **0.16** | 0.68 / 0.62 / **0.65** | 93 |
| learned logreg, LOOO-CV | 7 | 0.79 / 0.67 / **0.73** | 0.35 / 0.42 / **0.38** | 0.70 / 0.64 / **0.67** | 45 |

Models: `logreg` = logistic regression on standardised features, trained on *all* pairs; `mlp` = 2-layer MLP (hidden 64) on the same features; `gbt` = histogram gradient-boosted trees (numpy, 60 trees, depth 3); `emb_mlp` = the scalar features plus the interaction `[p_a, p_b, |p_a-p_b|, p_a*p_b]` of a trained 64-d projection of the frozen e5 label + full-text vectors (the closest thing to GraphMatcher without a GNN); `gcn_mlp` = the same after one training-free GCN step `e' = normalise(e + 0.5 mean_neighbours e)`.  `mlp`, `gbt` and the embedding models train on hard negatives (every positive, the top-5 per entity under the lexical score and the e5 CSLS cosines, plus 300 random pairs per training pair); scoring and decoding always cover every cross pair.  Seeds are fixed; two runs give identical numbers.

### Operating points chosen by the inner selection

- `emb_mlp` (LOPO): decoders ['hungarian', 'mutual'], thresholds 0.70-0.90
- `gcn_mlp` (LOPO): decoders ['hungarian', 'mutual'], thresholds 0.60-0.98
- `mlp` (LOPO): decoders ['hungarian', 'mutual'], thresholds 0.60-0.85
- `gbt` (LOPO): decoders ['hungarian', 'mutual'], thresholds 0.85-0.90
- `logreg` (LOPO): decoders ['greedy', 'hungarian', 'mutual'], thresholds 0.85-0.95

## Ablations (LOPO-CV, micro F1 M1 / M2 / M3)

| Base model | features | M1 | M2 | M3 | ΔM3 vs all features |
|---|---|---|---|---|---|
| logreg | all | 0.735 | 0.393 | 0.674 | - |
| logreg | lexical_only | 0.705 | 0.255 | 0.627 | -0.047 |
| logreg | no_hybrid | 0.730 | 0.375 | 0.671 | -0.003 |
| logreg | no_embeddings | 0.739 | 0.400 | 0.675 | +0.001 |
| logreg | no_structure | 0.719 | 0.346 | 0.654 | -0.020 |
| logreg | no_rank | 0.742 | 0.432 | 0.684 | +0.009 |
| logreg | single_encoder | 0.730 | 0.404 | 0.674 | -0.001 |
| gbt | all | 0.734 | 0.156 | 0.668 | - |
| gbt | lexical_only | 0.702 | 0.159 | 0.644 | -0.024 |
| gbt | no_hybrid | 0.722 | 0.235 | 0.663 | -0.005 |
| gbt | no_embeddings | 0.719 | 0.222 | 0.664 | -0.004 |
| gbt | no_structure | 0.713 | 0.219 | 0.650 | -0.017 |
| gbt | no_rank | 0.719 | 0.315 | 0.660 | -0.008 |
| gbt | single_encoder | 0.711 | 0.179 | 0.641 | -0.027 |

`lexical_only` keeps the lexical scores, their rank features and the meta flags; `no_embeddings` drops every encoder cosine and the ranks derived from them; `no_structure` drops the neighbourhood / depth features; `no_rank` drops all rank / margin / mutual-best features; `single_encoder` keeps only e5-small-v2; `no_hybrid` drops the hand-tuned cascade scores.

## Feature importance

**logreg** - mean standardised coefficient over the 21 fold models (sign = direction):

| feature | value |
|---|---|
| `meta:is_class` | +0.634 |
| `rank:all-MiniLM-L6-v2:col_rank` | +0.478 |
| `rank:e5-small-v2:col_rank` | -0.451 |
| `struct:domains_present` | -0.444 |
| `rank:all-MiniLM-L6-v2:row_margin` | +0.406 |
| `struct:ranges_present` | +0.361 |
| `rank:all-MiniLM-L6-v2:row_rank` | -0.299 |
| `lex:context_jaccard` | -0.291 |
| `rank:bge-small-en-v1.5:col_margin` | +0.282 |
| `struct:domains` | +0.246 |
| `rank:all-MiniLM-L6-v2:col_margin` | -0.238 |
| `emb:all-MiniLM-L6-v2:label_cos` | +0.238 |
| `emb:bge-small-en-v1.5:full_cos` | +0.232 |
| `struct:parents_present` | -0.226 |
| `hybrid:cascade` | +0.205 |

**gbt** - share of total split gain, mean over the 21 fold models:

| feature | value |
|---|---|
| `hybrid:cascade_alpha02` | +0.340 |
| `hybrid:cascade` | +0.142 |
| `struct:neighbour_agg` | +0.128 |
| `emb:bge-small-en-v1.5:full_csls` | +0.061 |
| `rank:all-MiniLM-L6-v2:row_margin` | +0.038 |
| `emb:all-MiniLM-L6-v2:label_cos` | +0.034 |
| `emb:all-MiniLM-L6-v2:full_csls` | +0.031 |
| `rank:bge-small-en-v1.5:row_rank` | +0.015 |
| `struct:neighbour_gain` | +0.014 |
| `rank:bge-small-en-v1.5:row_margin` | +0.014 |
| `lex:levenshtein` | +0.014 |
| `emb:bge-small-en-v1.5:label_csls` | +0.013 |
| `rank:lex:row_margin` | +0.010 |
| `rank:bge-small-en-v1.5:col_margin` | +0.010 |
| `struct:children_present` | +0.009 |

## Per-pair results (headline model `logreg`, LOPO-CV)

| pair | ref | sys | decoder @ th | M1 P / R / F1 | M2 P / R / F1 | M3 P / R / F1 |
|---|---|---|---|---|---|---|
| cmt-conference | 15 | 16 | mutual @ 0.90 | 0.62 / 0.67 / **0.64** | 0.33 / 0.33 / **0.33** | 0.56 / 0.60 / **0.58** |
| cmt-confOf | 16 | 8 | mutual @ 0.90 | 0.67 / 0.40 / **0.50** | 1.00 / 0.33 / **0.50** | 0.75 / 0.38 / **0.50** |
| cmt-edas | 13 | 13 | greedy @ 0.95 | 1.00 / 1.00 / **1.00** | 0.40 / 0.40 / **0.40** | 0.77 / 0.77 / **0.77** |
| cmt-ekaw | 11 | 12 | greedy @ 0.90 | 0.67 / 0.75 / **0.71** | 0.00 / 0.00 / **0.00** | 0.50 / 0.55 / **0.52** |
| cmt-iasted | 4 | 6 | greedy @ 0.90 | 0.80 / 1.00 / **0.89** | 0.00 / 0.00 / **0.00** | 0.67 / 1.00 / **0.80** |
| cmt-sigkdd | 12 | 13 | greedy @ 0.90 | 0.80 / 0.80 / **0.80** | 0.67 / 1.00 / **0.80** | 0.77 / 0.83 / **0.80** |
| conference-confOf | 15 | 16 | hungarian @ 0.90 | 0.69 / 0.82 / **0.75** | 0.67 / 0.50 / **0.57** | 0.69 / 0.73 / **0.71** |
| conference-edas | 17 | 21 | greedy @ 0.90 | 0.69 / 0.64 / **0.67** | 0.25 / 0.67 / **0.36** | 0.52 / 0.65 / **0.58** |
| conference-ekaw | 25 | 19 | mutual @ 0.95 | 0.67 / 0.52 / **0.59** | 1.00 / 0.50 / **0.67** | 0.68 / 0.52 / **0.59** |
| conference-iasted | 14 | 6 | greedy @ 0.95 | 0.83 / 0.38 / **0.53** | 0.00 / 0.00 / **0.00** | 0.83 / 0.36 / **0.50** |
| conference-sigkdd | 15 | 13 | greedy @ 0.95 | 0.91 / 0.83 / **0.87** | 0.50 / 0.33 / **0.40** | 0.85 / 0.73 / **0.79** |
| confOf-edas | 19 | 23 | greedy @ 0.90 | 0.67 / 0.71 / **0.69** | 0.25 / 0.40 / **0.31** | 0.52 / 0.63 / **0.57** |
| confOf-ekaw | 20 | 21 | greedy @ 0.90 | 0.89 / 0.85 / **0.87** | 0.00 / 0.00 / **0.00** | 0.81 / 0.85 / **0.83** |
| confOf-iasted | 9 | 10 | greedy @ 0.90 | 0.56 / 0.56 / **0.56** | 0.00 / 0.00 / **0.00** | 0.50 / 0.56 / **0.53** |
| confOf-sigkdd | 7 | 7 | greedy @ 0.85 | 1.00 / 0.83 / **0.91** | 0.50 / 1.00 / **0.67** | 0.86 / 0.86 / **0.86** |
| edas-ekaw | 23 | 31 | mutual @ 0.90 | 0.75 / 0.95 / **0.84** | 0.43 / 0.75 / **0.55** | 0.68 / 0.91 / **0.78** |
| edas-iasted | 19 | 11 | mutual @ 0.90 | 0.80 / 0.42 / **0.55** | 0.00 / 0.00 / **0.00** | 0.73 / 0.42 / **0.53** |
| edas-sigkdd | 15 | 10 | hungarian @ 0.95 | 1.00 / 0.64 / **0.78** | 0.67 / 0.50 / **0.57** | 0.90 / 0.60 / **0.72** |
| ekaw-iasted | 10 | 12 | hungarian @ 0.95 | 0.73 / 0.80 / **0.76** | 0.00 / 0.00 / **0.00** | 0.67 / 0.80 / **0.73** |
| ekaw-sigkdd | 11 | 10 | hungarian @ 0.95 | 0.89 / 0.73 / **0.80** | 0.00 / 0.00 / **0.00** | 0.80 / 0.73 / **0.76** |
| iasted-sigkdd | 15 | 22 | greedy @ 0.90 | 0.78 / 0.93 / **0.85** | 0.00 / 0.00 / **0.00** | 0.64 / 0.93 / **0.76** |

## Comparison with published OAEI 2023 Conference results (ra1, micro)

| System (M1) | P | R | F1 |
|---|---|---|---|
| GraphMatcher | 0.78 | 0.89 | 0.83 |
| SORBETMtch | 0.78 | 0.75 | 0.76 |
| **olaya hand-tuned hybrid LOPO-CV** (this repo) | 0.81 | 0.69 | 0.75 |
| **learned logreg LOPO-CV** (this repo) | 0.77 | 0.71 | 0.73 |
| **learned gbt LOPO-CV** (this repo) | 0.77 | 0.70 | 0.73 |
| **learned mlp LOPO-CV** (this repo) | 0.76 | 0.69 | 0.72 |
| LogMap | 0.84 | 0.63 | 0.72 |
| Matcha | 0.76 | 0.68 | 0.72 |
| **learned emb_mlp LOPO-CV** (this repo) | 0.72 | 0.72 | 0.72 |
| **learned gcn_mlp LOPO-CV** (this repo) | 0.75 | 0.69 | 0.72 |
| OLaLa | 0.66 | 0.73 | 0.69 |
| ALIN | 0.88 | 0.55 | 0.68 |
| edna | 0.88 | 0.54 | 0.67 |
| LogMapLt | 0.84 | 0.54 | 0.66 |
| AMD | 0.87 | 0.50 | 0.64 |
| LSMatch | 0.88 | 0.50 | 0.64 |
| StringEquiv | 0.88 | 0.50 | 0.64 |
| TOMATO | 0.73 | 0.56 | 0.63 |

| System (M3) | P | R | F1 |
|---|---|---|---|
| GraphMatcher | 0.76 | 0.80 | 0.78 |
| SORBETMtch | 0.78 | 0.64 | 0.70 |
| LogMap | 0.81 | 0.58 | 0.68 |
| **olaya hand-tuned hybrid LOPO-CV** (this repo) | 0.69 | 0.66 | 0.68 |
| **learned logreg LOPO-CV** (this repo) | 0.68 | 0.67 | 0.67 |
| **learned gcn_mlp LOPO-CV** (this repo) | 0.71 | 0.63 | 0.67 |
| Matcha | 0.70 | 0.64 | 0.67 |
| **learned emb_mlp LOPO-CV** (this repo) | 0.66 | 0.68 | 0.67 |
| **learned gbt LOPO-CV** (this repo) | 0.73 | 0.61 | 0.67 |
| **learned mlp LOPO-CV** (this repo) | 0.68 | 0.64 | 0.66 |
| ALIN | 0.88 | 0.47 | 0.61 |
| OLaLa | 0.56 | 0.66 | 0.61 |
| LogMapLt | 0.73 | 0.50 | 0.59 |
| edna | 0.79 | 0.47 | 0.59 |
| AMD | 0.87 | 0.43 | 0.58 |
| LSMatch | 0.88 | 0.42 | 0.57 |
| StringEquiv | 0.80 | 0.43 | 0.56 |
| TOMATO | 0.61 | 0.50 | 0.55 |
| PropMatch | 0.83 | 0.08 | 0.15 |

## Where we land

The best cross-validated learned model is `logreg`: M1 0.77 / 0.71 / **0.735**, M3 0.68 / 0.67 / **0.674** (LOPO-CV). Against the hand-tuned hybrid under the same LOPO protocol (M1 0.746, M3 0.676) that is -0.011 F1 on M1 and -0.001 on M3; recall moves +0.015 (M1) / +0.010 (M3). On M1 it is **0.095 F1 below** GraphMatcher's published 0.83 (P 0.78 / R 0.89); it would rank 3 of 13 in the published table. On M3 it is **0.106 F1 below** GraphMatcher's published 0.78 (P 0.76 / R 0.80); it would rank 4 of 14 in the published table. The published numbers are single runs on all 21 pairs; ours are 21 held-out folds summed, which is the fair protocol for a trained system but not the same experiment.  The ablation table says what carries the result; the per-pair table says where the recall is lost.

Reading: no learned fusion of the *same evidence* (lexical scores, three small encoders, the ontology graph) beats the hand-tuned cascade by more than noise, including the embedding-interaction and GCN-style heads and the hand-tuned score itself as a feature.  The ablations show the features are not redundant (lexical-only and no-structure lose 0.02-0.05 M3) but the ceiling is shared: the residual errors are plausible-but-unreferenced pairs and reference pairs with no lexical, semantic or structural evidence in these small ontologies.  Closing the gap to GraphMatcher's recall needs different evidence (a stronger text model or training signal), not a better combiner.

## Reproduce

```
.venv/bin/python scripts/benchmark_oaei_learned.py
```

Features are cached under `$OLAYA_CACHE_DIR/learned-features` (default `~/.cache/olaya/learned-features`), embeddings under `$OLAYA_CACHE_DIR/embeddings`; with warm caches the run is pure training.  `--models`, `--no-looo`, `--no-ablations`, `--inner-splits` and `--workers` cut the run down.
