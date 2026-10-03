# OAEI Conference track (ra1) — honest evaluation

Date: 2026-10-03  ·  Commit: `c2b189e24312`  ·  Runtime: 12.3 s (matching + 101-point threshold sweep, single CPU)

Protocol: standard OAEI alignment evaluation.  For each of the 21 OntoFarm pairs the matcher scores every
class/class and property/property cross pair, keeps the mutual-best 1:1 correspondences, applies a confidence
threshold and the alignment is scored against the ra1 reference alignment.  P / R / F1 are reported for classes
(M1), properties (M2) and both (M3).  Micro = absolute TP/FP/FN over the 21 pairs (the OAEI convention);
macro = mean of per-pair scores (pairs with no reference correspondences of that kind are skipped).
The gold target is never placed in the candidate set; no metric is derived from synthetic distributions.

## Results (micro P / R / F1)

| Run | threshold | M1 (classes) | M2 (properties) | M3 (all) |
|---|---|---|---|---|
| StringEquiv baseline (this harness) | 1.00 | 0.88 / 0.54 / **0.67** | 0.12 / 0.04 / **0.06** | 0.81 / 0.46 / **0.59** |
| olaya-lexical, a-priori threshold | 0.50 | 0.83 / 0.64 / **0.73** | 0.31 / 0.33 / **0.32** | 0.73 / 0.60 / **0.66** |
| olaya-lexical, best threshold on the test set (optimistic) | 0.51 | 0.85 / 0.64 / **0.73** | 0.31 / 0.33 / **0.32** | 0.74 / 0.59 / **0.66** |
| olaya-lexical + OWL 2 RL coherence repair | 0.50 | 0.84 / 0.61 / **0.71** | 0.31 / 0.33 / **0.32** | 0.73 / 0.57 / **0.64** |

Macro (mean over pairs) for the a-priori run: M1 P 0.86 / R 0.67 / F1 0.74 (21 pairs), M2 P 0.29 / R 0.30 / F1 0.27 (14 pairs), M3 P 0.76 / R 0.61 / F1 0.67 (21 pairs).

Notes:

- The threshold was fixed at 0.50 *before* sweeping (weighted-Jaccard "shared evidence outweighs unshared").
  The sweep's best M3 threshold is 0.51; selecting a threshold on the test set is optimistic,
  so the a-priori row is the number to quote.
- Harness sanity check: our StringEquiv re-implementation lands within a few points of the published
  StringEquiv baseline (ra1-M1 0.88 / 0.50 / 0.64, ra1-M3 0.80 / 0.43 / 0.56); the small gap comes from
  normalisation details of the organisers' implementation that are not published.
- The coherence repair (drop a correspondence when it maps a sub-class pair onto a disjoint pair, using the
  repository's OWL 2 RL closure) *lowers* F1 on ra1.  ra1 is known to contain such incoherences, which is
  precisely why OAEI introduced the repaired references ra2 / rar2; the projection removes true positives.
  It is kept as an opt-in flag and is not part of the headline number.

## Comparison with published OAEI 2023 results (ra1, micro)

Sources: https://oaei.ontologymatching.org/2023/results/conference/index.html, https://oaei.ontologymatching.org/2023/results/conference/eval.html.  Published numbers use each system's organiser-selected
threshold; ours use the a-priori 0.50.  Ranked by F1.

| System (M1) | P | R | F1 |
|---|---|---|---|
| GraphMatcher | 0.78 | 0.89 | 0.83 |
| SORBETMtch | 0.78 | 0.75 | 0.76 |
| **olaya-lexical** (this repo) | 0.83 | 0.64 | 0.73 |
| LogMap | 0.84 | 0.63 | 0.72 |
| Matcha | 0.76 | 0.68 | 0.72 |
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
| Matcha | 0.70 | 0.64 | 0.67 |
| **olaya-lexical** (this repo) | 0.73 | 0.60 | 0.66 |
| ALIN | 0.88 | 0.47 | 0.61 |
| OLaLa | 0.56 | 0.66 | 0.61 |
| LogMapLt | 0.73 | 0.50 | 0.59 |
| edna | 0.79 | 0.47 | 0.59 |
| AMD | 0.87 | 0.43 | 0.58 |
| LSMatch | 0.88 | 0.42 | 0.57 |
| StringEquiv | 0.80 | 0.43 | 0.56 |
| TOMATO | 0.61 | 0.50 | 0.55 |
| PropMatch | 0.83 | 0.08 | 0.15 |

Where we land: rank 3 of 13 on ra1-M1 and rank 5 of 14 on ra1-M3 when sorted by the
two-decimal F1 values.  Published numbers are rounded to two decimals, so a 0.01 gap (e.g. to LogMap and
Matcha on M1) is a tie, not a win.  This is a purely lexical matcher: it beats the string baselines and the
simpler lexical systems, is on par with LogMap / Matcha on classes, and is clearly below GraphMatcher and
SORBETMtch, which use embeddings and structural propagation.  It is not state of the art and we do not
claim it is.

## Per-pair results (a-priori run)

| Pair | ref | sys | M1 P / R / F1 | M2 P / R / F1 | M3 P / R / F1 |
|---|---|---|---|---|---|
| cmt-conference | 15 | 15 | 0.67 / 0.67 / 0.67 | 0.00 / 0.00 / 0.00 | 0.53 / 0.53 / 0.53 |
| cmt-confOf | 16 | 9 | 0.80 / 0.40 / 0.53 | 0.75 / 0.50 / 0.60 | 0.78 / 0.44 / 0.56 |
| cmt-edas | 13 | 13 | 1.00 / 1.00 / 1.00 | 0.40 / 0.40 / 0.40 | 0.77 / 0.77 / 0.77 |
| cmt-ekaw | 11 | 8 | 0.86 / 0.75 / 0.80 | 0.00 / 0.00 / 0.00 | 0.75 / 0.55 / 0.63 |
| cmt-iasted | 4 | 5 | 0.80 / 1.00 / 0.89 | 0.00 / 0.00 / 0.00 | 0.80 / 1.00 / 0.89 |
| cmt-sigkdd | 12 | 14 | 0.90 / 0.90 / 0.90 | 0.50 / 1.00 / 0.67 | 0.79 / 0.92 / 0.85 |
| conference-confOf | 15 | 12 | 0.80 / 0.73 / 0.76 | 1.00 / 0.50 / 0.67 | 0.83 / 0.67 / 0.74 |
| conference-edas | 17 | 17 | 0.73 / 0.57 / 0.64 | 0.33 / 0.67 / 0.44 | 0.59 / 0.59 / 0.59 |
| conference-ekaw | 25 | 19 | 0.71 / 0.52 / 0.60 | 0.50 / 0.50 / 0.50 | 0.68 / 0.52 / 0.59 |
| conference-iasted | 14 | 8 | 0.86 / 0.46 / 0.60 | 0.00 / 0.00 / 0.00 | 0.75 / 0.43 / 0.55 |
| conference-sigkdd | 15 | 11 | 0.91 / 0.83 / 0.87 | 0.00 / 0.00 / 0.00 | 0.91 / 0.67 / 0.77 |
| confOf-edas | 19 | 20 | 0.90 / 0.64 / 0.75 | 0.20 / 0.40 / 0.27 | 0.55 / 0.58 / 0.56 |
| confOf-ekaw | 20 | 20 | 0.83 / 0.75 / 0.79 | 0.00 / 0.00 / 0.00 | 0.75 / 0.75 / 0.75 |
| confOf-iasted | 9 | 5 | 1.00 / 0.44 / 0.62 | 0.00 / 0.00 / 0.00 | 0.80 / 0.44 / 0.57 |
| confOf-sigkdd | 7 | 4 | 1.00 / 0.67 / 0.80 | 0.00 / 0.00 / 0.00 | 1.00 / 0.57 / 0.73 |
| edas-ekaw | 23 | 18 | 0.80 / 0.63 / 0.71 | 0.33 / 0.25 / 0.29 | 0.72 / 0.57 / 0.63 |
| edas-iasted | 19 | 11 | 0.80 / 0.42 / 0.55 | 0.00 / 0.00 / 0.00 | 0.73 / 0.42 / 0.53 |
| edas-sigkdd | 15 | 8 | 1.00 / 0.64 / 0.78 | 0.00 / 0.00 / 0.00 | 0.88 / 0.47 / 0.61 |
| ekaw-iasted | 10 | 7 | 1.00 / 0.60 / 0.75 | 0.00 / 0.00 / 0.00 | 0.86 / 0.60 / 0.71 |
| ekaw-sigkdd | 11 | 8 | 0.88 / 0.64 / 0.74 | 0.00 / 0.00 / 0.00 | 0.88 / 0.64 / 0.74 |
| iasted-sigkdd | 15 | 17 | 0.80 / 0.80 / 0.80 | 0.00 / 0.00 / 0.00 | 0.71 / 0.80 / 0.75 |

## Matcher

`olaya.benchmarks.oaei.LexicalMatcher`: camelCase / underscore splitting, lowercase, light suffix stemming,
IDF-weighted Jaccard over name tokens (IDF over both ontologies of the pair), a Levenshtein term that only
counts for near-identical spellings (>= 0.90), and a 0.2-weight Jaccard over comment / parent / domain /
range tokens.  Mutual-best 1:1 selection, then the threshold.  Deterministic, standard library only.
Ablations that were tried and rejected: stop-word removal (-0.05 F1 on properties), ungated Levenshtein
(-0.1 F1), TF-IDF cosine instead of weighted Jaccard (slightly worse and threshold-sensitive), non-mutual
selection (-0.08 F1), coherence repair (-0.02 F1, see above).

Dataset: 28 upstream files, hashes in the JSON and in `tests/fixtures/oaei/README.md`.
