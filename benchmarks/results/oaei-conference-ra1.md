# OAEI Conference track (ra1) — honest evaluation

Date: 2026-10-04  ·  Commit: `11015830579a`  ·  Runtime: 311.7 s (lexical 10.6 s; embeddings 301.1 s including model loading, encoding, the configuration grid and the cross-validation; 4-thread CPU)

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
| olaya-embedding only (intfloat/e5-small-v2), a-priori threshold | 0.50 | 0.85 / 0.61 / **0.71** | 0.29 / 0.30 / **0.29** | 0.73 / 0.57 / **0.64** |
| olaya-embedding only (intfloat/e5-small-v2), best threshold (optimistic) | 0.46 | 0.84 / 0.63 / **0.72** | 0.27 / 0.35 / **0.30** | 0.71 / 0.59 / **0.64** |
| olaya-hybrid (intfloat/e5-small-v2), a-priori config | 0.50 | 0.82 / 0.68 / **0.74** | 0.29 / 0.37 / **0.32** | 0.71 / 0.63 / **0.67** |
| olaya-hybrid (intfloat/e5-small-v2), best config on ra1 (optimistic) | 0.51 | 0.83 / 0.69 / **0.75** | 0.33 / 0.48 / **0.39** | 0.71 / 0.66 / **0.68** |
| **olaya-hybrid (intfloat/e5-small-v2), leave-one-pair-out CV (headline)** | CV | 0.82 / 0.69 / **0.75** | 0.32 / 0.48 / **0.39** | 0.70 / 0.66 / **0.68** |
| olaya-hybrid (intfloat/e5-small-v2), leave-one-ontology-out CV | CV | 0.83 / 0.69 / **0.75** | 0.31 / 0.45 / **0.37** | 0.70 / 0.65 / **0.68** |

Macro (mean over pairs) for the lexical a-priori run: M1 P 0.86 / R 0.67 / F1 0.74 (21 pairs), M2 P 0.29 / R 0.30 / F1 0.27 (14 pairs), M3 P 0.76 / R 0.61 / F1 0.67 (21 pairs).

Notes:

- The lexical threshold was fixed at 0.50 *before* sweeping (weighted-Jaccard "shared evidence outweighs unshared").
  The sweep's best M3 threshold is 0.51; selecting a threshold on the test set is optimistic.
- Harness sanity check: our StringEquiv re-implementation lands within a few points of the published
  StringEquiv baseline (ra1-M1 0.88 / 0.50 / 0.64, ra1-M3 0.80 / 0.43 / 0.56); the small gap comes from
  normalisation details of the organisers' implementation that are not published.
- The coherence repair (drop a correspondence when it maps a sub-class pair onto a disjoint pair, using the
  repository's OWL 2 RL closure) *lowers* F1 on ra1.  ra1 is known to contain such incoherences, which is
  precisely why OAEI introduced the repaired references ra2 / rar2; the projection removes true positives.
  It is kept as an opt-in flag and is not part of the headline number.

## Embedding and hybrid matchers

Encoder: `intfloat/e5-small-v2` (revision `main`, commit `ffb93f3bd404`, 384-d,
mean pooling, L2-normalised, max 128 tokens, prefix `'query: '`), run locally on the CPU
with `transformers` only (`olaya.embeddings.TransformerEncoder`).  Entity text = label (camelCase split) +
comment + parent labels (classes) or domain / range labels (properties); the matcher uses the mean of the
label-only and full-description cosines.  Raw cosines have a high floor on this data (unrelated entities
score 0.3-0.5, and more for e5), so they are hubness-corrected with CSLS (Conneau et al. 2018): the mean
similarity of each entity's 10 nearest neighbours is the background and the score is rescaled so that
background = 0 and a perfect match = 1.  No gold data is involved in the calibration.

**Hybrid, a-priori config** (fixed before any grid was run): *cascade* - the lexical matcher runs first
(mutual-best, threshold 0.50) and the calibrated embedding scores only the entities it left unmatched,
same threshold 0.50, no structural propagation.  Weighted sums `(1-w)*lex + w*emb` were also tried and are
in the grid below.

**Honesty rule.**  ra1 is the only test set there is.  Every number selected by looking at ra1 is
optimistic, so three rows are reported: the a-priori config, the best grid point on ra1 (optimistic),
and the *cross-validated* number, which is the headline: for every held-out pair (leave-one-pair-out,
21 folds) the grid point with the best M3 micro-F1 on the other 20 pairs is selected and applied to the
held-out pair; the held-out TP/FP/FN are then summed.  Leave-one-ontology-out (7 folds, all 6 pairs of an
ontology held out, selection on the remaining 15) is the stricter variant.  Two caveats remain and are not
hidden: (1) the *design* choices (cascade rather than weighted sum, CSLS rather than raw cosine, averaging
label and full text) and the choice of the primary encoder were made after looking at ra1 numbers, and
the CV only protects the numeric knobs (combination, text, alpha, threshold); (2) 21 pairs from 7
ontologies are a small sample, so differences of 0.01-0.02 F1 between any two rows are within noise.

### Cross-validated selection (primary encoder)

- Leave-one-pair-out: M1 0.824 / 0.687 / **0.750**, M3 0.704 / 0.656 / **0.679**.  Configs chosen across the 21 folds: `cascade,w=0.0,text=both,alpha=0.2,th=0.50`, `cascade,w=0.0,text=both,alpha=0.2,th=0.51`.
- Leave-one-ontology-out: M1 0.826 / 0.685 / **0.749**, M3 0.705 / 0.649 / **0.676**.  Configs chosen across the 7 folds: `cascade,w=0.0,text=both,alpha=0.2,th=0.50`, `cascade,w=0.0,text=both,alpha=0.2,th=0.51`, `cascade,w=0.0,text=both,alpha=0.3,th=0.55`.
- Best single grid point on all 21 pairs (optimistic): `cascade,w=0.0,text=both,alpha=0.2,th=0.51`.
- Grid: combination in {lexical only, sum w=0.3, sum w=0.5, cascade} x text in {label, both, full} x alpha in {0.0, 0.1, 0.2, 0.3} x threshold 0.30-0.80 step 0.01 (2040 points); selection criterion: micro-F1 M3.

### Grid sweep at the a-priori threshold 0.50 (primary encoder, micro F1 M1 / M3)

| combination | text | alpha 0.0 | alpha 0.1 | alpha 0.2 | alpha 0.3 |
|---|---|---|---|---|---|
| lexical | - | 0.728 / 0.657 | 0.727 / 0.658 | 0.727 / 0.657 | 0.723 / 0.661 |
| sum w=0.3 | label | 0.710 / 0.636 | 0.717 / 0.644 | 0.719 / 0.645 | 0.717 / 0.644 |
| sum w=0.3 | both | 0.720 / 0.647 | 0.727 / 0.652 | 0.728 / 0.651 | 0.728 / 0.653 |
| sum w=0.3 | full | 0.715 / 0.648 | 0.723 / 0.662 | 0.723 / 0.659 | 0.719 / 0.655 |
| sum w=0.5 | label | 0.692 / 0.621 | 0.698 / 0.625 | 0.709 / 0.634 | 0.710 / 0.634 |
| sum w=0.5 | both | 0.696 / 0.628 | 0.703 / 0.633 | 0.710 / 0.637 | 0.710 / 0.643 |
| sum w=0.5 | full | 0.693 / 0.641 | 0.696 / 0.643 | 0.697 / 0.641 | 0.695 / 0.642 |
| cascade | label | 0.726 / 0.645 | 0.726 / 0.644 | 0.726 / 0.649 | 0.735 / 0.658 |
| cascade | both | 0.744 / 0.668 | 0.747 / 0.671 | 0.755 / 0.684 | 0.743 / 0.674 |
| cascade | full | 0.737 / 0.662 | 0.737 / 0.662 | 0.735 / 0.662 | 0.720 / 0.650 |

Reading the sweep: at 0.50 the weighted sum never beats the lexical matcher on M1 and at most matches it
on M3 (the embedding's residual floor dilutes exact lexical evidence); the cascade adds recall for
entities with no lexical evidence
(Place/Location, Trip/Excursion, AccommodationPlace/Hotel, Chairman/Chair) at roughly break-even
precision, which is why the gain is small.  Structural propagation (alpha) helps by about 0.01 on both M1
and M3 for the cascade and is kept as a tunable knob; its benefit is within noise.

### Encoder ablation (cascade, text=both, CSLS; micro F1 M1 / M3)

| encoder | commit | embedding only @0.50 | embedding only, best th | cascade alpha 0 @0.50 | cascade alpha 0.2 @0.50 | cascade, best th (optimistic) |
|---|---|---|---|---|---|---|
| `intfloat/e5-small-v2` | `ffb93f3b` | 0.711 / 0.638 | 0.721 / 0.643 (th 0.46) | 0.744 / 0.668 | 0.755 / 0.684 | 0.743 / 0.670 (th 0.43) |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243` | 0.690 / 0.624 | 0.699 / 0.634 (th 0.52) | 0.734 / 0.666 | 0.740 / 0.671 | 0.739 / 0.672 (th 0.52) |
| `BAAI/bge-small-en-v1.5` | `5c38ec7c` | 0.696 / 0.631 | 0.722 / 0.646 (th 0.39) | 0.731 / 0.668 | 0.736 / 0.669 | 0.739 / 0.677 (th 0.49) |

All small encoders land within 0.01-0.02 of each other.  Two base-size encoders were also measured once
with the same code during development (not part of the default run): `sentence-transformers/all-mpnet-base-v2`
cascade @0.50 M1 0.750 / M3 0.684 (alpha 0), `BAAI/bge-base-en-v1.5` 0.737 / 0.669 (alpha 0.2).  A larger
encoder does not move the ceiling; the limiting factor is that the leftover candidates the embedding has to
decide on are semantically plausible pairs ra1 does not contain (Reviewer/Possible_Reviewer, date/startDate).

## Comparison with published OAEI 2023 results (ra1, micro)

Sources: https://oaei.ontologymatching.org/2023/results/conference/index.html, https://oaei.ontologymatching.org/2023/results/conference/eval.html.  Published numbers use each system's organiser-selected
threshold; ours are the lexical a-priori run and, for the hybrid, the leave-one-pair-out cross-validated
number (no pair selects its own configuration).  Ranked by F1.

| System (M1) | P | R | F1 |
|---|---|---|---|
| GraphMatcher | 0.78 | 0.89 | 0.83 |
| SORBETMtch | 0.78 | 0.75 | 0.76 |
| **olaya-hybrid (LOPO-CV)** (this repo) | 0.82 | 0.69 | 0.75 |
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
| **olaya-hybrid (LOPO-CV)** (this repo) | 0.70 | 0.66 | 0.68 |
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

Where we land (headline = hybrid, leave-one-pair-out CV, M1 F1 0.75 / M3 F1 0.68):
rank 3 of 13 on ra1-M1 and rank 3 of 14 on ra1-M3 by two-decimal F1 (a 0.00 gap in the
two-decimal table is a tie, not a win).  That is 0.08 (M1) / 0.10 (M3)
F1 below GraphMatcher and 0.01 (M1) / 0.02 (M3) below SORBETMtch;
tied with LogMap on M3 and 0.03 above it on M1; above Matcha and the string-based systems.  The hybrid improves on
the purely lexical matcher by +0.022 (M1) / +0.022 (M3) under cross-validation, which is a real but small
gain, within the noise band of a 21-pair test set.  We do not beat or tie the published best on either
modality, and we do not claim state of the art.  GraphMatcher / SORBETMtch additionally learn from the
task (graph attention over the ontology, fine-tuned encoders); nothing here is trained on OntoFarm.

## Per-pair results (hybrid a-priori run)

| Pair | ref | sys | M1 P / R / F1 | M2 P / R / F1 | M3 P / R / F1 |
|---|---|---|---|---|---|
| cmt-conference | 15 | 17 | 0.62 / 0.67 / 0.64 | 0.25 / 0.33 / 0.29 | 0.53 / 0.60 / 0.56 |
| cmt-confOf | 16 | 9 | 0.80 / 0.40 / 0.53 | 0.75 / 0.50 / 0.60 | 0.78 / 0.44 / 0.56 |
| cmt-edas | 13 | 14 | 1.00 / 1.00 / 1.00 | 0.33 / 0.40 / 0.36 | 0.71 / 0.77 / 0.74 |
| cmt-ekaw | 11 | 10 | 0.88 / 0.88 / 0.88 | 0.00 / 0.00 / 0.00 | 0.70 / 0.64 / 0.67 |
| cmt-iasted | 4 | 5 | 0.80 / 1.00 / 0.89 | 0.00 / 0.00 / 0.00 | 0.80 / 1.00 / 0.89 |
| cmt-sigkdd | 12 | 14 | 0.90 / 0.90 / 0.90 | 0.50 / 1.00 / 0.67 | 0.79 / 0.92 / 0.85 |
| conference-confOf | 15 | 13 | 0.80 / 0.73 / 0.76 | 0.67 / 0.50 / 0.57 | 0.77 / 0.67 / 0.71 |
| conference-edas | 17 | 18 | 0.73 / 0.57 / 0.64 | 0.29 / 0.67 / 0.40 | 0.56 / 0.59 / 0.57 |
| conference-ekaw | 25 | 21 | 0.68 / 0.57 / 0.62 | 0.50 / 0.50 / 0.50 | 0.67 / 0.56 / 0.61 |
| conference-iasted | 14 | 8 | 0.86 / 0.46 / 0.60 | 0.00 / 0.00 / 0.00 | 0.75 / 0.43 / 0.55 |
| conference-sigkdd | 15 | 12 | 0.91 / 0.83 / 0.87 | 1.00 / 0.33 / 0.50 | 0.92 / 0.73 / 0.81 |
| confOf-edas | 19 | 20 | 0.90 / 0.64 / 0.75 | 0.20 / 0.40 / 0.27 | 0.55 / 0.58 / 0.56 |
| confOf-ekaw | 20 | 21 | 0.83 / 0.75 / 0.79 | 0.00 / 0.00 / 0.00 | 0.71 / 0.75 / 0.73 |
| confOf-iasted | 9 | 6 | 1.00 / 0.56 / 0.71 | 0.00 / 0.00 / 0.00 | 0.83 / 0.56 / 0.67 |
| confOf-sigkdd | 7 | 4 | 1.00 / 0.67 / 0.80 | 0.00 / 0.00 / 0.00 | 1.00 / 0.57 / 0.73 |
| edas-ekaw | 23 | 22 | 0.78 / 0.74 / 0.76 | 0.25 / 0.25 / 0.25 | 0.68 / 0.65 / 0.67 |
| edas-iasted | 19 | 11 | 0.80 / 0.42 / 0.55 | 0.00 / 0.00 / 0.00 | 0.73 / 0.42 / 0.53 |
| edas-sigkdd | 15 | 8 | 1.00 / 0.64 / 0.78 | 0.00 / 0.00 / 0.00 | 0.88 / 0.47 / 0.61 |
| ekaw-iasted | 10 | 11 | 0.89 / 0.80 / 0.84 | 0.00 / 0.00 / 0.00 | 0.73 / 0.80 / 0.76 |
| ekaw-sigkdd | 11 | 9 | 0.89 / 0.73 / 0.80 | 0.00 / 0.00 / 0.00 | 0.89 / 0.73 / 0.80 |
| iasted-sigkdd | 15 | 20 | 0.76 / 0.87 / 0.81 | 0.00 / 0.00 / 0.00 | 0.65 / 0.87 / 0.74 |

## Matchers

`olaya.benchmarks.oaei.LexicalMatcher`: camelCase / underscore splitting, lowercase, light suffix stemming,
IDF-weighted Jaccard over name tokens (IDF over both ontologies of the pair), a Levenshtein term that only
counts for near-identical spellings (>= 0.90), and a 0.2-weight Jaccard over comment / parent / domain /
range tokens.  Mutual-best 1:1 selection, then the threshold.  Deterministic, standard library only.
Ablations that were tried and rejected: stop-word removal (-0.05 F1 on properties), ungated Levenshtein
(-0.1 F1), TF-IDF cosine instead of weighted Jaccard (slightly worse and threshold-sensitive), non-mutual
selection (-0.08 F1), coherence repair (-0.02 F1, see above).

`olaya.benchmarks.oaei.EmbeddingMatcher` / `HybridMatcher`: see the embedding section.  Rejected on ra1:
raw (uncalibrated) cosine in the cascade (-0.15 F1 at 0.50, best threshold 0.70), `max(lex, emb)` fusion
(-0.1 F1), weighted sums (never above the lexical matcher at the a-priori threshold), label-only text
(-0.02 F1 vs label + description).  Embeddings are cached on disk under `$OLAYA_CACHE_DIR/embeddings`
(default `~/.cache/olaya/embeddings`), keyed by model commit and text sha256; `--no-embeddings` reproduces the
lexical rows offline.

Dataset: 28 upstream files, hashes in the JSON and in `tests/fixtures/oaei/README.md`.
