# OAEI Conference track (ra1) — honest evaluation

Date: 2026-10-04  ·  Commit: `954a65894888`  ·  Runtime: 613.6 s (lexical 20.9 s; embeddings 592.7 s including model loading, encoding, the configuration grid and the cross-validation; 4-thread CPU)

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
| olaya-lexical + structure (flooding), a-priori threshold | 0.50 | 0.89 / 0.42 / **0.57** | 0.18 / 0.04 / **0.07** | 0.83 / 0.36 / **0.51** |
| olaya-lexical + structure (flooding_relative), a-priori threshold | 0.50 | 0.53 / 0.73 / **0.61** | 0.10 / 0.48 / **0.16** | 0.36 / 0.70 / **0.48** |
| olaya-lexical + structure (neighbour), a-priori threshold | 0.50 | 0.86 / 0.63 / **0.73** | 0.35 / 0.33 / **0.34** | 0.77 / 0.58 / **0.66** |
| olaya-embedding only (intfloat/e5-small-v2), a-priori threshold | 0.50 | 0.85 / 0.61 / **0.71** | 0.29 / 0.30 / **0.29** | 0.73 / 0.57 / **0.64** |
| olaya-embedding only (intfloat/e5-small-v2), best threshold (optimistic) | 0.46 | 0.84 / 0.63 / **0.72** | 0.27 / 0.35 / **0.30** | 0.71 / 0.59 / **0.64** |
| olaya-hybrid (intfloat/e5-small-v2), a-priori config | 0.50 | 0.82 / 0.68 / **0.74** | 0.29 / 0.37 / **0.32** | 0.71 / 0.63 / **0.67** |
| olaya-hybrid (intfloat/e5-small-v2, path text), a-priori threshold | 0.50 | 0.82 / 0.69 / **0.75** | 0.33 / 0.48 / **0.39** | 0.70 / 0.66 / **0.68** |
| olaya-hybrid (intfloat/e5-small-v2, path text), best threshold (optimistic) | 0.40 | 0.80 / 0.73 / **0.76** | 0.32 / 0.52 / **0.40** | 0.69 / 0.70 / **0.69** |
| olaya-hybrid (intfloat/e5-small-v2), best config on ra1 (optimistic) | 0.51 | 0.84 / 0.69 / **0.76** | 0.35 / 0.48 / **0.41** | 0.73 / 0.66 / **0.69** |
| **olaya-hybrid (intfloat/e5-small-v2), leave-one-pair-out CV (headline)** | CV | 0.81 / 0.69 / **0.75** | 0.32 / 0.48 / **0.38** | 0.69 / 0.66 / **0.68** |
| olaya-hybrid (intfloat/e5-small-v2), leave-one-ontology-out CV | CV | 0.81 / 0.68 / **0.74** | 0.33 / 0.46 / **0.38** | 0.70 / 0.65 / **0.67** |
| olaya-hybrid + structure (intfloat/e5-small-v2, neighbour w=0.3), a-priori config | 0.50 | 0.86 / 0.65 / **0.74** | 0.38 / 0.43 / **0.40** | 0.76 / 0.62 / **0.68** |
| olaya-hybrid + structure (intfloat/e5-small-v2), best structural config on ra1 (optimistic) | 0.45 | 0.84 / 0.68 / **0.75** | 0.37 / 0.46 / **0.41** | 0.74 / 0.64 / **0.69** |
| olaya-hybrid + structure (intfloat/e5-small-v2), leave-one-pair-out CV (structural grid) | CV | 0.82 / 0.66 / **0.73** | 0.34 / 0.46 / **0.39** | 0.71 / 0.63 / **0.67** |
| olaya-hybrid + structure (intfloat/e5-small-v2), leave-one-ontology-out CV (structural grid) | CV | 0.83 / 0.67 / **0.74** | 0.32 / 0.41 / **0.36** | 0.72 / 0.63 / **0.67** |
| **olaya-hybrid (+structure) (intfloat/e5-small-v2), leave-one-pair-out CV, union grid** | CV | 0.81 / 0.69 / **0.75** | 0.32 / 0.48 / **0.38** | 0.69 / 0.66 / **0.68** |
| olaya-hybrid (+structure) (intfloat/e5-small-v2), leave-one-ontology-out CV, union grid | CV | 0.81 / 0.68 / **0.74** | 0.33 / 0.46 / **0.38** | 0.70 / 0.65 / **0.67** |

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
comment + parent labels (classes) or domain / range labels (properties); the original matcher uses the mean
of the label-only and full-description cosines (`text=both`).  The *path* text added after the error
analysis (`benchmarks/results/oaei-error-analysis.md`) encodes the entity as its chain from the root,
`Conference > Document > Paper` for a class and `Paper > has Author > Person` (domain > label > range) for a
property; `text=path` uses it alone, `text=label+path` averages it with the label cosine.  Raw cosines have a high floor on this data (unrelated entities
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
the CV only protects the numeric knobs (combination, text, alpha, threshold); the *path* text in the grid was
designed after the error analysis of the a-priori hybrid on ra1 (`oaei-error-analysis.md`), which is the same
kind of caveat; (2) 21 pairs from 7
ontologies are a small sample, so differences of 0.01-0.02 F1 between any two rows are within noise.

### Cross-validated selection (primary encoder)

- Leave-one-pair-out: M1 0.810 / 0.691 / **0.746**, M3 0.693 / 0.659 / **0.676**.  Configs chosen across the 21 folds: `cascade,w=0.0,text=label+path,alpha=0.0,th=0.42`, `cascade,w=0.0,text=label+path,alpha=0.2,th=0.51`, `cascade,w=0.0,text=path,alpha=0.0,th=0.40`.
- Leave-one-ontology-out: M1 0.808 / 0.683 / **0.741**, M3 0.698 / 0.649 / **0.673**.  Configs chosen across the 7 folds: `cascade,w=0.0,text=label+path,alpha=0.0,th=0.38`, `cascade,w=0.0,text=label+path,alpha=0.0,th=0.42`, `cascade,w=0.0,text=label+path,alpha=0.0,th=0.51`, `cascade,w=0.0,text=label+path,alpha=0.2,th=0.51`, `cascade,w=0.0,text=path,alpha=0.0,th=0.42`, `sum,w=0.5,text=path,alpha=0.1,th=0.44`.
- Best single grid point on all 21 pairs (optimistic): `cascade,w=0.0,text=label+path,alpha=0.2,th=0.51`.
- Path cascade alone (text=path, alpha 0), only its threshold cross-validated (51 points): leave-one-pair-out M1 0.756 / M3 0.686, leave-one-ontology-out M1 0.755 / M3 0.684.  The full grid above contains these points, but with five texts x four alphas x four combinations the selection on 20 pairs hops
  between `path` and `label+path` configurations and the held-out gain shrinks to the noise band: the path text
  is the one change the error analysis produced that survives cross-validation on its own sub-grid (+0.04 M3),
  but on the full grid the headline is unchanged within noise (0.750 / 0.679 before).  The headline stays the full-grid number.
- Grid: combination in {lexical only, sum w=0.3, sum w=0.5, cascade} x text in {label, both, full, path, label+path} x alpha in {0.0, 0.1, 0.2, 0.3} x threshold 0.30-0.80 step 0.01 (3264 points); selection criterion: micro-F1 M3.

### Grid sweep at the a-priori threshold 0.50 (primary encoder, micro F1 M1 / M3)

| combination | text | alpha 0.0 | alpha 0.1 | alpha 0.2 | alpha 0.3 |
|---|---|---|---|---|---|
| lexical | - | 0.728 / 0.657 | 0.727 / 0.658 | 0.727 / 0.657 | 0.723 / 0.661 |
| sum w=0.3 | label | 0.710 / 0.636 | 0.717 / 0.644 | 0.719 / 0.645 | 0.717 / 0.644 |
| sum w=0.3 | both | 0.720 / 0.647 | 0.727 / 0.652 | 0.728 / 0.651 | 0.728 / 0.653 |
| sum w=0.3 | full | 0.715 / 0.648 | 0.723 / 0.662 | 0.723 / 0.659 | 0.719 / 0.655 |
| sum w=0.3 | path | 0.718 / 0.658 | 0.719 / 0.663 | 0.720 / 0.662 | 0.718 / 0.658 |
| sum w=0.3 | label+path | 0.723 / 0.654 | 0.724 / 0.656 | 0.725 / 0.656 | 0.725 / 0.659 |
| sum w=0.5 | label | 0.692 / 0.621 | 0.698 / 0.625 | 0.709 / 0.634 | 0.710 / 0.634 |
| sum w=0.5 | both | 0.696 / 0.628 | 0.703 / 0.633 | 0.710 / 0.637 | 0.710 / 0.643 |
| sum w=0.5 | full | 0.693 / 0.641 | 0.696 / 0.643 | 0.697 / 0.641 | 0.695 / 0.642 |
| sum w=0.5 | path | 0.694 / 0.647 | 0.702 / 0.653 | 0.703 / 0.655 | 0.691 / 0.645 |
| sum w=0.5 | label+path | 0.696 / 0.629 | 0.704 / 0.635 | 0.707 / 0.640 | 0.705 / 0.642 |
| cascade | label | 0.726 / 0.645 | 0.726 / 0.644 | 0.726 / 0.649 | 0.735 / 0.658 |
| cascade | both | 0.744 / 0.668 | 0.747 / 0.671 | 0.755 / 0.684 | 0.743 / 0.674 |
| cascade | full | 0.737 / 0.662 | 0.737 / 0.662 | 0.735 / 0.662 | 0.720 / 0.650 |
| cascade | path | 0.749 / 0.680 | 0.754 / 0.683 | 0.748 / 0.678 | 0.740 / 0.669 |
| cascade | label+path | 0.746 / 0.684 | 0.751 / 0.689 | 0.756 / 0.692 | 0.749 / 0.685 |

Reading the sweep: at 0.50 the weighted sum never beats the lexical matcher on M1 and at most matches it
on M3 (the embedding's residual floor dilutes exact lexical evidence); the cascade adds recall for
entities with no lexical evidence
(Place/Location, Trip/Excursion, AccommodationPlace/Hotel, Chairman/Chair) at roughly break-even
precision, which is why the gain is small.  Structural propagation (alpha) helps by about 0.01 on both M1
and M3 for the cascade and is kept as a tunable knob; its benefit is within noise.

### Encoder ablation (cascade, text=both, CSLS; micro F1 M1 / M3)

| encoder | commit | embedding only @0.50 | embedding only, best th | cascade alpha 0 @0.50 | cascade alpha 0.2 @0.50 | cascade, best th (optimistic) | cascade + neighbour w=0.3 @0.50 | cascade, path text @0.50 | cascade, path text, best th (optimistic) |
|---|---|---|---|---|---|---|---|---|---|
| `intfloat/e5-small-v2` | `ffb93f3b` | 0.711 / 0.638 | 0.721 / 0.643 (th 0.46) | 0.744 / 0.668 | 0.755 / 0.684 | 0.743 / 0.670 (th 0.43) | 0.741 / 0.681 | 0.749 / 0.680 | 0.764 / 0.693 (th 0.40) |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243` | 0.690 / 0.624 | 0.699 / 0.634 (th 0.52) | 0.734 / 0.666 | 0.740 / 0.671 | 0.739 / 0.672 (th 0.52) | 0.740 / 0.678 | 0.744 / 0.680 | 0.749 / 0.685 (th 0.52) |
| `BAAI/bge-small-en-v1.5` | `5c38ec7c` | 0.696 / 0.631 | 0.722 / 0.646 (th 0.39) | 0.731 / 0.668 | 0.736 / 0.669 | 0.739 / 0.677 (th 0.49) | 0.738 / 0.679 | 0.741 / 0.677 | 0.744 / 0.683 (th 0.52) |

All small encoders land within 0.01-0.02 of each other.  Two base-size encoders were also measured once
with the same code during development (not part of the default run): `sentence-transformers/all-mpnet-base-v2`
cascade @0.50 M1 0.750 / M3 0.684 (alpha 0), `BAAI/bge-base-en-v1.5` 0.737 / 0.669 (alpha 0.2).  A larger
encoder does not move the ceiling; the limiting factor is that the leftover candidates the embedding has to
decide on are semantically plausible pairs ra1 does not contain (Reviewer/Possible_Reviewer, date/startDate).

## Structural propagation

GraphMatcher's edge on this track is *recall* (0.89 on M1 against our 0.69) and it gets it from the
ontology graph.  `olaya.benchmarks.structural` adds that evidence without training: the pairwise similarity
matrix of the a-priori hybrid (cascade, CSLS, alpha 0) seeds an iterative propagation over the two
ontologies' graphs (subClassOf, equivalentClass, disjointWith, property domain / range, inverseOf,
subPropertyOf), after which the usual mutual-best 1:1 selection and threshold apply.  Three rules were
implemented and compared on the same seed:

- `flooding`: Similarity Flooding (Melnik et al. 2002), pairwise connectivity graph with inverse-degree
  propagation coefficients, fixpoint variant C `normalise(s0 + s_i + phi(s0 + s_i))`, global-max
  normalisation; final score `(1 - w) seed + w fixpoint`.  `flooding_relative` rescales the fixpoint to the
  paper's relative similarity (divide by the larger of the row and column maximum) before mixing, because
  the global-max fixpoint sits far below the fixed 0.50 threshold for all but the strongest pairs.
- `neighbour`: `sim' = (1 - w) seed + w * mean over neighbourhood types of max-matched neighbour similarity`
  (parents, children, siblings, equivalents, disjoints, properties a class is domain / range of; domains,
  ranges, inverses, super- / sub-properties of a property), iterated to convergence (a contraction).
- `smooth`: GraphMatcher-style message passing on the embedding vectors without the learned attention:
  `e' = normalise(e + beta * mean(neighbour e))` for a fixed number of rounds, then the usual CSLS cosine
  inside the cascade.

**A-priori structural config** (fixed before the structural grid was swept): `neighbour`, w = 0.3,
at most 10 iterations, threshold 0.50 on the a-priori hybrid seed; the other rules' a-priori points are
`flooding` w = 0.5, `flooding_relative` w = 0.5 (added after the global-normalised fixpoint was seen to
collapse the scale, so not strictly a-priori) and `smooth` beta = 0.5, one round.  The rules were
smoke-tested at a handful of weights before the grid; as with the hybrid, the CV protects only the numeric
knobs (rule, weight, rounds, threshold), not the design choices.

### A-priori rows at threshold 0.50 (micro P / R / F1)

| seed | rule | M1 (classes) | M2 (properties) | M3 (all) |
|---|---|---|---|---|
| lexical | none | 0.835 / 0.645 / **0.728** | 0.306 / 0.326 / **0.316** | 0.731 / 0.597 / **0.657** |
| lexical | flooding | 0.886 / 0.421 / **0.571** | 0.182 / 0.043 / **0.070** | 0.828 / 0.364 / **0.506** |
| lexical | flooding_relative | 0.529 / 0.734 / **0.615** | 0.098 / 0.478 / **0.162** | 0.363 / 0.695 / **0.477** |
| lexical | neighbour | 0.862 / 0.629 / **0.728** | 0.349 / 0.326 / **0.337** | 0.767 / 0.584 / **0.663** |
| hybrid | none | 0.822 / 0.679 / **0.744** | 0.288 / 0.370 / **0.324** | 0.707 / 0.633 / **0.668** |
| hybrid | flooding | 0.889 / 0.556 / **0.684** | 0.280 / 0.152 / **0.197** | 0.807 / 0.495 / **0.614** |
| hybrid | flooding_relative | 0.424 / 0.799 / **0.554** | 0.106 / 0.696 / **0.183** | 0.302 / 0.784 / **0.436** |
| hybrid | neighbour | 0.858 / 0.652 / **0.741** | 0.377 / 0.435 / **0.404** | 0.756 / 0.620 / **0.681** |
| hybrid | smooth | 0.830 / 0.679 / **0.747** | 0.298 / 0.370 / **0.330** | 0.718 / 0.633 / **0.672** |

### Structural grid at the a-priori threshold 0.50 (hybrid seed; micro M1 P / R / F1, M3 F1)

| rule | w = 0.1 | w = 0.2 | w = 0.3 | w = 0.5 |
|---|---|---|---|---|
| flooding | 0.84 / 0.64 / 0.728, M3 0.653 | 0.86 / 0.62 / 0.721, M3 0.649 | 0.88 / 0.59 / 0.708, M3 0.642 | 0.89 / 0.56 / 0.684, M3 0.614 |
| flooding_relative | 0.80 / 0.69 / 0.740, M3 0.665 | 0.76 / 0.71 / 0.739, M3 0.666 | 0.67 / 0.76 / 0.714, M3 0.635 | 0.42 / 0.80 / 0.554, M3 0.436 |
| neighbour | 0.84 / 0.67 / 0.744, M3 0.671 | 0.85 / 0.65 / 0.740, M3 0.677 | 0.86 / 0.65 / 0.741, M3 0.681 | 0.83 / 0.62 / 0.712, M3 0.664 |

| rule | beta = 0.1 | beta = 0.3 | beta = 0.5 | beta = 1.0 |
|---|---|---|---|---|
| smooth, 1 round | 0.82 / 0.68 / 0.747, M3 0.671 | 0.83 / 0.68 / 0.745, M3 0.669 | 0.83 / 0.68 / 0.747, M3 0.672 | 0.82 / 0.66 / 0.732, M3 0.665 |
| smooth, 2 rounds | 0.82 / 0.68 / 0.747, M3 0.670 | 0.83 / 0.68 / 0.746, M3 0.674 | 0.82 / 0.66 / 0.731, M3 0.664 | 0.78 / 0.66 / 0.716, M3 0.650 |

No structure, same seed and threshold: M1 0.744 / M3 0.668.

### Cross-validated selection (structural)

- Structural grid alone (1020 points: 20 rule / weight / rounds settings x threshold 0.30-0.80), leave-one-pair-out: M1 0.823 / 0.664 / **0.735**, M3 0.712 / 0.633 / **0.670**; configs chosen: `structure=neighbour,w=0.20,it=10,th=0.43`, `structure=neighbour,w=0.20,it=10,th=0.44`, `structure=neighbour,w=0.20,it=10,th=0.47`, `structure=neighbour,w=0.30,it=10,th=0.44`, `structure=neighbour,w=0.30,it=10,th=0.45`.
- Structural grid alone, leave-one-ontology-out: M1 0.832 / 0.670 / **0.742**, M3 0.720 / 0.631 / **0.672**; configs chosen: `structure=flooding,w=0.30,it=50,th=0.38`, `structure=neighbour,w=0.20,it=10,th=0.43`, `structure=neighbour,w=0.20,it=10,th=0.45`, `structure=neighbour,w=0.30,it=10,th=0.44`, `structure=neighbour,w=0.30,it=10,th=0.45`, `structure=smooth,w=0.50,it=1,th=0.47`.
- Union of the hybrid grid and the structural grid (4284 points; the selection may pick "no structure"), leave-one-pair-out: M1 0.810 / 0.691 / **0.746**, M3 0.693 / 0.659 / **0.676**; configs chosen: `cascade,w=0.0,text=label+path,alpha=0.0,th=0.42`, `cascade,w=0.0,text=label+path,alpha=0.2,th=0.51`, `cascade,w=0.0,text=path,alpha=0.0,th=0.40`.
- Union grid, leave-one-ontology-out: M1 0.808 / 0.683 / **0.741**, M3 0.698 / 0.649 / **0.673**; configs chosen: `cascade,w=0.0,text=label+path,alpha=0.0,th=0.38`, `cascade,w=0.0,text=label+path,alpha=0.0,th=0.42`, `cascade,w=0.0,text=label+path,alpha=0.0,th=0.51`, `cascade,w=0.0,text=label+path,alpha=0.2,th=0.51`, `cascade,w=0.0,text=path,alpha=0.0,th=0.42`, `sum,w=0.5,text=path,alpha=0.1,th=0.44`.
- For reference, the hybrid grid alone (previous headline), leave-one-pair-out: M1 **0.746** / M3 **0.676**.
- Best structural grid point on all 21 pairs (optimistic): `structure=neighbour,w=0.30,it=10,th=0.45`; best union point: `cascade,w=0.0,text=label+path,alpha=0.2,th=0.51`.

### Does structure buy recall at a lower threshold?  (micro P / R, M1 and M3)

| threshold | no_structure (hybrid a-priori): M1 P / R, M3 P / R | cv_selected (structure=neighbour,w=0.30,it=10): M1 P / R, M3 P / R | flooding_relative,w=0.50,it=50: M1 P / R, M3 P / R |
|---|---|---|---|
| 0.30 | 0.67 / 0.78 (F1 0.719), 0.56 / 0.74 (F1 0.635) | 0.68 / 0.76 (F1 0.718), 0.56 / 0.73 (F1 0.635) | 0.37 / 0.84 (F1 0.513), 0.26 / 0.82 (F1 0.392) |
| 0.35 | 0.72 / 0.74 (F1 0.730), 0.61 / 0.70 (F1 0.652) | 0.75 / 0.73 (F1 0.740), 0.64 / 0.70 (F1 0.665) | 0.38 / 0.83 (F1 0.521), 0.27 / 0.81 (F1 0.399) |
| 0.40 | 0.78 / 0.71 (F1 0.746), 0.65 / 0.68 (F1 0.668) | 0.81 / 0.70 (F1 0.754), 0.70 / 0.67 (F1 0.681) | 0.39 / 0.83 (F1 0.531), 0.28 / 0.81 (F1 0.411) |
| 0.45 | 0.80 / 0.69 (F1 0.739), 0.68 / 0.65 (F1 0.664) | 0.84 / 0.68 (F1 0.750), 0.74 / 0.64 (F1 0.688) | 0.40 / 0.81 (F1 0.538), 0.29 / 0.80 (F1 0.421) |
| 0.50 | 0.82 / 0.68 (F1 0.744), 0.71 / 0.63 (F1 0.668) | 0.86 / 0.65 (F1 0.741), 0.76 / 0.62 (F1 0.681) | 0.42 / 0.80 (F1 0.554), 0.30 / 0.78 (F1 0.436) |
| 0.55 | 0.85 / 0.64 (F1 0.730), 0.74 / 0.58 (F1 0.654) | 0.86 / 0.61 (F1 0.718), 0.77 / 0.58 (F1 0.658) | 0.52 / 0.77 (F1 0.622), 0.40 / 0.75 (F1 0.519) |
| 0.60 | 0.86 / 0.59 (F1 0.697), 0.76 / 0.54 (F1 0.628) | 0.87 / 0.59 (F1 0.705), 0.77 / 0.54 (F1 0.637) | 0.60 / 0.75 (F1 0.669), 0.48 / 0.72 (F1 0.581) |
| 0.65 | 0.88 / 0.58 (F1 0.695), 0.79 / 0.52 (F1 0.626) | 0.89 / 0.51 (F1 0.645), 0.81 / 0.46 (F1 0.587) | 0.70 / 0.72 (F1 0.712), 0.59 / 0.69 (F1 0.636) |
| 0.70 | 0.88 / 0.57 (F1 0.690), 0.79 / 0.50 (F1 0.616) | 0.91 / 0.41 (F1 0.568), 0.88 / 0.37 (F1 0.524) | 0.80 / 0.67 (F1 0.728), 0.67 / 0.64 (F1 0.658) |
| 0.75 | 0.88 / 0.54 (F1 0.671), 0.78 / 0.47 (F1 0.588) | 0.91 / 0.34 (F1 0.499), 0.88 / 0.31 (F1 0.456) | 0.84 / 0.64 (F1 0.725), 0.72 / 0.59 (F1 0.652) |
| 0.80 | 0.88 / 0.54 (F1 0.671), 0.78 / 0.47 (F1 0.586) | 0.91 / 0.24 (F1 0.374), 0.89 / 0.21 (F1 0.335) | 0.89 / 0.55 (F1 0.678), 0.78 / 0.50 (F1 0.614) |

Reading the structural rows: at the a-priori threshold the change against the hybrid is `flooding` -0.060 (M1) / -0.054 (M3), `flooding_relative` -0.190 (M1) / -0.232 (M3), `neighbour` -0.003 (M1) / +0.013 (M3), `smooth` +0.003 (M1) / +0.005 (M3); `neighbour` is the best a-priori rule and the heavy-handed rules (global-normalised flooding, relative
flooding at w = 0.5) are clearly worse because they move the score scale away from the fixed threshold.
The best structural grid point on all 21 pairs is `structure=neighbour,w=0.30,it=10,th=0.45` at M1 0.750 / M3 0.688 (optimistic), which looks like a gain over the best hybrid point, but it does not survive
cross-validation: the union-grid LOPO-CV picks a structural config in 0 of 21 folds
(the structural points win on the 20 training pairs) and the held-out pairs then score
M1 0.746 / M3 0.676, against M1 0.746 / M3 0.676 for the hybrid grid alone (+0.000 M1 / +0.000 M3; the structural grid alone gives M1 0.735 / M3 0.670).  The structural gain on ra1 is a threshold-and-weight fit to the
particular pairs rather than a transferable improvement; the differences are within the noise band of 21 pairs.

Recall versus precision: structure *does* raise recall.  Similarity Flooding (relative, w = 0.5) reaches
M1 recall 0.84 on the curve above (GraphMatcher: 0.89), but at precision 0.37; the extra correspondences it floods in are mostly
structurally plausible pairs ra1 does not contain (sibling classes under a matched parent, properties with
matched domain and range but a different meaning), so precision falls faster than recall rises and F1 does
not improve.  Holding micro precision at >= 0.80 on M1, the best recall on the curve is 0.68 (th 0.50) without structure, 0.70 (th 0.40) with the CV-selected structural config and 0.67 (th 0.70) with flooding.
GraphMatcher's recall comes from an attention network *trained* on the Conference ontologies, which learns
which structural patterns are reliable; a training-free propagation cannot tell a matching sibling pair
from a merely parallel one, and the mutual-best 1:1 selection plus a single global threshold is a blunt
instrument for the recall it adds.

## Comparison with published OAEI 2023 results (ra1, micro)

Sources: https://oaei.ontologymatching.org/2023/results/conference/index.html, https://oaei.ontologymatching.org/2023/results/conference/eval.html.  Published numbers use each system's organiser-selected
threshold; ours are the lexical a-priori run and, for the hybrid, the leave-one-pair-out cross-validated
number (no pair selects its own configuration).  Ranked by F1.

| System (M1) | P | R | F1 |
|---|---|---|---|
| GraphMatcher | 0.78 | 0.89 | 0.83 |
| SORBETMtch | 0.78 | 0.75 | 0.76 |
| **olaya-hybrid (LOPO-CV)** (this repo) | 0.81 | 0.69 | 0.75 |
| **olaya-hybrid+structure (LOPO-CV, union grid)** (this repo) | 0.81 | 0.69 | 0.75 |
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
| **olaya-hybrid (LOPO-CV)** (this repo) | 0.69 | 0.66 | 0.68 |
| **olaya-hybrid+structure (LOPO-CV, union grid)** (this repo) | 0.69 | 0.66 | 0.68 |
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
two-decimal table is a tie, not a win).  GraphMatcher: 0.08 below on M1, 0.10 below on M3; SORBETMtch: 0.01 below on M1, 0.02 below on M3; LogMap: 0.03 above on M1, tied on M3;
above Matcha, OLaLa, ALIN, edna, LogMapLt, StringEquiv, LSMatch, AMD, TOMATO on both modalities.  The headline improves on
the purely lexical matcher by +0.018 (M1) / +0.019 (M3) under cross-validation, which is a real but small
gain, within the noise band of a 21-pair test set.  Structure did not help under cross-validation: the union-grid CV (M1 0.746 / M3 0.676) is +0.000 (M1) / +0.000 (M3) against the hybrid-grid CV, so the headline stays the hybrid; see the structural section for why (recall rises, precision falls faster).
We do not beat or tie the published best on either modality, and we do not claim state of the art.  GraphMatcher / SORBETMtch additionally learn from the
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

`olaya.benchmarks.structural.StructuralMatcher`: see the structural section.  Seeded by the hybrid (or by the
lexical matcher alone with `--no-embeddings`), propagates over subClassOf / equivalentClass / disjointWith /
domain / range / inverseOf / subPropertyOf with Similarity Flooding, neighbour aggregation or embedding
smoothing; dense numpy, well under a second per pair.

Dataset: 28 upstream files, hashes in the JSON and in `tests/fixtures/oaei/README.md`.
