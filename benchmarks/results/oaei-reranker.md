# OAEI Conference track (ra1) - cross-encoder reranker trained on the Anatomy track

Date: 2026-10-04T01:40:29Z  ·  Commit: `7cbc576a3e11`  ·  Runtime: 195.3 s (base matchers 14.8 s, cross-encoder scoring + grid 89.4 s, ablation 90.1 s; 2 CPU threads)

## What was done

A small cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, 22.7 M parameters, one logit) was
fine-tuned on the **OAEI Anatomy track** (mouse / human reference alignment, 1516 equivalences;
`tests/fixtures/oaei/anatomy/README.md`) and applied, **without any Conference training**, as a reranker
on top of the hybrid matcher's candidate pairs on the 21 Conference ra1 pairs.  The input of the
cross-encoder is `text(A) [SEP] text(B)` with the same entity descriptions the embedding matcher uses
(`build_entity_texts`: label + comment + parents / domain + range; underscores mapped to spaces).

Pipeline per ontology pair: base scores -> top-k candidates per entity (k = 5, union over both sides) ->
cross-encoder probability per candidate -> fused score `(1 - w) * base + w * ce` -> mutual-best 1:1 ->
threshold.  Pairs outside the candidate set are dropped, so the candidate recall is the reranker's ceiling:
cascade: mean reference recall 0.896 · sum: mean reference recall 0.953.

## Training on Anatomy (held-out validation)

- Data: 18716 training pairs (1224 positives) from 2194 mouse source entities; 4848 validation pairs (292 positives) from 549 held-out source entities (split by source entity, fraction 0.8, seed 0).
  Hard negatives: top-5 non-reference candidates per entity under the CSLS-calibrated `intfloat/e5-small-v2` cosine (reference recall inside the candidate set 0.941).  Train / validation pair digests: `a8e610da3d44` / `eb3c29ef38a3`.
- Hyper-parameters: lr 2e-05, batch 32, max length 96, 2 epoch(s) planned, warm-up 0.1, weight decay 0.01, seed 0, time budget 900.0 s, 2 threads.
- Run: 1170 steps, 2.00 epochs in 918.0 s; mean BCE per epoch [0.1331, 0.06706]; loss curve (every 20 steps): 20:0.671, 100:0.310, 180:0.229, 260:0.191, 340:0.169, 420:0.153, 500:0.141, 580:0.134, 660:0.081, 740:0.067, 820:0.064, 900:0.068, 980:0.068, 1060:0.067, 1140:0.066.

| Anatomy validation (held-out source entities) | AUC | hits@1 |
|---|---|---|
| embedding base score (`intfloat/e5-small-v2`, CSLS) | 0.9527 | 0.9135 |
| cross-encoder before fine-tuning (MS MARCO) | 0.9484 | 0.8997 |
| cross-encoder after fine-tuning | 0.9787 | 0.9343 |
| fused w=0.5 after fine-tuning | 0.975 | 0.9239 |

hits@1 = fraction of held-out mouse entities that have a reference match whose best-scored candidate is that match.

## Conference ra1 results (micro P / R / F1; M1 classes, M2 properties, M3 all)

| Run | M1 | M2 | M3 |
|---|---|---|---|
| hybrid base `cascade`, a-priori threshold 0.50 | 0.82 / 0.68 / **0.74** | 0.29 / 0.37 / **0.32** | 0.71 / 0.63 / **0.67** |
| hybrid base `cascade`, best threshold on ra1 (optimistic, `th=0.43`) | 0.79 / 0.70 / **0.74** | 0.30 / 0.48 / **0.37** | 0.67 / 0.67 / **0.67** |
| hybrid base `cascade`, leave-one-pair-out threshold selection | 0.77 / 0.67 / **0.72** | 0.27 / 0.41 / **0.32** | 0.65 / 0.63 / **0.64** |
| hybrid base `sum`, a-priori threshold 0.50 | 0.85 / 0.59 / **0.70** | 0.30 / 0.28 / **0.29** | 0.74 / 0.54 / **0.63** |
| hybrid base `sum`, best threshold on ra1 (optimistic, `th=0.39`) | 0.78 / 0.67 / **0.72** | 0.27 / 0.37 / **0.31** | 0.67 / 0.63 / **0.65** |
| hybrid base `sum`, leave-one-pair-out threshold selection | 0.77 / 0.64 / **0.70** | 0.25 / 0.33 / **0.28** | 0.66 / 0.59 / **0.62** |
| reranker, a-priori config (base=cascade, k=5, w=0.5, th=0.5) | 0.84 / 0.49 / **0.62** | 0.39 / 0.35 / **0.37** | 0.75 / 0.47 / **0.58** |
| reranker, best grid point on ra1 (optimistic, `base=cascade,k=5,w=0.00,th=0.43`) | 0.79 / 0.70 / **0.74** | 0.30 / 0.48 / **0.37** | 0.67 / 0.67 / **0.67** |
| reranker, leave-one-pair-out CV over the a-priori base only | 0.77 / 0.67 / **0.72** | 0.27 / 0.41 / **0.32** | 0.65 / 0.63 / **0.64** |
| **reranker, leave-one-pair-out CV (headline)** | 0.77 / 0.67 / **0.72** | 0.27 / 0.41 / **0.32** | 0.65 / 0.63 / **0.64** |
| ablation: un-fine-tuned MS MARCO cross-encoder, a-priori config | 0.52 / 0.70 / **0.60** | 0.15 / 0.57 / **0.24** | 0.40 / 0.68 / **0.51** |
| ablation: un-fine-tuned MS MARCO cross-encoder, leave-one-pair-out CV | 0.77 / 0.67 / **0.72** | 0.27 / 0.41 / **0.32** | 0.65 / 0.63 / **0.64** |

Configs chosen across the 21 CV folds (fine-tuned reranker): `base=cascade,k=5,w=0.00,th=0.39`, `base=cascade,k=5,w=0.00,th=0.42`, `base=cascade,k=5,w=0.00,th=0.43`, `base=cascade,k=5,w=0.00,th=0.51`, `base=cascade,k=5,w=0.00,th=0.52`.

### Grid slice at the a-priori threshold 0.50 (micro F1 M1 / M2 / M3)

| base | w=0.00 (base only) | w=0.25 | w=0.50 | w=0.75 | w=1.00 (cross-encoder only) |
|---|---|---|---|---|---|
| cascade | 0.744 / 0.324 / 0.668 | 0.706 / 0.330 / 0.638 | 0.623 / 0.368 / 0.578 | 0.587 / 0.354 / 0.549 | 0.573 / 0.325 / 0.531 |
| sum | 0.696 / 0.292 / 0.628 | 0.696 / 0.356 / 0.638 | 0.615 / 0.390 / 0.577 | 0.584 / 0.364 / 0.549 | 0.570 / 0.321 / 0.528 |

### Best threshold per (base, w) on ra1 (optimistic; micro F1 M1 / M2 / M3)

| base, w | best grid point | M1 | M2 | M3 |
|---|---|---|---|---|
| base=cascade,w=0.00 | `base=cascade,k=5,w=0.00,th=0.43` | 0.743 | 0.370 | 0.670 |
| base=cascade,w=0.25 | `base=cascade,k=5,w=0.25,th=0.38` | 0.747 | 0.336 | 0.669 |
| base=cascade,w=0.50 | `base=cascade,k=5,w=0.50,th=0.30` | 0.720 | 0.304 | 0.639 |
| base=cascade,w=0.75 | `base=cascade,k=5,w=0.75,th=0.31` | 0.643 | 0.340 | 0.589 |
| base=cascade,w=1.00 | `base=cascade,k=5,w=1.00,th=0.30` | 0.597 | 0.289 | 0.541 |
| base=sum,w=0.00 | `base=sum,k=5,w=0.00,th=0.39` | 0.721 | 0.312 | 0.645 |
| base=sum,w=0.25 | `base=sum,k=5,w=0.25,th=0.34` | 0.718 | 0.355 | 0.650 |
| base=sum,w=0.50 | `base=sum,k=5,w=0.50,th=0.39` | 0.697 | 0.358 | 0.637 |
| base=sum,w=0.75 | `base=sum,k=5,w=0.75,th=0.30` | 0.628 | 0.337 | 0.575 |
| base=sum,w=1.00 | `base=sum,k=5,w=1.00,th=0.41` | 0.583 | 0.306 | 0.535 |

## Where we land

| System | ra1-M1 F1 | ra1-M3 F1 |
|---|---|---|
| GraphMatcher (published, OAEI 2023) | 0.83 | 0.78 |
| SORBETMtch (published, OAEI 2023) | 0.76 | 0.70 |
| olaya hybrid `cascade` base, LOPO threshold CV (this report) | 0.72 | 0.64 |
| olaya hybrid + Anatomy-trained reranker, LOPO CV (this report) | 0.72 | 0.64 |

Cross-validated, the reranker moves the hybrid base from M1 0.718 / M3 0.641 to M1 0.718 / M3 0.641 (+0.000 / +0.000). That is still below both published systems (GraphMatcher 0.83 / 0.78, SORBETMtch 0.76 / 0.70); this is **not** a state-of-the-art claim. The change versus the base matcher is within the noise of a 21-pair test set.

## Interpretation

- The fine-tuning *did* transfer in the sense that the Anatomy-trained model is clearly better than the
  off-the-shelf MS MARCO cross-encoder on Conference at the a-priori config (M3 F1 0.578 vs 0.507), and on the held-out Anatomy entities it beats the embedding base score.
- It does **not** beat the hybrid base on Conference: in every CV fold the selection picks `w=0.00`
  (base only).  Fusing in the cross-encoder raises precision (and M2) but costs class recall: the
  Anatomy positives are near-identical strings, so the model learned a *strict* notion of equivalence
  and gives low probability to Conference correspondences whose surface forms differ (the kind of
  pair the embedding stage exists to recover).  A single track with one vocabulary is too narrow a
  training signal for cross-domain reranking; Bio-ML or multiple tracks would be the next thing to try.
- The `cascade` base rows here only cross-validate the threshold (no alpha / text grid), which is why
  its LOPO number is a little below the full-grid headline of `oaei-conference-ra1.md`; the a-priori
  row (0.74 / 0.32 / 0.67) matches that report exactly.

## Honesty notes

- The Conference track was never used for training; the only Conference-derived choices are the numeric
  knobs (base, w, threshold), and those are protected by the leave-one-pair-out selection.  The a-priori
  configuration (cascade base, k = 5, w = 0.5, threshold 0.50) was fixed before the grid was run.
- The choice of the Anatomy track as training data, of k = 5, of the base cross-encoder and of the
  fine-tuning recipe were made once and not tuned on Conference.  The hybrid base and its a-priori
  configuration are those of `benchmarks/results/oaei-conference-ra1.md`, recomputed here.
- 21 pairs from 7 ontologies are a small sample; differences of 0.01-0.02 F1 between rows are within noise.
- The reranker only sees the candidate set; a reference correspondence outside the top-5 candidates of
  both entities can never be recovered (candidate recall above).
- The checkpoint is not committed (~90 MB); `scripts/train_reranker.py` reproduces it deterministically
  (seed, thread count and data hashes in the manifest, embedded in the JSON next to this file).

Reproduce: `.venv/bin/python scripts/train_reranker.py && .venv/bin/python scripts/benchmark_oaei_reranker.py`.
