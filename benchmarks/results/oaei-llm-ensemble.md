# OAEI Conference track (ra1) - ensemble of cached LLM judges

Date: 2026-10-07T05:35:21Z  ·  Commit: `eb03924f4ffb`  ·  Runtime: 148.3 s (candidates 16.9 s; 187572 decodings)

**Category: LLM-based ensemble.**  Judges: `qwen__qwen3.8-max` (qwen3.8-max), `qwen__qwen3.8-flash` (qwen3.8-flash), `qwen__glm-5.2` (glm-5.2); anchor `qwen__qwen3.8-max`.  **All verdicts are cached**
(no model was called for this run): prompt version `oaei-judge-v1` (cache versions oaei-judge-v1), system prompt sha256 `874534030e78`, files `benchmarks/results/llm-judgments/qwen__qwen3.8-max.jsonl`, `benchmarks/results/llm-judgments/qwen__qwen3.8-flash.jsonl`, `benchmarks/results/llm-judgments/qwen__glm-5.2.jsonl`.  Candidate generator: `candidates[olaya-hybrid[olaya-embedding[intfloat/e5-small-v2,both,csls10],w=0.50,cascade,alpha=0.00],k=3,min=0.15,rich_k=2]` on `intfloat/e5-small-v2` (k = 3, min = 0.15, rich_k = 2), the same set the single-judge runs were judged on.

## Design

A single judge is recall-strong and precision-weak on the candidate set; the ensemble turns agreement across
judges into precision.  Each judge's cache is read by *pair identity* (source / target ontology and entity:
the record `key` digest encodes the model name and differs between judges), lined up per candidate, and one
combiner turns the row of verdicts into an ordinary `Judgment`, so the single-judge decodings and the
cross-validation run unchanged.  A judge without a verdict for a pair abstains; a pair nobody judged is
`not equivalent, confidence 0` (the `repair` decoding then keeps the hybrid decision).

Combiners (`olaya.benchmarks.llm_ensemble`):

- `majority`: equivalent iff a strict majority of the non-abstaining judges say so; confidence = mean of the
  majority side (with two judges this is unanimity).
- `unanimous`: equivalent iff every non-abstaining judge says so; confidence = min.
- `weighted`: weighted mean of each judge's P(equivalent) (confidence if equivalent, else 1 - confidence),
  equivalent iff >= 0.5; each weight is the judge's own M3 F1 on the candidates of the **training** pairs
  (nested inside the cross-validation; the a-priori / best rows below use weights from all pairs).
- `anchor`: `qwen__qwen3.8-max` says equivalent AND at least one other judge agrees; confidence = mean of the
  agreeing judges.

Decoding and selection are those of `scripts/benchmark_oaei_llm.py`: `judge`, `repair` and `fused` decodings,
threshold 0.30-0.80 (step 0.01), fusion weight in {0.25, 0.5, 0.75}; the headline is cross-validated over the
union grid of every combiner (1020 points): on each fold the combiner, the mode and the
threshold (and the `weighted` judge weights) are chosen on the training pairs only, and the held-out pair(s)
are scored under that choice.  The a-priori row `combiner=majority,mode=repair,th=0.50` was fixed before the grid was run.

## Coverage of the caches

9560 candidate pairs over 21 ontology pairs (ra1 recall ceiling of the set on the pairs used: M1 0.965, M3 0.961).

| judge | records in cache | candidates judged | fraction | on the pairs used |
|---|---|---|---|---|
| `qwen__qwen3.8-max` | 9560 | 9560 / 9560 | 1.000 | 9560 / 9560 (1.000) |
| `qwen__qwen3.8-flash` | 9560 | 9560 / 9560 | 1.000 | 9560 / 9560 (1.000) |
| `qwen__glm-5.2` | 9560 | 9560 / 9560 | 1.000 | 9560 / 9560 (1.000) |

All 21 ontology pairs are covered by every judge.

## Results (micro P / R / F1; M1 classes, M2 properties, M3 all)

### Per judge (reference)

| Run | M1 | M2 | M3 |
|---|---|---|---|
| hybrid alignment alone (cascade, threshold 0.50) | 0.82 / 0.68 / **0.74** | 0.29 / 0.37 / **0.32** | 0.71 / 0.63 / **0.67** |
| `qwen__qwen3.8-max` own verdicts on the candidate set (equivalent, confidence >= 0.5) | 0.70 / 0.87 / **0.78** | 0.44 / 0.70 / **0.54** | 0.65 / 0.85 / **0.74** |
| `qwen__qwen3.8-max` a-priori `mode=repair,th=0.50` | 0.75 / 0.85 / **0.80** | 0.47 / 0.70 / **0.56** | 0.70 / 0.83 / **0.76** |
| `qwen__qwen3.8-max` LOPO-CV over its own grid | 0.79 / 0.82 / **0.80** | 0.48 / 0.59 / **0.53** | 0.73 / 0.79 / **0.76** |
| `qwen__qwen3.8-max` LOOO-CV over its own grid | 0.78 / 0.83 / **0.81** | 0.50 / 0.62 / **0.55** | 0.73 / 0.80 / **0.77** |
| `qwen__qwen3.8-flash` own verdicts on the candidate set (equivalent, confidence >= 0.5) | 0.70 / 0.81 / **0.75** | 0.31 / 0.72 / **0.44** | 0.60 / 0.80 / **0.68** |
| `qwen__qwen3.8-flash` a-priori `mode=repair,th=0.50` | 0.75 / 0.80 / **0.77** | 0.36 / 0.67 / **0.47** | 0.66 / 0.78 / **0.71** |
| `qwen__qwen3.8-flash` LOPO-CV over its own grid | 0.78 / 0.75 / **0.76** | 0.35 / 0.50 / **0.41** | 0.69 / 0.71 / **0.70** |
| `qwen__qwen3.8-flash` LOOO-CV over its own grid | 0.79 / 0.76 / **0.77** | 0.36 / 0.57 / **0.44** | 0.69 / 0.73 / **0.71** |
| `qwen__glm-5.2` own verdicts on the candidate set (equivalent, confidence >= 0.5) | 0.74 / 0.75 / **0.74** | 0.39 / 0.78 / **0.52** | 0.65 / 0.75 / **0.70** |
| `qwen__glm-5.2` a-priori `mode=repair,th=0.50` | 0.77 / 0.74 / **0.75** | 0.46 / 0.78 / **0.58** | 0.69 / 0.74 / **0.72** |
| `qwen__glm-5.2` LOPO-CV over its own grid | 0.83 / 0.71 / **0.76** | 0.45 / 0.54 / **0.49** | 0.75 / 0.68 / **0.71** |
| `qwen__glm-5.2` LOOO-CV over its own grid | 0.82 / 0.72 / **0.77** | 0.40 / 0.51 / **0.45** | 0.73 / 0.69 / **0.71** |

### Per combiner

| Run | M1 | M2 | M3 |
|---|---|---|---|
| `majority` ensemble verdicts on the candidate set (equivalent, confidence >= 0.5) | 0.77 / 0.83 / **0.80** | 0.42 / 0.76 / **0.54** | 0.69 / 0.82 / **0.75** |
| `majority` a-priori `combiner=majority,mode=repair,th=0.50` | 0.79 / 0.82 / **0.80** | 0.47 / 0.76 / **0.58** | 0.72 / 0.81 / **0.76** |
| `majority` best grid point on ra1 (optimistic, `combiner=majority,mode=judge,th=0.30`) | 0.79 / 0.82 / **0.80** | 0.47 / 0.76 / **0.58** | 0.72 / 0.81 / **0.76** |
| `majority` LOPO-CV over its grid | 0.79 / 0.78 / **0.78** | 0.41 / 0.59 / **0.48** | 0.71 / 0.75 / **0.73** |
| `majority` LOOO-CV over its grid | 0.81 / 0.79 / **0.80** | 0.45 / 0.59 / **0.51** | 0.74 / 0.76 / **0.75** |
| `unanimous` ensemble verdicts on the candidate set (equivalent, confidence >= 0.5) | 0.80 / 0.71 / **0.75** | 0.53 / 0.61 / **0.57** | 0.75 / 0.70 / **0.72** |
| `unanimous` a-priori `combiner=unanimous,mode=repair,th=0.50` | 0.81 / 0.70 / **0.75** | 0.54 / 0.61 / **0.57** | 0.76 / 0.69 / **0.72** |
| `unanimous` best grid point on ra1 (optimistic, `combiner=unanimous,mode=fused,w=0.50,th=0.50`) | 0.85 / 0.70 / **0.77** | 0.59 / 0.57 / **0.58** | 0.80 / 0.68 / **0.74** |
| `unanimous` LOPO-CV over its grid | 0.83 / 0.69 / **0.76** | 0.58 / 0.57 / **0.57** | 0.79 / 0.67 / **0.73** |
| `unanimous` LOOO-CV over its grid | 0.85 / 0.69 / **0.76** | 0.49 / 0.53 / **0.51** | 0.78 / 0.67 / **0.72** |
| `weighted` ensemble verdicts on the candidate set (equivalent, confidence >= 0.5) | 0.77 / 0.83 / **0.80** | 0.42 / 0.76 / **0.54** | 0.69 / 0.82 / **0.75** |
| `weighted` a-priori `combiner=weighted,mode=repair,th=0.50` | 0.79 / 0.82 / **0.80** | 0.47 / 0.76 / **0.58** | 0.72 / 0.81 / **0.76** |
| `weighted` best grid point on ra1 (optimistic, `combiner=weighted,mode=judge,th=0.61`) | 0.80 / 0.81 / **0.81** | 0.49 / 0.76 / **0.59** | 0.73 / 0.81 / **0.77** |
| `weighted` LOPO-CV over its grid | 0.80 / 0.81 / **0.81** | 0.49 / 0.76 / **0.59** | 0.73 / 0.81 / **0.77** |
| `weighted` LOOO-CV over its grid | 0.80 / 0.81 / **0.80** | 0.47 / 0.74 / **0.58** | 0.73 / 0.80 / **0.76** |
| `anchor` ensemble verdicts on the candidate set (equivalent, confidence >= 0.5) | 0.78 / 0.83 / **0.80** | 0.46 / 0.67 / **0.55** | 0.72 / 0.80 / **0.76** |
| `anchor` a-priori `combiner=anchor,mode=repair,th=0.50` | 0.80 / 0.81 / **0.81** | 0.50 / 0.67 / **0.57** | 0.74 / 0.79 / **0.77** |
| `anchor` best grid point on ra1 (optimistic, `combiner=anchor,mode=fused,w=0.75,th=0.71`) | 0.83 / 0.79 / **0.81** | 0.52 / 0.59 / **0.55** | 0.78 / 0.76 / **0.77** |
| `anchor` LOPO-CV over its grid | 0.80 / 0.78 / **0.79** | 0.45 / 0.52 / **0.48** | 0.74 / 0.74 / **0.74** |
| `anchor` LOOO-CV over its grid | 0.81 / 0.79 / **0.80** | 0.50 / 0.57 / **0.53** | 0.76 / 0.76 / **0.76** |

### Headline

| Run | M1 | M2 | M3 |
|---|---|---|---|
| a-priori `combiner=majority,mode=repair,th=0.50` (fixed before the grid) | 0.79 / 0.82 / **0.80** | 0.47 / 0.76 / **0.58** | 0.72 / 0.81 / **0.76** |
| best grid point on ra1 (optimistic, `combiner=anchor,mode=fused,w=0.75,th=0.71`) | 0.83 / 0.79 / **0.81** | 0.52 / 0.59 / **0.55** | 0.78 / 0.76 / **0.77** |
| **leave-one-pair-out CV over the union grid (1020 points)** | 0.80 / 0.78 / **0.79** | 0.43 / 0.52 / **0.47** | 0.73 / 0.74 / **0.73** |
| leave-one-ontology-out CV over the union grid (7 folds) | 0.81 / 0.79 / **0.80** | 0.49 / 0.57 / **0.53** | 0.76 / 0.76 / **0.76** |

Configs chosen across the 21 LOPO folds: `combiner=anchor,mode=fused,w=0.50,th=0.55`, `combiner=anchor,mode=fused,w=0.75,th=0.66`, `combiner=anchor,mode=fused,w=0.75,th=0.71`, `combiner=anchor,mode=fused,w=0.75,th=0.72`, `combiner=anchor,mode=repair,th=0.78`, `combiner=weighted,mode=judge,th=0.61`.
Configs chosen across the 7 LOOO folds: `combiner=anchor,mode=fused,w=0.50,th=0.55`, `combiner=anchor,mode=fused,w=0.75,th=0.71`, `combiner=anchor,mode=fused,w=0.75,th=0.72`, `combiner=anchor,mode=repair,th=0.78`, `combiner=weighted,mode=judge,th=0.61`.

## Where we land

Published OAEI 2023 Conference ra1 F1 against this run's LOPO-CV row (ranked by M3 F1, then M1):

| rank | system | M1 F1 | M3 F1 |
|---|---|---|---|
| 1 | GraphMatcher | 0.83 | 0.78 |
| 2 | **olaya (this run)** | 0.79 | 0.73 |
| 3 | SORBETMtch | 0.76 | 0.70 |
| 4 | LogMap | 0.72 | 0.68 |
| 5 | Matcha | 0.72 | 0.67 |
| 6 | edna | 0.67 | 0.59 |

Rank 2 of 6 by M3 F1, behind GraphMatcher and ahead of SORBETMtch, LogMap, Matcha, edna.  These are LLM-ensemble numbers: the matcher is in the LLM-based category, every judge was run once over the same
candidate set with the same prompt and its verdicts committed, the judges never saw the reference alignment, and
the combiner / decoding mode / threshold (and the `weighted` weights) are chosen by leave-one-pair-out
cross-validation.  The best-on-ra1 row is optimistic (selected on the test pairs) and is not the claim.  The
published numbers are the OAEI 2023 campaign results; our evaluation uses the same ra1 reference and the same
micro P / R / F1 but the local scorer, so small differences in the evaluation script are possible.

**Ensemble versus the best single judge, same pairs and protocol:** `qwen__qwen3.8-max` alone reaches LOPO-CV M1 0.804 / M3 0.759; the union-grid ensemble reaches M1 0.787 / M3 0.734 (-0.025 M3), and the best single combiner `weighted` reaches M1 0.807 / M3 0.768 (+0.008 M3).  On this evidence the ensemble does not help yet: the agreement step raises the judge-only precision (see the per-combiner table) but a weaker judge also vetoes true pairs, and a larger union grid costs selection noise in cross-validation.
