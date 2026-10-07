# OAEI Conference track (ra1) - LLM judge stage

Date: 2026-10-07T03:00:48Z  ·  Commit: `9a64abbe5a3a`  ·  Runtime: 3874.4 s (candidates 15.9 s, judging 3847.3 s)

**Category: LLM-based matcher.**  Judge: `qwen[qwen3.8-max,json=on,thinking=off,oaei-judge-v1]` (model `qwen3.8-max`, prompt `oaei-judge-v1`, system prompt sha256 `874534030e78`), judgments in `benchmarks/results/llm-judgments/qwen__qwen3.8-max.jsonl`.  Candidate generator: `candidates[olaya-hybrid[olaya-embedding[intfloat/e5-small-v2,both,csls10],w=0.50,cascade,alpha=0.00],k=3,min=0.15,rich_k=2]` on `intfloat/e5-small-v2`.

## Design

The hybrid matcher misses 112 of the 305 ra1 correspondences (`benchmarks/results/oaei-error-analysis.md`):
49 are mutual-best pairs scoring 0.30-0.49 (under the threshold), 39 are synonyms the encoder scores near
zero (Paper / Contribution, Event / Activity, Attendee / Delegate), 15 are entities the lexical stage
consumed with the wrong partner; and 34 of its 80 false positives are identical-label pairs ra1 rejects.
All of these are decisions a reader of the two ontologies makes easily, so a language-model *judge* is
asked, with the ontology context of both entities, whether each candidate pair is equivalent.

1. **Candidates** (`olaya.benchmarks.llm_judge.CandidateGenerator`): the a-priori hybrid alignment
   (cascade, mutual-best, threshold 0.50; to be *verified*) plus, per entity, the top-k same-kind pairs by
   the dense hybrid score (`0.5 lexical + 0.5 CSLS cosine`) above a minimum, the top-k by the path-text
   CSLS cosine, and the top-`rich_k` by the *raw* cosine of the rich text (the CSLS correction zeroes the
   synonym pairs because the encoder puts them in the hub region; the raw cosine keeps them near the top).
   Config: k = 3, min = 0.15, rich_k = 2.
2. **Prompt**: both entities with label (camelCase split), kind, path from the root, parents / children /
   siblings, properties with domain -> range (classes), domain / range (properties), comment and ontology
   name; 1 pairs per request, deterministic order; structured JSON verdicts
   (`equivalent`, `confidence`, `relation` in {equivalent, subsumes, subsumed_by, related, unrelated},
   `rationale`) through `output_config.format`; system prompt cached (`cache_control`).  Model
   `claude-opus-5-5` by default (`--model claude-sonnet-5-5` / `claude-haiku-4-5`), adaptive thinking,
   `output_config.effort` = `--effort` (default medium), `max_retries=5`, 300 s timeout, key from the
   environment only.
   **OpenAI-compatible backend** (`--judge openai`, alias `--judge zai` for Z.ai): the same prompt through
   `POST <base-url>/chat/completions` (`--base-url` / `LLM_JUDGE_BASE_URL`; key in the variable named by
   `--api-key-env`, default `LLM_JUDGE_API_KEY`; `--provider` label; `--model`; `--list-models`), `temperature 0`,
   `response_format json_object` (`--json-mode`, dropped automatically on a 400 naming it), `thinking` field
   (`--thinking`, Z.ai style; unset unless given), fence / prose-tolerant JSON parser validated against the
   same schema, 180 s timeout, 5 retries with backoff on 429 / 5xx, `--concurrency` (default 4) parallel
   requests; judgments in `llm-judgments/<provider>__<model>.jsonl`.
3. **Cache**: one JSON line per verdict keyed by `sha256(model, prompt version, rendered pair)` under
   `benchmarks/results/llm-judgments/<model>.jsonl`; a rerun sends only unjudged pairs; the per-request
   token usage is kept in the results JSON (`usage_ledger`).
4. **Decoding**: `judge` (judge score = confidence when equivalent, else 0; mutual-best + threshold),
   `repair` (hybrid alignment minus judge-rejected pairs, plus judge-accepted candidates; mutual-best) and
   `fused` (`(1 - w) hybrid + w judge`; mutual-best + threshold).  Threshold 0.30-0.80 (step 0.01) and
   w in {0.25, 0.5, 0.75}; the headline is leave-one-pair-out cross-validated over the grid; the a-priori
   row is `mode=repair,th=0.50`, fixed before the grid was run.

## Candidate recall ceiling (no judge involved)

9560 candidate pairs over 21 ontology pairs (accepted: 273, hybrid: 2001, path: 1315, rich: 8545; a pair can have several origins).  Fraction of ra1 correspondences inside the candidate set (micro): M1 **0.965**, M2 **0.935**, M3 **0.961** (target >= 0.95).  No decoding can exceed this recall.

How the sources were chosen (measured on the 21 pairs with `intfloat/e5-small-v2` before any judging; the
candidate count is the cost driver): hybrid top-3 >= 0.15 plus path top-3 alone reaches M3 recall 0.898
with 2,489 candidates and does not improve with k = 5 (the synonym pairs score under the minimum, not
below the rank cut); lowering the minimum to 0.05 with k = 10 reaches 0.954 at 9,385 candidates; adding
the raw rich-text cosine top-1 / top-2 / top-3 per entity reaches 0.938 / 0.961 / 0.971 at 5,845 / 9,560 /
13,311 candidates.  `rich_k = 2` is the default.  The 12 reference pairs still outside the set at that
setting are mostly property paraphrases (`hasBeenAssigned` / `reviewes`, `assignedTo` / `isReviewedBy`).

| pair | candidates | M1 | M2 | M3 |
|---|---|---|---|---|
| cmt-conference | 421 | 1.000 | 0.667 | 0.933 |
| cmt-confOf | 296 | 1.000 | 0.833 | 0.938 |
| cmt-edas | 467 | 1.000 | 1.000 | 1.000 |
| cmt-ekaw | 404 | 1.000 | 1.000 | 1.000 |
| cmt-iasted | 535 | 1.000 | 1.000 | 1.000 |
| cmt-sigkdd | 333 | 1.000 | 1.000 | 1.000 |
| conference-confOf | 384 | 1.000 | 1.000 | 1.000 |
| conference-edas | 563 | 1.000 | 0.667 | 0.941 |
| conference-ekaw | 476 | 0.957 | 1.000 | 0.960 |
| conference-iasted | 620 | 0.692 | 1.000 | 0.714 |
| conference-sigkdd | 407 | 1.000 | 1.000 | 1.000 |
| confOf-edas | 430 | 0.929 | 1.000 | 0.947 |
| confOf-ekaw | 356 | 1.000 | 1.000 | 1.000 |
| confOf-iasted | 492 | 0.889 | 1.000 | 0.889 |
| confOf-sigkdd | 280 | 1.000 | 1.000 | 1.000 |
| edas-ekaw | 547 | 1.000 | 1.000 | 1.000 |
| edas-iasted | 676 | 0.947 | 1.000 | 0.947 |
| edas-sigkdd | 431 | 1.000 | 1.000 | 1.000 |
| ekaw-iasted | 573 | 1.000 | 1.000 | 1.000 |
| ekaw-sigkdd | 349 | 1.000 | 1.000 | 1.000 |
| iasted-sigkdd | 520 | 0.933 | 1.000 | 0.933 |

## Expected cost of judging every candidate once

9560 requests of 1 pairs; input tokens 5,687,900 (chars/4 estimate), output assumed 573,600 (60 tokens per verdict).  Prices: `qwen` rates per million tokens; 0 = not pinned, fill from the provider's pricing page.

| model | input $/M | output $/M | estimated USD |
|---|---|---|---|
| `qwen3.8-max` | 0.00 | 0.00 | 0.00 |

This run: 9139 verdicts requested, 421 already cached; actual usage 5,888,545 input / 796,809 output tokens (0 cache reads), USD 0.0000.

## Results (micro P / R / F1; M1 classes, M2 properties, M3 all)

| Run | M1 | M2 | M3 |
|---|---|---|---|
| hybrid alignment alone (cascade, threshold 0.50; the candidates' `accepted` set) | 0.82 / 0.68 / **0.74** | 0.29 / 0.37 / **0.32** | 0.71 / 0.63 / **0.67** |
| judge's own verdicts on the candidate set (equivalent, confidence >= 0.5; FN includes pairs outside the set) | 0.70 / 0.87 / **0.78** | 0.44 / 0.70 / **0.54** | 0.65 / 0.85 / **0.74** |
| a-priori decoding `mode=repair,th=0.50` | 0.75 / 0.85 / **0.80** | 0.47 / 0.70 / **0.56** | 0.70 / 0.83 / **0.76** |
| best grid point on ra1 (optimistic, `mode=fused,w=0.75,th=0.71`) | 0.79 / 0.83 / **0.81** | 0.50 / 0.63 / **0.56** | 0.74 / 0.80 / **0.77** |
| **leave-one-pair-out CV over the decoding grid (255 points)** | 0.79 / 0.82 / **0.80** | 0.48 / 0.59 / **0.53** | 0.73 / 0.79 / **0.76** |
| leave-one-pair-out CV, `judge` decoding only | 0.75 / 0.85 / **0.79** | 0.47 / 0.70 / **0.56** | 0.70 / 0.82 / **0.75** |
| leave-one-pair-out CV, `repair` decoding only | 0.75 / 0.85 / **0.80** | 0.47 / 0.70 / **0.56** | 0.70 / 0.83 / **0.76** |
| leave-one-pair-out CV, `fused` decoding only | 0.79 / 0.82 / **0.80** | 0.48 / 0.59 / **0.53** | 0.73 / 0.79 / **0.76** |

Configs chosen across the 21 folds: `mode=fused,w=0.50,th=0.61`, `mode=fused,w=0.75,th=0.71`.

Published (OAEI 2023, ra1 F1): GraphMatcher M1 0.83 / M3 0.78, SORBETMtch M1 0.76 / M3 0.70.

## Reading

These are LLM-judge numbers: the matcher is in the LLM-based category (as OLaLa 2023 and the other
language-model matchers of the Conference track), the model has no access to the reference alignment,
the verdicts are committed so the run is reproducible offline, and the threshold / fusion knobs are
chosen by leave-one-pair-out cross-validation.  The candidate recall ceiling bounds the recall.
