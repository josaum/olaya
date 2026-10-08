# Candidate routers on the LLM-distilled corpora: summary

Labels are **LLM-distilled and filtered, not human** (see `decision-head-summary.md`). Same corpora, same
batch-disjoint splits and same test rows as the decision-head reports. Every encoder / variant choice is made
on the validation split; the test split is scored once per configuration. The ceiling is the fraction of test
rows whose gold class the router puts in its top-8; top-1 / recall@5 use the router's own score over the full
ontology (no gold candidate injected); ECE is on the decision head's `concept` test tasks (gold + 5 hard
negatives) after a temperature fitted on the calibration split.

| domain | system | ceiling@8 | leaf ceiling (top-10) | top-1 | recall@5 | concept task top-1 | ECE |
|---|---|---|---|---|---|---|---|
| gs1 | lexical | 0.562 | 0.562 | 0.281 | 0.531 | 0.524 | 0.136 |
| gs1 | lexical router + trained head (committed) | - | - | 0.141 | 0.469 | 0.190 | 0.208 |
| gs1 | zero-shot e5-small cosine | 0.641 | 0.641 | 0.266 | 0.547 | 0.651 | 0.179 |
| gs1 | zero-shot bge-small cosine (a-priori default) | 0.766 | 0.797 | 0.375 | 0.688 | 0.698 | 0.112 |
| gs1 | best zero-shot by validation: bge-base cosine | 0.734 | 0.797 | 0.406 | 0.703 | 0.746 | 0.122 |
| gs1 | best hybrid by validation: bge-base hybrid-union | 0.812 | 0.812 | 0.281 | 0.719 | 0.746 | 0.122 |
| gs1 | fine-tuned embeddinggemma-2 cosine (`3a74d3b0d903`) | 1.000 | 0.938 | 0.859 | 0.969 | 0.984 | 0.026 |
| incident | lexical | 0.875 | 0.750 | 0.562 | 0.875 | 0.688 | 0.316 |
| incident | lexical router + trained head (committed) | - | - | 0.188 | 0.688 | 0.438 | 0.261 |
| incident | zero-shot e5-small cosine | 0.938 | 0.812 | 0.438 | 0.875 | 0.688 | 0.245 |
| incident | zero-shot bge-small cosine (a-priori default) | 0.938 | 0.812 | 0.562 | 0.938 | 0.875 | 0.125 |
| incident | best zero-shot by validation: bge-base cosine | 0.938 | 0.812 | 0.625 | 0.938 | 1.000 | 0.000 |
| incident | best hybrid by validation: bge-base hybrid-rrf | 0.938 | 0.812 | 0.688 | 0.938 | 1.000 | 0.000 |
| incident | fine-tuned embeddinggemma-2 cosine (`e88f9496e337`) | 1.000 | 0.812 | 0.812 | 0.938 | 0.938 | 0.062 |
| schemaorg | lexical | 0.208 | 0.225 | 0.100 | 0.175 | 0.419 | 0.109 |
| schemaorg | lexical router + trained head (committed) | - | - | 0.075 | 0.142 | 0.179 | 0.111 |
| schemaorg | zero-shot e5-small cosine | 0.617 | 0.600 | 0.275 | 0.483 | 0.803 | 0.046 |
| schemaorg | zero-shot bge-small cosine (a-priori default) | 0.525 | 0.550 | 0.308 | 0.475 | 0.778 | 0.050 |
| schemaorg | best zero-shot by validation: embeddinggemma-2 cosine | 0.800 | 0.742 | 0.450 | 0.683 | 0.923 | 0.080 |
| schemaorg | best hybrid by validation: e5-large hybrid-union | 0.633 | 0.625 | 0.100 | 0.558 | 0.889 | 0.049 |
| schemaorg | fine-tuned embeddinggemma-2 cosine (`e322518d5377`) | 0.900 | 0.767 | 0.642 | 0.850 | 0.966 | 0.021 |

## Deployment path (`OLaya.system_one`, router top-10, leaves scored) on the test split

| domain | router | scorer | top-1 | recall@5 | gold among scored leaves |
|---|---|---|---|---|---|
| gs1 | lexical | lexical top-1 | 0.281 | 0.531 | 0.562 |
| gs1 | lexical | trained head (committed) | 0.141 | 0.469 | - |
| gs1 | lexical | bge-small cosine | 0.453 | 0.547 | 0.547 |
| gs1 | cosine | bge-small cosine | 0.375 | 0.688 | 0.719 |
| gs1 | csls | bge-small cosine | 0.359 | 0.672 | 0.672 |
| gs1 | hybrid-union | bge-small cosine | 0.375 | 0.734 | 0.734 |
| gs1 | hybrid-rrf | bge-small cosine | 0.406 | 0.625 | 0.625 |
| gs1 | lexical [routed] | bge-small cosine | 0.453 | 0.547 | 0.547 |
| gs1 | cosine [routed] | bge-small cosine | 0.375 | 0.688 | 0.719 |
| gs1 | csls [routed] | bge-small cosine | 0.359 | 0.672 | 0.672 |
| gs1 | hybrid-union [routed] | bge-small cosine | 0.375 | 0.750 | 0.766 |
| gs1 | hybrid-rrf [routed] | bge-small cosine | 0.406 | 0.641 | 0.656 |
| gs1 | lexical | bge-base cosine | 0.359 | 0.547 | 0.547 |
| gs1 | cosine | bge-base cosine | 0.406 | 0.703 | 0.719 |
| gs1 | csls | bge-base cosine | 0.406 | 0.703 | 0.734 |
| gs1 | hybrid-union | bge-base cosine | 0.406 | 0.734 | 0.750 |
| gs1 | hybrid-rrf | bge-base cosine | 0.359 | 0.562 | 0.625 |
| gs1 | lexical [routed] | bge-base cosine | 0.359 | 0.547 | 0.547 |
| gs1 | cosine [routed] | bge-base cosine | 0.406 | 0.703 | 0.719 |
| gs1 | csls [routed] | bge-base cosine | 0.406 | 0.703 | 0.734 |
| gs1 | hybrid-union [routed] | bge-base cosine | 0.406 | 0.766 | 0.781 |
| gs1 | hybrid-rrf [routed] | bge-base cosine | 0.359 | 0.578 | 0.656 |
| gs1 | lexical | fine-tuned embeddinggemma-2 | 0.531 | 0.547 | 0.547 |
| gs1 | cosine | fine-tuned embeddinggemma-2 | 0.859 | 0.938 | 0.938 |
| gs1 | csls | fine-tuned embeddinggemma-2 | 0.859 | 0.938 | 0.938 |
| gs1 | hybrid-union | fine-tuned embeddinggemma-2 | 0.859 | 0.906 | 0.906 |
| gs1 | hybrid-rrf | fine-tuned embeddinggemma-2 | 0.625 | 0.641 | 0.641 |
| gs1 | lexical [routed] | fine-tuned embeddinggemma-2 | 0.531 | 0.547 | 0.547 |
| gs1 | cosine [routed] | fine-tuned embeddinggemma-2 | 0.859 | 0.969 | 0.984 |
| gs1 | csls [routed] | fine-tuned embeddinggemma-2 | 0.859 | 0.969 | 1.000 |
| gs1 | hybrid-union [routed] | fine-tuned embeddinggemma-2 | 0.859 | 0.938 | 0.953 |
| gs1 | hybrid-rrf [routed] | fine-tuned embeddinggemma-2 | 0.641 | 0.672 | 0.688 |
| incident | lexical | lexical top-1 | 0.562 | 0.875 | 0.750 |
| incident | lexical | trained head (committed) | 0.188 | 0.688 | - |
| incident | lexical | bge-small cosine | 0.625 | 0.750 | 0.750 |
| incident | cosine | bge-small cosine | 0.562 | 0.812 | 0.812 |
| incident | csls | bge-small cosine | 0.562 | 0.812 | 0.812 |
| incident | hybrid-union | bge-small cosine | 0.562 | 0.750 | 0.812 |
| incident | hybrid-rrf | bge-small cosine | 0.562 | 0.750 | 0.812 |
| incident | lexical [routed] | bge-small cosine | 0.688 | 0.875 | 0.875 |
| incident | cosine [routed] | bge-small cosine | 0.562 | 0.875 | 0.938 |
| incident | csls [routed] | bge-small cosine | 0.562 | 0.875 | 0.875 |
| incident | hybrid-union [routed] | bge-small cosine | 0.562 | 0.812 | 0.938 |
| incident | hybrid-rrf [routed] | bge-small cosine | 0.562 | 0.812 | 0.938 |
| incident | lexical | bge-base cosine | 0.688 | 0.750 | 0.750 |
| incident | cosine | bge-base cosine | 0.625 | 0.750 | 0.812 |
| incident | csls | bge-base cosine | 0.625 | 0.750 | 0.812 |
| incident | hybrid-union | bge-base cosine | 0.625 | 0.750 | 0.812 |
| incident | hybrid-rrf | bge-base cosine | 0.625 | 0.750 | 0.812 |
| incident | lexical [routed] | bge-base cosine | 0.750 | 0.875 | 0.875 |
| incident | cosine [routed] | bge-base cosine | 0.625 | 0.875 | 0.938 |
| incident | csls [routed] | bge-base cosine | 0.625 | 0.875 | 0.938 |
| incident | hybrid-union [routed] | bge-base cosine | 0.625 | 0.875 | 0.938 |
| incident | hybrid-rrf [routed] | bge-base cosine | 0.688 | 0.875 | 0.938 |
| incident | lexical | fine-tuned embeddinggemma-2 | 0.688 | 0.750 | 0.750 |
| incident | cosine | fine-tuned embeddinggemma-2 | 0.750 | 0.812 | 0.812 |
| incident | csls | fine-tuned embeddinggemma-2 | 0.750 | 0.812 | 0.812 |
| incident | hybrid-union | fine-tuned embeddinggemma-2 | 0.750 | 0.812 | 0.812 |
| incident | hybrid-rrf | fine-tuned embeddinggemma-2 | 0.750 | 0.812 | 0.812 |
| incident | lexical [routed] | fine-tuned embeddinggemma-2 | 0.812 | 0.875 | 0.875 |
| incident | cosine [routed] | fine-tuned embeddinggemma-2 | 0.812 | 0.938 | 1.000 |
| incident | csls [routed] | fine-tuned embeddinggemma-2 | 0.812 | 0.938 | 0.938 |
| incident | hybrid-union [routed] | fine-tuned embeddinggemma-2 | 0.812 | 0.938 | 0.938 |
| incident | hybrid-rrf [routed] | fine-tuned embeddinggemma-2 | 0.875 | 0.938 | 0.938 |
| schemaorg | lexical | lexical top-1 | 0.100 | 0.175 | 0.225 |
| schemaorg | lexical | trained head (committed) | 0.075 | 0.142 | - |
| schemaorg | lexical | bge-small cosine | 0.133 | 0.158 | 0.158 |
| schemaorg | cosine | bge-small cosine | 0.300 | 0.458 | 0.475 |
| schemaorg | csls | bge-small cosine | 0.267 | 0.367 | 0.367 |
| schemaorg | hybrid-union | bge-small cosine | 0.300 | 0.400 | 0.400 |
| schemaorg | hybrid-rrf | bge-small cosine | 0.192 | 0.317 | 0.317 |
| schemaorg | lexical [routed] | bge-small cosine | 0.133 | 0.167 | 0.167 |
| schemaorg | cosine [routed] | bge-small cosine | 0.308 | 0.475 | 0.483 |
| schemaorg | csls [routed] | bge-small cosine | 0.275 | 0.375 | 0.375 |
| schemaorg | hybrid-union [routed] | bge-small cosine | 0.308 | 0.400 | 0.400 |
| schemaorg | hybrid-rrf [routed] | bge-small cosine | 0.192 | 0.325 | 0.325 |
| schemaorg | lexical | embeddinggemma-2 cosine | 0.150 | 0.158 | 0.158 |
| schemaorg | cosine | embeddinggemma-2 cosine | 0.475 | 0.658 | 0.692 |
| schemaorg | csls | embeddinggemma-2 cosine | 0.450 | 0.567 | 0.567 |
| schemaorg | hybrid-union | embeddinggemma-2 cosine | 0.475 | 0.592 | 0.592 |
| schemaorg | hybrid-rrf | embeddinggemma-2 cosine | 0.250 | 0.350 | 0.350 |
| schemaorg | lexical [routed] | embeddinggemma-2 cosine | 0.142 | 0.167 | 0.167 |
| schemaorg | cosine [routed] | embeddinggemma-2 cosine | 0.450 | 0.683 | 0.708 |
| schemaorg | csls [routed] | embeddinggemma-2 cosine | 0.442 | 0.583 | 0.583 |
| schemaorg | hybrid-union [routed] | embeddinggemma-2 cosine | 0.450 | 0.617 | 0.617 |
| schemaorg | hybrid-rrf [routed] | embeddinggemma-2 cosine | 0.242 | 0.358 | 0.367 |
| schemaorg | lexical | fine-tuned embeddinggemma-2 | 0.158 | 0.158 | 0.158 |
| schemaorg | cosine | fine-tuned embeddinggemma-2 | 0.625 | 0.742 | 0.767 |
| schemaorg | csls | fine-tuned embeddinggemma-2 | 0.625 | 0.750 | 0.767 |
| schemaorg | hybrid-union | fine-tuned embeddinggemma-2 | 0.625 | 0.725 | 0.725 |
| schemaorg | hybrid-rrf | fine-tuned embeddinggemma-2 | 0.358 | 0.367 | 0.367 |
| schemaorg | lexical [routed] | fine-tuned embeddinggemma-2 | 0.167 | 0.167 | 0.167 |
| schemaorg | cosine [routed] | fine-tuned embeddinggemma-2 | 0.642 | 0.858 | 0.875 |
| schemaorg | csls [routed] | fine-tuned embeddinggemma-2 | 0.642 | 0.833 | 0.867 |
| schemaorg | hybrid-union [routed] | fine-tuned embeddinggemma-2 | 0.642 | 0.792 | 0.792 |
| schemaorg | hybrid-rrf [routed] | fine-tuned embeddinggemma-2 | 0.392 | 0.417 | 0.417 |

## Choices

- A-priori default encoder (best mean validation top-1 among the small encoders e5-small, bge-small, minilm): **bge-small** (bge-small 0.524, e5-small 0.408, minilm 0.447)
- gs1: default_small = {"encoder": "minilm", "variant": "cosine", "validation_top1": 0.4745762711864407}; best_zero_shot = {"encoder": "bge-base", "variant": "cosine", "validation_top1": 0.5084745762711864}; best_hybrid = {"encoder": "bge-base", "variant": "hybrid-union", "validation_ceiling_at_8": 0.7966101694915254}
- incident: default_small = {"encoder": "bge-small", "variant": "cosine", "validation_top1": 0.8235294117647058}; best_zero_shot = {"encoder": "bge-base", "variant": "cosine", "validation_top1": 0.9411764705882353}; best_hybrid = {"encoder": "bge-base", "variant": "hybrid-rrf", "validation_ceiling_at_8": 1.0}
- schemaorg: default_small = {"encoder": "bge-small", "variant": "cosine", "validation_top1": 0.32575757575757575}; best_zero_shot = {"encoder": "embeddinggemma-2", "variant": "cosine", "validation_top1": 0.45454545454545453}; best_hybrid = {"encoder": "e5-large", "variant": "hybrid-union", "validation_ceiling_at_8": 0.7803030303030303}

## Class-disjoint split (transfer to classes never seen in training)

Same corpora, split 60/15/10/15 over *classes* (seeded), so no test class has a training row. The fine-tuned
encoder here is trained on the class-disjoint train split; the committed checkpoints are not reused.

| domain | test rows (classes) | system | ceiling@8 | top-1 | recall@5 | engine top-1 (leaves) | engine top-1 [routed] |
|---|---|---|---|---|---|---|---|
| gs1 | 61 (9) | lexical | 0.443 | 0.131 | 0.426 | - | - |
| gs1 | 61 (9) | bge-small cosine | 0.721 | 0.344 | 0.607 | 0.344 | 0.344 |
| gs1 | 61 (9) | fine-tuned embeddinggemma-2 (class-disjoint train) cosine | 0.869 | 0.557 | 0.820 | 0.574 | 0.557 |
| incident | 23 (2) | lexical | 0.870 | 0.478 | 0.826 | - | - |
| incident | 23 (2) | bge-small cosine | 1.000 | 0.609 | 1.000 | 0.609 | 0.609 |
| incident | 23 (2) | fine-tuned embeddinggemma-2 (class-disjoint train) cosine | 1.000 | 0.739 | 1.000 | 0.913 | 0.739 |
| schemaorg | 127 (18) | lexical | 0.394 | 0.205 | 0.386 | - | - |
| schemaorg | 127 (18) | bge-small cosine | 0.535 | 0.291 | 0.457 | 0.283 | 0.291 |
| schemaorg | 127 (18) | fine-tuned embeddinggemma-2 (class-disjoint train) cosine | 0.858 | 0.598 | 0.835 | 0.614 | 0.598 |

## What is and is not established

- gs1: zero-shot bge-small cosine top-1 0.375 vs lexical 0.281 on 64 test rows (better, no training on the corpus; a-priori choice); fine-tuned embeddinggemma-2 top-1 0.859 (selected on validation, test scored once; 4 of the test classes were unseen in training, every other test class was seen through other generation batches).
- incident: zero-shot bge-small cosine top-1 0.562 vs lexical 0.562 on 16 test rows (not better, no training on the corpus; a-priori choice); fine-tuned embeddinggemma-2 top-1 0.812 (selected on validation, test scored once; 1 of the test classes were unseen in training, every other test class was seen through other generation batches); the calibration-split temperature fit saturated for bge-small, minilm, mpnet, bge-base, e5-large, bge-large, embeddinggemma-2 (9 calibration tasks), so their ECE is not a calibrated-probability claim.
- schemaorg: zero-shot bge-small cosine top-1 0.308 vs lexical 0.100 on 120 test rows (better, no training on the corpus; a-priori choice); fine-tuned embeddinggemma-2 top-1 0.642 (selected on validation, test scored once; 9 of the test classes were unseen in training, every other test class was seen through other generation batches).

Established on these corpora: an off-the-shelf sentence encoder over path-context class texts raises the
candidate ceiling and the zero-shot top-1 above the lexical router without any training, the hybrid routers
do not beat the best embedding ranking on the ceiling once the embedding is decent, and a short contrastive
fine-tune of a small encoder is the best scorer on every domain. Not established: human-judged accuracy
(labels are LLM-distilled and filtered), behaviour on ambiguous or no-answer inputs, transfer to classes the
fine-tuned encoder never saw (the split is by generation batch, not by class), robustness of the small-split
numbers (16 incident test rows), and anything about the shipped Rust / Node runtimes, which remain lexical.

Per-domain reports: `router-eval-gs1.md`, `router-eval-incident.md`, `router-eval-schemaorg.md`.
Fine-tuned checkpoints are gitignored; `models/checkpoints/MANIFEST.json` (`embedding_checkpoints`) records their hashes and metrics.
