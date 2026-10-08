# Joint bi-encoder over the three LLM-distilled corpora (class-disjoint split)

Labels are LLM-distilled and filtered, **not human**. One encoder is fine-tuned on the union of the three
domains' train splits and selected on the mean validation top-1 over domains; each domain's test split is
scored once. Engine rows use `OLaya.system_one` over the full ontology (router top-10); `[routed]` scores every
routed concept instead of the routed leaves only.

- Base `google/embeddinggemma-2`, 819 train pairs over 736 classes, 8 epochs in 1575.9 s (budget 60.0 min), best epoch 1
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query + 3 mined hard negatives per query re-mined each epoch
- Mean validation top-1: fine-tuned 0.676 vs zero-shot base 0.564
- Checkpoint `models/checkpoints/olaya_embed_joint-gemma-classdisjoint.pt` sha256 `be9705e9d6775158` (542157663 bytes; gitignored)
- Commit `0c8a4e38c59d`, runtime 1627.3 s

| domain | test rows | unseen test classes | system | ceiling@8 | top-1 | recall@5 | task top-1 | ECE | engine top-1 | engine top-1 [routed] | engine recall@5 [routed] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gs1 | 61 | 9 | lexical | 0.443 | 0.131 | 0.426 | 0.483 | 0.150 | - | - | - |
| gs1 | 61 | 9 | per-domain fine-tuned | 0.885 | 0.459 | 0.803 | 0.817 | 0.084 | 0.475 | 0.459 | 0.803 |
| gs1 | 61 | 9 | **joint fine-tuned** | 0.951 | 0.607 | 0.820 | 0.883 | 0.062 | 0.623 | 0.607 | 0.820 |
| incident | 23 | 2 | lexical | 0.870 | 0.478 | 0.826 | 0.783 | 0.217 | - | - | - |
| incident | 23 | 2 | per-domain fine-tuned | 1.000 | 0.696 | 1.000 | 0.957 | 0.153 | 0.913 | 0.696 | 1.000 |
| incident | 23 | 2 | **joint fine-tuned** | 1.000 | 0.652 | 1.000 | 1.000 | 0.102 | 1.000 | 0.652 | 1.000 |
| schemaorg | 127 | 18 | lexical | 0.394 | 0.205 | 0.386 | 0.548 | 0.086 | - | - | - |
| schemaorg | 127 | 18 | per-domain fine-tuned | 0.780 | 0.417 | 0.732 | 0.911 | 0.027 | 0.417 | 0.417 | 0.724 |
| schemaorg | 127 | 18 | **joint fine-tuned** | 0.882 | 0.591 | 0.843 | 0.960 | 0.009 | 0.598 | 0.591 | 0.843 |

## History

- epoch 1 loss 1.414 val 0.676; epoch 2 loss 0.477 val 0.607; epoch 3 loss 0.188 val 0.531; epoch 4 loss 0.091 val 0.482; epoch 5 loss 0.044 val 0.455; epoch 6 loss 0.018 val 0.415; epoch 7 loss 0.007 val 0.391; epoch 8 loss 0.003 val 0.336

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out whole classes per domain; no test class has a training row.
- The per-domain fine-tuned reference rows come from `router-eval-<domain>.json` and were trained with the same budget but without mined negatives.
- Nothing here changes the shipped Rust / Node runtimes, which remain lexical.
