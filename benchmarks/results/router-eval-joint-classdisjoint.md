# Joint bi-encoder over the three LLM-distilled corpora (class-disjoint split)

Labels are LLM-distilled and filtered, **not human**. One encoder is fine-tuned on the union of the three
domains' train splits and selected on the mean validation top-1 over domains; each domain's test split is
scored once. Engine rows use `OLaya.system_one` over the full ontology (router top-10); `[routed]` scores every
routed concept instead of the routed leaves only.

- Base `BAAI/bge-small-en-v1.5`, 819 train pairs over 736 classes, 5 epochs in 1826.8 s (budget 30.0 min), best epoch 1
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query + 2 mined hard negatives per query re-mined each epoch
- Mean validation top-1: fine-tuned 0.646 vs zero-shot base 0.450
- Checkpoint `models/checkpoints/olaya_embed_joint-classdisjoint.pt` sha256 `b3339f6f374115bb` (133508777 bytes; gitignored)
- Commit `c14ff2f5eacd`, runtime 1958.5 s

| domain | test rows | unseen test classes | system | ceiling@8 | top-1 | recall@5 | task top-1 | ECE | engine top-1 | engine top-1 [routed] | engine recall@5 [routed] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gs1 | 61 | 9 | lexical | 0.443 | 0.131 | 0.426 | 0.483 | 0.150 | - | - | - |
| gs1 | 61 | 9 | zero-shot bge-small | 0.721 | 0.344 | 0.607 | 0.717 | 0.164 | 0.344 | 0.344 | 0.607 |
| gs1 | 61 | 9 | per-domain fine-tuned | 0.885 | 0.459 | 0.803 | 0.817 | 0.084 | 0.475 | 0.459 | 0.803 |
| gs1 | 61 | 9 | **joint fine-tuned** | 0.869 | 0.410 | 0.820 | 0.833 | 0.062 | 0.410 | 0.410 | 0.820 |
| incident | 23 | 2 | lexical | 0.870 | 0.478 | 0.826 | 0.783 | 0.217 | - | - | - |
| incident | 23 | 2 | zero-shot bge-small | 1.000 | 0.609 | 1.000 | 0.696 | 0.203 | 0.609 | 0.609 | 1.000 |
| incident | 23 | 2 | per-domain fine-tuned | 1.000 | 0.696 | 1.000 | 0.957 | 0.153 | 0.913 | 0.696 | 1.000 |
| incident | 23 | 2 | **joint fine-tuned** | 1.000 | 0.696 | 1.000 | 1.000 | 0.085 | 1.000 | 0.696 | 1.000 |
| schemaorg | 127 | 18 | lexical | 0.394 | 0.205 | 0.386 | 0.548 | 0.086 | - | - | - |
| schemaorg | 127 | 18 | zero-shot bge-small | 0.535 | 0.291 | 0.457 | 0.806 | 0.066 | 0.283 | 0.291 | 0.449 |
| schemaorg | 127 | 18 | per-domain fine-tuned | 0.780 | 0.417 | 0.732 | 0.911 | 0.027 | 0.417 | 0.417 | 0.724 |
| schemaorg | 127 | 18 | **joint fine-tuned** | 0.764 | 0.402 | 0.693 | 0.919 | 0.042 | 0.409 | 0.402 | 0.693 |

## History

- epoch 1 loss 1.437 val 0.646; epoch 2 loss 0.840 val 0.567; epoch 3 loss 0.450 val 0.507; epoch 4 loss 0.231 val 0.467; epoch 5 loss 0.137 val 0.464

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out whole classes per domain; no test class has a training row.
- The per-domain fine-tuned reference rows come from `router-eval-<domain>.json` and were trained with the same budget but without mined negatives.
- Nothing here changes the shipped Rust / Node runtimes, which remain lexical.
