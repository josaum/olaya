# Joint bi-encoder over the three LLM-distilled corpora (batch-disjoint split)

Labels are LLM-distilled and filtered, **not human**. One encoder is fine-tuned on the union of the three
domains' train splits and selected on the mean validation top-1 over domains; each domain's test split is
scored once. Engine rows use `OLaya.system_one` over the full ontology (router top-10); `[routed]` scores every
routed concept instead of the routed leaves only.

- Base `BAAI/bge-small-en-v1.5`, 849 train pairs over 736 classes, 8 epochs in 3626.1 s (budget 60.0 min), best epoch 5
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query + 2 mined hard negatives per query re-mined each epoch
- Mean validation top-1: fine-tuned 0.735 vs zero-shot base 0.524
- Checkpoint `models/checkpoints/olaya_embed_joint60.pt` sha256 `7824ad7266535489` (133506253 bytes; gitignored)
- Commit `3ec52f194508`, runtime 3743.6 s

| domain | test rows | unseen test classes | system | ceiling@8 | top-1 | recall@5 | task top-1 | ECE | engine top-1 | engine top-1 [routed] | engine recall@5 [routed] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gs1 | 64 | 4 | lexical | 0.562 | 0.281 | 0.531 | 0.524 | 0.136 | - | - | - |
| gs1 | 64 | 4 | zero-shot bge-small | 0.766 | 0.375 | 0.688 | 0.698 | 0.112 | 0.375 | 0.375 | 0.688 |
| gs1 | 64 | 4 | per-domain fine-tuned | 0.938 | 0.609 | 0.922 | 0.889 | 0.065 | 0.625 | 0.609 | 0.922 |
| gs1 | 64 | 4 | **joint fine-tuned** | 0.922 | 0.656 | 0.828 | 0.873 | 0.093 | 0.672 | 0.656 | 0.828 |
| incident | 16 | 1 | lexical | 0.875 | 0.562 | 0.875 | 0.688 | 0.316 | - | - | - |
| incident | 16 | 1 | zero-shot bge-small | 0.938 | 0.562 | 0.938 | 0.875 | 0.125 | 0.562 | 0.562 | 0.875 |
| incident | 16 | 1 | per-domain fine-tuned | 1.000 | 0.688 | 1.000 | 0.875 | 0.125 | 0.688 | 0.688 | 0.938 |
| incident | 16 | 1 | **joint fine-tuned** | 0.938 | 0.812 | 0.938 | 1.000 | 0.000 | 0.750 | 0.812 | 0.938 |
| schemaorg | 120 | 9 | lexical | 0.208 | 0.100 | 0.175 | 0.419 | 0.109 | - | - | - |
| schemaorg | 120 | 9 | zero-shot bge-small | 0.525 | 0.308 | 0.475 | 0.778 | 0.050 | 0.300 | 0.308 | 0.475 |
| schemaorg | 120 | 9 | per-domain fine-tuned | 0.867 | 0.558 | 0.833 | 0.949 | 0.049 | 0.558 | 0.558 | 0.833 |
| schemaorg | 120 | 9 | **joint fine-tuned** | 0.867 | 0.642 | 0.817 | 0.957 | 0.042 | 0.642 | 0.642 | 0.825 |

## History

- epoch 1 loss 1.450 val 0.648; epoch 2 loss 0.898 val 0.710; epoch 3 loss 0.525 val 0.712; epoch 4 loss 0.289 val 0.701; epoch 5 loss 0.183 val 0.735; epoch 6 loss 0.109 val 0.715; epoch 7 loss 0.067 val 0.718; epoch 8 loss 0.042 val 0.729

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out generation batches, not classes; the encoder has seen almost every test class in training.
- The per-domain fine-tuned reference rows come from `router-eval-<domain>.json` and were trained with the same budget but without mined negatives.
- Nothing here changes the shipped Rust / Node runtimes, which remain lexical.
