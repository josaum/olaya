# Joint bi-encoder over the three LLM-distilled corpora (batch-disjoint split)

Labels are LLM-distilled and filtered, **not human**. One encoder is fine-tuned on the union of the three
domains' train splits and selected on the mean validation top-1 over domains; each domain's test split is
scored once. Engine rows use `OLaya.system_one` over the full ontology (router top-10); `[routed]` scores every
routed concept instead of the routed leaves only.

- Base `BAAI/bge-base-en-v1.5`, 849 train pairs over 736 classes, 20 epochs in 1109.1 s (budget 60.0 min), best epoch 3
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query + 3 mined hard negatives per query re-mined each epoch
- Mean validation top-1: fine-tuned 0.763 vs zero-shot base 0.587
- Checkpoint `models/checkpoints/olaya_embed_joint-base.pt` sha256 `1b5e293a6afd2d51` (437995892 bytes; gitignored)
- Commit `bcbbbc2ae284`, runtime 1151.7 s

| domain | test rows | unseen test classes | system | ceiling@8 | top-1 | recall@5 | task top-1 | ECE | engine top-1 | engine top-1 [routed] | engine recall@5 [routed] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gs1 | 64 | 4 | lexical | 0.562 | 0.281 | 0.531 | 0.524 | 0.136 | - | - | - |
| gs1 | 64 | 4 | zero-shot bge-base | 0.734 | 0.406 | 0.703 | 0.746 | 0.122 | 0.406 | 0.406 | 0.703 |
| gs1 | 64 | 4 | per-domain fine-tuned | 0.938 | 0.609 | 0.922 | 0.889 | 0.065 | 0.625 | 0.609 | 0.922 |
| gs1 | 64 | 4 | **joint fine-tuned** | 0.953 | 0.672 | 0.953 | 0.937 | 0.025 | 0.688 | 0.672 | 0.953 |
| incident | 16 | 1 | lexical | 0.875 | 0.562 | 0.875 | 0.688 | 0.316 | - | - | - |
| incident | 16 | 1 | zero-shot bge-base | 0.938 | 0.625 | 0.938 | 1.000 | 0.000 | 0.625 | 0.625 | 0.875 |
| incident | 16 | 1 | per-domain fine-tuned | 1.000 | 0.688 | 1.000 | 0.875 | 0.125 | 0.688 | 0.688 | 0.938 |
| incident | 16 | 1 | **joint fine-tuned** | 0.938 | 0.750 | 0.938 | 1.000 | 0.000 | 0.750 | 0.750 | 0.938 |
| schemaorg | 120 | 9 | lexical | 0.208 | 0.100 | 0.175 | 0.419 | 0.109 | - | - | - |
| schemaorg | 120 | 9 | zero-shot bge-base | 0.583 | 0.267 | 0.525 | 0.803 | 0.106 | 0.283 | 0.267 | 0.525 |
| schemaorg | 120 | 9 | per-domain fine-tuned | 0.867 | 0.558 | 0.833 | 0.949 | 0.049 | 0.558 | 0.558 | 0.833 |
| schemaorg | 120 | 9 | **joint fine-tuned** | 0.900 | 0.658 | 0.858 | 0.949 | 0.022 | 0.625 | 0.658 | 0.858 |

## History

- epoch 1 loss 1.350 val 0.733; epoch 2 loss 0.609 val 0.711; epoch 3 loss 0.237 val 0.763; epoch 4 loss 0.090 val 0.743; epoch 5 loss 0.039 val 0.746; epoch 6 loss 0.014 val 0.751; epoch 7 loss 0.005 val 0.731; epoch 8 loss 0.003 val 0.722; epoch 9 loss 0.002 val 0.719; epoch 10 loss 0.002 val 0.722; epoch 11 loss 0.001 val 0.709; epoch 12 loss 0.001 val 0.721; epoch 13 loss 0.001 val 0.720; epoch 14 loss 0.001 val 0.721; epoch 15 loss 0.001 val 0.715; epoch 16 loss 0.001 val 0.712; epoch 17 loss 0.001 val 0.717; epoch 18 loss 0.001 val 0.723; epoch 19 loss 0.001 val 0.726; epoch 20 loss 0.001 val 0.723

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out generation batches, not classes; the encoder has seen almost every test class in training.
- The per-domain fine-tuned reference rows come from `router-eval-<domain>.json` and were trained with the same budget but without mined negatives.
- Nothing here changes the shipped Rust / Node runtimes, which remain lexical.
