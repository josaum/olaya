# Joint bi-encoder over the three LLM-distilled corpora (batch-disjoint split)

Labels are LLM-distilled and filtered, **not human**. One encoder is fine-tuned on the union of the three
domains' train splits and selected on the mean validation top-1 over domains; each domain's test split is
scored once. Engine rows use `OLaya.system_one` over the full ontology (router top-10); `[routed]` scores every
routed concept instead of the routed leaves only.

- Base `google/embeddinggemma-2`, 849 train pairs over 736 classes, 8 epochs in 1648.4 s (budget 60.0 min), best epoch 5
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query + 3 mined hard negatives per query re-mined each epoch
- Mean validation top-1: fine-tuned 0.795 vs zero-shot base 0.593
- Checkpoint `models/checkpoints/olaya_embed_joint-gemma.pt` sha256 `019bd5f922e1b7ce` (542151733 bytes; gitignored)
- Commit `0c8a4e38c59d`, runtime 1696.9 s

| domain | test rows | unseen test classes | system | ceiling@8 | top-1 | recall@5 | task top-1 | ECE | engine top-1 | engine top-1 [routed] | engine recall@5 [routed] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gs1 | 64 | 4 | lexical | 0.562 | 0.281 | 0.531 | 0.524 | 0.136 | - | - | - |
| gs1 | 64 | 4 | per-domain fine-tuned | 0.938 | 0.609 | 0.922 | 0.889 | 0.065 | 0.625 | 0.609 | 0.922 |
| gs1 | 64 | 4 | **joint fine-tuned** | 1.000 | 0.797 | 0.953 | 0.937 | 0.049 | 0.781 | 0.797 | 0.953 |
| incident | 16 | 1 | lexical | 0.875 | 0.562 | 0.875 | 0.688 | 0.316 | - | - | - |
| incident | 16 | 1 | per-domain fine-tuned | 1.000 | 0.688 | 1.000 | 0.875 | 0.125 | 0.688 | 0.688 | 0.938 |
| incident | 16 | 1 | **joint fine-tuned** | 1.000 | 0.812 | 0.938 | 1.000 | 0.000 | 0.812 | 0.812 | 0.938 |
| schemaorg | 120 | 9 | lexical | 0.208 | 0.100 | 0.175 | 0.419 | 0.109 | - | - | - |
| schemaorg | 120 | 9 | per-domain fine-tuned | 0.867 | 0.558 | 0.833 | 0.949 | 0.049 | 0.558 | 0.558 | 0.833 |
| schemaorg | 120 | 9 | **joint fine-tuned** | 0.942 | 0.783 | 0.917 | 0.983 | 0.015 | 0.708 | 0.783 | 0.908 |

## History

- epoch 1 loss 1.402 val 0.766; epoch 2 loss 0.559 val 0.753; epoch 3 loss 0.267 val 0.779; epoch 4 loss 0.129 val 0.778; epoch 5 loss 0.069 val 0.795; epoch 6 loss 0.033 val 0.784; epoch 7 loss 0.016 val 0.762; epoch 8 loss 0.008 val 0.771

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out generation batches, not classes; the encoder has seen almost every test class in training.
- The per-domain fine-tuned reference rows come from `router-eval-<domain>.json` and were trained with the same budget but without mined negatives.
- Nothing here changes the shipped Rust / Node runtimes, which remain lexical.
