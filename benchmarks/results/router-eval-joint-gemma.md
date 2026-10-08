# Joint bi-encoder over the three LLM-distilled corpora (batch-disjoint split)

Labels are LLM-distilled and filtered, **not human**. One encoder is fine-tuned on the union of the three
domains' train splits and selected on the mean validation top-1 over domains; each domain's test split is
scored once. Engine rows use `OLaya.system_one` over the full ontology (router top-10); `[routed]` scores every
routed concept instead of the routed leaves only.

- Base `google/embeddinggemma-2`, 849 train pairs over 736 classes, 8 epochs in 1588.8 s (budget 60.0 min), best epoch 8
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query + 3 mined hard negatives per query re-mined each epoch
- Recipe: float32 weights, AdamW lr 3e-05 with linear warmup over 10% of the planned steps then linear decay, batch 16, early stop patience 3, token embeddings trainable, LoRA rank 16 merged into the weights; stopped on epochs
- Mean validation top-1: fine-tuned 0.780 vs zero-shot base 0.593
- Checkpoint `models/checkpoints/olaya_embed_joint-gemma.pt` sha256 `902eea4a52d752c1` (1084157045 bytes; gitignored)
- Commit `0f519c8eca15`, runtime 1639.7 s

| domain | test rows | unseen test classes | system | ceiling@8 | top-1 | recall@5 | task top-1 | ECE | engine top-1 | engine top-1 [routed] | engine recall@5 [routed] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gs1 | 64 | 4 | lexical | 0.562 | 0.281 | 0.531 | 0.524 | 0.136 | - | - | - |
| gs1 | 64 | 4 | per-domain fine-tuned | 0.938 | 0.609 | 0.922 | 0.889 | 0.065 | 0.625 | 0.609 | 0.922 |
| gs1 | 64 | 4 | **joint fine-tuned** | 0.969 | 0.797 | 0.953 | 0.937 | 0.037 | 0.797 | 0.797 | 0.953 |
| incident | 16 | 1 | lexical | 0.875 | 0.562 | 0.875 | 0.688 | 0.316 | - | - | - |
| incident | 16 | 1 | per-domain fine-tuned | 1.000 | 0.688 | 1.000 | 0.875 | 0.125 | 0.688 | 0.688 | 0.938 |
| incident | 16 | 1 | **joint fine-tuned** | 1.000 | 0.812 | 0.938 | 1.000 | 0.000 | 0.750 | 0.812 | 0.938 |
| schemaorg | 120 | 9 | lexical | 0.208 | 0.100 | 0.175 | 0.419 | 0.109 | - | - | - |
| schemaorg | 120 | 9 | per-domain fine-tuned | 0.867 | 0.558 | 0.833 | 0.949 | 0.049 | 0.558 | 0.558 | 0.833 |
| schemaorg | 120 | 9 | **joint fine-tuned** | 0.917 | 0.750 | 0.900 | 0.974 | 0.015 | 0.700 | 0.750 | 0.900 |

## History

- epoch 1 loss 2.405 val 0.648; epoch 2 loss 1.483 val 0.731; epoch 3 loss 0.960 val 0.775; epoch 4 loss 0.698 val 0.766; epoch 5 loss 0.542 val 0.777; epoch 6 loss 0.437 val 0.777; epoch 7 loss 0.369 val 0.775; epoch 8 loss 0.331 val 0.780

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out generation batches, not classes; the encoder has seen almost every test class in training.
- The per-domain fine-tuned reference rows come from `router-eval-<domain>.json` and were trained with the same budget but without mined negatives.
- Nothing here changes the shipped Rust / Node runtimes, which remain lexical.
