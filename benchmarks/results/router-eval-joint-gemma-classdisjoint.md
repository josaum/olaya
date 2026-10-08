# Joint bi-encoder over the three LLM-distilled corpora (class-disjoint split)

Labels are LLM-distilled and filtered, **not human**. One encoder is fine-tuned on the union of the three
domains' train splits and selected on the mean validation top-1 over domains; each domain's test split is
scored once. Engine rows use `OLaya.system_one` over the full ontology (router top-10); `[routed]` scores every
routed concept instead of the routed leaves only.

- Base `google/embeddinggemma-2`, 819 train pairs over 736 classes, 5 epochs in 965.9 s (budget 60.0 min), best epoch 2
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query + 3 mined hard negatives per query re-mined each epoch
- Recipe: float32 weights, AdamW lr 3e-05 with linear warmup over 10% of the planned steps then linear decay, batch 16, early stop patience 3, token embeddings trainable, LoRA rank 16 merged into the weights; stopped on patience
- Mean validation top-1: fine-tuned 0.722 vs zero-shot base 0.564
- Checkpoint `models/checkpoints/olaya_embed_joint-gemma-classdisjoint.pt` sha256 `e3af553e716bdd66` (1084162911 bytes; gitignored)
- Commit `0f519c8eca15`, runtime 1017.5 s

| domain | test rows | unseen test classes | system | ceiling@8 | top-1 | recall@5 | task top-1 | ECE | engine top-1 | engine top-1 [routed] | engine recall@5 [routed] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gs1 | 61 | 9 | lexical | 0.443 | 0.131 | 0.426 | 0.483 | 0.150 | - | - | - |
| gs1 | 61 | 9 | zero-shot embeddinggemma-2 | 0.754 | 0.410 | 0.656 | 0.800 | 0.105 | 0.410 | 0.410 | 0.656 |
| gs1 | 61 | 9 | per-domain fine-tuned | 0.869 | 0.557 | 0.820 | 0.917 | 0.081 | 0.574 | 0.557 | 0.787 |
| gs1 | 61 | 9 | **joint fine-tuned** | 0.902 | 0.656 | 0.820 | 0.883 | 0.056 | 0.672 | 0.656 | 0.820 |
| incident | 23 | 2 | lexical | 0.870 | 0.478 | 0.826 | 0.783 | 0.217 | - | - | - |
| incident | 23 | 2 | zero-shot embeddinggemma-2 | 1.000 | 0.826 | 1.000 | 1.000 | 0.097 | 1.000 | 0.826 | 1.000 |
| incident | 23 | 2 | per-domain fine-tuned | 1.000 | 0.739 | 1.000 | 0.913 | 0.088 | 0.913 | 0.739 | 1.000 |
| incident | 23 | 2 | **joint fine-tuned** | 1.000 | 0.826 | 1.000 | 1.000 | 0.066 | 1.000 | 0.826 | 1.000 |
| schemaorg | 127 | 18 | lexical | 0.394 | 0.205 | 0.386 | 0.548 | 0.086 | - | - | - |
| schemaorg | 127 | 18 | zero-shot embeddinggemma-2 | 0.811 | 0.512 | 0.772 | 0.927 | 0.034 | 0.520 | 0.512 | 0.772 |
| schemaorg | 127 | 18 | per-domain fine-tuned | 0.858 | 0.598 | 0.835 | 0.960 | 0.024 | 0.614 | 0.598 | 0.835 |
| schemaorg | 127 | 18 | **joint fine-tuned** | 0.874 | 0.606 | 0.827 | 0.968 | 0.038 | 0.622 | 0.606 | 0.827 |

## History

- epoch 1 loss 2.438 val 0.670; epoch 2 loss 1.519 val 0.722; epoch 3 loss 0.960 val 0.714; epoch 4 loss 0.660 val 0.700; epoch 5 loss 0.460 val 0.673

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out whole classes per domain; no test class has a training row.
- The per-domain fine-tuned reference rows come from `router-eval-<domain>.json` and were trained with the same budget but without mined negatives.
- Nothing here changes the shipped Rust / Node runtimes, which remain lexical.
