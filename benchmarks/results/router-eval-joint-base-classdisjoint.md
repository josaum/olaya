# Joint bi-encoder over the three LLM-distilled corpora (class-disjoint split)

Labels are LLM-distilled and filtered, **not human**. One encoder is fine-tuned on the union of the three
domains' train splits and selected on the mean validation top-1 over domains; each domain's test split is
scored once. Engine rows use `OLaya.system_one` over the full ontology (router top-10); `[routed]` scores every
routed concept instead of the routed leaves only.

- Base `BAAI/bge-base-en-v1.5`, 819 train pairs over 736 classes, 20 epochs in 1055.4 s (budget 60.0 min), best epoch 1
- InfoNCE (temperature 0.05) over in-batch gold class texts + one sibling hard negative per query + 3 mined hard negatives per query re-mined each epoch
- Mean validation top-1: fine-tuned 0.631 vs zero-shot base 0.478
- Checkpoint `models/checkpoints/olaya_embed_joint-base-classdisjoint.pt` sha256 `ee3418ced65dda83` (437998762 bytes; gitignored)
- Commit `bcbbbc2ae284`, runtime 1086.9 s

| domain | test rows | unseen test classes | system | ceiling@8 | top-1 | recall@5 | task top-1 | ECE | engine top-1 | engine top-1 [routed] | engine recall@5 [routed] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gs1 | 61 | 9 | lexical | 0.443 | 0.131 | 0.426 | 0.483 | 0.150 | - | - | - |
| gs1 | 61 | 9 | zero-shot bge-base | 0.590 | 0.279 | 0.541 | 0.633 | 0.215 | 0.279 | 0.279 | 0.541 |
| gs1 | 61 | 9 | per-domain fine-tuned | 0.885 | 0.459 | 0.803 | 0.817 | 0.084 | 0.475 | 0.459 | 0.803 |
| gs1 | 61 | 9 | **joint fine-tuned** | 0.951 | 0.574 | 0.869 | 0.933 | 0.091 | 0.574 | 0.574 | 0.869 |
| incident | 23 | 2 | lexical | 0.870 | 0.478 | 0.826 | 0.783 | 0.217 | - | - | - |
| incident | 23 | 2 | zero-shot bge-base | 1.000 | 0.826 | 1.000 | 0.870 | 0.109 | 0.826 | 0.826 | 1.000 |
| incident | 23 | 2 | per-domain fine-tuned | 1.000 | 0.696 | 1.000 | 0.957 | 0.153 | 0.913 | 0.696 | 1.000 |
| incident | 23 | 2 | **joint fine-tuned** | 1.000 | 0.652 | 1.000 | 0.957 | 0.051 | 0.913 | 0.652 | 1.000 |
| schemaorg | 127 | 18 | lexical | 0.394 | 0.205 | 0.386 | 0.548 | 0.086 | - | - | - |
| schemaorg | 127 | 18 | zero-shot bge-base | 0.661 | 0.276 | 0.551 | 0.855 | 0.044 | 0.283 | 0.276 | 0.551 |
| schemaorg | 127 | 18 | per-domain fine-tuned | 0.780 | 0.417 | 0.732 | 0.911 | 0.027 | 0.417 | 0.417 | 0.724 |
| schemaorg | 127 | 18 | **joint fine-tuned** | 0.795 | 0.362 | 0.732 | 0.944 | 0.051 | 0.354 | 0.362 | 0.732 |

## History

- epoch 1 loss 1.326 val 0.631; epoch 2 loss 0.501 val 0.410; epoch 3 loss 0.162 val 0.503; epoch 4 loss 0.062 val 0.247; epoch 5 loss 0.022 val 0.428; epoch 6 loss 0.009 val 0.270; epoch 7 loss 0.005 val 0.246; epoch 8 loss 0.002 val 0.272; epoch 9 loss 0.002 val 0.244; epoch 10 loss 0.001 val 0.246; epoch 11 loss 0.001 val 0.234; epoch 12 loss 0.001 val 0.236; epoch 13 loss 0.001 val 0.234; epoch 14 loss 0.001 val 0.231; epoch 15 loss 0.001 val 0.224; epoch 16 loss 0.001 val 0.200; epoch 17 loss 0.000 val 0.231; epoch 18 loss 0.000 val 0.200; epoch 19 loss 0.000 val 0.202; epoch 20 loss 0.000 val 0.188

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out whole classes per domain; no test class has a training row.
- The per-domain fine-tuned reference rows come from `router-eval-<domain>.json` and were trained with the same budget but without mined negatives.
- Nothing here changes the shipped Rust / Node runtimes, which remain lexical.
