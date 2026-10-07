# Decision head on the LLM-distilled schemaorg corpus

Labels are LLM-distilled and filtered (generated for a class, label-substring filtered, kept only when an
independent labeller call agreed). They are **not human labels**. The numbers below show whether the
trained head recovers the class a language model wrote the text for; they do not establish real-world
accuracy.

- Corpus: `data/corpora/schemaorg.jsonl`, 862 rows, 119 classes, sha256 `3253d80a8d766154`
- Split by generation batch (not by class): train 524 / val 132 / calib 86 / test 120 rows; expanded tasks {'train': 2601, 'validation': 655, 'calibration': 429, 'test': 597}
- Test classes unseen in training: 9
- Training: custom_bert (4 layers, hidden 256, 4 heads), 6 epochs, 1519.5 s, selected val acc 0.498
- Checkpoint sha256 `f2c8f4503f6313aa` (63607962 bytes), engine model id `checkpoint-sha256:f2c8f4503f63`
- Temperature 1.209 fitted on 429 calibration tasks

## Test task metrics (gold always among candidates)

| | raw | temperature-scaled |
|---|---|---|
| top-1 | 0.524 | 0.524 |
| top-3 | 0.863 | 0.863 |
| NLL | 1.163 | 1.105 |
| ECE | 0.103 | 0.078 |
| Brier | 0.584 | 0.575 |

| task | n | top-1 | ECE |
|---|---|---|---|
| choice | 120 | 0.342 | 0.043 |
| concept | 117 | 0.179 | 0.111 |
| noul | 120 | 0.617 | 0.160 |
| score | 120 | 0.683 | 0.111 |
| subsumes | 120 | 0.792 | 0.117 |

Abstention curve (temperature-scaled, all tasks):

| threshold | coverage | accuracy when answered |
|---|---|---|
| 0.0 | 1.00 | 0.524 |
| 0.3 | 0.74 | 0.619 |
| 0.5 | 0.61 | 0.698 |
| 0.6 | 0.47 | 0.742 |
| 0.7 | 0.39 | 0.768 |
| 0.8 | 0.34 | 0.785 |
| 0.9 | 0.26 | 0.818 |
| 0.95 | 0.17 | 0.843 |

## Full-ontology retrieval on the same test rows (deployment path)

| system | top-1 | recall@5 |
|---|---|---|
| lexical router (abstains on ties / no evidence) | 0.100 | 0.175 |
| router top-k + trained model (`OLaya.system_one`) | 0.075 | 0.142 |

Gold was inside the routed subgraph for 0.292 of test rows, which bounds the model's reachable top-1.
The trained model **does not beat** the lexical baseline on top-1 over the full ontology on this test set.

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out generation batches (register x seed), not classes; some test classes may be unseen in training (see split.report).
- Task metrics have the gold candidate present; full_ontology_retrieval is the deployment path and the number to compare with the lexical baseline.
