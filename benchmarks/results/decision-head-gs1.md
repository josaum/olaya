# Decision head on the LLM-distilled gs1 corpus

Labels are LLM-distilled and filtered (generated for a class, label-substring filtered, kept only when an
independent labeller call agreed). They are **not human labels**. The numbers below show whether the
trained head recovers the class a language model wrote the text for; they do not establish real-world
accuracy.

- Corpus: `data/corpora/gs1.jsonl`, 422 rows, 63 classes, sha256 `c442463da3977bc0`
- Split by generation batch (not by class): train 258 / val 59 / calib 41 / test 64 rows; expanded tasks {'train': 1286, 'validation': 295, 'calibration': 205, 'test': 319}
- Test classes unseen in training: 4
- Training: custom_bert (4 layers, hidden 256, 4 heads), 12 epochs, 1024.4 s, selected val acc 0.569
- Checkpoint sha256 `c1ec2e0aa46b4214` (63607060 bytes), engine model id `checkpoint-sha256:c1ec2e0aa46b`
- Temperature 1.261 fitted on 205 calibration tasks

## Test task metrics (gold always among candidates)

| | raw | temperature-scaled |
|---|---|---|
| top-1 | 0.558 | 0.558 |
| top-3 | 0.846 | 0.846 |
| NLL | 1.514 | 1.324 |
| ECE | 0.191 | 0.164 |
| Brier | 0.600 | 0.579 |

| task | n | top-1 | ECE |
|---|---|---|---|
| choice | 64 | 0.266 | 0.180 |
| concept | 63 | 0.190 | 0.208 |
| noul | 64 | 0.719 | 0.215 |
| score | 64 | 0.703 | 0.225 |
| subsumes | 64 | 0.906 | 0.055 |

Abstention curve (temperature-scaled, all tasks):

| threshold | coverage | accuracy when answered |
|---|---|---|
| 0.0 | 1.00 | 0.558 |
| 0.3 | 0.92 | 0.592 |
| 0.5 | 0.70 | 0.701 |
| 0.6 | 0.62 | 0.747 |
| 0.7 | 0.58 | 0.773 |
| 0.8 | 0.54 | 0.786 |
| 0.9 | 0.48 | 0.831 |
| 0.95 | 0.37 | 0.890 |

## Full-ontology retrieval on the same test rows (deployment path)

| system | top-1 | recall@5 |
|---|---|---|
| lexical router (abstains on ties / no evidence) | 0.281 | 0.531 |
| router top-k + trained model (`OLaya.system_one`) | 0.141 | 0.469 |

Gold was inside the routed subgraph for 0.562 of test rows, which bounds the model's reachable top-1.
The trained model **does not beat** the lexical baseline on top-1 over the full ontology on this test set.

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out generation batches (register x seed), not classes; some test classes may be unseen in training (see split.report).
- Task metrics have the gold candidate present; full_ontology_retrieval is the deployment path and the number to compare with the lexical baseline.
