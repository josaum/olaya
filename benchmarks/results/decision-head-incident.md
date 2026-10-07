# Decision head on the LLM-distilled incident corpus

Labels are LLM-distilled and filtered (generated for a class, label-substring filtered, kept only when an
independent labeller call agreed). They are **not human labels**. The numbers below show whether the
trained head recovers the class a language model wrote the text for; they do not establish real-world
accuracy.

- Corpus: `data/corpora/incident.jsonl`, 109 rows, 13 classes, sha256 `289385a90d5bca12`
- Split by generation batch (not by class): train 67 / val 17 / calib 9 / test 16 rows; expanded tasks {'train': 335, 'validation': 85, 'calibration': 45, 'test': 80}
- Test classes unseen in training: 1
- Training: custom_bert (4 layers, hidden 256, 4 heads), 12 epochs, 330.8 s, selected val acc 0.506
- Checkpoint sha256 `f8271b8ce94785bd` (63607289 bytes), engine model id `checkpoint-sha256:f8271b8ce947`
- Temperature 1.077 fitted on 45 calibration tasks

## Test task metrics (gold always among candidates)

| | raw | temperature-scaled |
|---|---|---|
| top-1 | 0.388 | 0.388 |
| top-3 | 0.837 | 0.837 |
| NLL | 1.115 | 1.092 |
| ECE | 0.256 | 0.248 |
| Brier | 0.694 | 0.682 |

| task | n | top-1 | ECE |
|---|---|---|---|
| choice | 16 | 0.188 | 0.401 |
| concept | 16 | 0.438 | 0.261 |
| noul | 16 | 0.375 | 0.317 |
| score | 16 | 0.438 | 0.133 |
| subsumes | 16 | 0.500 | 0.270 |

Abstention curve (temperature-scaled, all tasks):

| threshold | coverage | accuracy when answered |
|---|---|---|
| 0.0 | 1.00 | 0.387 |
| 0.3 | 0.97 | 0.397 |
| 0.5 | 0.81 | 0.431 |
| 0.6 | 0.41 | 0.515 |
| 0.7 | 0.36 | 0.517 |
| 0.8 | 0.14 | 0.636 |
| 0.9 | 0.07 | 1.000 |
| 0.95 | 0.04 | 1.000 |

## Full-ontology retrieval on the same test rows (deployment path)

| system | top-1 | recall@5 |
|---|---|---|
| lexical router (abstains on ties / no evidence) | 0.562 | 0.875 |
| router top-k + trained model (`OLaya.system_one`) | 0.188 | 0.688 |

Gold was inside the routed subgraph for 1.000 of test rows, which bounds the model's reachable top-1.
The trained model **does not beat** the lexical baseline on top-1 over the full ontology on this test set.

## Limitations

- Labels are LLM-distilled (generated for a class by one model, kept only when a second model agreed) and filtered; they are not human labels.
- The split holds out generation batches (register x seed), not classes; some test classes may be unseen in training (see split.report).
- Task metrics have the gold candidate present; full_ontology_retrieval is the deployment path and the number to compare with the lexical baseline.
