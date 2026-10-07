# Decision head on LLM-distilled corpora: summary

Labels are **LLM-distilled and filtered, not human**: one model wrote observations for a class without
naming it, texts containing the label / CURIE were dropped, and a second model, shown only the text and a
shuffled candidate list, had to pick the same class for the row to be kept. Agreement shows the text is
recognisable as its generating class by a language model; it does not establish human-judged accuracy,
real-world distribution, or performance on ambiguous / no-answer inputs. Splits hold out generation
batches (register x seed), not classes. Task metrics have the gold candidate present; the retrieval
columns are the deployment path (`OLaya.system_one`: lexical router top-k, then the model) over the
full ontology, and the lexical baseline is scored on the same test rows. LLM cost is reported as 0
because the prices of the models used are not pinned in this repository.

| domain | rows (classes) | train/val/calib/test rows | lexical top-1 | model top-1 (retrieval) | retrieval recall@5 (lex / model) | task top-1 / top-3 | ECE raw -> scaled | runtime (train + eval) |
|---|---|---|---|---|---|---|---|---|
| gs1 | 422 (63) | 258/59/41/64 | 0.281 | 0.141 | 0.531 / 0.469 | 0.558 / 0.846 | 0.191 -> 0.164 | 1024 s + 2 s |
| incident | 109 (13) | 67/17/9/16 | 0.562 | 0.188 | 0.875 / 0.688 | 0.388 / 0.837 | 0.256 -> 0.248 | 331 s + 1 s |
| schemaorg | 862 (119) | 524/132/86/120 | 0.100 | 0.075 | 0.175 / 0.142 | 0.524 / 0.863 | 0.103 -> 0.078 | 1520 s + 4 s |

Per-domain reports: `decision-head-gs1.md`, `decision-head-incident.md`, `decision-head-schemaorg.md`.

Checkpoints: gs1 `c1ec2e0aa46b`, incident `f8271b8ce947`, schemaorg `f2c8f4503f63` (gitignored; `models/checkpoints/MANIFEST.json` is committed).

## Why the model loses to the lexical router: the router bounds it

`OLaya.system_one` scores only the most specific (leaf) concepts of the routed subgraph; ancestors are
context. The fraction of test rows whose gold class is among those scored leaves (model-independent,
same test rows, router top-10) is the ceiling of the model's deployed top-1:

| domain | gold is a scored leaf | gold present only as ancestor context |
|---|---|---|
| incident | 0.812 | 0.188 |
| gs1 | 0.562 | 0.000 |
| schemaorg | 0.200 | 0.092 |

Within that ceiling the head still under-performs the router's own ranking (its concept-task top-1
with the gold present and five hard negatives is 0.18-0.19 on gs1 / Schema.org), so the head trained
from scratch on a few hundred distilled rows is not yet a useful re-ranker. The verdict on all three
domains is that **the trained model does not beat the lexical baseline**; the committed checkpoint
manifest documents what was trained, not a recommended model.

Not done: a multi-domain run (time budget), ONNX export (`onnx` / `onnxruntime` are not installed in
this environment).
