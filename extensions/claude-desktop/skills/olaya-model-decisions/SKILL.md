---
name: olaya-model-decisions
description: Run and interpret O-Laya Python model decisions using five typed primitives in the complete offline Claude Desktop extension.
---

Use `olaya_capabilities` to establish the installed runtime and then
`olaya_system_one(tbox, state, questions)` for actual model inference. `tbox` is an
ontology object; omitting it uses the incident ontology. State is JSON and questions
is a named map. Inspect actual answers rather than fabricating expected scores.

The five primitive types are `concept`, `subsumes`, `choice`, `score`, and `noul`.
Python `concept` binds a `root`, `subsumes` binds a `concept`, `choice` uses named
`criteria`, `score` uses ordered `criteria`, and `noul` takes statement instructions.
Do not substitute the HTTP edge `options` or Node rejection semantics. For
ontology-bound policy execution, use the workflow tools and
[workflow contract](references/workflow-dsl-proposal.md) with the
[example TBox](references/storage-tbox.json).

Interpret scores according to the primitive: a score is an expected ordinal level,
not normalized business risk; Python noul represents statement probability rather
than Node out-of-domain rejection. Confidence and ontology-consistency checks do
not establish empirical calibration or independently verified correctness.

The complete macOS Apple Silicon bundle embeds CPython 3.12, Torch/Transformers,
local tokenizer, and the existing `olaya_decision_model.pt` checkpoint. Strict
loading establishes compatibility. Its historical checkpoint validation accuracy
of 0.3793 is not a new validation run or SOTA claim. No external model/API download
or API key is needed. State meaningful limitations beside the actual result.

A workflow terminal emits an action intent, not an external connector effect.
Do not claim model inference dispatched goods, changed infrastructure, or called
an application. Use explicit user intent and supported application mechanisms for
any separately requested effects; core workflow tools do not implement them.
