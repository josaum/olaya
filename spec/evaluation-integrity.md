# Evaluation integrity

Source observations are assigned stable source and group IDs before expansion into
concept, choice, score, noul, and subsumption tasks. Synthetic variants of one template
family stay in one partition. Text-equivalent sources join connected groups, including
transitive duplicates. Conflicting labels or IDs fail closed. At least four independent
groups are required. Seeded group allocation targets 60/15/10/15 percent for training,
validation, calibration, and test, with at least one group per partition; these ratios
apply to groups, not task counts. The sample budget is approximate expanded-task count;
finite source space and invalid tasks can yield fewer samples.

Model input state omits the target CURIE. Generic property wording with no term or
explicit target mapping is excluded. Both noul outcomes use the same question form;
negative queried classes fall outside the target ancestry. This is a closed-world
synthetic label convention, not proof of logical disjointness or natural-language truth.
Specificity uses longest parent-path depth capped at three, rejecting cycles and missing
parents. Concept and choice tasks still include the gold candidate: their task accuracy
cannot measure full-ontology retrieval.

Validation chooses the best checkpoint, which is reloaded before calibration. A distinct
calibration partition fits temperature. Final test reports accuracy, NLL, ECE, Brier,
and task metrics after selection and fitting. A shared scalar temperature applies to
the mixed task set. Candidate masks remain excluded after scaling, even at extreme
temperatures. ECE on the calibration fit set is separately
labeled. Reports include source/group IDs, ontology/source/tokenizer/checkpoint hashes,
hyperparameters, seed, and software/device metadata. Raw text is omitted. Seeded runs
do not guarantee bitwise determinism across devices. A new test corpus is necessary
if repeated experiments inform decisions using the current test results.

Legacy ONNX export now requires explicit `--export-onnx`; exported tokenizer and
calibration parity remain unverified. These reports concern raw Python model logits,
not ontology-projected deployed probabilities or the lexical HTTP handler.

## Reproduction

```bash
.venv/bin/python scripts/evaluate_lexical_baseline.py \
  --ontology docs/public/ontologies/schemaorg_ontology.json --domain schemaorg \
  --num-samples 500 --seed 42 \
  --output benchmarks/results/schemaorg-lexical-evaluation.json

OMP_NUM_THREADS=2 .venv/bin/python scripts/train_real_ontologies.py \
  --domain schemaorg --ontology-path docs/public/ontologies/schemaorg_ontology.json \
  --samples 40 --epochs 1 --batch-size 4 --device cpu --seed 42 \
  --no-real-abox --output-dir /tmp/olaya-evaluation-smoke

.venv/bin/pytest tests/test_evaluation_integrity.py tests/test_lexical_baseline.py
```

The checked-in training smoke report used two Torch CPU threads and temporary checkpoint
storage. Its checkpoint hash records that run; the temporary weights are not retained.
The CLI reproduction may differ with host thread settings. The lexical report searches
the full 660-class ontology using query text alone; targets never enter retrieval.
Top1 ties and absent lexical evidence abstain, and fallback candidates do not count
as recall evidence. Macro F1 includes the union of target and predicted classes.

The baseline scored top1 50%, recall@5 90%, macro F1 0.4762, and coverage 100%,
on just twenty synthetic test observations from one held-out template family. Local
latency excludes generation/index/file IO and has no warm-up. The training smoke
scored 60% on five test tasks with ECE 0.4000, after only one epoch. These runs use
different sample budgets and tasks and cannot establish a relative model advantage.
The tiny calibration set drove temperature to its 0.001 lower bound, giving 100%
confidence and poor test calibration; this is evidence to enlarge calibration data,
not an improvement claim.
Neither result is an independently labeled real-world or competitive benchmark.

The next meaningful accuracy gate is a frozen human-labeled corpus with source-document
and ontology-family separation, ambiguity/no-answer cases, full-ontology retrieval,
independent calibration, and deployed inference parity. Model upgrades should be
compared on that corpus with fixed latency and memory budgets.
