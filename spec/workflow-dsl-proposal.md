# TBox-bound workflow DSL: implemented version 1 contract

The local Python compiler and interpreter expose `compile_workflow(document,
ontology)` and `run_workflow(compiled, engine, inputs, run_id=None)`. Version 1
accepts a JSON object describing an acyclic graph of typed decision steps and
terminal action intents. Load the document as JSON; YAML and a compact text
language are not implemented. This is operational policy execution over a
supplied engine and TBox, not evidence of calibrated real-world semantics.

## Storage example

The complete example is [storage.json](../examples/workflows/storage.json), with
its supplied [TBox](../examples/workflows/storage-tbox.json). Its identifiers use
`ex: https://example.org/storage#`. The workflow ID and TBox ID are absolute IRIs;
workflow, decision activity, option, and emitted intent types are actual classes
in this TBox. The emitted `ex:note` property is declared with `ex:ActionIntent`
as its domain. Dispatch and review intent classes inherit from that domain.

```json
{
  "version": 1,
  "id": "https://example.org/storage/workflows/shipment",
  "type": "ex:Workflow",
  "prefixes": {"ex": "https://example.org/storage#"},
  "tbox": "https://example.org/storage/ontology",
  "start": "storage",
  "steps": {
    "storage": {
      "type": "ex:StorageDecision",
      "decide": {
        "state": "input.shipment",
        "question": {
          "type": "choice",
          "root": "ex:StorageMode",
          "options": ["ex:Cold", "ex:Ambient"]
        }
      },
      "accept": {"min_confidence": 0.8, "min_margin": 0.15},
      "route": {"ex:Cold": "cold_dispatch", "ex:Ambient": "ambient_dispatch"},
      "on_abstain": "review",
      "on_error": "review"
    },
    "cold_dispatch": {
      "emit": {"type": "ex:DispatchCold", "properties": {"ex:note": "Cold dispatch requested"}}
    },
    "ambient_dispatch": {
      "emit": {"type": "ex:DispatchAmbient", "properties": {"ex:note": "Ambient dispatch requested"}}
    },
    "review": {
      "emit": {"type": "ex:Review", "properties": {"ex:note": "Human review requested"}}
    }
  }
}
```

The confidence and margin thresholds demonstrate policy syntax. They have not
been validated as operating thresholds. Exact ontology option identifiers drive
routing; they are not free-text action names. This DSL adapter targets Python
`OLaya.system_one`: its choice adapter translates ontology-bound options to the
Python `criteria` contract. It does not call the HTTP edge API or Node runtime,
whose primitive semantics differ.

## Compile and run boundaries

Compile against the explicitly supplied `Ontology`. Reject unknown types and
properties, unbound prefixes, invalid graph targets, cycles, malformed decision
questions, and policy fields rather than silently guessing their meaning.
Declare prefixes for every compact class/property identifier in the supplied TBox.
Full IRIs and compact aliases resolve to native ontology identifiers. The compiler
rejects ambiguous expanded identifiers. Workflow and step class references, scoped option classes, and emitted properties
must resolve against the TBox. Property domains are checked through class ancestry;
property ranges, OWL restrictions, datatype facets, and full SHACL conformance are
not validated by this version. A TBox identifier labels that supplied ontology;
it does not fetch or authenticate a remote document. The trace binds its content
hash as well as its identity.

At runtime, the engine must use the compiled TBox. State references use validated
`input` paths; they are not arbitrary Python expressions. Decision responses pass
through policy checks before routing. Insufficient confidence, a tie or
insufficient probability margin sends this example to `on_abstain`. Missing
input and inference failures send it to `on_error`. Terminal `emit` constructs a
record of an action intent. It performs no dispatch, network request, shell
command, physical control, or application side effect.

## JSON-LD execution evidence

Each run returns a JSON-LD document with an inline local context and an `@graph`.
It links the workflow definition, supplied TBox identity and SHA-256 digest,
run activity, decision activities and answer entities, transitions, and terminal
intent. The local workflow vocabulary is `urn:olaya:workflow:`; PROV uses
`http://www.w3.org/ns/prov#`. The schema is documented in
[workflow-vocabulary.jsonld](workflow-vocabulary.jsonld). There is no remote
context resolution requirement.

Predicted class references describe the decision record using `wf:predictedType`.
They do not assert that the input shipment has that `rdf:type`. A terminal record
has its declared intent type, which describes the requested operation rather
than proof that it occurred. The `wf:modelIdentifier` value is the engine-reported model name, not a weights
checksum or calibration certificate. Scores, confidence, and that name establish
bounded engineering provenance; they do not establish probability calibration,
semantic truth, or independent business acceptance. Raw shipment text is excluded
from the emitted trace by default.

## Run the demonstration

From the repository root with its Python dependencies installed:

```bash
python examples/workflows/run_storage.py
python examples/workflows/run_storage.py --shipment "Office stationery for ambient storage."
python examples/workflows/run_storage.py --replay examples/workflows/storage-run.jsonld
```

[run_storage.py](../examples/workflows/run_storage.py) reads fixture paths relative
to its own location and loads `models/tokenizer.json` locally. It constructs an
explicitly **untrained** CPU engine using `weights_path=False` and a fixed seed.
No trained checkpoint is silently loaded. Its result demonstrates integration
and policy gates; it is not a storage classification accuracy benchmark. Review
is a legitimate and expected terminal result with this untrained model.

Tests with controlled engine responses establish graph validation, scoped
routing, abstention/error fallbacks, typed property checks, and JSON-LD reference
structure. Independently labeled end-to-end cases and trained-model evaluation
remain necessary before claiming operational decision accuracy. No deployment,
SOTA comparison, or calibrated warehouse automation claim follows from this demo.

## Replay recorded decisions without inference

`replay_workflow(compiled, trace)` in `olaya.workflow_replay` accepts the exact
first-generation live inline-context execution document emitted by this implementation.
Expanded or differently compacted JSON-LD, reordered graph arrays, and replay-of-replay
documents are outside this replay contract. It requires
no engine, trained weights, tokenizer, raw shipment input, or model call. It
reconstructs recorded sanitized decisions and recorded backend-error outcomes,
then reruns policy gates and transitions against the supplied compiled workflow.

Replay regenerates the complete original trace under its recorded run ID and
checks equality with the supplied trace. A mismatch in answers, gates,
transitions, intent, context, identities, or provenance rejects replay. On
success it emits a new run UUID with `wf:executionMode: "replay"` and an IRI
reference `wf:replayOf` to the original run. Live execution records use
`wf:executionMode: "live"`.

```python
import json
from pathlib import Path
from olaya import Ontology
from olaya.workflow import compile_workflow
from olaya.workflow_replay import replay_workflow

fixtures = Path("examples/workflows")
ontology = Ontology.from_json((fixtures / "storage-tbox.json").read_text())
compiled = compile_workflow(json.loads((fixtures / "storage.json").read_text()), ontology)
trace = json.loads((fixtures / "storage-run.jsonld").read_text())
print(json.dumps(replay_workflow(compiled, trace), indent=2))
```

The checked-in trace demonstrates recorded decisions from the explicitly
untrained engine. Replay checks consistency with compiled policy; it does not
rerun inference or certify the original observation, model result, or occurrence
of a recorded backend error. Workflow and TBox hashes identify content. They
are not signatures and do not authenticate a trace. An internally consistent
fabricated trace can pass these checks. Use independent authenticity controls
if provenance must prove who actually performed an execution.

## Format and provenance references

The emitter uses the [W3C JSON-LD 1.1](https://www.w3.org/TR/json-ld11/)
format with inline prefix mappings, explicit IRI links, and ordered lists.
Activities and generated entities use the
[W3C PROV-O](https://www.w3.org/TR/prov-o/) vocabulary.
The workflow-specific vocabulary is local and defined in this repository;
these links do not claim full OWL reasoning or a standards conformance certificate.
