# Complete O-Laya extension for Claude Desktop

`olaya-desktop-complete-macos-arm64-0.2.0.mcpb` includes the native macOS Apple
Silicon runtime: embedded CPython 3.12, Torch/Transformers, local tokenizer, and
existing `olaya_decision_model.pt` checkpoint. It executes Python model decisions,
workflow compilation/execution, and trace replay offline, without a separate
Python installation, model download, or API key. The full server does not expose
the earlier lexical-grounding tool.

Strict checkpoint loading establishes runtime/parameter compatibility. The existing
checkpoint reports historical validation accuracy **0.3793**; this is not a new
quality or calibration validation, production acceptance, or SOTA claim.

## Install the extension

Double-click the reviewed `.mcpb` artifact and follow Claude Desktop's installation
screen. If unavailable, use **Settings → Extensions → Advanced → Install Extension**
and select it. This artifact targets macOS Apple Silicon; it is not a portable
Windows, Linux, or Intel Mac runtime. Installation occurs in the profile chosen
by the user. These repository instructions do not edit profile configuration.

## Tools and guidance

Inspect `olaya_capabilities` for the installed surface:

| Tool | Input and behavior |
|---|---|
| `olaya_system_one` | Optional TBox object (incident ontology by default), JSON state, named questions; real model inference for `concept`, `subsumes`, `choice`, `score`, and `noul` |
| `olaya_workflow_compile` | Workflow and TBox objects; validation with canonical document, workflow hash, and TBox hash |
| `olaya_workflow_run` | Workflow, TBox, inputs, optional run ID; actual model-backed policy execution with JSON-LD trace |
| `olaya_workflow_replay` | Workflow, TBox, trace; replay consistency checks without model inference |
| `olaya_read_reference` | Bundled relative reference path; reads local guidance/examples |

Prompts `olaya-tbox-workflows`, `olaya-evidence-review`, and `olaya-model-decisions`
provide companion guidance through the MCPB surface. Use actual tool results as
execution evidence; loading prompts/resources does not itself run inference.

Workflow predictions are recorded as decision relationships, not asserted input
class membership. Terminal emissions are action intents. Core tools do not
execute external application connectors, dispatch goods, or change infrastructure.
Declared infrastructure/application resources do not prove reachability or
ownership. Replay checks policy consistency, not authenticity or underlying truth;
content hashes are not signatures.

## Companion skills

Self-contained native skills are supplied under `skills/`:

- **olaya-tbox-workflows**: author, compile, run, and replay TBox-bound workflows.
- **olaya-evidence-review**: inspect decision provenance, resource links, replay,
  and benchmark/calibration limits.
- **olaya-model-decisions**: run and interpret the five Python decision primitives.

Each skill includes `SKILL.md` and its own references. Storage live/replay fixtures
were generated with an explicitly untrained local engine and illustrate the trace
format; fresh complete-bundle execution uses its existing trained checkpoint.

Native skill ZIP import is separate from `.mcpb` installation. Where supported,
import each named skill archive through Claude Desktop's skills interface. Each
ZIP contains one skill folder with `SKILL.md` and `references/`; a skill ZIP is
not an MCPB extension. All guidance is also exposed as prompts/resources in the
complete bundle, so those remain available without native skill import.
