# Complete O-Laya extension for Claude Desktop

`olaya-desktop-complete-macos-arm64-0.2.0.mcpb` includes the native macOS Apple
Silicon runtime: embedded CPython 3.12, Torch/Transformers, local tokenizer, and
existing `olaya_decision_model.pt` checkpoint. It executes Python model decisions,
workflow compilation/execution, and trace replay offline, without a separate
Python installation, model download, or API key. The full server does not expose
the earlier lexical-grounding tool.

Strict checkpoint loading establishes runtime/parameter compatibility. **No
checkpoint is committed to the repository** (`*.pt` is gitignored); the build
script expects one at `models/olaya_decision_model.pt`, so train or supply a
checkpoint first (`scripts/train_real_ontologies.py`). The checkpoint used for the
original packaging run reported historical validation accuracy **0.3793**; this
is not a quality or calibration validation, production acceptance, or SOTA claim.

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

## Double-click skill files and rebuild

The release also includes three `.skill` files (standard skill ZIP containers).
The installed Claude Desktop app registers this file type, so double-click one to
open its skill import flow. ZIP copies are supplied for the documented upload
route through Customize → Skills. The extension itself already contains the same
skill content as prompts/resources; native skill import enables skill discovery.

Build the complete release with `.venv/bin/python scripts/build-claude-desktop-complete.py`
on the tested macOS ARM64 Python 3.12 environment, then run
`python3 scripts/test-claude-desktop-complete.py`. The build copies a relocatable
standalone Python distribution, the dependency closure (with license metadata),
the source runtime, the tokenizer and the checkpoint. Build tooling uses the
pinned MCPB CLI 2.1.2; the resulting extension does not run npm, pip or uv.

`olaya_workflow_run` and `olaya_workflow_replay` return their JSON-LD document in
`structuredContent.trace`; pass that trace object into replay. Direct model
inference returns `structuredContent.result`, `jsonld`, and model provenance.
The neural workflow trace model identifier includes the checkpoint SHA-256.
Build output is unsigned; Keychain signing produces a separate `-signed.mcpb` variant. Build provenance and artifact checksums accompany it.

## Sign with an existing Keychain identity

Use `security find-identity -v -p codesigning` to select a valid certificate
fingerprint, then run:

```bash
.venv/bin/python scripts/sign-claude-desktop-keychain.py dist/olaya-desktop-complete-macos-arm64-0.2.0.mcpb --identity-sha1 CERTIFICATE_SHA1
```

The helper matches the identity by its exact certificate fingerprint, signs through
macOS Security.framework and appends the supported MCPB detached CMS signature.
It never exports the private key or changes Keychain ACLs. OpenSSL checks the
content signature, macOS checks the certificate for code-signing trust, and an
altered-payload test must fail. A signature report and checksum accompany the
signed artifact. MCPB CLI 2.1.2 reports unsigned because its node-forge PKCS#7
verification function is unimplemented; native verification is recorded separately.
Do not interpret that CLI result as evidence that the native signature is absent,
or claim that Desktop's publisher UI was verified from native verification alone.
