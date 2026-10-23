# Changelog

All notable changes to **O-Laya** (`leio-decision`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Breaking: the default encoder of the ontology-conditioned engine is EmbeddingGemma 2 (installed by `ml`), and client-specific modules were removed from the package. See *Changed* and *Removed*.

### Added
- **Bring-your-own ontology (`olaya.ontology_io`):** `load_ontology` reads Turtle, RDF/XML, N-Triples, JSON-LD or O-Laya JSON (or a directory of RDF files) in pure Python, with no Rust toolchain; `ontology_fingerprint` hashes the canonical TBox (format- and order-independent) so a checkpoint can be bound to the ontology it was trained on. New `rdf` extra (`rdflib`), included in `all`.
- **`OLaya.from_ontology(...)`:** builds an engine conditioned on a user-supplied ontology at instantiation: class texts are embedded once, `concept` questions are routed (hybrid lexical + embedding RRF, or embedding only) and scored zero-shot, then projected through the verified axiom kernel. An optional fine-tuned checkpoint is verified against its SHA-256 and the ontology fingerprint.
- **Domain packs (`olaya.domain_pack`, `OLaya.from_pack(path)`):** a `pack.json` manifest (ontology, fingerprint, encoder / checkpoint pin, router, candidate policy) kept in the consumer's own repository.
- **Caller-supplied evidence:** `generate_real_ontology_observations(..., evidence=[(curie, text), ...])`, a `"generic"` template family for any ontology, `--ontology` / `--evidence` flags on the training and calibration scripts, and `OLAYA_EXTRA_ONTOLOGY` for `build_master_ontology.py`.
- **Hermetic tests for the model-backed paths:** tiny randomly initialised BERT and prompted sentence-transformers models stand in for Hugging Face checkpoints (encoders, full and LoRA fine-tuning, checkpoint reload, cross-encoder reranker, decision-head training), plus table-driven workflow, replay and ontology validation tests; line coverage 83% → 90% without downloads.

### Changed
- **BREAKING — EmbeddingGemma 2 is the default and ships with `ml`:** `OLaya.from_ontology()` (and domain packs without an explicit model) default to `google/embeddinggemma-2` (`olaya.embeddings.DEFAULT_ENCODER`); `sentence-transformers[image]`, `peft` and `transformers>=5.19` move into the `ml` extra (and therefore `all` / `dev`). Install `torch` and `torchvision` from the same index (CI and the release workflow now do). `TransformerEncoder` keeps `DEFAULT_MODEL` (MiniLM) as its own default.
- **Turtle/OWL loading in `load_ontology_file`:** no longer shells out to `cargo run`; it uses `olaya.ontology_io` and names RDF classes like `OLaya.from_ontology` (`prefix:Local`), so evidence, corpora and engine answers agree. `curie_style="local"` (`--curie-style local` on the scripts) reproduces the Rust converter's bare local names; O-Laya JSON is unchanged.
- **Observation generators default to the `"generic"` domain** instead of a host-specific one.
- **Source archive allow-list:** the sdist ships only `src/`, `tests/`, `spec/`, `README.md`, `CHANGELOG.md` and `pyproject.toml`; scripts, benchmarks, outputs and the docs site stay in the repository.

### Deprecated
- **`use_real_abox`:** `use_real_abox=True` is ignored with a `DeprecationWarning`; pass `evidence=` instead.
- **`leio-decision[gemma]`:** an alias of `ml`, kept so existing installs resolve.

### Removed
- **BREAKING — host- and client-specific code:** the `olaya.real_abox` module and its package exports (a host-specific evidence harvester), its training domain, the M&A deal-scenario script with its outputs and ontology, and the corresponding playground demo. Supply your own ontology via `load_ontology` / domain packs and your own observations via `evidence=`.

## [0.6.0] - 2026-10-20

### Added
- **Standalone stdlib-only base install (`leio-decision`):** Split optional dependencies into composable extras (`pii`, `reasoning`, `ml`, `llm`, `gemma`, and `all`) so consumers can install `leio-decision` or `leio-decision[pii]` without pulling PyTorch, Transformers, or NumPy (#32).
- **PII / MNPI obfuscator validator hook & hardening:** Added `candidate_filter(entity_type, text)` prior to overlap resolution (enabling external Luhn/SSA validators), environment/per-process salt (`OLAYA_PII_SALT`), single-pass longest-token-first `deanonymize`, non-reversible `mask`/`redact` vault semantics, and expanded street/title patterns.
- **LoRA fine-tuning recipe for EmbeddingGemma 2:** `finetune_bi_encoder` now defaults to `google/embeddinggemma-2` with LoRA (`rank=16`, `lr=3e-5`, float32 master weights, linear warmup/decay, patience-based early stopping), improving class-disjoint test top-1 to `0.656` (GS1), `0.826` (incident), and `0.606` (Schema.org), plus per-domain fine-tunes (`0.859` GS1, `0.812` incident, `0.642` Schema.org on batch splits) (#27).
- **RevOps Bowtie routing benchmark:** End-to-end evaluation of four encoders on `pt-BR` vertical utterances against `revops.ttl` with import-closure resolution (`embeddinggemma-300m` reaching `10/10` including boundary cases) (`benchmarks/results/revops-routing-summary.md`).
- **Node demo server (`examples/node-page`):** Single-file Node server classifying observations against bundled ontologies (and optional `OLAYA_EXTRA_ONTOLOGY`) with lineage, candidates, and abstention display (#27).

### Changed
- **Ontology conversion & Node DAG loading:** `convert_ontology` preserves source IRIs; `OntologyDAG.fromJSON` resolves bare local names only when unambiguous across prefixes and never shadows full CURIEs (#27, #28).

### Fixed
- **Per-domain EmbeddingGemma checkpoint manifest:** Retrained `olaya_embed_gs1.pt` and `olaya_embed_schemaorg.pt` with the deterministic LoRA recipe to reproduce the reported test metrics (`0.859` / `0.984` on GS1, `0.642` / `0.966` on Schema.org) and updated `models/checkpoints/MANIFEST.json` (#31).
- **Node demo server prototype key handling:** Checked own properties on `catalog[domain]` so inherited `Object.prototype` keys (`__proto__`, `constructor`) return HTTP 400 instead of crashing the server (#29).
- **Local source distribution build:** Added `.venv-*/` to `.gitignore` so secondary local virtual environments are excluded when building source distributions.

## [0.5.0] - 2026-10-08

### Added
- **EmbeddingGemma 2 encoder backend (`olaya[gemma]`):** `SentenceTransformerEncoder` supporting prompted query/document sides (`google/embeddinggemma-2`), contrastive fine-tuning, and activation checkpointing on Apple Silicon (`mps`) (#22, #24).
- **Routed candidate policy & class-disjoint evaluation protocol:** Opt-in `candidate_policy="routed"` restricting projection to the top-$k$ routed concepts and their ancestors, class-disjoint train/val/test splits (`split_mode="class"`), and joint multi-domain bi-encoder training (`scripts/finetune_joint_encoder.py`) with mined hard negatives (#20, #23).
- **Device selection & Apple Silicon training guide:** Automatic/explicit `cpu` / `mps` / `cuda` selection across encoders and fine-tuners, documented in `docs/training-on-apple-silicon.md` (#20).
- **Router inference latency benchmark:** Added `benchmarks/bench_router_latency.py` and per-configuration latency reports (`benchmarks/results/router-latency.md`) (#25).

### Fixed
- **First-run encoder downloads:** Removed forced Hugging Face offline environment defaults in evaluation/training scripts so fresh machines can download pretrained encoders (#21).

## [0.4.0] - 2026-10-07

### Added
- **Embedding router & bi-encoder scorer inside ontology projection:** Injectable `EmbeddingCandidateRouter` and `EmbeddingScorer` using cosine similarity over path-context class descriptions inside the existing axiom projection, plus InfoNCE contrastive fine-tuning (#18).
- **LLM-judge OAEI matcher & multi-judge ensemble:** Cached LLM-judge backend (`src/olaya/benchmarks/llm_judge.py`) and multi-judge ensemble (`llm_ensemble.py`) evaluated across all 21 OAEI Conference `ra1` pairs (`qwen3.8-max`, `qwen3.8-flash`, `glm-5.2`, `deepseek-v4.1-flash`), achieving `0.80` M1 F1 / `0.76` M3 F1 under LOPO-CV (#18).
- **LLM-distilled corpora & from-scratch decision head benchmark:** Committed provenance-tracked distilled corpora (`data/corpora/`) and end-to-end training script (`scripts/train_from_corpus.py`) (#18).

### Fixed
- **LLM-judge truncation resilience:** Treated truncated single-pair judge responses as abstentions rather than aborting evaluation runs (#18).

## [0.3.2] - 2026-10-06

### Changed
- **Unified `leio-decision` package identity:** Renamed both the PyPI distribution and npm package to `leio-decision` (retaining `import olaya` in Python and CLI entry points `leio-decision`, `leio`, `leio-mcp`, `olaya-mcp`) with automated PyPI trusted publishing and npm staged publishing in `release.yml`.

## [0.3.1] - 2026-10-06

### Added
- **Lean 4 formal proof core (`lean/OlayaAxioms.lean`):** Formalized Steps 3–4 of `spec/AXIOMS.md` and the axiom verifier in Lean 4 (16 theorems, zero `sorry`, audited via `lean4checker` in CI, and cross-checked against the Python reference in `tests/test_lean_oracle.py`) (#13).
- **Registry publication workflow:** Added PyPI OIDC trusted publishing and npm staged publishing jobs to `.github/workflows/release.yml` (#14).

## [0.3.0] - 2026-10-05

### Added
- **OWL 2 RL instance reasoning & ABox individuals:** Exported `Individual`, `ObjectProperty`, `OWL2RLReasoner`, and `abox_consistency_report` with per-rule conformance specifications (#11).
- **OAEI Conference `ra1` benchmark suite:** Vendored upstream OAEI Conference track ontologies and reference alignments with lexical, hybrid embedding, structural propagation, cross-encoder reranker, and learned pairwise matchers under nested leave-one-pair-out cross-validation.
- **Sovereign PII/MNPI shield:** Added deterministic reversible PII tokenization (`olaya.pii`).
- **Cross-runtime decision parity (`spec/decision_parity.json`):** Unified confidence calculation and verified identical concept selection across Rust, WASM, Python, and Node runtimes.

## [0.2.0] - 2026-10-03

### Added
- **Automated release pipeline (`.github/workflows/release.yml`):** End-to-end CI release workflow building and verifying the Python wheel, source archive, Node tarball, and offline macOS ARM64 Claude Desktop `.mcpb` bundle.
- **Local Ollama MCP bridge:** Added `scripts/ollama_mcp_server.py` and PII obfuscation integration.

### Fixed
- **Clean-checkout Studio ontology fixture:** Included the Mycelia Studio TBox fixture in tracked repository data so clean CI checkouts pass all TBox tests.

## [0.1.0] - 2026-10-03

### Added
- **Multi-language O-Laya workspace:** Initial release of the ontology-grounded System 1 decision engine across Rust (`olaya-core`, `owl-fast-core`, `olaya-wasm`, `mycelia-ort-utils`), Python (`olaya`), and Node/TypeScript (`packages/olaya-node`).
- **Shared axiom kernel & golden vectors (`spec/AXIOMS.md`, `spec/axioms_golden.json`):** Order-filter upward closure, disjointness resolution, and monotonicity projection verified across all runtimes.
- **TBox-bound workflows & inference-free JSON-LD replay:** Workflow compiler, executor (`olaya.workflow`), and deterministic JSON-LD trace replay (`olaya.workflow_replay`) (#6).
- **Complete macOS ARM64 Claude Desktop bundle (`.mcpb`):** Offline self-contained CPython/PyTorch runtime, neural checkpoint, MCP server, and three native skills (`olaya-model-decisions`, `olaya-tbox-workflows`, `olaya-evidence-review`).

[0.6.0]: https://github.com/josaum/olaya/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/josaum/olaya/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/josaum/olaya/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/josaum/olaya/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/josaum/olaya/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/josaum/olaya/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/josaum/olaya/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/josaum/olaya/releases/tag/v0.1.0
