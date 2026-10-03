# owl-fast-core

OWL / RDF / OBO / SKOS entity extraction on top of the Oxigraph component parser stack — the centralized ontology parser for `office-parsers-rs`.

This is a reusable parser crate included in the O-Laya workspace. See the
[repository README](../../README.md) for runtime architecture and verification
boundaries.

**Role boundary:** this crate is a **parser / extractor / RAG chunker**, not an OWL2-RL reasoner. It does **not** expand `equivalentClass`, restrictions, or property chains into inferred triples — platform reasoning stays in Crepe Datalog (`mycelia-gateway`). Asserted `owl:sameAs` / `owl:equivalentProperty` links are preserved on the entity model (never collapsed).

## HTTP edge boundary

The edge `noul` contract now distinguishes supported/contradicted evidence from
`unmatched`, `conflicting`, `ambiguous_scope`, and `unsupported_question`.
Unknown cases return `abstained: true`, `noul: 0.5`, and confidence `0`.
Structured-state evidence uses separate string values, excluding keys and scalar
booleans/numbers, with no proposition matches spanning fields. This edge-only
behavior does not change the runtime/package/parser APIs described here.


The separate [System One edge handler](../../docs/public/api/systemone.js) does
not call this parser or load a formal ontology DAG. Its abstention and negation
behavior is specified in [the edge contract](../../spec/systemone-reliability.md);
those response changes do not alter this crate's parser API.

## What it does

Parses ontology documents (Turtle, RDF/XML, OWL/TriG/N-Quads, JSON-LD, SKOS/OBO annotations) into:

1. A flat, JSON-serializable **entity model** — classes, properties, individuals, labels, synonyms, hierarchy, disjointness, `sameAs`, `equivalentProperty`.
2. **Structural triples** for alignment (`subClassOf`, `domain`, `range`, `sameAs`, `equivalentProperty`, …).
3. **Per-subject retrieval chunks** (`chunk_by_subject`) for RAG ingest, with named-graph provenance preserved.
4. **Streaming quad iterators** (`quads_from_slice` / `quads_from_reader`) for incremental consumers. Entity/chunk extraction still materializes quads (label/synonym cross-refs need a full pass).

Uses only the `oxrdf` + `oxrdfio` parser + `Quad` model (no Oxigraph `Store` / SPARQL), dropping ~41 transitive crates versus full `oxigraph`.

## Usage

```rust
use owl_fast_core::{OntologyParser, ParseOptions, RdfFormat};

let ttl = br#"
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<http://example.org/Animal> a owl:Class ; rdfs:label "Animal" .
"#;

let entities = OntologyParser::parse_bytes(ttl, RdfFormat::Turtle)?;
for e in &entities {
    println!("{} :: {}", e.uri, e.label);
}

// Relative IRIs:
let opts = ParseOptions {
    base_iri: Some("http://example.org/onto".into()),
};
let entities = OntologyParser::parse_bytes_with_options(
    br#"<#Paper> a <http://www.w3.org/2002/07/owl#Class> ."#,
    RdfFormat::Turtle,
    Default::default(),
    &opts,
)?;

let chunks = OntologyParser::chunk_by_subject(ttl, RdfFormat::Turtle)?;
# Ok::<(), anyhow::Error>(())
```

## Features

- `serde` (default) — JSON round-trip for entities (alignment jobs, REST, persisted benches). Disable with `--no-default-features` for a minimal core.
- OAEI eval variants: `M1` / `M2` / `M3` / `Full` (annotations + named individuals).
- Language-aware labels (`@en` preferred); `skos:prefLabel` accepted as primary label.
- Named-graph provenance on `SubjectChunk.graph` for TriG/N-Quads.
- [`ParseOptions::base_iri`] for relative-IRI resolution on Turtle / TriG / RDF/XML / JSON-LD.

## License

MIT.
