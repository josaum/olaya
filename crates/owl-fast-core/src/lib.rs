//! OWL / RDF / OBO / SKOS entity extraction.
//!
//! Centralized parser surface for the workspace: do not duplicate this
//! logic in downstream apps (see workspace CLAUDE.md, invariant #7).
//!
//! ## Surface
//!
//! - [`OntologyParser`] — entry points: `parse_file`, `parse_bytes`,
//!   variants with `_with_eval`, `_with_structure`, and `_with_options`
//!   (base IRI for relative-IRI resolution).
//! - [`Entity`] — extracted class / property record (uri, label, synonyms,
//!   parent/child URIs, disjointWith, sameAs, equivalentProperty, depth,
//!   siblings).
//! - [`EntityType`] — `Class | ObjectProperty | DataProperty | …`
//! - [`ConferenceEvalVariant`] — OAEI M1/M2/M3 (+ Full) entity-set selector.
//! - [`ParseOptions`] — base IRI and related parse knobs.
//! - Streaming quad iterators via [`OntologyParser::quads_from_slice`] /
//!   [`OntologyParser::quads_from_reader`] (oxrdfio `SliceQuadParser` /
//!   `ReaderQuadParser`). Entity/chunk extraction still materializes quads
//!   because label/synonym resolution needs a full pass.
//!
//! ## Role boundary (not a reasoner)
//!
//! This crate is a **parser / extractor / RAG chunker**. It does **not**
//! expand OWL axioms (`equivalentClass`, property chains, restrictions,
//! inverseOf, …) into inferred triples. Platform OWL2-RL reasoning stays in
//! Crepe Datalog (`mycelia-gateway`). Asserted `owl:sameAs` /
//! `owl:equivalentProperty` links are **preserved as entity fields**, never
//! collapsed — alignment consumers need the explicit link set.
//!
//! ## Error boundary
//!
//! Returns `anyhow::Result`. Downstream crates with typed errors should
//! add a `From<anyhow::Error>` impl on their side (see `mycelia-align`).
//!
//! ## Backend
//!
//! Backed by the component crates `oxrdf` (model) + `oxrdfio` (parser
//! dispatcher). We never use a triple store: the common path collects a
//! `Vec<Quad>` and extraction is a single linear pass that buckets into the
//! same hash maps a SPARQL pattern query would build. Callers that only need
//! quads can stream via `quads_from_*` without that materialization.

#![deny(missing_docs)]
use anyhow::{anyhow, Context, Result};
use oxrdf::{GraphName, Literal, NamedOrBlankNode, Quad, Term};
use oxrdfio::RdfParser;

// Re-exports so downstream consumers (mycelia-align api.rs, job_queue.rs, bins)
// don't need a direct oxigraph dep to construct an `RdfFormat`.
pub use oxrdfio::{JsonLdProfileSet, RdfFormat, RdfParseError, ReaderQuadParser, SliceQuadParser};

/// Formal Concept Analysis (Fast FCA) module for lattice generation, Galois derivation, and implications.
pub mod fca;
pub use fca::{BitSet, ConceptLattice, FormalConcept, FormalContext, Implication};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use tracing::{debug, info};

// ─── Namespace constants ─────────────────────────────────────────────────
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const OWL_CLASS: &str = "http://www.w3.org/2002/07/owl#Class";
const RDFS_CLASS: &str = "http://www.w3.org/2000/01/rdf-schema#Class";
const OWL_OBJECT_PROPERTY: &str = "http://www.w3.org/2002/07/owl#ObjectProperty";
const OWL_DATATYPE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#DatatypeProperty";
const OWL_ANNOTATION_PROPERTY: &str = "http://www.w3.org/2002/07/owl#AnnotationProperty";
const OWL_NAMED_INDIVIDUAL: &str = "http://www.w3.org/2002/07/owl#NamedIndividual";
const OWL_THING: &str = "http://www.w3.org/2002/07/owl#Thing";
const OWL_NOTHING: &str = "http://www.w3.org/2002/07/owl#Nothing";
const OWL_DISJOINT_WITH: &str = "http://www.w3.org/2002/07/owl#disjointWith";
const OWL_SAME_AS: &str = "http://www.w3.org/2002/07/owl#sameAs";
const OWL_EQUIVALENT_PROPERTY: &str = "http://www.w3.org/2002/07/owl#equivalentProperty";
const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
const RDFS_SUBCLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
const RDFS_SUB_PROPERTY_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf";
const RDFS_DOMAIN: &str = "http://www.w3.org/2000/01/rdf-schema#domain";
const RDFS_RANGE: &str = "http://www.w3.org/2000/01/rdf-schema#range";
const OBO_EXACT_SYNONYM: &str = "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym";
const OBO_RELATED_SYNONYM: &str = "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym";
const SKOS_PREF_LABEL: &str = "http://www.w3.org/2004/02/skos/core#prefLabel";
const SKOS_ALT_LABEL: &str = "http://www.w3.org/2004/02/skos/core#altLabel";
const SKOS_HIDDEN_LABEL: &str = "http://www.w3.org/2004/02/skos/core#hiddenLabel";
const RDFS_COMMENT: &str = "http://www.w3.org/2000/01/rdf-schema#comment";
const SKOS_DEFINITION: &str = "http://www.w3.org/2004/02/skos/core#definition";
const SCHEMA_DESCRIPTION_HTTP: &str = "http://schema.org/description";
const SCHEMA_DESCRIPTION_HTTPS: &str = "https://schema.org/description";
const DC_DESCRIPTION: &str = "http://purl.org/dc/elements/1.1/description";
const DCTERMS_DESCRIPTION: &str = "http://purl.org/dc/terms/description";
const OBO_DEFINITION: &str = "http://purl.obolibrary.org/obo/IAO_0000115";

/// An entity extracted from an ontology.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Entity {
    /// URI of the entity.
    pub uri: String,

    /// Primary label (rdfs:label or local name).
    pub label: String,

    /// Human-readable definition, comment, or description.
    #[cfg_attr(feature = "serde", serde(default))]
    pub description: Option<String>,

    /// Synonyms (hasExactSynonym, hasRelatedSynonym, etc.).
    pub synonyms: Vec<String>,

    /// Parent class URIs (rdfs:subClassOf).
    pub parents: Vec<String>,

    /// Child class URIs.
    pub children: Vec<String>,

    /// Entity type (Class, ObjectProperty, etc.).
    pub entity_type: EntityType,

    /// Kind string for category structure.
    pub kind: Option<String>,

    /// Hierarchy depth (0 = root, computed after parsing).
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth: usize,

    /// URIs of disjoint classes (owl:disjointWith).
    #[cfg_attr(feature = "serde", serde(default))]
    pub disjoint_with: Vec<String>,

    /// URIs of sibling entities (entities sharing a parent).
    #[cfg_attr(feature = "serde", serde(default))]
    pub siblings: Vec<String>,

    /// URIs linked by asserted `owl:sameAs` (symmetric; **not** collapsed).
    ///
    /// Align / merge consumers need the explicit equivalence link set; this
    /// crate never merges sameAs individuals into one entity record.
    #[cfg_attr(feature = "serde", serde(default))]
    pub same_as: Vec<String>,

    /// URIs linked by asserted `owl:equivalentProperty` (symmetric; **not**
    /// collapsed into a single property record).
    #[cfg_attr(feature = "serde", serde(default))]
    pub equivalent_properties: Vec<String>,
}

/// Options for RDF parse (relative-IRI resolution and related knobs).
///
/// Passed through to oxrdfio's [`RdfParser::with_base_iri`]. N-Triples and
/// N-Quads ignore `base_iri` (absolute IRIs only).
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParseOptions {
    /// Base IRI used to resolve relative IRIs in Turtle / TriG / RDF/XML /
    /// JSON-LD / N3. `None` leaves relative IRIs unresolved (parse may error).
    pub base_iri: Option<String>,
}

/// Type of ontology entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EntityType {
    /// `owl:Class` — a class / concept.
    Class,
    /// `owl:ObjectProperty` — a relation between individuals.
    ObjectProperty,
    /// `owl:DatatypeProperty` — a relation to a literal value.
    DataProperty,
    /// `owl:AnnotationProperty` — non-logical annotation metadata.
    AnnotationProperty,
    /// `owl:NamedIndividual` — an instance.
    Individual,
    /// Entity kind could not be determined from the source triples.
    Unknown,
}

/// A structural triple emitted by `OntologyParser::parse_*_with_structure`.
///
/// `(subject_uri, object_uri, predicate_short_name)` where `predicate_short_name`
/// is one of: `"subClassOf" | "subPropertyOf" | "domain" | "range" |
/// "disjointWith" | "sameAs" | "equivalentProperty"`.
pub type StructuralTriple = (String, String, String);

/// A per-subject retrieval chunk produced by [`OntologyParser::chunk_by_subject`].
///
/// One chunk per **named** subject: the subject's human label as a header line,
/// followed by every `predicate: object` line. Literal objects (notably
/// `rdfs:comment` / `skos:definition`) are rendered verbatim so the regulatory
/// definition text survives into the embedding; object URIs are rendered as
/// their label (or derived local name). This is the embedding-ready text the
/// RAG index ingests — it replaces the regex TTL chunkers that previously lived
/// in downstream apps (workspace CLAUDE.md, invariant #7).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SubjectChunk {
    /// Subject URI (citation handle; maps to the Milvus `article` field).
    pub uri: String,
    /// Named-graph IRI when the quad is not in the default graph.
    ///
    /// `None` for TriG/N-Quads default-graph quads and for triple-only formats
    /// (Turtle, N-Triples, RDF/XML). Preserving this is required so multi-graph
    /// ontologies do not silently collapse provenance into one subject chunk.
    #[cfg_attr(feature = "serde", serde(default))]
    pub graph: Option<String>,
    /// Human label (`rdfs:label` / `skos:prefLabel`) or derived local name.
    pub label: String,
    /// Entity kind: `"Class" | "ObjectProperty" | "DataProperty" |
    /// "AnnotationProperty" | "Individual" | "Resource"`.
    pub kind: String,
    /// Embedding-ready chunk text (label header + `predicate: object` lines).
    pub text: String,
    /// Number of predicate/object lines folded into `text`.
    pub triple_count: usize,
}

/// OAEI Conference-style evaluation variant (which OWL entities participate in alignment scoring).
///
/// - **M1**: classes (`owl:Class`) only — legacy bench default.
/// - **M2**: object + datatype properties only.
/// - **M3**: classes + properties (matches common OAEI “M3” / rar2-M3 entity union).
/// - **Full**: M3 plus `owl:AnnotationProperty` and `owl:NamedIndividual`
///   (non-OAEI extraction; used when the caller needs the full TBox/ABox surface).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ConferenceEvalVariant {
    /// M1: classes only.
    #[default]
    M1,
    /// M2: `owl:ObjectProperty` + `owl:DatatypeProperty`.
    M2,
    /// M3: classes + object + datatype properties.
    M3,
    /// Full: M3 + annotation properties + named individuals.
    Full,
}

/// Parser for OWL/RDF ontologies.
pub struct OntologyParser;

impl OntologyParser {
    /// Parse an ontology file and extract entities (OAEI M1: classes only).
    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Vec<Entity>> {
        Self::parse_file_with_eval(path, ConferenceEvalVariant::default())
    }

    /// Parse an ontology file and extract entities for the given Conference evaluation variant.
    pub fn parse_file_with_eval<P: AsRef<Path>>(
        path: P,
        variant: ConferenceEvalVariant,
    ) -> Result<Vec<Entity>> {
        Self::parse_file_with_options(path, variant, &ParseOptions::default())
    }

    /// Parse an ontology file with [`ParseOptions`] (e.g. base IRI).
    pub fn parse_file_with_options<P: AsRef<Path>>(
        path: P,
        variant: ConferenceEvalVariant,
        options: &ParseOptions,
    ) -> Result<Vec<Entity>> {
        let path = path.as_ref();
        let format = Self::detect_format(path)?;
        info!("Parsing {:?} as {:?}", path, format);

        let quads = Self::parse_to_quads_from_path(path, format, options)?;
        Ok(Self::extract_entities_from_quads(&quads, variant))
    }

    /// Parse ontology from bytes (OAEI M1 by default).
    pub fn parse_bytes(data: &[u8], format: RdfFormat) -> Result<Vec<Entity>> {
        Self::parse_bytes_with_eval(data, format, ConferenceEvalVariant::default())
    }

    /// Parse ontology from bytes with a Conference evaluation variant.
    pub fn parse_bytes_with_eval(
        data: &[u8],
        format: RdfFormat,
        variant: ConferenceEvalVariant,
    ) -> Result<Vec<Entity>> {
        Self::parse_bytes_with_options(data, format, variant, &ParseOptions::default())
    }

    /// Parse ontology from bytes with [`ParseOptions`] (e.g. base IRI for relative IRIs).
    pub fn parse_bytes_with_options(
        data: &[u8],
        format: RdfFormat,
        variant: ConferenceEvalVariant,
        options: &ParseOptions,
    ) -> Result<Vec<Entity>> {
        let quads = Self::parse_to_quads_from_slice(data, format, options)?;
        Ok(Self::extract_entities_from_quads(&quads, variant))
    }

    /// Parse ontology from bytes and extract entities with structural triples (OAEI M1).
    ///
    /// Bytes counterpart of [`parse_file_with_structure`]: avoids a temp-file
    /// round-trip when the caller already holds the ontology in memory
    /// (align job queue, API upload path).
    pub fn parse_bytes_with_structure(
        data: &[u8],
        format: RdfFormat,
    ) -> Result<(Vec<Entity>, Vec<StructuralTriple>)> {
        Self::parse_bytes_with_structure_and_eval(data, format, ConferenceEvalVariant::default())
    }

    /// Parse ontology from bytes with structural triples for the given Conference evaluation variant.
    pub fn parse_bytes_with_structure_and_eval(
        data: &[u8],
        format: RdfFormat,
        variant: ConferenceEvalVariant,
    ) -> Result<(Vec<Entity>, Vec<StructuralTriple>)> {
        Self::parse_bytes_with_structure_and_options(
            data,
            format,
            variant,
            &ParseOptions::default(),
        )
    }

    /// Parse ontology from bytes with structural triples and [`ParseOptions`].
    pub fn parse_bytes_with_structure_and_options(
        data: &[u8],
        format: RdfFormat,
        variant: ConferenceEvalVariant,
        options: &ParseOptions,
    ) -> Result<(Vec<Entity>, Vec<StructuralTriple>)> {
        let quads = Self::parse_to_quads_from_slice(data, format, options)?;
        let entities = Self::extract_entities_from_quads(&quads, variant);
        let entity_uris: HashSet<&str> = entities.iter().map(|e| e.uri.as_str()).collect();
        let triples = Self::extract_structural_triples(&quads, &entity_uris);
        Ok((entities, triples))
    }

    /// Parse an ontology file and extract entities with structural triples (OAEI M1 entity set).
    /// Returns (entities, triples) where triples are (subject, object, predicate_type).
    pub fn parse_file_with_structure<P: AsRef<Path>>(
        path: P,
    ) -> Result<(Vec<Entity>, Vec<StructuralTriple>)> {
        Self::parse_file_with_structure_and_eval(path, ConferenceEvalVariant::default())
    }

    /// Parse an ontology file with structural triples for the given Conference evaluation variant.
    pub fn parse_file_with_structure_and_eval<P: AsRef<Path>>(
        path: P,
        variant: ConferenceEvalVariant,
    ) -> Result<(Vec<Entity>, Vec<StructuralTriple>)> {
        Self::parse_file_with_structure_and_options(path, variant, &ParseOptions::default())
    }

    /// Parse an ontology file with structural triples and [`ParseOptions`].
    pub fn parse_file_with_structure_and_options<P: AsRef<Path>>(
        path: P,
        variant: ConferenceEvalVariant,
        options: &ParseOptions,
    ) -> Result<(Vec<Entity>, Vec<StructuralTriple>)> {
        let path = path.as_ref();
        let format = Self::detect_format(path)?;
        info!("Parsing {:?} as {:?}", path, format);

        let quads = Self::parse_to_quads_from_path(path, format, options)?;
        let entities = Self::extract_entities_from_quads(&quads, variant);
        let entity_uris: HashSet<&str> = entities.iter().map(|e| e.uri.as_str()).collect();
        let triples = Self::extract_structural_triples(&quads, &entity_uris);
        Ok((entities, triples))
    }

    /// Stream quads from an in-memory slice without collecting into a `Vec`.
    ///
    /// Prefer this when the caller can process quads incrementally. Entity and
    /// subject-chunk extraction still need a full materialization for
    /// label/synonym cross-references — use [`parse_bytes`] / [`chunk_by_subject`]
    /// for those paths.
    pub fn quads_from_slice<'a>(
        data: &'a [u8],
        format: RdfFormat,
        options: &ParseOptions,
    ) -> Result<SliceQuadParser<'a>> {
        Ok(Self::build_parser(format, options)?.for_slice(data))
    }

    /// Stream quads from any [`Read`] without collecting into a `Vec`.
    pub fn quads_from_reader<R: Read>(
        reader: R,
        format: RdfFormat,
        options: &ParseOptions,
    ) -> Result<ReaderQuadParser<R>> {
        Ok(Self::build_parser(format, options)?.for_reader(reader))
    }

    /// Group an ontology into one retrieval chunk per **named** subject × graph.
    ///
    /// Two linear passes over the quad stream (no triple store): pass 1
    /// collects `rdfs:label` / `skos:prefLabel` and `rdf:type` for every subject
    /// key (so object URIs can be rendered as their label); pass 2 buckets
    /// `predicate: object` lines per `(named subject, graph)` and folds them
    /// into embedding-ready text. Literal objects (notably `rdfs:comment` /
    /// `skos:definition`) are kept verbatim; object URIs are rendered as their
    /// label or derived local name. Blank-node subjects and blank-node objects
    /// (OWL restriction / list nodes) are skipped — they carry no retrievable
    /// text.
    ///
    /// Named-graph provenance is preserved: TriG/N-Quads quads in a named graph
    /// produce a distinct chunk from the same subject in the default graph
    /// (workspace invariant: no silent erosion of provenance).
    ///
    /// This is the embedding-ready chunker the RAG ingest path consumes; it
    /// supersedes the regex TTL chunkers downstream apps used to carry
    /// (workspace CLAUDE.md, invariant #7).
    pub fn chunk_by_subject(data: &[u8], format: RdfFormat) -> Result<Vec<SubjectChunk>> {
        Self::chunk_by_subject_with_options(data, format, &ParseOptions::default())
    }

    /// Like [`chunk_by_subject`], with [`ParseOptions`] (base IRI for relative IRIs).
    pub fn chunk_by_subject_with_options(
        data: &[u8],
        format: RdfFormat,
        options: &ParseOptions,
    ) -> Result<Vec<SubjectChunk>> {
        let quads = Self::parse_to_quads_from_slice(data, format, options)?;

        // Pass 1: label + kind by subject key (named OR blank, for object deref).
        // Labels are graph-agnostic so object-URI rendering stays stable across
        // graphs; language preference still applies (en > untagged > other).
        let mut labels: HashMap<String, String> = HashMap::new();
        let mut label_rank: HashMap<String, u8> = HashMap::new();
        let mut kinds: HashMap<String, String> = HashMap::new();
        for quad in &quads {
            let key = match &quad.subject {
                NamedOrBlankNode::NamedNode(s) => s.as_str(),
                NamedOrBlankNode::BlankNode(b) => b.as_str(),
            };
            match quad.predicate.as_str() {
                RDFS_LABEL | SKOS_PREF_LABEL => {
                    if let Term::Literal(lit) = &quad.object {
                        Self::prefer_label(&mut labels, &mut label_rank, key, lit);
                    }
                }
                RDF_TYPE => {
                    if let Term::NamedNode(o) = &quad.object {
                        let k = match o.as_str() {
                            OWL_CLASS | RDFS_CLASS => "Class",
                            OWL_OBJECT_PROPERTY => "ObjectProperty",
                            OWL_DATATYPE_PROPERTY => "DataProperty",
                            OWL_ANNOTATION_PROPERTY => "AnnotationProperty",
                            OWL_NAMED_INDIVIDUAL => "Individual",
                            t if t.ends_with("NamedIndividual") => "Individual",
                            _ => continue,
                        };
                        kinds
                            .entry(key.to_string())
                            .or_insert_with(|| k.to_string());
                    }
                }
                _ => {}
            }
        }

        // Pass 2: bucket predicate/object lines per (NAMED subject, graph), stream order.
        let mut order: Vec<(String, Option<String>)> = Vec::new();
        let mut lines: HashMap<(String, Option<String>), Vec<String>> = HashMap::new();
        for quad in &quads {
            let su = match &quad.subject {
                NamedOrBlankNode::NamedNode(s) => s.as_str(),
                _ => continue,
            };
            let graph = Self::graph_iri(&quad.graph_name);
            let pred = Self::extract_local_name(quad.predicate.as_str());
            let obj = match &quad.object {
                Term::Literal(lit) => lit.value().trim().to_string(),
                Term::NamedNode(o) => labels
                    .get(o.as_str())
                    .cloned()
                    .unwrap_or_else(|| Self::extract_local_name(o.as_str())),
                // blank-node objects (restrictions / RDF lists) carry no text
                _ => continue,
            };
            if obj.is_empty() {
                continue;
            }
            let key = (su.to_string(), graph.clone());
            if !lines.contains_key(&key) {
                order.push(key.clone());
            }
            lines.entry(key).or_default().push(format!("{pred}: {obj}"));
        }

        let chunks = order
            .into_iter()
            .map(|(uri, graph)| {
                let label = labels
                    .get(&uri)
                    .cloned()
                    .unwrap_or_else(|| Self::extract_local_name(&uri));
                let kind = kinds
                    .get(&uri)
                    .cloned()
                    .unwrap_or_else(|| "Resource".to_string());
                let body = lines
                    .remove(&(uri.clone(), graph.clone()))
                    .unwrap_or_default();
                let triple_count = body.len();
                let text = format!("{label}\n{}", body.join("\n"));
                SubjectChunk {
                    uri,
                    graph,
                    label,
                    kind,
                    text,
                    triple_count,
                }
            })
            .collect();

        Ok(chunks)
    }

    // ─── format detection ────────────────────────────────────────────────

    fn detect_format(path: &Path) -> Result<RdfFormat> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());

        match ext.as_deref() {
            Some("owl") | Some("rdf") | Some("xml") => Ok(RdfFormat::RdfXml),
            Some("ttl") => Ok(RdfFormat::Turtle),
            Some("nt") => Ok(RdfFormat::NTriples),
            Some("nq") => Ok(RdfFormat::NQuads),
            Some("trig") => Ok(RdfFormat::TriG),
            Some("jsonld") | Some("json") => Ok(RdfFormat::JsonLd {
                profile: JsonLdProfileSet::empty(),
            }),
            _ => Err(anyhow!("Unknown file format: {:?}", path)),
        }
    }

    // ─── parsing helpers (oxrdfio → Vec<Quad> or streaming iterators) ────

    fn build_parser(format: RdfFormat, options: &ParseOptions) -> Result<RdfParser> {
        let mut parser = RdfParser::from_format(format);
        if let Some(base) = options.base_iri.as_deref() {
            parser = parser
                .with_base_iri(base)
                .map_err(|e| anyhow!("invalid base IRI {base:?}: {e}"))?;
        }
        Ok(parser)
    }

    fn parse_to_quads_from_slice(
        data: &[u8],
        format: RdfFormat,
        options: &ParseOptions,
    ) -> Result<Vec<Quad>> {
        match Self::build_parser(format, options)?.for_slice(data).collect::<std::result::Result<Vec<_>, _>>() {
            Ok(quads) => Ok(quads),
            Err(e) => {
                let alt = match format {
                    RdfFormat::RdfXml => Some(RdfFormat::Turtle),
                    RdfFormat::Turtle => Some(RdfFormat::RdfXml),
                    _ => None,
                };
                if let Some(alt_format) = alt {
                    if let Ok(parser) = Self::build_parser(alt_format, options) {
                        if let Ok(quads) = parser.for_slice(data).collect::<std::result::Result<Vec<_>, _>>() {
                            return Ok(quads);
                        }
                    }
                }
                Err(anyhow::anyhow!("RDF parse error: {e}"))
            }
        }
    }

    fn parse_to_quads_from_path(
        path: &Path,
        format: RdfFormat,
        options: &ParseOptions,
    ) -> Result<Vec<Quad>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        match Self::build_parser(format, options)?.for_reader(reader).collect::<std::result::Result<Vec<_>, _>>() {
            Ok(quads) => Ok(quads),
            Err(e) => {
                let alt = match format {
                    RdfFormat::RdfXml => Some(RdfFormat::Turtle),
                    RdfFormat::Turtle => Some(RdfFormat::RdfXml),
                    _ => None,
                };
                if let Some(alt_format) = alt {
                    let file = File::open(path)?;
                    let reader = BufReader::new(file);
                    if let Ok(parser) = Self::build_parser(alt_format, options) {
                        if let Ok(quads) = parser.for_reader(reader).collect::<std::result::Result<Vec<_>, _>>() {
                            return Ok(quads);
                        }
                    }
                }
                Err(anyhow::anyhow!("RDF parse error: {e}"))
            }
        }
    }

    // ─── extraction (single-pass over quads) ─────────────────────────────

    fn extract_entities_from_quads(quads: &[Quad], variant: ConferenceEvalVariant) -> Vec<Entity> {
        let mut class_uris: HashSet<String> = HashSet::new();
        let mut object_prop_uris: HashSet<String> = HashSet::new();
        let mut datatype_prop_uris: HashSet<String> = HashSet::new();
        let mut annotation_prop_uris: HashSet<String> = HashSet::new();
        let mut individual_uris: HashSet<String> = HashSet::new();
        let mut entity_types: HashMap<String, EntityType> = HashMap::new();

        // Collect ALL labels by subject key (named OR blank), so OBO blank-node
        // synonym references (e.g. `_:b1 rdfs:label "foo"`) can be resolved.
        // Rank tracks language preference so `@en` / untagged win over other langs.
        let mut all_labels: HashMap<String, String> = HashMap::new();
        let mut all_descriptions: HashMap<String, String> = HashMap::new();
        let mut label_rank: HashMap<String, u8> = HashMap::new();

        // Synonym candidates captured during the pass, resolved against
        // `all_labels` and filtered to `entity_uris` afterwards.
        let mut syn_pending: Vec<(String, Term)> = Vec::new();

        // Subclass / subproperty edges captured raw, filtered later.
        let mut subclass_edges: Vec<(String, String)> = Vec::new();
        let mut subprop_edges: Vec<(String, String)> = Vec::new();
        let mut disjoint_edges: Vec<(String, String)> = Vec::new();
        // Asserted equivalence links — preserved, never collapsed (align needs
        // the explicit sameAs / equivalentProperty set; Crepe owns reasoning).
        let mut same_as_edges: Vec<(String, String)> = Vec::new();
        let mut equiv_prop_edges: Vec<(String, String)> = Vec::new();

        // Single linear pass — same dispatch logic the old `quads_for_pattern`
        // calls did sequentially, but without the indexed-Store overhead.
        for quad in quads {
            let pred = quad.predicate.as_str();
            match pred {
                RDF_TYPE => {
                    if let (NamedOrBlankNode::NamedNode(s), Term::NamedNode(o)) =
                        (&quad.subject, &quad.object)
                    {
                        let uri = s.as_str();
                        match o.as_str() {
                            OWL_CLASS | RDFS_CLASS if !Self::should_skip(uri) => {
                                class_uris.insert(uri.to_string());
                                entity_types.insert(uri.to_string(), EntityType::Class);
                            }
                            OWL_OBJECT_PROPERTY => {
                                object_prop_uris.insert(uri.to_string());
                                entity_types.insert(uri.to_string(), EntityType::ObjectProperty);
                            }
                            OWL_DATATYPE_PROPERTY => {
                                datatype_prop_uris.insert(uri.to_string());
                                entity_types.insert(uri.to_string(), EntityType::DataProperty);
                            }
                            OWL_ANNOTATION_PROPERTY => {
                                annotation_prop_uris.insert(uri.to_string());
                                entity_types
                                    .insert(uri.to_string(), EntityType::AnnotationProperty);
                            }
                            OWL_NAMED_INDIVIDUAL => {
                                individual_uris.insert(uri.to_string());
                                entity_types.insert(uri.to_string(), EntityType::Individual);
                            }
                            other => {
                                if variant == ConferenceEvalVariant::Full
                                    && !Self::should_skip(uri)
                                    && !other.starts_with("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
                                    && !other.starts_with("http://www.w3.org/2000/01/rdf-schema#")
                                    && !other.starts_with("http://www.w3.org/2002/07/owl#")
                                {
                                    individual_uris.insert(uri.to_string());
                                    entity_types.insert(uri.to_string(), EntityType::Individual);
                                    subclass_edges.push((uri.to_string(), other.to_string()));
                                }
                            }
                        }
                    }
                }
                RDFS_LABEL | SKOS_PREF_LABEL => {
                    if let Term::Literal(lit) = &quad.object {
                        let key = match &quad.subject {
                            NamedOrBlankNode::NamedNode(s) => Some(s.as_str()),
                            NamedOrBlankNode::BlankNode(b) => Some(b.as_str()),
                        };
                        if let Some(k) = key {
                            Self::prefer_label(&mut all_labels, &mut label_rank, k, lit);
                        }
                    }
                }
                RDFS_SUBCLASS_OF => {
                    if let (NamedOrBlankNode::NamedNode(s), Term::NamedNode(o)) =
                        (&quad.subject, &quad.object)
                    {
                        subclass_edges.push((s.as_str().to_string(), o.as_str().to_string()));
                    }
                }
                RDFS_SUB_PROPERTY_OF => {
                    if let (NamedOrBlankNode::NamedNode(s), Term::NamedNode(o)) =
                        (&quad.subject, &quad.object)
                    {
                        subprop_edges.push((s.as_str().to_string(), o.as_str().to_string()));
                    }
                }
                OWL_DISJOINT_WITH => {
                    if let (NamedOrBlankNode::NamedNode(s), Term::NamedNode(o)) =
                        (&quad.subject, &quad.object)
                    {
                        disjoint_edges.push((s.as_str().to_string(), o.as_str().to_string()));
                    }
                }
                OWL_SAME_AS => {
                    if let (NamedOrBlankNode::NamedNode(s), Term::NamedNode(o)) =
                        (&quad.subject, &quad.object)
                    {
                        same_as_edges.push((s.as_str().to_string(), o.as_str().to_string()));
                    }
                }
                OWL_EQUIVALENT_PROPERTY => {
                    if let (NamedOrBlankNode::NamedNode(s), Term::NamedNode(o)) =
                        (&quad.subject, &quad.object)
                    {
                        equiv_prop_edges.push((s.as_str().to_string(), o.as_str().to_string()));
                    }
                }
                OBO_EXACT_SYNONYM | OBO_RELATED_SYNONYM | SKOS_ALT_LABEL | SKOS_HIDDEN_LABEL => {
                    if let NamedOrBlankNode::NamedNode(s) = &quad.subject {
                        syn_pending.push((s.as_str().to_string(), quad.object.clone()));
                    }
                }
                RDFS_COMMENT
                | SKOS_DEFINITION
                | SCHEMA_DESCRIPTION_HTTP
                | SCHEMA_DESCRIPTION_HTTPS
                | DC_DESCRIPTION
                | DCTERMS_DESCRIPTION
                | OBO_DEFINITION => {
                    if let Term::Literal(lit) = &quad.object {
                        let key = match &quad.subject {
                            NamedOrBlankNode::NamedNode(s) => Some(s.as_str()),
                            NamedOrBlankNode::BlankNode(b) => Some(b.as_str()),
                        };
                        if let Some(k) = key {
                            let val = lit.value().trim();
                            if !val.is_empty() {
                                all_descriptions
                                    .entry(k.to_string())
                                    .or_insert_with(|| val.to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Resolve entity set per variant.
        let entity_uris: HashSet<String> = match variant {
            ConferenceEvalVariant::M1 => class_uris,
            ConferenceEvalVariant::M2 => {
                let mut s = HashSet::new();
                s.extend(object_prop_uris);
                s.extend(datatype_prop_uris);
                s
            }
            ConferenceEvalVariant::M3 => {
                let mut s = class_uris;
                s.extend(object_prop_uris);
                s.extend(datatype_prop_uris);
                s
            }
            ConferenceEvalVariant::Full => {
                let mut s = class_uris;
                s.extend(object_prop_uris);
                s.extend(datatype_prop_uris);
                s.extend(annotation_prop_uris);
                s.extend(individual_uris);
                s
            }
        };

        match variant {
            ConferenceEvalVariant::M1 => {
                info!("Found {} class entities (OAEI M1)", entity_uris.len());
            }
            ConferenceEvalVariant::M2 => {
                info!(
                    "Found {} property entities (OAEI M2: object+datatype)",
                    entity_uris.len()
                );
            }
            ConferenceEvalVariant::M3 => {
                info!(
                    "Found {} entities (OAEI M3: classes+properties)",
                    entity_uris.len()
                );
            }
            ConferenceEvalVariant::Full => {
                info!(
                    "Found {} entities (Full: classes+properties+annotations+individuals)",
                    entity_uris.len()
                );
            }
        }

        // Restrict labels to entity set.
        let labels: HashMap<String, String> = all_labels
            .iter()
            .filter(|(k, _)| entity_uris.contains(k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Resolve synonyms (literal direct, or blank/named-node label dereference).
        let mut synonyms: HashMap<String, Vec<String>> = HashMap::new();
        for (uri, syn_obj) in syn_pending {
            if !entity_uris.contains(&uri) {
                continue;
            }
            match syn_obj {
                Term::Literal(lit) => {
                    synonyms
                        .entry(uri)
                        .or_default()
                        .push(lit.value().to_string());
                }
                Term::NamedNode(n) => {
                    if let Some(syn_label) = all_labels.get(n.as_str()) {
                        synonyms.entry(uri).or_default().push(syn_label.clone());
                    }
                }
                Term::BlankNode(b) => {
                    if let Some(syn_label) = all_labels.get(b.as_str()) {
                        synonyms.entry(uri).or_default().push(syn_label.clone());
                    }
                }
                // RDF 1.2 triple terms (object position only). Skip.
                Term::Triple(_) => {}
            }
        }

        debug!("Extracted {} entities with synonyms", synonyms.len());

        // Subclass + subproperty parents (both feed the same parent/child maps;
        // subproperty matters for M2/M3).
        let mut parents: HashMap<String, Vec<String>> = HashMap::new();
        let mut children: HashMap<String, Vec<String>> = HashMap::new();
        for (child, parent) in subclass_edges.into_iter().chain(subprop_edges) {
            if entity_uris.contains(&child) {
                parents
                    .entry(child.clone())
                    .or_default()
                    .push(parent.clone());
                children.entry(parent).or_default().push(child);
            }
        }

        // Symmetric disjoint.
        let mut disjoint_with: HashMap<String, Vec<String>> = HashMap::new();
        for (s, o) in disjoint_edges {
            if entity_uris.contains(&s) && entity_uris.contains(&o) {
                disjoint_with.entry(s.clone()).or_default().push(o.clone());
                disjoint_with.entry(o).or_default().push(s);
            }
        }

        // Symmetric sameAs — keep both entity records; only record the link.
        let mut same_as: HashMap<String, Vec<String>> = HashMap::new();
        for (s, o) in same_as_edges {
            if entity_uris.contains(&s) && entity_uris.contains(&o) && s != o {
                same_as.entry(s.clone()).or_default().push(o.clone());
                same_as.entry(o).or_default().push(s);
            }
        }

        // Symmetric equivalentProperty — same policy as sameAs.
        let mut equivalent_properties: HashMap<String, Vec<String>> = HashMap::new();
        for (s, o) in equiv_prop_edges {
            if entity_uris.contains(&s) && entity_uris.contains(&o) && s != o {
                equivalent_properties
                    .entry(s.clone())
                    .or_default()
                    .push(o.clone());
                equivalent_properties.entry(o).or_default().push(s);
            }
        }

        // Build entities.
        let mut entities: Vec<Entity> = entity_uris
            .into_iter()
            .map(|uri| {
                let label = labels
                    .get(&uri)
                    .cloned()
                    .unwrap_or_else(|| Self::extract_local_name(&uri));

                let etype = entity_types.get(&uri).copied().unwrap_or(EntityType::Class);
                Entity {
                    entity_type: etype,
                    label,
                    description: all_descriptions.get(&uri).cloned(),
                    synonyms: synonyms.get(&uri).cloned().unwrap_or_default(),

                    parents: parents.get(&uri).cloned().unwrap_or_default(),
                    children: children.get(&uri).cloned().unwrap_or_default(),
                    kind: Some(match etype {
                        EntityType::Class => "Class".to_string(),
                        EntityType::ObjectProperty
                        | EntityType::DataProperty
                        | EntityType::AnnotationProperty => "Property".to_string(),
                        EntityType::Individual => "Individual".to_string(),
                        EntityType::Unknown => "Class".to_string(),
                    }),
                    depth: 0,
                    disjoint_with: disjoint_with.get(&uri).cloned().unwrap_or_default(),
                    siblings: vec![],
                    same_as: same_as.get(&uri).cloned().unwrap_or_default(),
                    equivalent_properties: equivalent_properties
                        .get(&uri)
                        .cloned()
                        .unwrap_or_default(),
                    uri,
                }
            })
            .collect();

        Self::compute_hierarchy_metadata(&mut entities);

        debug!("Extracted {} entities", entities.len());
        entities
    }

    fn extract_structural_triples(
        quads: &[Quad],
        entity_uris: &HashSet<&str>,
    ) -> Vec<StructuralTriple> {
        let mut triples = Vec::new();
        for quad in quads {
            let pred_type = match quad.predicate.as_str() {
                RDFS_SUBCLASS_OF => "subClassOf",
                RDFS_SUB_PROPERTY_OF => "subPropertyOf",
                RDFS_DOMAIN => "domain",
                RDFS_RANGE => "range",
                OWL_DISJOINT_WITH => "disjointWith",
                OWL_SAME_AS => "sameAs",
                OWL_EQUIVALENT_PROPERTY => "equivalentProperty",
                _ => continue,
            };
            if let (NamedOrBlankNode::NamedNode(s), Term::NamedNode(o)) =
                (&quad.subject, &quad.object)
            {
                let su = s.as_str();
                let ou = o.as_str();
                if entity_uris.contains(su) && entity_uris.contains(ou) {
                    triples.push((su.to_string(), ou.to_string(), pred_type.to_string()));
                }
            }
        }
        triples
    }

    /// Compute depth and sibling information for entities.
    fn compute_hierarchy_metadata(entities: &mut [Entity]) {
        use std::collections::VecDeque;

        let n = entities.len();
        if n == 0 {
            return;
        }

        let uri_to_idx: HashMap<String, usize> = entities
            .iter()
            .enumerate()
            .map(|(i, e)| (e.uri.clone(), i))
            .collect();

        let parent_indices: Vec<Vec<usize>> = entities
            .iter()
            .map(|e| {
                e.parents
                    .iter()
                    .filter_map(|p| uri_to_idx.get(p).copied())
                    .collect()
            })
            .collect();

        let mut child_indices = vec![Vec::new(); n];
        for (child_idx, parents) in parent_indices.iter().enumerate() {
            for &parent_idx in parents {
                child_indices[parent_idx].push(child_idx);
            }
        }

        let mut depths = vec![usize::MAX; n];
        let mut queue = VecDeque::new();
        for i in 0..n {
            if parent_indices[i].is_empty() {
                depths[i] = 0;
                queue.push_back(i);
            }
        }
        while let Some(node) = queue.pop_front() {
            let current_depth = depths[node];
            for &child in &child_indices[node] {
                let new_depth = current_depth + 1;
                if new_depth < depths[child] {
                    depths[child] = new_depth;
                    queue.push_back(child);
                }
            }
        }

        let mut sibling_indices: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for i in 0..n {
            for &parent_idx in &parent_indices[i] {
                for &sibling_idx in &child_indices[parent_idx] {
                    if sibling_idx != i {
                        sibling_indices[i].insert(sibling_idx);
                    }
                }
            }
        }

        let sibling_uris: Vec<Vec<String>> = (0..n)
            .map(|i| {
                sibling_indices[i]
                    .iter()
                    .map(|&idx| entities[idx].uri.clone())
                    .collect()
            })
            .collect();

        for (i, entity) in entities.iter_mut().enumerate() {
            entity.depth = if depths[i] == usize::MAX {
                0
            } else {
                depths[i]
            };
            entity.siblings = sibling_uris[i].clone();
        }
    }

    fn should_skip(uri: &str) -> bool {
        // Exact IRI match only: substring `contains("Thing")` previously
        // dropped legitimate domain classes (e.g. `ex:SomethingElse`).
        // Obsolete / meta-vocabulary filtering stays substring-based because
        // OBO uses many `*obsolete*` local names under varying namespaces.
        uri == OWL_THING
            || uri == OWL_NOTHING
            || uri.contains("oboInOwl")
            || uri.contains("obsolete")
    }

    /// Named-graph IRI for provenance, or `None` for the default graph.
    fn graph_iri(graph: &GraphName) -> Option<String> {
        match graph {
            GraphName::NamedNode(n) => Some(n.as_str().to_string()),
            GraphName::BlankNode(b) => Some(format!("_:{}", b.as_str())),
            GraphName::DefaultGraph => None,
        }
    }

    /// Language preference rank for label selection (higher wins).
    ///
    /// English (`@en` / `@en-*`) is preferred for OAEI / RAG English pipelines;
    /// untagged literals beat other languages so a bare `rdfs:label` is not
    /// overwritten by a late `@fr` triple in stream order.
    fn label_rank(lit: &Literal) -> u8 {
        match lit.language() {
            Some(lang) if lang == "en" || lang.starts_with("en-") => 2,
            None => 1,
            Some(_) => 0,
        }
    }

    /// Insert `lit` as the preferred label for `key` when its language rank
    /// is strictly better than the current one (or none exists yet).
    fn prefer_label(
        labels: &mut HashMap<String, String>,
        ranks: &mut HashMap<String, u8>,
        key: &str,
        lit: &Literal,
    ) {
        let rank = Self::label_rank(lit);
        let better = ranks.get(key).map(|&r| rank > r).unwrap_or(true);
        if better {
            labels.insert(key.to_string(), lit.value().to_string());
            ranks.insert(key.to_string(), rank);
        }
    }

    fn extract_local_name(uri: &str) -> String {
        let start = uri
            .rfind('#')
            .or_else(|| uri.rfind('/'))
            .map(|i| i + 1)
            .unwrap_or(0);

        uri[start..].replace('_', " ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(suffix: &str, contents: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new()
            .suffix(suffix)
            .tempfile()
            .expect("tempfile");
        f.write_all(contents).expect("write");
        f.flush().expect("flush");
        f
    }

    const MIN_TTL: &[u8] = br#"
        @prefix ex: <http://example.org/> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:A a owl:Class ; rdfs:label "A" .
    "#;

    const MIN_RDFXML: &[u8] = br#"<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:owl="http://www.w3.org/2002/07/owl#"
                 xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
          <owl:Class rdf:about="http://example.org/A">
            <rdfs:label>A</rdfs:label>
          </owl:Class>
        </rdf:RDF>
    "#;

    #[test]
    fn test_extract_local_name() {
        assert_eq!(
            OntologyParser::extract_local_name("http://example.org#MyClass"),
            "MyClass"
        );
        assert_eq!(
            OntologyParser::extract_local_name("http://example.org/ontology/My_Class"),
            "My Class"
        );
    }

    #[test]
    fn parses_skos_and_blank_node_synonyms() {
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
            @prefix obo: <http://www.geneontology.org/formats/oboInOwl#> .

            ex:Paper a owl:Class ;
                rdfs:label "Paper" ;
                skos:altLabel "Contribution" ;
                skos:hiddenLabel "Submission" ;
                obo:hasExactSynonym _:paperSynonym .

            _:paperSynonym rdfs:label "Manuscript" .
        "#;

        let entities = OntologyParser::parse_bytes(ttl, RdfFormat::Turtle).unwrap();
        let paper = entities
            .iter()
            .find(|entity| entity.uri == "http://example.org/Paper")
            .unwrap();

        assert!(paper.synonyms.contains(&"Contribution".to_string()));
        assert!(paper.synonyms.contains(&"Submission".to_string()));
        assert!(paper.synonyms.contains(&"Manuscript".to_string()));
    }

    #[test]
    fn chunk_by_subject_keeps_definitions_and_resolves_object_labels() {
        // Regulatory-style ontology: the retrievable value lives in rdfs:comment,
        // and a relation points at another class by URI. The chunk must (a) emit
        // one chunk per named subject, (b) keep the comment text verbatim so RAG
        // grounding has the definition, and (c) render the object URI as its
        // label, not a bare IRI — otherwise retrieval scores against noise.
        let ttl = br#"
            @prefix ex: <http://example.org/cvm#> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

            ex:InsiderTrading a owl:Class ;
                rdfs:label "Insider Trading" ;
                rdfs:comment "Trading on material non-public information." ;
                rdfs:subClassOf ex:MarketAbuse .

            ex:MarketAbuse a owl:Class ; rdfs:label "Market Abuse" .
        "#;

        let chunks = OntologyParser::chunk_by_subject(ttl, RdfFormat::Turtle).unwrap();
        assert_eq!(chunks.len(), 2, "one chunk per named subject");

        let it = chunks
            .iter()
            .find(|c| c.uri == "http://example.org/cvm#InsiderTrading")
            .expect("InsiderTrading chunk");
        assert_eq!(it.label, "Insider Trading");
        assert_eq!(it.kind, "Class");
        assert!(
            it.text
                .contains("Trading on material non-public information."),
            "rdfs:comment must survive into chunk text: {}",
            it.text
        );
        assert!(
            it.text.contains("Market Abuse"),
            "object URI must resolve to its label, not a bare IRI: {}",
            it.text
        );
    }

    #[test]
    fn conference_eval_variants_select_entity_sets() {
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

            ex:A a owl:Class ; rdfs:label "A" .
            ex:writes a owl:ObjectProperty ; rdfs:label "writes" .
            ex:age a owl:DatatypeProperty ; rdfs:range xsd:int ; rdfs:label "age" .
        "#;

        let m1 = OntologyParser::parse_bytes_with_eval(
            ttl,
            RdfFormat::Turtle,
            ConferenceEvalVariant::M1,
        )
        .unwrap();
        assert_eq!(m1.len(), 1);

        let m2 = OntologyParser::parse_bytes_with_eval(
            ttl,
            RdfFormat::Turtle,
            ConferenceEvalVariant::M2,
        )
        .unwrap();
        assert_eq!(m2.len(), 2);

        let m3 = OntologyParser::parse_bytes_with_eval(
            ttl,
            RdfFormat::Turtle,
            ConferenceEvalVariant::M3,
        )
        .unwrap();
        assert_eq!(m3.len(), 3);
    }

    // ─── Contract tests pinning the public surface used by consumers

    #[test]
    fn parse_file_detects_turtle_by_extension() {
        let f = write_tmp(".ttl", MIN_TTL);
        let entities = OntologyParser::parse_file(f.path()).expect("parse_file ttl");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].uri, "http://example.org/A");
    }

    #[test]
    fn parse_file_detects_rdfxml_by_extension() {
        for ext in [".owl", ".rdf", ".xml"] {
            let f = write_tmp(ext, MIN_RDFXML);
            let entities =
                OntologyParser::parse_file(f.path()).unwrap_or_else(|e| panic!("{ext}: {e:?}"));
            assert_eq!(entities.len(), 1, "ext {ext}");
            assert_eq!(entities[0].uri, "http://example.org/A");
        }
    }

    #[test]
    fn parse_file_unknown_extension_errors() {
        let f = write_tmp(".xyz", MIN_TTL);
        let err = OntologyParser::parse_file(f.path()).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("Unknown file format"),
            "expected unknown-format error, got: {msg}"
        );
    }

    #[test]
    fn subclass_populates_parents_children_depth_siblings() {
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            ex:Root  a owl:Class ; rdfs:label "Root" .
            ex:Mid   a owl:Class ; rdfs:label "Mid"   ; rdfs:subClassOf ex:Root .
            ex:Leaf1 a owl:Class ; rdfs:label "Leaf1" ; rdfs:subClassOf ex:Mid .
            ex:Leaf2 a owl:Class ; rdfs:label "Leaf2" ; rdfs:subClassOf ex:Mid .
        "#;
        let entities = OntologyParser::parse_bytes(ttl, RdfFormat::Turtle).unwrap();
        let by_label = |lbl: &str| {
            entities
                .iter()
                .find(|e| e.label == lbl)
                .unwrap_or_else(|| panic!("missing {lbl}"))
        };

        let root = by_label("Root");
        let mid = by_label("Mid");
        let l1 = by_label("Leaf1");
        let l2 = by_label("Leaf2");

        assert!(root.parents.is_empty());
        assert!(root
            .children
            .contains(&"http://example.org/Mid".to_string()));
        assert_eq!(root.depth, 0);

        assert!(mid.parents.contains(&"http://example.org/Root".to_string()));
        assert!(mid
            .children
            .contains(&"http://example.org/Leaf1".to_string()));
        assert!(mid
            .children
            .contains(&"http://example.org/Leaf2".to_string()));
        assert_eq!(mid.depth, 1);

        assert!(l1.parents.contains(&"http://example.org/Mid".to_string()));
        assert_eq!(l1.depth, 2);
        assert!(
            l1.siblings
                .contains(&"http://example.org/Leaf2".to_string()),
            "Leaf1 siblings = {:?}",
            l1.siblings
        );
        assert!(l2
            .siblings
            .contains(&"http://example.org/Leaf1".to_string()));
    }

    #[test]
    fn disjoint_with_is_symmetric() {
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            ex:A a owl:Class ; rdfs:label "A" ; owl:disjointWith ex:B .
            ex:B a owl:Class ; rdfs:label "B" .
        "#;
        let entities = OntologyParser::parse_bytes(ttl, RdfFormat::Turtle).unwrap();
        let a = entities.iter().find(|e| e.label == "A").unwrap();
        let b = entities.iter().find(|e| e.label == "B").unwrap();
        assert!(a
            .disjoint_with
            .contains(&"http://example.org/B".to_string()));
        assert!(b
            .disjoint_with
            .contains(&"http://example.org/A".to_string()));
    }

    #[test]
    fn label_falls_back_to_local_name_when_rdfs_label_absent() {
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            ex:Foo_Bar a owl:Class .
        "#;
        let entities = OntologyParser::parse_bytes(ttl, RdfFormat::Turtle).unwrap();
        let foo = entities
            .iter()
            .find(|e| e.uri == "http://example.org/Foo_Bar")
            .unwrap();
        assert_eq!(foo.label, "Foo Bar");
    }

    #[test]
    fn entity_kind_classifies_by_type() {
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            ex:C  a owl:Class ; rdfs:label "C" .
            ex:OP a owl:ObjectProperty ; rdfs:label "OP" .
            ex:DP a owl:DatatypeProperty ; rdfs:label "DP" .
        "#;
        let entities = OntologyParser::parse_bytes_with_eval(
            ttl,
            RdfFormat::Turtle,
            ConferenceEvalVariant::M3,
        )
        .unwrap();
        let kind_of = |lbl: &str| {
            entities
                .iter()
                .find(|e| e.label == lbl)
                .unwrap_or_else(|| panic!("missing {lbl}"))
                .kind
                .clone()
                .expect("kind set")
        };
        assert_eq!(kind_of("C"), "Class");
        assert_eq!(kind_of("OP"), "Property");
        assert_eq!(kind_of("DP"), "Property");

        let etype_of = |lbl: &str| {
            entities
                .iter()
                .find(|e| e.label == lbl)
                .unwrap()
                .entity_type
        };
        assert_eq!(etype_of("C"), EntityType::Class);
        assert_eq!(etype_of("OP"), EntityType::ObjectProperty);
        assert_eq!(etype_of("DP"), EntityType::DataProperty);
    }

    #[test]
    fn m2_uses_subproperty_for_property_hierarchy() {
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            ex:parentProp a owl:ObjectProperty ; rdfs:label "parentProp" .
            ex:childProp  a owl:ObjectProperty ; rdfs:label "childProp" ;
                rdfs:subPropertyOf ex:parentProp .
        "#;
        let entities = OntologyParser::parse_bytes_with_eval(
            ttl,
            RdfFormat::Turtle,
            ConferenceEvalVariant::M2,
        )
        .unwrap();
        let child = entities.iter().find(|e| e.label == "childProp").unwrap();
        assert!(
            child
                .parents
                .contains(&"http://example.org/parentProp".to_string()),
            "child.parents = {:?}",
            child.parents
        );
    }

    #[test]
    fn parse_file_with_structure_returns_filtered_triples() {
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            ex:A a owl:Class ; rdfs:label "A" .
            ex:B a owl:Class ; rdfs:label "B" ; rdfs:subClassOf ex:A ; owl:disjointWith ex:A .
        "#;
        let f = write_tmp(".ttl", ttl);
        let (entities, triples) = OntologyParser::parse_file_with_structure(f.path()).unwrap();
        assert_eq!(entities.len(), 2);

        assert!(
            triples.iter().any(|(s, o, p)| s == "http://example.org/B"
                && o == "http://example.org/A"
                && p == "subClassOf"),
            "missing subClassOf triple, got {triples:?}"
        );
        assert!(
            triples.iter().any(|(_, _, p)| p == "disjointWith"),
            "missing disjointWith triple, got {triples:?}"
        );
    }

    #[test]
    fn owl_thing_is_filtered_from_classes() {
        let ttl = br#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .
            owl:Thing   a owl:Class .
            owl:Nothing a owl:Class .
            ex:Real     a owl:Class ; rdfs:label "Real" .
            # Must NOT be dropped by a substring "Thing"/"Nothing" filter -
            # domain IRIs legitimately contain those tokens.
            ex:SomethingElse a owl:Class ; rdfs:label "SomethingElse" .
            ex:DoNothing a owl:Class ; rdfs:label "DoNothing" .
        "#;
        let entities = OntologyParser::parse_bytes(ttl, RdfFormat::Turtle).unwrap();
        let labels: HashSet<_> = entities.iter().map(|e| e.label.as_str()).collect();
        assert!(labels.contains("Real"));
        assert!(
            labels.contains("SomethingElse"),
            "substring Thing filter must not drop domain classes: {labels:?}"
        );
        assert!(
            labels.contains("DoNothing"),
            "substring Nothing filter must not drop domain classes: {labels:?}"
        );
        assert!(!labels.contains("Thing"));
        assert!(!labels.contains("Nothing"));
        assert_eq!(entities.len(), 3);
    }

    #[test]
    fn prefers_english_label_over_other_languages() {
        // Multilingual ontologies emit several rdfs:label triples; stream order
        // must not silently pick a non-English label when @en is present —
        // OAEI / RAG pipelines assume English primary labels.
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            ex:Paper a owl:Class ;
                rdfs:label "Papier"@fr ;
                rdfs:label "Paper"@en ;
                rdfs:label "Papel"@es .
        "#;
        let entities = OntologyParser::parse_bytes(ttl, RdfFormat::Turtle).unwrap();
        let paper = entities
            .iter()
            .find(|e| e.uri == "http://example.org/Paper")
            .unwrap();
        assert_eq!(paper.label, "Paper");
    }

    #[test]
    fn skos_pref_label_is_accepted_as_primary_label() {
        // SKOS vocabularies often omit rdfs:label and only publish skos:prefLabel.
        // Without this, extract falls back to a local-name and alignment recall drops.
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
            ex:Concept a owl:Class ; skos:prefLabel "Preferred Concept" .
        "#;
        let entities = OntologyParser::parse_bytes(ttl, RdfFormat::Turtle).unwrap();
        let c = entities
            .iter()
            .find(|e| e.uri == "http://example.org/Concept")
            .unwrap();
        assert_eq!(c.label, "Preferred Concept");
    }

    #[test]
    fn full_variant_includes_annotation_properties_and_individuals() {
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            ex:C a owl:Class ; rdfs:label "C" .
            ex:OP a owl:ObjectProperty ; rdfs:label "OP" .
            ex:AP a owl:AnnotationProperty ; rdfs:label "AP" .
            ex:alice a owl:NamedIndividual ; rdfs:label "alice" .
        "#;
        let m3 = OntologyParser::parse_bytes_with_eval(
            ttl,
            RdfFormat::Turtle,
            ConferenceEvalVariant::M3,
        )
        .unwrap();
        assert_eq!(m3.len(), 2, "M3 must stay OAEI-compatible (classes+props)");

        let full = OntologyParser::parse_bytes_with_eval(
            ttl,
            RdfFormat::Turtle,
            ConferenceEvalVariant::Full,
        )
        .unwrap();
        assert_eq!(full.len(), 4);
        let by = |lbl: &str| full.iter().find(|e| e.label == lbl).unwrap();
        assert_eq!(by("AP").entity_type, EntityType::AnnotationProperty);
        assert_eq!(by("alice").entity_type, EntityType::Individual);
        assert_eq!(by("alice").kind.as_deref(), Some("Individual"));
    }

    #[test]
    fn chunk_by_subject_preserves_named_graph_provenance() {
        // TriG named graphs are the provenance boundary for multi-tenant /
        // multi-source ontology merges. Collapsing them into one subject chunk
        // silently erodes which graph asserted the triples.
        let trig = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

            ex:g1 {
                ex:A a owl:Class ; rdfs:label "A" ; rdfs:comment "from g1" .
            }
            ex:g2 {
                ex:A a owl:Class ; rdfs:label "A" ; rdfs:comment "from g2" .
            }
        "#;
        let chunks = OntologyParser::chunk_by_subject(trig, RdfFormat::TriG).unwrap();
        assert_eq!(
            chunks.len(),
            2,
            "same subject in two named graphs must yield two chunks: {chunks:?}"
        );
        let graphs: HashSet<_> = chunks.iter().filter_map(|c| c.graph.as_deref()).collect();
        assert!(graphs.contains("http://example.org/g1"));
        assert!(graphs.contains("http://example.org/g2"));
        for c in &chunks {
            assert_eq!(c.uri, "http://example.org/A");
            assert!(
                c.text.contains("from g1") || c.text.contains("from g2"),
                "graph-local comment must survive: {}",
                c.text
            );
        }
    }

    #[test]
    fn parse_bytes_with_structure_matches_file_path() {
        // In-memory path must not diverge from the file path — align job queue
        // holds bytes and must not re-serialize to a tempfile just for structure.
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            ex:A a owl:Class ; rdfs:label "A" .
            ex:B a owl:Class ; rdfs:label "B" ; rdfs:subClassOf ex:A .
        "#;
        let (ents_bytes, triples_bytes) =
            OntologyParser::parse_bytes_with_structure(ttl, RdfFormat::Turtle).unwrap();
        let f = write_tmp(".ttl", ttl);
        let (ents_file, triples_file) =
            OntologyParser::parse_file_with_structure(f.path()).unwrap();
        assert_eq!(ents_bytes.len(), ents_file.len());
        assert_eq!(triples_bytes.len(), triples_file.len());
        assert!(triples_bytes
            .iter()
            .any(|(s, o, p)| s == "http://example.org/B"
                && o == "http://example.org/A"
                && p == "subClassOf"));
    }

    #[test]
    fn same_as_links_are_preserved_not_collapsed() {
        // Align needs the explicit owl:sameAs link set. Collapsing A and B into
        // one entity would erase the asserted equivalence edge that repair /
        // merge / gold matching consume. Crepe owns transitive closure — we
        // only surface the asserted link.
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            ex:A a owl:Class ; rdfs:label "A" ; owl:sameAs ex:B .
            ex:B a owl:Class ; rdfs:label "B" .
        "#;
        let entities = OntologyParser::parse_bytes(ttl, RdfFormat::Turtle).unwrap();
        assert_eq!(
            entities.len(),
            2,
            "sameAs must not collapse entities: {entities:?}"
        );
        let a = entities.iter().find(|e| e.label == "A").unwrap();
        let b = entities.iter().find(|e| e.label == "B").unwrap();
        assert!(
            a.same_as.contains(&"http://example.org/B".to_string()),
            "A.same_as = {:?}",
            a.same_as
        );
        assert!(
            b.same_as.contains(&"http://example.org/A".to_string()),
            "sameAs must be recorded symmetrically: B.same_as = {:?}",
            b.same_as
        );

        let (_, triples) =
            OntologyParser::parse_bytes_with_structure(ttl, RdfFormat::Turtle).unwrap();
        assert!(
            triples.iter().any(|(_, _, p)| p == "sameAs"),
            "structural triples must expose sameAs for align: {triples:?}"
        );
    }

    #[test]
    fn equivalent_property_links_are_preserved_not_collapsed() {
        // Parallel to sameAs: equivalentProperty is an asserted link for
        // property alignment, not a cue to merge property records.
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            ex:writes a owl:ObjectProperty ; rdfs:label "writes" ;
                owl:equivalentProperty ex:authors .
            ex:authors a owl:ObjectProperty ; rdfs:label "authors" .
        "#;
        let entities = OntologyParser::parse_bytes_with_eval(
            ttl,
            RdfFormat::Turtle,
            ConferenceEvalVariant::M2,
        )
        .unwrap();
        assert_eq!(entities.len(), 2);
        let writes = entities.iter().find(|e| e.label == "writes").unwrap();
        let authors = entities.iter().find(|e| e.label == "authors").unwrap();
        assert!(writes
            .equivalent_properties
            .contains(&"http://example.org/authors".to_string()));
        assert!(authors
            .equivalent_properties
            .contains(&"http://example.org/writes".to_string()));
    }

    #[test]
    fn base_iri_resolves_relative_iris() {
        // Relative IRIs are common in Turtle snippets and RDF/XML without an
        // xml:base. Without an exposed base, parse fails or yields unusable
        // relative subjects — align upload paths need this knob.
        let ttl = br#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            <#Paper> a owl:Class ; rdfs:label "Paper" .
        "#;
        let opts = ParseOptions {
            base_iri: Some("http://example.org/onto".to_string()),
        };
        let entities = OntologyParser::parse_bytes_with_options(
            ttl,
            RdfFormat::Turtle,
            ConferenceEvalVariant::M1,
            &opts,
        )
        .unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].uri, "http://example.org/onto#Paper");
        assert_eq!(entities[0].label, "Paper");
    }

    #[test]
    fn quads_from_slice_streams_without_collect_api() {
        // Streaming surface for callers that process quads incrementally.
        // Entity extraction still materializes (label cross-refs), but the
        // quad iterator itself must not force a Vec up front.
        let ttl = br#"
            @prefix ex: <http://example.org/> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            ex:A a owl:Class .
            ex:B a owl:Class .
        "#;
        let mut count = 0usize;
        for q in OntologyParser::quads_from_slice(ttl, RdfFormat::Turtle, &ParseOptions::default())
            .unwrap()
        {
            q.unwrap();
            count += 1;
        }
        assert!(count >= 2, "expected at least two type quads, got {count}");
    }
}
