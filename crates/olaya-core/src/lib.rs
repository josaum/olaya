//! O-Laya Core: Arrow-native, zero-copy ontology decision engine in Rust.

pub mod arrow_schema;
pub mod axioms;
pub mod router;
pub mod engine;

#[cfg(feature = "onnx")]
pub mod neural;

pub use arrow_schema::{
    ontology_schema, decision_output_schema, write_ontology_ipc, read_ontology_ipc,
    OntologyBatchBuilder, extract_curie,
};
pub use axioms::{
    calibrate_and_project_inplace, enforce_subsumption_monotonicity_inplace,
    enforce_disjointness_inplace, normalized_entropy_confidence,
};
pub use router::{ArrowCandidateRouter, ActiveDecisionSubgraph, tokenize_text};
pub use engine::{OlayaEngine, DecisionResult};

#[cfg(feature = "onnx")]
pub use neural::NeuralScorer;

