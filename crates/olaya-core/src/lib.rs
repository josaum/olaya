//! O-Laya Core: Arrow-native, zero-copy ontology decision engine in Rust.

pub mod arrow_schema;
pub mod axioms;
pub mod engine;
pub mod router;

#[cfg(feature = "onnx")]
pub mod neural;

pub use arrow_schema::{
    decision_output_schema, extract_curie, ontology_schema, read_ontology_ipc, write_ontology_ipc,
    OntologyBatchBuilder,
};
pub use axioms::{
    calibrate_and_project_inplace, enforce_disjointness_inplace,
    enforce_subsumption_monotonicity_inplace, normalized_entropy_confidence, verify_axioms,
    DISJOINT_THRESHOLD,
};
pub use engine::{DecisionResult, OlayaEngine};
pub use router::{
    tokenize_text, ActiveDecisionSubgraph, ArrowCandidateRouter, RowTokens, MAX_QUERY_POSTINGS,
};

#[cfg(feature = "onnx")]
pub use neural::NeuralScorer;
