//! High-performance O-Laya engine producing zero-copy Arrow decision outputs.

use std::sync::Arc;
use anyhow::{Context, Result};
use arrow::array::{Array, BooleanBuilder, Float32Builder, StringArray, StringBuilder, UInt32Array, UInt32Builder};
use arrow::record_batch::RecordBatch;

use crate::arrow_schema::decision_output_schema;
use crate::axioms::calibrate_and_project_inplace;
use crate::router::ArrowCandidateRouter;

/// High-level result of a single decision evaluation.
#[derive(Debug, Clone)]
pub struct DecisionResult {
    pub selected_concept_id: u32,
    pub selected_curie: String,
    pub selected_label: String,
    pub confidence: f32,
    pub probabilities: Vec<(String, f32)>,
    pub axioms_verified: bool,
}

/// The Arrow-native O-Laya decision engine.
pub struct OlayaEngine {
    ontology_batch: RecordBatch,
    router: ArrowCandidateRouter,
    #[cfg(feature = "onnx")]
    neural_scorer: Option<Arc<crate::neural::NeuralScorer>>,
}

impl OlayaEngine {
    /// Initializes engine from an Arrow ontology RecordBatch.
    pub fn new(ontology_batch: RecordBatch) -> Result<Self> {
        let router = ArrowCandidateRouter::build(&ontology_batch)
            .context("Failed to build router from ontology RecordBatch")?;
        Ok(Self {
            ontology_batch,
            router,
            #[cfg(feature = "onnx")]
            neural_scorer: None,
        })
    }

    #[cfg(feature = "onnx")]
    /// Attach an ONNX neural cross-encoder scorer.
    pub fn with_neural_scorer(mut self, scorer: Arc<crate::neural::NeuralScorer>) -> Self {
        self.neural_scorer = Some(scorer);
        self
    }


    /// Evaluates a decision query against an input state text.
    pub fn system_one(
        &self,
        state_text: &str,
        max_candidates: usize,
    ) -> Result<DecisionResult> {
        let subgraph = self.router.extract_subgraph(&self.ontology_batch, state_text, max_candidates)?;
        let m = subgraph.row_indices.len();
        if m == 0 {
            anyhow::bail!("Active subgraph is empty");
        }

        let concept_ids = self.ontology_batch
            .column_by_name("concept_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();

        let curies = self.ontology_batch
            .column_by_name("curie")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let labels = self.ontology_batch
            .column_by_name("label")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let descriptions = self.ontology_batch
            .column_by_name("description")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());

        #[cfg(feature = "onnx")]
        let mut logits = if let Some(scorer) = &self.neural_scorer {
            let candidate_texts: Vec<String> = subgraph
                .row_indices
                .iter()
                .map(|&row| {
                    if let Some(desc_arr) = descriptions {
                        if !desc_arr.is_null(row) && !desc_arr.value(row).is_empty() {
                            return format!("{} - {}", labels.value(row), desc_arr.value(row));
                        }
                    }
                    labels.value(row).to_string()
                })
                .collect();
            scorer.score_candidates(state_text, &candidate_texts)?
        } else {
            let state_lower = state_text.to_lowercase();
            let query_tokens: std::collections::HashSet<String> =
                crate::router::tokenize_text(&state_lower).into_iter().collect();

            let mut l = vec![1.0f32; m];
            for (slot, &row) in subgraph.row_indices.iter().enumerate() {
                for tok in crate::router::tokenize_text(labels.value(row)) {
                    if query_tokens.contains(&tok) {
                        l[slot] += 3.0;
                    }
                }
                if let Some(desc_arr) = descriptions {
                    if !desc_arr.is_null(row) {
                        for tok in crate::router::tokenize_text(desc_arr.value(row)) {
                            if query_tokens.contains(&tok) {
                                l[slot] += 1.5;
                            }
                        }
                    }
                }
            }
            l
        };

        #[cfg(not(feature = "onnx"))]
        let mut logits = {
            let state_lower = state_text.to_lowercase();
            let query_tokens: std::collections::HashSet<String> =
                crate::router::tokenize_text(&state_lower).into_iter().collect();

            let mut l = vec![1.0f32; m];
            for (slot, &row) in subgraph.row_indices.iter().enumerate() {
                for tok in crate::router::tokenize_text(labels.value(row)) {
                    if query_tokens.contains(&tok) {
                        l[slot] += 3.0;
                    }
                }
                if let Some(desc_arr) = descriptions {
                    if !desc_arr.is_null(row) {
                        for tok in crate::router::tokenize_text(desc_arr.value(row)) {
                            if query_tokens.contains(&tok) {
                                l[slot] += 1.5;
                            }
                        }
                    }
                }
            }
            l
        };

        // Apply in-place temperature, subsumption monotonicity, and entropy calibration
        let confidence = calibrate_and_project_inplace(
            &mut logits,
            &subgraph.parent_edges,
            &subgraph.disjoint_edges,
        );

        // Find argmax slot, breaking ties in favor of more specific leaf descendants
        let mut best_slot = 0;
        let mut best_prob = -1.0f32;
        for (slot, &p) in logits.iter().enumerate() {
            if p > best_prob + 1e-5 {
                best_prob = p;
                best_slot = slot;
            } else if (p - best_prob).abs() <= 1e-5 {
                // If tied, check if current slot is a child of the current best_slot
                let is_child = subgraph.parent_edges.iter().any(|&(c, parent)| c == slot && parent == best_slot);
                if is_child {
                    best_slot = slot;
                }
            }
        }

        let best_row = subgraph.row_indices[best_slot];
        let selected_concept_id = concept_ids.value(best_row);
        let selected_curie = curies.value(best_row).to_string();
        let selected_label = labels.value(best_row).to_string();

        let probabilities = subgraph.row_indices.iter().enumerate().map(|(slot, &r)| {
            (curies.value(r).to_string(), logits[slot])
        }).collect();

        Ok(DecisionResult {
            selected_concept_id,
            selected_curie,
            selected_label,
            confidence,
            probabilities,
            axioms_verified: true,
        })
    }

    /// Evaluates a batch of states and outputs an Arrow RecordBatch directly.
    pub fn process_batch_to_arrow(
        &self,
        states: &[&str],
        question_id: &str,
    ) -> Result<RecordBatch> {
        let mut state_id_builder = UInt32Builder::with_capacity(states.len());
        let mut question_id_builder = StringBuilder::with_capacity(states.len(), states.len() * 16);
        let mut selected_id_builder = UInt32Builder::with_capacity(states.len());
        let mut curie_builder = StringBuilder::with_capacity(states.len(), states.len() * 32);
        let mut label_builder = StringBuilder::with_capacity(states.len(), states.len() * 32);
        let mut confidence_builder = Float32Builder::with_capacity(states.len());
        let mut axioms_builder = BooleanBuilder::with_capacity(states.len());

        for (i, &state) in states.iter().enumerate() {
            let res = self.system_one(state, 8)?;
            state_id_builder.append_value(i as u32);
            question_id_builder.append_value(question_id);
            selected_id_builder.append_value(res.selected_concept_id);
            curie_builder.append_value(&res.selected_curie);
            label_builder.append_value(&res.selected_label);
            confidence_builder.append_value(res.confidence);
            axioms_builder.append_value(res.axioms_verified);
        }

        let schema = decision_output_schema();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(state_id_builder.finish()),
                Arc::new(question_id_builder.finish()),
                Arc::new(selected_id_builder.finish()),
                Arc::new(curie_builder.finish()),
                Arc::new(label_builder.finish()),
                Arc::new(confidence_builder.finish()),
                Arc::new(axioms_builder.finish()),
            ],
        ).context("Failed to construct decision output RecordBatch")
    }
}
