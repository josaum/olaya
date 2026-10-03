//! High-performance O-Laya engine producing zero-copy Arrow decision outputs.

use anyhow::{Context, Result};
use arrow::array::{
    Array, BooleanBuilder, Float32Builder, StringArray, StringBuilder, UInt32Array, UInt32Builder,
};
use arrow::record_batch::RecordBatch;
use std::collections::HashSet;
use std::sync::Arc;

use crate::arrow_schema::decision_output_schema;
use crate::axioms::{calibrate_and_project_inplace, verify_axioms, DISJOINT_THRESHOLD};
use crate::router::{tokenize_text, ActiveDecisionSubgraph, ArrowCandidateRouter};

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
    router: ArrowCandidateRouter,
    concept_ids: UInt32Array,
    curies: StringArray,
    labels: StringArray,
    #[cfg_attr(not(feature = "onnx"), allow(dead_code))]
    descriptions: Option<StringArray>,
    distilled_neural_scorer: Option<Arc<crate::distilled_neural::DistilledNeuralScorer>>,
    /// Ontology row -> concept index inside the distilled scorer (validated when attached).
    distilled_rows: Vec<usize>,
    #[cfg(feature = "onnx")]
    neural_scorer: Option<Arc<crate::neural::NeuralScorer>>,
}

fn required_column<T: Array + Clone + 'static>(batch: &RecordBatch, name: &str) -> Result<T> {
    batch
        .column_by_name(name)
        .with_context(|| format!("Missing '{name}' column"))?
        .as_any()
        .downcast_ref::<T>()
        .cloned()
        .with_context(|| format!("Invalid '{name}' column type"))
}

impl OlayaEngine {
    /// Initializes engine from an Arrow ontology RecordBatch.
    ///
    /// Validates the schema and caches typed column handles, so per-decision work never
    /// re-resolves columns.
    pub fn new(ontology_batch: RecordBatch) -> Result<Self> {
        let router = ArrowCandidateRouter::build(&ontology_batch)
            .context("Failed to build router from ontology RecordBatch")?;
        let descriptions = match ontology_batch.column_by_name("description") {
            Some(col) => Some(
                col.as_any()
                    .downcast_ref::<StringArray>()
                    .cloned()
                    .context("Invalid 'description' column type")?,
            ),
            None => None,
        };
        Ok(Self {
            router,
            concept_ids: required_column(&ontology_batch, "concept_id")?,
            curies: required_column(&ontology_batch, "curie")?,
            labels: required_column(&ontology_batch, "label")?,
            descriptions,
            distilled_neural_scorer: None,
            distilled_rows: Vec::new(),
            #[cfg(feature = "onnx")]
            neural_scorer: None,
        })
    }

    /// Attach a distilled native neural head scorer (no external dependencies).
    ///
    /// Every CURIE in the ontology must be known to the scorer; otherwise the scorer was trained on
    /// a different ontology and its concept embeddings would be read at the wrong index, so this is
    /// an error rather than a silent fallback.
    pub fn with_distilled_neural_scorer(
        mut self,
        scorer: Arc<crate::distilled_neural::DistilledNeuralScorer>,
    ) -> Result<Self> {
        let mut rows = Vec::with_capacity(self.curies.len());
        let mut unknown = Vec::new();
        for row in 0..self.curies.len() {
            let curie = self.curies.value(row);
            match scorer.curie_to_idx.get(curie) {
                Some(&idx) => rows.push(idx),
                None => unknown.push(curie.to_string()),
            }
        }
        if !unknown.is_empty() {
            let shown: Vec<&str> = unknown.iter().take(5).map(String::as_str).collect();
            anyhow::bail!(
                "distilled scorer does not know {} of {} ontology CURIEs (e.g. {}); it was trained \
                 on a different ontology",
                unknown.len(),
                self.curies.len(),
                shown.join(", ")
            );
        }
        self.distilled_rows = rows;
        self.distilled_neural_scorer = Some(scorer);
        Ok(self)
    }

    #[cfg(feature = "onnx")]
    /// Attach an ONNX neural cross-encoder scorer.
    pub fn with_neural_scorer(mut self, scorer: Arc<crate::neural::NeuralScorer>) -> Self {
        self.neural_scorer = Some(scorer);
        self
    }

    /// Lexical logits: label, synonym and description token overlap with the state.
    fn lexical_logits(&self, state_text: &str, subgraph: &ActiveDecisionSubgraph) -> Vec<f32> {
        let query_tokens: HashSet<String> = tokenize_text(state_text).into_iter().collect();
        let overlap =
            |tokens: &[String]| tokens.iter().filter(|t| query_tokens.contains(*t)).count() as f32;

        subgraph
            .row_indices
            .iter()
            .map(|&row| {
                let toks = self.router.row_tokens(row);
                1.0 + 3.0 * overlap(&toks.label)
                    + 2.5 * overlap(&toks.synonyms)
                    + 1.5 * overlap(&toks.description)
            })
            .collect()
    }

    #[cfg(feature = "onnx")]
    /// Candidate option text in the same format the Python sequence builder uses:
    /// `CURIE: label (subClassOf: parent labels) - description`.
    fn option_text(&self, row: usize, subgraph_rows: &HashSet<usize>) -> String {
        let mut text = format!("{}: {}", self.curies.value(row), self.labels.value(row));
        let parent_labels: Vec<&str> = self
            .router
            .parent_rows(row)
            .iter()
            .filter(|p| subgraph_rows.contains(p))
            .map(|&p| self.labels.value(p))
            .collect();
        if !parent_labels.is_empty() {
            text.push_str(&format!(" (subClassOf: {})", parent_labels.join(", ")));
        }
        if let Some(desc) = &self.descriptions {
            if !desc.is_null(row) && !desc.value(row).is_empty() {
                text.push_str(&format!(" - {}", desc.value(row)));
            }
        }
        text
    }

    fn candidate_logits(
        &self,
        state_text: &str,
        subgraph: &ActiveDecisionSubgraph,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "onnx")]
        if let Some(scorer) = &self.neural_scorer {
            let rows: HashSet<usize> = subgraph.row_indices.iter().copied().collect();
            let candidate_texts: Vec<String> = subgraph
                .row_indices
                .iter()
                .map(|&row| self.option_text(row, &rows))
                .collect();
            let mut logits = scorer.score_candidates(state_text, &candidate_texts)?;
            logits.resize(subgraph.row_indices.len(), f32::NEG_INFINITY);
            return Ok(logits);
        }

        let mut lexical = self.lexical_logits(state_text, subgraph);

        if let Some(distilled) = &self.distilled_neural_scorer {
            let tokens = tokenize_text(state_text);
            let tok_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
            // Smoothed IDF over the ontology batch, as at training time.
            let idfs: Vec<f32> = tokens.iter().map(|t| self.router.idf(t)).collect();
            let q_vec = distilled.encode_query(&tok_refs, &idfs);

            // Every row was mapped to a scorer index when the scorer was attached.
            let to_idx = |row: usize| self.distilled_rows[row];
            let candidate_indices: Vec<usize> = subgraph
                .row_indices
                .iter()
                .map(|&row| to_idx(row))
                .collect();

            let parents: Vec<Vec<usize>> = subgraph
                .row_indices
                .iter()
                .map(|&row| {
                    self.router
                        .parent_rows(row)
                        .iter()
                        .map(|&p| to_idx(p))
                        .collect()
                })
                .collect();

            let disjoint: Vec<Vec<usize>> = subgraph
                .row_indices
                .iter()
                .map(|&row| {
                    self.router
                        .disjoint_rows(row)
                        .iter()
                        .map(|&d| to_idx(d))
                        .collect()
                })
                .collect();

            let neural_scores = distilled.score_candidates_with_morphisms(
                &q_vec,
                &candidate_indices,
                &parents,
                &disjoint,
            );

            for (i, &neu) in neural_scores.iter().enumerate() {
                lexical[i] += 2.5 * neu;
            }
        }

        Ok(lexical)
    }

    /// Evaluates a decision query against an input state text.
    pub fn system_one(&self, state_text: &str, max_candidates: usize) -> Result<DecisionResult> {
        let subgraph = self.router.extract_subgraph(state_text, max_candidates);
        if subgraph.row_indices.is_empty() {
            anyhow::bail!("Active subgraph is empty");
        }

        let mut logits = self.candidate_logits(state_text, &subgraph)?;

        // Apply in-place temperature, subsumption monotonicity, and entropy calibration
        let confidence = calibrate_and_project_inplace(
            &mut logits,
            &subgraph.parent_edges,
            &subgraph.disjoint_edges,
        );

        // Argmax; ties (e.g. a parent lifted to its child's probability) go to the deepest
        // concept, then to the lowest row index. Independent of row order.
        let max_prob = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let best_slot = (0..logits.len())
            .filter(|&slot| logits[slot] >= max_prob - 1e-5)
            .max_by(|&a, &b| {
                let (ra, rb) = (subgraph.row_indices[a], subgraph.row_indices[b]);
                self.router
                    .depth(ra)
                    .cmp(&self.router.depth(rb))
                    .then(rb.cmp(&ra))
            })
            .context("No candidate slots")?;

        let best_row = subgraph.row_indices[best_slot];
        let axioms_ok = verify_axioms(
            &logits,
            &subgraph.parent_edges,
            &subgraph.disjoint_edges,
            DISJOINT_THRESHOLD,
        );
        let probabilities = subgraph
            .row_indices
            .iter()
            .enumerate()
            .map(|(slot, &r)| (self.curies.value(r).to_string(), logits[slot]))
            .collect();

        Ok(DecisionResult {
            selected_concept_id: self.concept_ids.value(best_row),
            selected_curie: self.curies.value(best_row).to_string(),
            selected_label: self.labels.value(best_row).to_string(),
            confidence,
            probabilities,
            axioms_verified: axioms_ok,
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
        )
        .context("Failed to construct decision output RecordBatch")
    }
}
