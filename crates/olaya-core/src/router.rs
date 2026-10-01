//! Fast candidate routing and active decision subgraph extraction over Arrow RecordBatches.

use std::collections::{HashMap, HashSet};
use anyhow::{Context, Result};
use arrow::array::{Array, ListArray, StringArray, UInt32Array};
use arrow::record_batch::RecordBatch;

/// Active decision subgraph mapped to contiguous slot indices [0 .. M-1].
#[derive(Debug, Clone)]
pub struct ActiveDecisionSubgraph {
    /// Row indices in the ontology RecordBatch
    pub row_indices: Vec<usize>,
    /// Subsumption edges between slots: (child_slot, parent_slot)
    pub parent_edges: Vec<(usize, usize)>,
    /// Disjointness edges between slots: (slot_a, slot_b)
    pub disjoint_edges: Vec<(usize, usize)>,
}

/// Tokenizer helper for CamelCase splitting and normalization.
pub fn tokenize_text(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() {
            if ch.is_uppercase() && !current.is_empty() {
                tokens.push(current.to_lowercase());
                current = String::new();
            }
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(current.to_lowercase());
            current = String::new();
        }
    }
    if !current.is_empty() {
        tokens.push(current.to_lowercase());
    }
    tokens
}

/// Fast candidate router indexing Arrow RecordBatches.
pub struct ArrowCandidateRouter {
    /// Token -> Set of row indices in the ontology RecordBatch
    inverted_index: HashMap<String, Vec<usize>>,
    /// Concept count
    num_concepts: usize,
}

impl ArrowCandidateRouter {
    /// Builds the router index directly from an Arrow RecordBatch.
    pub fn build(batch: &RecordBatch) -> Result<Self> {
        let mut inverted_index: HashMap<String, Vec<usize>> = HashMap::new();
        let num_concepts = batch.num_rows();

        let labels = batch
            .column_by_name("label")
            .context("Missing 'label' column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Invalid label column type")?;

        let descriptions = batch
            .column_by_name("description")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());

        let synonyms = batch
            .column_by_name("synonyms")
            .and_then(|c| c.as_any().downcast_ref::<ListArray>());

        for row in 0..num_concepts {
            let mut row_tokens = HashSet::new();

            // Index label
            if !labels.is_null(row) {
                for tok in tokenize_text(labels.value(row)) {
                    row_tokens.insert(tok);
                }
            }

            // Index description
            if let Some(desc_arr) = descriptions {
                if !desc_arr.is_null(row) {
                    for tok in tokenize_text(desc_arr.value(row)) {
                        row_tokens.insert(tok);
                    }
                }
            }

            // Index synonyms
            if let Some(syn_list) = synonyms {
                if !syn_list.is_null(row) {
                    let syn_values = syn_list.value(row);
                    if let Some(str_arr) = syn_values.as_any().downcast_ref::<StringArray>() {
                        for i in 0..str_arr.len() {
                            if !str_arr.is_null(i) {
                                for tok in tokenize_text(str_arr.value(i)) {
                                    row_tokens.insert(tok);
                                }
                            }
                        }
                    }
                }
            }

            for tok in row_tokens {
                inverted_index.entry(tok).or_default().push(row);
            }
        }

        Ok(Self {
            inverted_index,
            num_concepts,
        })
    }

    /// Extracts active decision subgraph for incoming state text.
    pub fn extract_subgraph(
        &self,
        batch: &RecordBatch,
        state_text: &str,
        max_candidates: usize,
    ) -> Result<ActiveDecisionSubgraph> {
        let query_tokens = tokenize_text(state_text);
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for tok in &query_tokens {
            if let Some(rows) = self.inverted_index.get(tok) {
                let idf = 1.0 / (1.0 + rows.len() as f32);
                for &row in rows {
                    *scores.entry(row).or_default() += idf * 2.0;
                }
            }
        }

        let mut ranked: Vec<(usize, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected_rows: HashSet<usize> = ranked
            .iter()
            .take(max_candidates)
            .map(|(row, _)| *row)
            .collect();

        // If no matches, take first few rows as fallback
        if selected_rows.is_empty() {
            for r in 0..self.num_concepts.min(max_candidates) {
                selected_rows.insert(r);
            }
        }

        // Expand to parents
        let parents_col = batch
            .column_by_name("parents")
            .and_then(|c| c.as_any().downcast_ref::<ListArray>());

        let concept_ids = batch
            .column_by_name("concept_id")
            .context("Missing concept_id column")?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("Invalid concept_id type")?;

        // Map concept_id to row index
        let mut id_to_row = HashMap::new();
        for r in 0..self.num_concepts {
            id_to_row.insert(concept_ids.value(r), r);
        }

        if let Some(parents_list) = parents_col {
            let mut parent_additions = Vec::new();
            for &row in &selected_rows {
                if !parents_list.is_null(row) {
                    let p_vals = parents_list.value(row);
                    if let Some(uint_arr) = p_vals.as_any().downcast_ref::<UInt32Array>() {
                        for i in 0..uint_arr.len() {
                            let p_id = uint_arr.value(i);
                            if let Some(&p_row) = id_to_row.get(&p_id) {
                                parent_additions.push(p_row);
                            }
                        }
                    }
                }
            }
            for p_row in parent_additions {
                selected_rows.insert(p_row);
            }
        }

        // Build sorted slot indices
        let mut row_indices: Vec<usize> = selected_rows.into_iter().collect();
        row_indices.sort_unstable();

        let row_to_slot: HashMap<usize, usize> = row_indices
            .iter()
            .enumerate()
            .map(|(slot, &row)| (row, slot))
            .collect();

        let mut parent_edges = Vec::new();
        let mut disjoint_edges = Vec::new();

        // Build parent edges
        if let Some(parents_list) = parents_col {
            for &row in &row_indices {
                let child_slot = row_to_slot[&row];
                if !parents_list.is_null(row) {
                    let p_vals = parents_list.value(row);
                    if let Some(uint_arr) = p_vals.as_any().downcast_ref::<UInt32Array>() {
                        for i in 0..uint_arr.len() {
                            let p_id = uint_arr.value(i);
                            if let Some(&p_row) = id_to_row.get(&p_id) {
                                if let Some(&parent_slot) = row_to_slot.get(&p_row) {
                                    parent_edges.push((child_slot, parent_slot));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Build disjoint edges
        let disjoint_col = batch
            .column_by_name("disjoint_with")
            .and_then(|c| c.as_any().downcast_ref::<ListArray>());

        if let Some(disjoint_list) = disjoint_col {
            for &row in &row_indices {
                let slot_a = row_to_slot[&row];
                if !disjoint_list.is_null(row) {
                    let d_vals = disjoint_list.value(row);
                    if let Some(uint_arr) = d_vals.as_any().downcast_ref::<UInt32Array>() {
                        for i in 0..uint_arr.len() {
                            let d_id = uint_arr.value(i);
                            if let Some(&d_row) = id_to_row.get(&d_id) {
                                if let Some(&slot_b) = row_to_slot.get(&d_row) {
                                    if slot_a < slot_b {
                                        disjoint_edges.push((slot_a, slot_b));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(ActiveDecisionSubgraph {
            row_indices,
            parent_edges,
            disjoint_edges,
        })
    }
}
