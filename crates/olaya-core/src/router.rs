//! Fast candidate routing and active decision subgraph extraction over Arrow RecordBatches.
//!
//! All per-concept structure (id → row map, parent / disjoint adjacency, depth, token sets) is
//! precomputed once in [`ArrowCandidateRouter::build`], so a query only touches the rows that
//! share a token with the state plus their ancestors — independent of total ontology size.

use anyhow::{Context, Result};
use arrow::array::{Array, ListArray, StringArray, UInt32Array};
use arrow::record_batch::RecordBatch;
use std::collections::{HashMap, HashSet};

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

/// Pre-tokenized text fields of one ontology row.
#[derive(Debug, Clone, Default)]
pub struct RowTokens {
    pub label: Vec<String>,
    pub synonyms: Vec<String>,
    pub description: Vec<String>,
}

/// Tokenizer helper: splits on non-alphanumerics and CamelCase boundaries, lowercases, and drops
/// single-character tokens.
///
/// Case boundaries follow `tokenize_normalize` in the Python package: a split happens between a
/// lowercase and an uppercase letter (`DatabaseOutage` → `database`, `outage`) and before the last
/// letter of an uppercase run that is followed by lowercase (`HTTPServer` → `http`, `server`).
/// Acronyms stay whole (`API` → `api`). When a word is split, the whole lowercased word is emitted
/// as well (`PostgreSQL` → `postgre`, `sql`, `postgresql`) so it still matches `POSTGRESQL`.
pub fn tokenize_text(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for word in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        let chars: Vec<char> = word.chars().collect();
        let mut start = 0;
        let mut parts = 0;
        for i in 1..chars.len() {
            let (prev, cur) = (chars[i - 1], chars[i]);
            let next_is_lower = chars.get(i + 1).is_some_and(|c| c.is_lowercase());
            if cur.is_uppercase() && (prev.is_lowercase() || (prev.is_uppercase() && next_is_lower))
            {
                push_token(&chars[start..i], &mut tokens);
                start = i;
                parts += 1;
            }
        }
        push_token(&chars[start..], &mut tokens);
        if parts > 0 {
            push_token(&chars, &mut tokens);
        }
    }
    tokens
}

fn push_token(chars: &[char], tokens: &mut Vec<String>) {
    if chars.len() > 1 {
        tokens.push(chars.iter().collect::<String>().to_lowercase());
    }
}

fn dedup_sorted(mut tokens: Vec<String>) -> Vec<String> {
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// Tokens indexed under more rows than this are treated as stopwords at query time. Their IDF
/// weight (`2 / (1 + postings)`) is negligible, yet scanning them would make a query O(N).
pub const MAX_QUERY_POSTINGS: usize = 1024;

/// Fast candidate router indexing Arrow RecordBatches.
pub struct ArrowCandidateRouter {
    /// Token -> row indices in the ontology RecordBatch
    inverted_index: HashMap<String, Vec<usize>>,
    /// Direct parent rows of each row
    parent_rows: Vec<Vec<usize>>,
    /// Direct child rows of each row
    child_rows: Vec<Vec<usize>>,
    /// Disjoint rows of each row
    disjoint_rows: Vec<Vec<usize>>,
    /// Longest subClassOf path from a top-level concept (0 = top level)
    depth: Vec<u32>,
    /// Tokenized label / synonyms / description of each row
    row_tokens: Vec<RowTokens>,
    /// Rows without parents, ascending
    top_level_rows: Vec<usize>,
}

fn id_list_to_rows(
    list: Option<&ListArray>,
    row: usize,
    id_to_row: &HashMap<u32, usize>,
) -> Vec<usize> {
    let Some(list) = list else { return Vec::new() };
    if list.is_null(row) {
        return Vec::new();
    }
    let values = list.value(row);
    let Some(ids) = values.as_any().downcast_ref::<UInt32Array>() else {
        return Vec::new();
    };
    let mut rows: Vec<usize> = (0..ids.len())
        .filter(|&i| !ids.is_null(i))
        .filter_map(|i| id_to_row.get(&ids.value(i)).copied())
        .filter(|&r| r != row)
        .collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

impl ArrowCandidateRouter {
    /// Builds the router index directly from an Arrow RecordBatch.
    pub fn build(batch: &RecordBatch) -> Result<Self> {
        let num_concepts = batch.num_rows();

        let concept_ids = batch
            .column_by_name("concept_id")
            .context("Missing 'concept_id' column")?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("Invalid concept_id column type")?;

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

        let parents_col = batch
            .column_by_name("parents")
            .and_then(|c| c.as_any().downcast_ref::<ListArray>());

        let disjoint_col = batch
            .column_by_name("disjoint_with")
            .and_then(|c| c.as_any().downcast_ref::<ListArray>());

        let mut id_to_row: HashMap<u32, usize> = HashMap::with_capacity(num_concepts);
        for r in 0..num_concepts {
            id_to_row.insert(concept_ids.value(r), r);
        }

        let mut inverted_index: HashMap<String, Vec<usize>> = HashMap::new();
        let mut row_tokens = Vec::with_capacity(num_concepts);
        let mut parent_rows = Vec::with_capacity(num_concepts);
        let mut disjoint_rows = Vec::with_capacity(num_concepts);

        for row in 0..num_concepts {
            let label = if labels.is_null(row) {
                Vec::new()
            } else {
                dedup_sorted(tokenize_text(labels.value(row)))
            };

            let description = match descriptions {
                Some(desc) if !desc.is_null(row) => dedup_sorted(tokenize_text(desc.value(row))),
                _ => Vec::new(),
            };

            let mut syn_tokens = Vec::new();
            if let Some(syn_list) = synonyms {
                if !syn_list.is_null(row) {
                    let syn_values = syn_list.value(row);
                    if let Some(str_arr) = syn_values.as_any().downcast_ref::<StringArray>() {
                        for i in (0..str_arr.len()).filter(|&i| !str_arr.is_null(i)) {
                            syn_tokens.extend(tokenize_text(str_arr.value(i)));
                        }
                    }
                }
            }
            let syn_tokens = dedup_sorted(syn_tokens);

            let all: HashSet<&String> = label
                .iter()
                .chain(&syn_tokens)
                .chain(&description)
                .collect();
            for tok in all {
                inverted_index.entry(tok.clone()).or_default().push(row);
            }

            row_tokens.push(RowTokens {
                label,
                synonyms: syn_tokens,
                description,
            });
            parent_rows.push(id_list_to_rows(parents_col, row, &id_to_row));
            disjoint_rows.push(id_list_to_rows(disjoint_col, row, &id_to_row));
        }

        // Disjointness is symmetric even if only asserted on one side.
        for row in 0..num_concepts {
            for i in 0..disjoint_rows[row].len() {
                let other = disjoint_rows[row][i];
                if !disjoint_rows[other].contains(&row) {
                    disjoint_rows[other].push(row);
                }
            }
        }

        let mut child_rows = vec![Vec::new(); num_concepts];
        for (row, parents) in parent_rows.iter().enumerate() {
            for &p in parents {
                child_rows[p].push(row);
            }
        }

        let top_level_rows = (0..num_concepts)
            .filter(|&r| parent_rows[r].is_empty())
            .collect();
        let depth = compute_depths(&parent_rows);

        Ok(Self {
            inverted_index,
            parent_rows,
            child_rows,
            disjoint_rows,
            depth,
            row_tokens,
            top_level_rows,
        })
    }

    /// Number of indexed concepts.
    pub fn num_concepts(&self) -> usize {
        self.parent_rows.len()
    }

    /// Longest subClassOf path from a top-level concept to `row` (0 = top level).
    pub fn depth(&self, row: usize) -> u32 {
        self.depth[row]
    }

    /// Direct parent rows of `row`.
    pub fn parent_rows(&self, row: usize) -> &[usize] {
        &self.parent_rows[row]
    }

    /// Mutually disjoint rows of `row`.
    pub fn disjoint_rows(&self, row: usize) -> &[usize] {
        &self.disjoint_rows[row]
    }

    /// Pre-tokenized text fields of `row`.
    pub fn row_tokens(&self, row: usize) -> &RowTokens {
        &self.row_tokens[row]
    }

    /// Ranks rows by lexical overlap with the state, highest first (ties by row index).
    fn rank_rows(&self, state_text: &str, max_candidates: usize) -> Vec<usize> {
        let query_tokens = dedup_sorted(tokenize_text(state_text));
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for tok in &query_tokens {
            if let Some(rows) = self
                .inverted_index
                .get(tok)
                .filter(|r| r.len() <= MAX_QUERY_POSTINGS)
            {
                let idf = 1.0 / (1.0 + rows.len() as f32);
                for &row in rows {
                    *scores.entry(row).or_default() += idf * 2.0;
                }
            }
        }

        let mut ranked: Vec<(usize, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked
            .into_iter()
            .take(max_candidates)
            .map(|(row, _)| row)
            .collect()
    }

    /// Fallback for states sharing no token with the ontology: the children of the single
    /// top-level concept, or the top-level concepts themselves.
    fn fallback_rows(&self, max_candidates: usize) -> Vec<usize> {
        let rows = match self.top_level_rows.as_slice() {
            [root] if !self.child_rows[*root].is_empty() => self.child_rows[*root].clone(),
            top => top.to_vec(),
        };
        rows.into_iter().take(max_candidates).collect()
    }

    /// Extracts the active decision subgraph for incoming state text: the top `max_candidates`
    /// lexical matches plus every ancestor up to the top level.
    pub fn extract_subgraph(
        &self,
        state_text: &str,
        max_candidates: usize,
    ) -> ActiveDecisionSubgraph {
        let mut seeds = self.rank_rows(state_text, max_candidates);
        if seeds.is_empty() {
            seeds = self.fallback_rows(max_candidates);
        }

        // Expand to the full ancestor chain.
        let mut selected: HashSet<usize> = seeds.iter().copied().collect();
        let mut stack = seeds;
        while let Some(row) = stack.pop() {
            for &p in &self.parent_rows[row] {
                if selected.insert(p) {
                    stack.push(p);
                }
            }
        }

        let mut row_indices: Vec<usize> = selected.into_iter().collect();
        row_indices.sort_unstable();
        let row_to_slot: HashMap<usize, usize> = row_indices
            .iter()
            .enumerate()
            .map(|(slot, &row)| (row, slot))
            .collect();

        let mut parent_edges = Vec::new();
        let mut disjoint_edges = Vec::new();
        for (slot, &row) in row_indices.iter().enumerate() {
            for p in &self.parent_rows[row] {
                if let Some(&parent_slot) = row_to_slot.get(p) {
                    parent_edges.push((slot, parent_slot));
                }
            }
            for d in &self.disjoint_rows[row] {
                if let Some(&other) = row_to_slot.get(d) {
                    if slot < other {
                        disjoint_edges.push((slot, other));
                    }
                }
            }
        }

        ActiveDecisionSubgraph {
            row_indices,
            parent_edges,
            disjoint_edges,
        }
    }
}

/// Longest path from a top-level concept, computed iteratively; rows on a cycle get the depth
/// reached before the cycle is detected.
fn compute_depths(parent_rows: &[Vec<usize>]) -> Vec<u32> {
    const UNVISITED: u8 = 0;
    const IN_PROGRESS: u8 = 1;
    const DONE: u8 = 2;

    let n = parent_rows.len();
    let mut depth = vec![0u32; n];
    let mut state = vec![UNVISITED; n];

    for start in 0..n {
        if state[start] != UNVISITED {
            continue;
        }
        let mut stack = vec![(start, 0usize)];
        state[start] = IN_PROGRESS;
        while let Some(top) = stack.last_mut() {
            let row = top.0;
            if let Some(&p) = parent_rows[row].get(top.1) {
                top.1 += 1;
                if state[p] == UNVISITED {
                    state[p] = IN_PROGRESS;
                    stack.push((p, 0));
                }
            } else {
                depth[row] = parent_rows[row]
                    .iter()
                    .filter(|&&p| state[p] == DONE)
                    .map(|&p| depth[p] + 1)
                    .max()
                    .unwrap_or(0);
                state[row] = DONE;
                stack.pop();
            }
        }
    }
    depth
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_keeps_acronyms_whole() {
        assert_eq!(
            tokenize_text("PostgreSQL HTTP API key leaked"),
            [
                "postgre",
                "sql",
                "postgresql",
                "http",
                "api",
                "key",
                "leaked"
            ]
        );
        assert_eq!(
            tokenize_text("HTTPServer DatabaseOutage"),
            [
                "http",
                "server",
                "httpserver",
                "database",
                "outage",
                "databaseoutage"
            ]
        );
        assert_eq!(
            tokenize_text("db_lock-wait 500 a"),
            ["db", "lock", "wait", "500"]
        );
        assert_eq!(tokenize_text("deadlock"), tokenize_text("DEADLOCK"));
    }

    #[test]
    fn depths_follow_longest_path_and_survive_cycles() {
        // 0 <- 1 <- 2, and 0 <- 2 directly: depth(2) = 2.
        let parents = vec![vec![], vec![0], vec![0, 1]];
        assert_eq!(compute_depths(&parents), vec![0, 1, 2]);
        // Cycle 0 <-> 1 must terminate.
        let cyclic = vec![vec![1], vec![0]];
        assert_eq!(compute_depths(&cyclic).len(), 2);
    }
}
