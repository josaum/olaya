//! O-Laya WASM: IDF-weighted lexical decision engine for the browser.
//!
//! Exposes the same candidate routing + axiom projection logic as the Rust Arrow engine,
//! compiled to WebAssembly for sub-microsecond in-browser inference without any JS overhead.
//!
//! API surface matches olaya-engine.js exports for a drop-in neural upgrade path.

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ── Concept types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConcept {
    pub id: u32,
    pub curie: String,
    pub label: String,
    pub description: String,
    pub synonyms: Vec<String>,
    pub parents: Vec<u32>,
    pub disjoint_with: Vec<u32>,
}

// ── Axiom constants (must match spec/AXIOMS.md) ────────────────────────────────

const DISJOINT_THRESHOLD: f32 = 0.6;
const K1_LABEL: f32 = 1.5;      // BM25 saturation for label tokens
const K1_DESC: f32 = 2.0;       // BM25 saturation for description tokens
const LABEL_BOOST: f32 = 4.0;
const SYNONYM_BOOST: f32 = 2.0;
const DESC_BOOST: f32 = 1.0;

// ── Tokenizer ─────────────────────────────────────────────────────────────────

const STOPWORDS: &[&str] = &[
    "a","an","the","and","or","but","in","on","at","to","for","with","by","about","between",
    "into","through","before","after","above","below","from","up","down","out","off","over",
    "under","then","once","here","there","when","where","why","how","all","any","both","each",
    "few","more","most","other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","can","will","just","should","now","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","of","it","its","that","this","also","made",
];

fn tokenize(text: &str) -> Vec<String> {
    let stopset: HashSet<&str> = STOPWORDS.iter().copied().collect();
    // Split CamelCase: insert space before uppercase letters preceded by lowercase
    let mut spaced = String::with_capacity(text.len() + 16);
    let chars: Vec<char> = text.chars().collect();
    for i in 0..chars.len() {
        if i > 0 && chars[i].is_uppercase() && chars[i-1].is_lowercase() {
            spaced.push(' ');
        }
        spaced.push(chars[i]);
    }
    spaced.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2 && !stopset.contains(*t))
        .map(|t| t.to_string())
        .collect()
}

// ── Softmax + confidence ──────────────────────────────────────────────────────

fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let t = temperature.max(0.0001);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| ((l - max) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

fn temperature_by_cardinality(k: usize) -> f32 {
    match k {
        0..=2 => 0.85,
        3..=5 => 0.95,
        6..=10 => 1.10,
        _ => 1.25,
    }
}

fn normalized_entropy_confidence(probs: &[f32]) -> f32 {
    let n = probs.len();
    if n <= 1 { return 1.0; }
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 { return 0.0; }
    let mut h = 0.0f32;
    for &p in probs {
        let pn = p / sum;
        if pn > 1e-12 { h -= pn * pn.ln(); }
    }
    let max_h = (n as f32).ln();
    (1.0 - h / max_h).max(0.0).min(1.0)
}

// ── Axiom projection ──────────────────────────────────────────────────────────

fn enforce_subsumption_monotonicity(probs: &mut [f32], parent_edges: &[(usize, usize)]) {
    let k = probs.len();
    for _ in 0..k {
        let mut changed = false;
        for &(child, parent) in parent_edges {
            if child < k && parent < k && probs[child] > probs[parent] {
                probs[parent] = probs[child];
                changed = true;
            }
        }
        if !changed { break; }
    }
}

fn zero_subtree(probs: &mut [f32], loser: usize, parent_edges: &[(usize, usize)]) {
    let mut queue = vec![loser];
    let mut visited = HashSet::new();
    visited.insert(loser);
    while let Some(node) = queue.pop() {
        if node < probs.len() { probs[node] = 0.0; }
        for &(child, parent) in parent_edges {
            if parent == node && !visited.contains(&child) {
                visited.insert(child);
                queue.push(child);
            }
        }
    }
}

fn enforce_disjointness(probs: &mut [f32], disjoint_edges: &[(usize, usize)], parent_edges: &[(usize, usize)]) {
    for &(a, b) in disjoint_edges {
        if a >= probs.len() || b >= probs.len() { continue; }
        let pa = probs[a];
        let pb = probs[b];
        if pa.max(pb) >= DISJOINT_THRESHOLD {
            let loser = if pb <= pa { b } else { a };
            zero_subtree(probs, loser, parent_edges);
        }
    }
}

fn verify_axioms(probs: &[f32], parent_edges: &[(usize, usize)], disjoint_edges: &[(usize, usize)]) -> bool {
    for &(child, parent) in parent_edges {
        if child < probs.len() && parent < probs.len() && probs[parent] < probs[child] - 1e-6 {
            return false;
        }
    }
    for &(a, b) in disjoint_edges {
        if a < probs.len() && b < probs.len() {
            let min = probs[a].min(probs[b]);
            let max = probs[a].max(probs[b]);
            if min > 1e-6 && max >= DISJOINT_THRESHOLD {
                return false;
            }
        }
    }
    true
}

// ── IDF-weighted candidate router ─────────────────────────────────────────────

struct ConceptTokens {
    label: Vec<String>,
    synonyms: Vec<String>,
    description: Vec<String>,
}

struct WasmRouter {
    /// concepts[i] corresponds to slot index i
    concepts: Vec<WasmConcept>,
    /// inverted_index[token] = set of concept ids
    inverted_index: HashMap<String, Vec<u32>>,
    /// Smooth IDF per token
    idf: HashMap<String, f32>,
    /// Token fields per concept id
    tokens: HashMap<u32, ConceptTokens>,
}

impl WasmRouter {
    fn build(concepts: &[WasmConcept]) -> Self {
        let n = concepts.len() as f32;
        let mut inverted: HashMap<String, Vec<u32>> = HashMap::new();
        let mut tokens_map: HashMap<u32, ConceptTokens> = HashMap::new();

        for concept in concepts {
            let label_toks = tokenize(&concept.label);
            let syn_toks: Vec<String> = concept.synonyms.iter().flat_map(|s| tokenize(s)).collect();
            let desc_toks = tokenize(&concept.description);

            // Build inverted index
            let all_toks: HashSet<String> = label_toks.iter()
                .chain(syn_toks.iter())
                .chain(desc_toks.iter())
                .cloned().collect();
            for tok in all_toks {
                inverted.entry(tok).or_default().push(concept.id);
            }

            tokens_map.insert(concept.id, ConceptTokens {
                label: label_toks,
                synonyms: syn_toks,
                description: desc_toks,
            });
        }

        // Compute smooth IDF
        let mut idf_map = HashMap::new();
        for (tok, postings) in &inverted {
            let df = postings.len() as f32;
            idf_map.insert(tok.clone(), ((n + 1.0) / (df + 1.0)).ln() + 1.0);
        }

        WasmRouter {
            concepts: concepts.to_vec(),
            inverted_index: inverted,
            idf: idf_map,
            tokens: tokens_map,
        }
    }

    fn score_candidates(&self, query_text: &str, top_k: usize) -> Vec<(u32, f32)> {
        let query_toks = tokenize(query_text);
        let mut scores: HashMap<u32, f32> = HashMap::new();

        for tok in &query_toks {
            let idf = *self.idf.get(tok).unwrap_or(&1.0);
            if let Some(postings) = self.inverted_index.get(tok) {
                for &id in postings {
                    let toks = self.tokens.get(&id).expect("consistent index");
                    let label_tf = toks.label.iter().filter(|t| *t == tok).count() as f32;
                    let syn_tf = toks.synonyms.iter().filter(|t| *t == tok).count() as f32;
                    let desc_tf = toks.description.iter().filter(|t| *t == tok).count() as f32;

                    // BM25-style saturation
                    let label_score = idf * LABEL_BOOST * (label_tf * (K1_LABEL + 1.0)) / (label_tf + K1_LABEL);
                    let syn_score = idf * SYNONYM_BOOST * (syn_tf * (K1_DESC + 1.0)) / (syn_tf + K1_DESC);
                    let desc_score = idf * DESC_BOOST * (desc_tf * (K1_DESC + 1.0)) / (desc_tf + K1_DESC);

                    *scores.entry(id).or_insert(0.0) += label_score + syn_score + desc_score;
                }
            }
        }

        // Sort by score descending
        let mut ranked: Vec<(u32, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);

        // If empty, fallback to root concepts
        if ranked.is_empty() {
            for concept in self.concepts.iter().take(top_k) {
                if concept.parents.is_empty() {
                    ranked.push((concept.id, 0.1));
                }
            }
        }
        ranked
    }

    fn extract_subgraph(&self, query_text: &str, top_k: usize) -> (Vec<u32>, Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let top = self.score_candidates(query_text, top_k);
        let mut slot_ids: Vec<u32> = top.iter().map(|(id, _)| *id).collect();

        // Include all ancestors of top-k
        let id_to_concept: HashMap<u32, &WasmConcept> = self.concepts.iter().map(|c| (c.id, c)).collect();
        let mut in_slots: HashSet<u32> = slot_ids.iter().cloned().collect();

        for &(id, _) in &top {
            let mut queue = vec![id];
            while let Some(cur) = queue.pop() {
                if let Some(concept) = id_to_concept.get(&cur) {
                    for &parent in &concept.parents {
                        if !in_slots.contains(&parent) {
                            in_slots.insert(parent);
                            slot_ids.push(parent);
                            queue.push(parent);
                        }
                    }
                }
            }
        }

        // Build slot index
        let slot_index: HashMap<u32, usize> = slot_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Build edges
        let mut parent_edges = vec![];
        let mut disjoint_edges = vec![];
        for (slot_i, &id) in slot_ids.iter().enumerate() {
            if let Some(concept) = id_to_concept.get(&id) {
                for &parent in &concept.parents {
                    if let Some(&parent_slot) = slot_index.get(&parent) {
                        parent_edges.push((slot_i, parent_slot));
                    }
                }
                for &disj in &concept.disjoint_with {
                    if let Some(&disj_slot) = slot_index.get(&disj) {
                        if slot_i < disj_slot {
                            disjoint_edges.push((slot_i, disj_slot));
                        }
                    }
                }
            }
        }

        (slot_ids, parent_edges, disjoint_edges)
    }
}

// ── WASM-exposed OntologyEngine ───────────────────────────────────────────────

#[wasm_bindgen]
pub struct OlayaWasm {
    router: WasmRouter,
}

#[derive(Serialize)]
pub struct WasmDecisionResult {
    pub curie: String,
    pub label: String,
    pub confidence: f32,
    pub axioms_verified: bool,
    pub probabilities: Vec<(String, f32)>,
}

#[wasm_bindgen]
impl OlayaWasm {
    /// Build an O-Laya engine from a JSON concept array.
    /// Concepts must have: id, curie, label, description, synonyms, parents, disjoint_with.
    #[wasm_bindgen(constructor)]
    pub fn new(concepts_json: &str) -> Result<OlayaWasm, JsValue> {
        let concepts: Vec<WasmConcept> = serde_json::from_str(concepts_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let router = WasmRouter::build(&concepts);
        Ok(OlayaWasm { router })
    }

    /// Run a System 1 concept decision. Returns JSON DecisionResult.
    #[wasm_bindgen]
    pub fn system_one(&self, state_text: &str, top_k: usize) -> Result<JsValue, JsValue> {
        let (slot_ids, parent_edges, disjoint_edges) = self.router.extract_subgraph(state_text, top_k);
        let k = slot_ids.len();
        if k == 0 {
            return Err(JsValue::from_str("Empty subgraph"));
        }

        // Lexical logits using IDF-weighted scores
        let query_toks: HashSet<String> = tokenize(state_text).into_iter().collect();
        let logits: Vec<f32> = slot_ids.iter().map(|&id| {
            let toks = self.router.tokens.get(&id);
            let mut score = 1.0f32;
            if let Some(t) = toks {
                let idf_score = |tok_list: &[String], boost: f32| -> f32 {
                    tok_list.iter().map(|t| {
                        if query_toks.contains(t) {
                            boost * self.router.idf.get(t).cloned().unwrap_or(1.0)
                        } else { 0.0 }
                    }).sum::<f32>()
                };
                score += idf_score(&t.label, LABEL_BOOST);
                score += idf_score(&t.synonyms, SYNONYM_BOOST);
                score += idf_score(&t.description, DESC_BOOST);
            }
            score
        }).collect();

        // Axiom projection
        let t = temperature_by_cardinality(k);
        let mut probs = softmax(&logits, t);
        enforce_subsumption_monotonicity(&mut probs, &parent_edges);
        enforce_disjointness(&mut probs, &disjoint_edges, &parent_edges);

        // Select winner: prefer more specific concept on tie
        let mut best_slot = 0;
        let mut best_prob = f32::NEG_INFINITY;
        for (i, &p) in probs.iter().enumerate() {
            if p > best_prob {
                best_prob = p;
                best_slot = i;
            }
        }

        let best_id = slot_ids[best_slot];
        let best_concept = self.router.concepts.iter().find(|c| c.id == best_id)
            .ok_or_else(|| JsValue::from_str("Concept not found"))?;

        let confidence = normalized_entropy_confidence(&probs);
        let verified = verify_axioms(&probs, &parent_edges, &disjoint_edges);

        let probabilities: Vec<(String, f32)> = slot_ids.iter().zip(probs.iter()).map(|(&id, &p)| {
            let curie = self.router.concepts.iter().find(|c| c.id == id)
                .map(|c| c.curie.clone()).unwrap_or_default();
            (curie, p)
        }).collect();

        let result = WasmDecisionResult {
            curie: best_concept.curie.clone(),
            label: best_concept.label.clone(),
            confidence,
            axioms_verified: verified,
            probabilities,
        };

        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Number of concepts loaded.
    #[wasm_bindgen]
    pub fn concept_count(&self) -> usize {
        self.router.concepts.len()
    }

    /// Score a query against named options (choice primitive).
    #[wasm_bindgen]
    pub fn choice(&self, state_text: &str, options_json: &str) -> Result<JsValue, JsValue> {
        let options: Vec<String> = serde_json::from_str(options_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let query_toks: HashSet<String> = tokenize(state_text).into_iter().collect();

        let logits: Vec<f32> = options.iter().map(|opt| {
            let toks = tokenize(opt);
            let score: f32 = toks.iter().map(|t| {
                if query_toks.contains(t) {
                    LABEL_BOOST * self.router.idf.get(t).cloned().unwrap_or(1.0)
                } else { 0.0 }
            }).sum();
            1.0 + score
        }).collect();

        let t = temperature_by_cardinality(options.len());
        let probs = softmax(&logits, t);
        let best = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);

        let result = serde_json::json!({
            "choice": options[best],
            "confidence": normalized_entropy_confidence(&probs),
            "probabilities": options.iter().zip(probs.iter()).map(|(o, p)| (o, p)).collect::<Vec<_>>()
        });
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
