//! O-Laya WASM: IDF-weighted lexical decision engine for the browser.
//!
//! A dependency-free port of the Node package (`packages/olaya-node/src`): the same tokenizer
//! (stopwords, tokens longer than two characters, CamelCase split), the same smoothed IDF, the same
//! router scoring and candidate logits, and the shared axiom kernel from `spec/AXIOMS.md`.
//!
//! Everything except the `#[wasm_bindgen]` wrappers at the bottom is plain Rust, so the unit tests
//! in this file run natively (`cargo test -p olaya-wasm`) against `spec/axioms_golden.json` and
//! `spec/decision_parity.json`. The wasm32 build itself is checked in CI with
//! `cargo build -p olaya-wasm --target wasm32-unknown-unknown --release`.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use wasm_bindgen::prelude::*;

// ── Concept types ─────────────────────────────────────────────────────────────

/// One ontology class, as the JS side passes it in (ids are the caller's slot ids).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConcept {
    pub id: u32,
    pub curie: String,
    pub label: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub synonyms: Vec<String>,
    #[serde(default)]
    pub parents: Vec<u32>,
    #[serde(default)]
    pub disjoint_with: Vec<u32>,
}

/// Active decision subgraph: slot `i` is `slot_ids[i]`, edges are between slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveSubgraph {
    pub slot_ids: Vec<u32>,
    pub parent_edges: Vec<(usize, usize)>,
    pub disjoint_edges: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WasmDecisionResult {
    pub curie: String,
    pub label: String,
    pub confidence: f32,
    pub axioms_verified: bool,
    pub probabilities: Vec<(String, f32)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WasmChoiceResult {
    pub choice: String,
    pub confidence: f32,
    pub probabilities: Vec<(String, f32)>,
}

// ── Axiom kernel (spec/AXIOMS.md) ─────────────────────────────────────────────

/// Disjointness threshold shared by all runtimes.
pub const DISJOINT_THRESHOLD: f32 = 0.6;

/// Calibrated softmax temperature by candidate cardinality (spec step 1).
pub fn temperature_by_cardinality(k: usize) -> f32 {
    match k {
        0..=2 => 0.85,
        3..=5 => 0.95,
        6..=10 => 1.10,
        _ => 1.25,
    }
}

/// Temperature-scaled softmax (spec step 2).
pub fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let t = temperature.max(1e-4);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| ((l - max) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

/// Normalised Shannon-entropy confidence in `[0, 1]`; the input is normalised to sum to 1 first.
pub fn normalized_entropy_confidence(probs: &[f32]) -> f32 {
    let n = probs.len();
    if n <= 1 {
        return 1.0;
    }
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0f32;
    for &p in probs {
        let pn = p / sum;
        if pn > 1e-12 {
            h -= pn * pn.ln();
        }
    }
    (1.0 - h / (n as f32).ln()).clamp(0.0, 1.0)
}

/// Spec step 3: lift every parent to at least its child's value.
pub fn enforce_subsumption_monotonicity(probs: &mut [f32], parent_edges: &[(usize, usize)]) {
    let k = probs.len();
    for _ in 0..=k {
        let mut changed = false;
        for &(child, parent) in parent_edges {
            if child < k && parent < k && probs[child] > probs[parent] {
                probs[parent] = probs[child];
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn zero_subtree(probs: &mut [f32], loser: usize, parent_edges: &[(usize, usize)]) {
    let mut stack = vec![loser];
    let mut visited = HashSet::new();
    visited.insert(loser);
    while let Some(node) = stack.pop() {
        if node < probs.len() {
            probs[node] = 0.0;
        }
        for &(child, parent) in parent_edges {
            if parent == node && child < probs.len() && visited.insert(child) {
                stack.push(child);
            }
        }
    }
}

/// Spec step 4: for each disjoint pair in order, the smaller side (on a tie, `b`) and its subtree
/// are zeroed when the larger side reaches the threshold.
pub fn enforce_disjointness(
    probs: &mut [f32],
    disjoint_edges: &[(usize, usize)],
    parent_edges: &[(usize, usize)],
) {
    for &(a, b) in disjoint_edges {
        if a >= probs.len() || b >= probs.len() {
            continue;
        }
        let (pa, pb) = (probs[a], probs[b]);
        if pa.max(pb) >= DISJOINT_THRESHOLD {
            let loser = if pa >= pb { b } else { a };
            zero_subtree(probs, loser, parent_edges);
        }
    }
}

/// Spec step 5: confidence of the calibrated (pre-lift) distribution restricted to the slots that
/// survived disjointness exclusion.
pub fn projected_confidence(calibrated: &[f32], projected: &[f32]) -> f32 {
    let masked: Vec<f32> = calibrated
        .iter()
        .zip(projected)
        .map(|(&p, &q)| if q > 0.0 { p } else { 0.0 })
        .collect();
    normalized_entropy_confidence(&masked)
}

/// The full kernel: returns the projected values and the step-5 confidence.
pub fn calibrate_and_project(
    logits: &[f32],
    parent_edges: &[(usize, usize)],
    disjoint_edges: &[(usize, usize)],
) -> (Vec<f32>, f32) {
    let calibrated = softmax(logits, temperature_by_cardinality(logits.len()));
    let mut probs = calibrated.clone();
    enforce_subsumption_monotonicity(&mut probs, parent_edges);
    enforce_disjointness(&mut probs, disjoint_edges, parent_edges);
    let confidence = projected_confidence(&calibrated, &probs);
    (probs, confidence)
}

/// Checks the kernel's guarantees with tolerance `1e-6`.
pub fn verify_axioms(
    probs: &[f32],
    parent_edges: &[(usize, usize)],
    disjoint_edges: &[(usize, usize)],
) -> bool {
    const TOL: f32 = 1e-6;
    let n = probs.len();
    let monotone = parent_edges
        .iter()
        .all(|&(c, p)| c >= n || p >= n || probs[p] >= probs[c] - TOL);
    let exclusive = disjoint_edges.iter().all(|&(a, b)| {
        a >= n
            || b >= n
            || probs[a].min(probs[b]) <= TOL
            || probs[a].max(probs[b]) < DISJOINT_THRESHOLD
    });
    monotone && exclusive
}

// ── Tokenizer (same rules as packages/olaya-node/src/router.ts) ────────────────

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "out", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "don", "should", "now", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "of", "it",
    "its", "that", "this", "also", "made",
];

/// Splits CamelCase, lowercases, keeps `[a-z0-9]` runs longer than two characters that are not
/// stopwords.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut spaced = String::with_capacity(text.len() + 16);
    let mut prev_lower = false;
    for ch in text.chars() {
        if ch.is_ascii_uppercase() && prev_lower {
            spaced.push(' ');
        }
        prev_lower = ch.is_ascii_lowercase();
        spaced.push(ch);
    }
    spaced
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() > 2 && !STOPWORDS.contains(t))
        .map(str::to_string)
        .collect()
}

// ── IDF-weighted candidate router ─────────────────────────────────────────────

struct ConceptTokens {
    label: Vec<String>,
    synonyms: Vec<String>,
    description: Vec<String>,
    label_set: HashSet<String>,
    synonym_set: HashSet<String>,
    description_set: HashSet<String>,
}

/// Lexical router over a concept list; see the module docs for what it mirrors.
pub struct WasmRouter {
    concepts: Vec<WasmConcept>,
    /// concept id -> position in `concepts`
    position: HashMap<u32, usize>,
    /// token -> positions of the concepts containing it, in concept order
    inverted_index: HashMap<String, Vec<usize>>,
    /// Smoothed IDF per indexed token: ln((N + 1) / (df + 1)) + 1
    idf: HashMap<String, f32>,
    tokens: Vec<ConceptTokens>,
    /// Longest subClassOf path from a root (0 = root); used to break argmax ties.
    depth: Vec<u32>,
}

impl WasmRouter {
    pub fn build(concepts: &[WasmConcept]) -> Self {
        let n = concepts.len() as f32;
        let position: HashMap<u32, usize> = concepts
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id, i))
            .collect();
        let mut inverted: HashMap<String, Vec<usize>> = HashMap::new();
        let mut tokens = Vec::with_capacity(concepts.len());

        for (pos, concept) in concepts.iter().enumerate() {
            let label = tokenize(&concept.label);
            let synonyms: Vec<String> = concept.synonyms.iter().flat_map(|s| tokenize(s)).collect();
            let description = tokenize(&concept.description);
            let label_set: HashSet<String> = label.iter().cloned().collect();
            let synonym_set: HashSet<String> = synonyms.iter().cloned().collect();
            let description_set: HashSet<String> = description.iter().cloned().collect();

            let mut all: Vec<&String> = label_set
                .iter()
                .chain(&synonym_set)
                .chain(&description_set)
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            all.sort();
            for tok in all {
                inverted.entry(tok.clone()).or_default().push(pos);
            }
            tokens.push(ConceptTokens {
                label,
                synonyms,
                description,
                label_set,
                synonym_set,
                description_set,
            });
        }

        let idf = inverted
            .iter()
            .map(|(tok, postings)| {
                let df = postings.len() as f32;
                (tok.clone(), ((n + 1.0) / (df + 1.0)).ln() + 1.0)
            })
            .collect();

        let depth = compute_depths(concepts, &position);

        WasmRouter {
            concepts: concepts.to_vec(),
            position,
            inverted_index: inverted,
            idf,
            tokens,
            depth,
        }
    }

    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    pub fn concept_by_id(&self, id: u32) -> Option<&WasmConcept> {
        self.position.get(&id).map(|&p| &self.concepts[p])
    }

    fn idf(&self, token: &str) -> f32 {
        self.idf.get(token).copied().unwrap_or(1.0)
    }

    /// Ranks concept positions by the Node router's score: label hit `4·idf`, synonym hit `2·idf`,
    /// description hit `idf` (doubled when the label shares no token with the query).
    fn rank(&self, query_tokens: &[String]) -> Vec<usize> {
        let query_set: HashSet<&String> = query_tokens.iter().collect();
        let mut scores: HashMap<usize, f32> = HashMap::new();
        let mut order: Vec<usize> = Vec::new();

        for tok in query_tokens {
            let Some(postings) = self.inverted_index.get(tok) else {
                continue;
            };
            let idf = self.idf(tok);
            for &pos in postings {
                let fields = &self.tokens[pos];
                let label_overlap = fields.label_set.iter().any(|t| query_set.contains(t));
                let desc_boost = if label_overlap { 1.0 } else { 2.0 };
                let mut score = 0.0;
                if fields.label_set.contains(tok) {
                    score += idf * 4.0;
                }
                if fields.synonym_set.contains(tok) {
                    score += idf * 2.0;
                }
                if fields.description_set.contains(tok) {
                    score += idf * desc_boost;
                }
                if score > 0.0 {
                    let entry = scores.entry(pos).or_insert_with(|| {
                        order.push(pos);
                        0.0
                    });
                    *entry += score;
                }
            }
        }

        // Stable sort by score descending; ties keep first-touch order, as in Node.
        order.sort_by(|a, b| scores[b].total_cmp(&scores[a]));
        order
    }

    /// The top `max_candidates` lexical matches plus all of their ancestors, in the Node
    /// package's slot order, or the root concepts when nothing matches.
    pub fn extract_subgraph(&self, state_text: &str, max_candidates: usize) -> ActiveSubgraph {
        let ranked = self.rank(&tokenize(state_text));

        let mut active: Vec<usize> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        for &pos in ranked.iter().take(max_candidates) {
            if seen.insert(pos) {
                active.push(pos);
            }
            let mut queue = VecDeque::from([pos]);
            while let Some(cur) = queue.pop_front() {
                for &parent in &self.concepts[cur].parents {
                    if let Some(&ppos) = self.position.get(&parent) {
                        if seen.insert(ppos) {
                            active.push(ppos);
                            queue.push_back(ppos);
                        }
                    }
                }
            }
        }
        if active.is_empty() {
            for (pos, c) in self.concepts.iter().enumerate() {
                if c.parents.is_empty() {
                    active.push(pos);
                    if active.len() >= max_candidates {
                        break;
                    }
                }
            }
        }

        let slot_of: HashMap<usize, usize> =
            active.iter().enumerate().map(|(s, &p)| (p, s)).collect();
        let mut parent_edges = Vec::new();
        let mut disjoint_edges = Vec::new();
        for (slot, &pos) in active.iter().enumerate() {
            let c = &self.concepts[pos];
            for parent in &c.parents {
                if let Some(&ps) = self.position.get(parent).and_then(|p| slot_of.get(p)) {
                    parent_edges.push((slot, ps));
                }
            }
            for other in &c.disjoint_with {
                if let Some(&os) = self.position.get(other).and_then(|p| slot_of.get(p)) {
                    if slot < os {
                        disjoint_edges.push((slot, os));
                    }
                }
            }
        }

        ActiveSubgraph {
            slot_ids: active.iter().map(|&p| self.concepts[p].id).collect(),
            parent_edges,
            disjoint_edges,
        }
    }

    /// Candidate logits of the Node engine: `1 + 6·idf` per matching label token, `5·idf` per
    /// matching synonym token, `idf` per matching description token.
    fn logits(&self, query_set: &HashSet<String>, slot_ids: &[u32]) -> Vec<f32> {
        slot_ids
            .iter()
            .map(|id| {
                let fields = &self.tokens[self.position[id]];
                let hit = |toks: &[String], boost: f32| -> f32 {
                    toks.iter()
                        .filter(|t| query_set.contains(*t))
                        .map(|t| boost * self.idf(t))
                        .sum()
                };
                1.0 + hit(&fields.label, 6.0)
                    + hit(&fields.synonyms, 5.0)
                    + hit(&fields.description, 1.0)
            })
            .collect()
    }

    /// System 1 concept decision.
    pub fn decide(&self, state_text: &str, top_k: usize) -> Result<WasmDecisionResult, String> {
        let subgraph = self.extract_subgraph(state_text, top_k);
        if subgraph.slot_ids.is_empty() {
            return Err("Empty subgraph".to_string());
        }
        let query_set: HashSet<String> = tokenize(state_text).into_iter().collect();
        let logits = self.logits(&query_set, &subgraph.slot_ids);
        let (probs, confidence) =
            calibrate_and_project(&logits, &subgraph.parent_edges, &subgraph.disjoint_edges);

        // Argmax; near-ties (a parent lifted to its child's value) go to the deepest concept,
        // then to the lowest slot.
        let max_prob = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let best_slot = (0..probs.len())
            .filter(|&s| probs[s] >= max_prob - 1e-5)
            .max_by(|&a, &b| {
                let da = self.depth[self.position[&subgraph.slot_ids[a]]];
                let db = self.depth[self.position[&subgraph.slot_ids[b]]];
                da.cmp(&db).then(b.cmp(&a))
            })
            .ok_or_else(|| "No candidate slots".to_string())?;

        let best = &self.concepts[self.position[&subgraph.slot_ids[best_slot]]];
        let probabilities = subgraph
            .slot_ids
            .iter()
            .zip(&probs)
            .map(|(id, &p)| (self.concepts[self.position[id]].curie.clone(), p))
            .collect();

        Ok(WasmDecisionResult {
            curie: best.curie.clone(),
            label: best.label.clone(),
            confidence,
            axioms_verified: verify_axioms(
                &probs,
                &subgraph.parent_edges,
                &subgraph.disjoint_edges,
            ),
            probabilities,
        })
    }

    /// Discrete choice over option labels: each option scores `1 + 6·idf` per label token found
    /// in the state (the Node engine's `choice`, which builds a flat ontology of the options).
    pub fn choice(&self, state_text: &str, options: &[String]) -> Result<WasmChoiceResult, String> {
        if options.is_empty() {
            return Err("choice needs at least one option".to_string());
        }
        let option_concepts: Vec<WasmConcept> = options
            .iter()
            .enumerate()
            .map(|(i, label)| WasmConcept {
                id: i as u32,
                curie: format!("OPT:{i}"),
                label: label.clone(),
                description: String::new(),
                synonyms: Vec::new(),
                parents: Vec::new(),
                disjoint_with: Vec::new(),
            })
            .collect();
        let flat = WasmRouter::build(&option_concepts);
        let query_set: HashSet<String> = tokenize(state_text).into_iter().collect();
        let slot_ids: Vec<u32> = (0..options.len() as u32).collect();
        let logits = flat.logits(&query_set, &slot_ids);
        let (probs, confidence) = calibrate_and_project(&logits, &[], &[]);
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(WasmChoiceResult {
            choice: options[best].clone(),
            confidence,
            probabilities: options.iter().cloned().zip(probs).collect(),
        })
    }
}

/// Longest path from a root for every concept position; rows on a cycle stop where the cycle is
/// detected.
fn compute_depths(concepts: &[WasmConcept], position: &HashMap<u32, usize>) -> Vec<u32> {
    const UNVISITED: u8 = 0;
    const IN_PROGRESS: u8 = 1;
    const DONE: u8 = 2;
    let n = concepts.len();
    let parents: Vec<Vec<usize>> = concepts
        .iter()
        .map(|c| {
            c.parents
                .iter()
                .filter_map(|p| position.get(p).copied())
                .collect()
        })
        .collect();
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
            if let Some(&p) = parents[row].get(top.1) {
                top.1 += 1;
                if state[p] == UNVISITED {
                    state[p] = IN_PROGRESS;
                    stack.push((p, 0));
                }
            } else {
                depth[row] = parents[row]
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

// ── WASM-exposed OntologyEngine ───────────────────────────────────────────────

#[wasm_bindgen]
pub struct OlayaWasm {
    router: WasmRouter,
}

#[wasm_bindgen]
impl OlayaWasm {
    /// Build an O-Laya engine from a JSON concept array.
    /// Concepts must have: id, curie, label, description, synonyms, parents, disjoint_with.
    #[wasm_bindgen(constructor)]
    pub fn new(concepts_json: &str) -> Result<OlayaWasm, JsValue> {
        let concepts: Vec<WasmConcept> =
            serde_json::from_str(concepts_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(OlayaWasm {
            router: WasmRouter::build(&concepts),
        })
    }

    /// Run a System 1 concept decision. Returns `{curie, label, confidence, axioms_verified,
    /// probabilities: [[curie, prob], ...]}`.
    #[wasm_bindgen]
    pub fn system_one(&self, state_text: &str, top_k: usize) -> Result<JsValue, JsValue> {
        let result = self
            .router
            .decide(state_text, top_k)
            .map_err(|e| JsValue::from_str(&e))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Number of concepts loaded.
    #[wasm_bindgen]
    pub fn concept_count(&self) -> usize {
        self.router.concept_count()
    }

    /// Score a state against named options (choice primitive). `options_json` is a JSON string
    /// array; returns `{choice, confidence, probabilities: [[option, prob], ...]}`.
    #[wasm_bindgen]
    pub fn choice(&self, state_text: &str, options_json: &str) -> Result<JsValue, JsValue> {
        let options: Vec<String> =
            serde_json::from_str(options_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let result = self
            .router
            .choice(state_text, &options)
            .map_err(|e| JsValue::from_str(&e))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// ── Native unit tests (no browser needed) ─────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_file(name: &str) -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../spec")
            .join(name);
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("cannot read {}: {e}", path.display());
        }))
        .unwrap()
    }

    fn edges(v: &serde_json::Value) -> Vec<(usize, usize)> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|e| {
                (
                    e[0].as_u64().unwrap() as usize,
                    e[1].as_u64().unwrap() as usize,
                )
            })
            .collect()
    }

    /// Builds the fixture ontology with slot ids 0..n in file order.
    fn parity_concepts(fixture: &serde_json::Value) -> Vec<WasmConcept> {
        let raw = fixture["ontology"]["concepts"].as_array().unwrap();
        let id_of: HashMap<&str, u32> = raw
            .iter()
            .enumerate()
            .map(|(i, c)| (c["curie"].as_str().unwrap(), i as u32))
            .collect();
        let ids = |v: &serde_json::Value| -> Vec<u32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| id_of[x.as_str().unwrap()])
                .collect()
        };
        let strings = |v: &serde_json::Value| -> Vec<String> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap().to_string())
                .collect()
        };
        raw.iter()
            .enumerate()
            .map(|(i, c)| WasmConcept {
                id: i as u32,
                curie: c["curie"].as_str().unwrap().to_string(),
                label: c["label"].as_str().unwrap().to_string(),
                description: c["description"].as_str().unwrap_or("").to_string(),
                synonyms: strings(&c["synonyms"]),
                parents: ids(&c["parents"]),
                disjoint_with: ids(&c["disjoint_with"]),
            })
            .collect()
    }

    #[test]
    fn kernel_matches_shared_golden_vectors() {
        let golden = spec_file("axioms_golden.json");
        assert_eq!(
            golden["disjoint_threshold"].as_f64().unwrap() as f32,
            DISJOINT_THRESHOLD
        );
        let tol = golden["tolerance"].as_f64().unwrap() as f32;
        for case in golden["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let logits: Vec<f32> = case["logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let parent_edges = edges(&case["parent_edges"]);
            let disjoint_edges = edges(&case["disjoint_edges"]);
            let (probs, confidence) =
                calibrate_and_project(&logits, &parent_edges, &disjoint_edges);
            for (slot, want) in case["expected_probs"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                let want = want.as_f64().unwrap() as f32;
                assert!(
                    (probs[slot] - want).abs() <= tol,
                    "{name}: slot {slot} got {} want {want}",
                    probs[slot]
                );
            }
            let want_conf = case["expected_confidence"].as_f64().unwrap() as f32;
            assert!(
                (confidence - want_conf).abs() <= tol,
                "{name}: confidence {confidence} want {want_conf}"
            );
            assert_eq!(
                verify_axioms(&probs, &parent_edges, &disjoint_edges),
                case["expected_axioms_verified"].as_bool().unwrap(),
                "{name}: axioms_verified"
            );
        }
    }

    #[test]
    fn tokenizer_follows_node_rules() {
        // CamelCase split, stopwords dropped, tokens of length <= 2 dropped, lowercased.
        // "PostgreSQL" splits at the lower->upper boundary exactly as Node's regex does.
        assert_eq!(
            tokenize("DatabaseDeadlock on the PostgreSQL db at 2am"),
            ["database", "deadlock", "postgre", "sql", "2am"]
        );
        assert_eq!(tokenize("HTTP 503 from API"), ["http", "503", "api"]);
        assert!(tokenize("a an the of").is_empty());
    }

    #[test]
    fn decisions_match_shared_parity_fixture() {
        let fixture = spec_file("decision_parity.json");
        let router = WasmRouter::build(&parity_concepts(&fixture));
        let top_k = fixture["max_candidates"].as_u64().unwrap() as usize;
        for case in fixture["queries"].as_array().unwrap() {
            let state = case["state"].as_str().unwrap();
            let want = case["expected_curie"].as_str().unwrap();
            let res = router.decide(state, top_k).unwrap();
            assert_eq!(res.curie, want, "{state:?}: {:?}", res.probabilities);
            assert!(res.axioms_verified, "{state:?}");
            assert!(
                res.confidence > 0.0,
                "{state:?}: confidence {}",
                res.confidence
            );
            // Disjointness: the listed concepts must be in the subgraph and zeroed.
            for excluded in case["expected_excluded"].as_array().into_iter().flatten() {
                let curie = excluded.as_str().unwrap();
                let prob = res
                    .probabilities
                    .iter()
                    .find(|(c, _)| c == curie)
                    .unwrap_or_else(|| panic!("{state:?}: {curie} not in subgraph"))
                    .1;
                assert_eq!(prob, 0.0, "{state:?}: {curie} should be excluded");
            }
        }
    }

    #[test]
    fn tie_break_prefers_most_specific_concept_and_confidence_survives_lifting() {
        // Child rows before parents, so slot order cannot be relied on.
        let concepts = vec![
            WasmConcept {
                id: 7,
                curie: "INC:DB_DEADLOCK".into(),
                label: "Database Deadlock".into(),
                description: String::new(),
                synonyms: vec!["deadlock".into()],
                parents: vec![3],
                disjoint_with: vec![],
            },
            WasmConcept {
                id: 3,
                curie: "INC:DATA".into(),
                label: "Data Incident".into(),
                description: String::new(),
                synonyms: vec![],
                parents: vec![1],
                disjoint_with: vec![],
            },
            WasmConcept {
                id: 1,
                curie: "INC:ROOT".into(),
                label: "Operational Incident".into(),
                description: String::new(),
                synonyms: vec![],
                parents: vec![],
                disjoint_with: vec![],
            },
        ];
        let router = WasmRouter::build(&concepts);
        let res = router.decide("deadlock", 8).unwrap();
        assert_eq!(res.curie, "INC:DB_DEADLOCK");
        // All three slots are lifted to the same value; confidence must come from the
        // calibrated distribution, not the uniform-looking lifted one.
        let lifted: Vec<f32> = res.probabilities.iter().map(|(_, p)| *p).collect();
        assert!(lifted.iter().all(|&p| (p - lifted[0]).abs() < 1e-6));
        assert!(res.confidence > 0.5, "confidence {}", res.confidence);
        assert!(res.axioms_verified);
    }

    #[test]
    fn choice_selects_lexically_matching_option() {
        let router = WasmRouter::build(&[]);
        let options = vec![
            "Process Refund".to_string(),
            "Restart Server".to_string(),
            "Deploy Database".to_string(),
        ];
        let res = router
            .choice(
                "Customer requested a refund for the duplicate charge",
                &options,
            )
            .unwrap();
        assert_eq!(res.choice, "Process Refund");
        assert!(res.confidence > 0.3);
        let total: f32 = res.probabilities.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-5);
        assert!(router.choice("anything", &[]).is_err());
    }

    #[test]
    fn empty_match_falls_back_to_roots() {
        let fixture = spec_file("decision_parity.json");
        let router = WasmRouter::build(&parity_concepts(&fixture));
        let sub = router.extract_subgraph("zzz qqq", 12);
        assert_eq!(sub.slot_ids, vec![0]);
        let res = router.decide("zzz qqq", 12).unwrap();
        assert_eq!(res.curie, "INC:ROOT");
        assert_eq!(res.confidence, 1.0);
    }
}
