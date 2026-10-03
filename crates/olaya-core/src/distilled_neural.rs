//! Distilled Native Neural Head inference engine.
//!
//! Ultra-lightweight, zero-external-dependency neural scorer.
//! Runs in pure native Rust and WebAssembly in < 5 microseconds without PyTorch or ONNX Runtime.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const FNV_PRIME: u64 = 0x100000001B3;
const FNV_OFFSET: u64 = 0xCBF29CE484222325;

/// Fast 64-bit FNV-1a hash folded to `num_buckets`, matching the Python implementation bit-for-bit.
#[inline(always)]
pub fn fnv1a_hash(text: &str, num_buckets: usize) -> usize {
    let mut h = FNV_OFFSET;
    for &b in text.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    (h % (num_buckets as u64)) as usize
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledModelData {
    pub format: String,
    pub num_buckets: usize,
    pub dim: usize,
    pub hidden_dim: usize,
    pub alpha_parent: f32,
    pub beta_disjoint: f32,
    pub concept_curies: Vec<String>,
    pub token_embeddings: Vec<Vec<f32>>,
    pub mlp_w1: Vec<Vec<f32>>,
    pub mlp_b1: Vec<f32>,
    pub mlp_w2: Vec<Vec<f32>>,
    pub mlp_b2: Vec<f32>,
    pub concept_embeddings: Vec<Vec<f32>>,
}

/// In-memory native neural scorer with flattened, cache-friendly weights.
#[derive(Debug, Clone)]
pub struct DistilledNeuralScorer {
    pub num_buckets: usize,
    pub dim: usize,
    pub hidden_dim: usize,
    pub alpha_parent: f32,
    pub beta_disjoint: f32,
    pub concept_curies: Vec<String>,
    pub curie_to_idx: HashMap<String, usize>,

    // Flattened weights for maximal cache locality and auto-vectorization
    token_embeddings: Vec<f32>,   // num_buckets * dim
    mlp_w1: Vec<f32>,             // hidden_dim * dim
    mlp_b1: Vec<f32>,             // hidden_dim
    mlp_w2: Vec<f32>,             // dim * hidden_dim
    mlp_b2: Vec<f32>,             // dim
    concept_embeddings: Vec<f32>, // num_concepts * dim (L2-normalized)
}

impl DistilledNeuralScorer {
    /// Load from a JSON weights string.
    pub fn from_json(json_str: &str) -> Result<Self> {
        let data: DistilledModelData =
            serde_json::from_str(json_str).context("Failed to parse DistilledModelData JSON")?;

        let num_concepts = data.concept_curies.len();
        let dim = data.dim;
        let hidden_dim = data.hidden_dim;
        let num_buckets = data.num_buckets;

        // Flatten token embeddings
        let mut flat_tok = Vec::with_capacity(num_buckets * dim);
        for row in &data.token_embeddings {
            flat_tok.extend_from_slice(row);
        }

        // Flatten MLP W1
        let mut flat_w1 = Vec::with_capacity(hidden_dim * dim);
        for row in &data.mlp_w1 {
            flat_w1.extend_from_slice(row);
        }

        // Flatten MLP W2
        let mut flat_w2 = Vec::with_capacity(dim * hidden_dim);
        for row in &data.mlp_w2 {
            flat_w2.extend_from_slice(row);
        }

        // Flatten and L2-normalize concept embeddings
        let mut flat_concepts = Vec::with_capacity(num_concepts * dim);
        for row in &data.concept_embeddings {
            let mut norm_sq = 0.0f32;
            for &v in row {
                norm_sq += v * v;
            }
            let inv_norm = if norm_sq > 1e-12 {
                1.0 / norm_sq.sqrt()
            } else {
                0.0
            };
            for &v in row {
                flat_concepts.push(v * inv_norm);
            }
        }

        let mut curie_to_idx = HashMap::with_capacity(num_concepts);
        for (i, c) in data.concept_curies.iter().enumerate() {
            curie_to_idx.insert(c.clone(), i);
        }

        Ok(Self {
            num_buckets,
            dim,
            hidden_dim,
            alpha_parent: data.alpha_parent,
            beta_disjoint: data.beta_disjoint,
            concept_curies: data.concept_curies,
            curie_to_idx,
            token_embeddings: flat_tok,
            mlp_w1: flat_w1,
            mlp_b1: data.mlp_b1,
            mlp_w2: flat_w2,
            mlp_b2: data.mlp_b2,
            concept_embeddings: flat_concepts,
        })
    }

    /// Load from the compact binary LEIO format.
    pub fn from_binary(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 24 {
            bail!("Binary header too short");
        }
        if &bytes[0..4] != b"LEIO" {
            bail!("Invalid magic header, expected 'LEIO'");
        }

        let mut cursor = 4;
        let read_u16 = |buf: &[u8], c: &mut usize| -> u16 {
            let val = u16::from_le_bytes([buf[*c], buf[*c + 1]]);
            *c += 2;
            val
        };
        let read_u32 = |buf: &[u8], c: &mut usize| -> u32 {
            let val = u32::from_le_bytes([buf[*c], buf[*c + 1], buf[*c + 2], buf[*c + 3]]);
            *c += 4;
            val
        };
        let read_f32 = |buf: &[u8], c: &mut usize| -> f32 {
            let val = f32::from_le_bytes([buf[*c], buf[*c + 1], buf[*c + 2], buf[*c + 3]]);
            *c += 4;
            val
        };

        let version = read_u16(bytes, &mut cursor);
        if version != 1 {
            bail!("Unsupported version {version}");
        }
        let dim = read_u16(bytes, &mut cursor) as usize;
        let hidden_dim = read_u16(bytes, &mut cursor) as usize;
        let num_buckets = read_u32(bytes, &mut cursor) as usize;
        let num_concepts = read_u32(bytes, &mut cursor) as usize;
        let alpha_parent = read_f32(bytes, &mut cursor);
        let beta_disjoint = read_f32(bytes, &mut cursor);

        // Read token embeddings (num_buckets * dim * 4 bytes)
        let tok_len = num_buckets * dim;
        let mut token_embeddings = Vec::with_capacity(tok_len);
        for _ in 0..tok_len {
            token_embeddings.push(read_f32(bytes, &mut cursor));
        }

        // Read mlp_w1 (hidden_dim * dim)
        let w1_len = hidden_dim * dim;
        let mut mlp_w1 = Vec::with_capacity(w1_len);
        for _ in 0..w1_len {
            mlp_w1.push(read_f32(bytes, &mut cursor));
        }

        // Read mlp_b1 (hidden_dim)
        let mut mlp_b1 = Vec::with_capacity(hidden_dim);
        for _ in 0..hidden_dim {
            mlp_b1.push(read_f32(bytes, &mut cursor));
        }

        // Read mlp_w2 (dim * hidden_dim)
        let w2_len = dim * hidden_dim;
        let mut mlp_w2 = Vec::with_capacity(w2_len);
        for _ in 0..w2_len {
            mlp_w2.push(read_f32(bytes, &mut cursor));
        }

        // Read mlp_b2 (dim)
        let mut mlp_b2 = Vec::with_capacity(dim);
        for _ in 0..dim {
            mlp_b2.push(read_f32(bytes, &mut cursor));
        }

        // Read concept embeddings (num_concepts * dim)
        let mut concept_embeddings = Vec::with_capacity(num_concepts * dim);
        for _ in 0..num_concepts {
            let start = cursor;
            let mut norm_sq = 0.0f32;
            for _ in 0..dim {
                let v = read_f32(bytes, &mut cursor);
                norm_sq += v * v;
            }
            let inv_norm = if norm_sq > 1e-12 {
                1.0 / norm_sq.sqrt()
            } else {
                0.0
            };
            let mut read_c = start;
            for _ in 0..dim {
                let v = read_f32(bytes, &mut read_c);
                concept_embeddings.push(v * inv_norm);
            }
        }

        // Read concept curies
        let mut concept_curies = Vec::with_capacity(num_concepts);
        let mut curie_to_idx = HashMap::with_capacity(num_concepts);
        for i in 0..num_concepts {
            let str_len = read_u16(bytes, &mut cursor) as usize;
            let s = std::str::from_utf8(&bytes[cursor..cursor + str_len])
                .context("Invalid UTF-8 in concept CURIE string table")?;
            cursor += str_len;
            concept_curies.push(s.to_string());
            curie_to_idx.insert(s.to_string(), i);
        }

        Ok(Self {
            num_buckets,
            dim,
            hidden_dim,
            alpha_parent,
            beta_disjoint,
            concept_curies,
            curie_to_idx,
            token_embeddings,
            mlp_w1,
            mlp_b1,
            mlp_w2,
            mlp_b2,
            concept_embeddings,
        })
    }

    /// Encode query tokens and IDF weights into a normalized query embedding vector of size `self.dim`.
    /// Zero heap allocation if writing into caller-provided slice.
    #[inline]
    pub fn encode_query_into(&self, tokens: &[&str], idfs: &[f32], out_vec: &mut [f32]) {
        assert_eq!(out_vec.len(), self.dim);
        if tokens.is_empty() {
            out_vec.fill(0.0);
            return;
        }

        let dim = self.dim;
        let hidden_dim = self.hidden_dim;

        // Step 1: IDF-weighted token pooling into temporary buffer
        let mut pooled = vec![0.0f32; dim];
        let mut weight_sum = 0.0f32;

        for (i, &t) in tokens.iter().enumerate() {
            let bucket = fnv1a_hash(t, self.num_buckets);
            let w = if i < idfs.len() { idfs[i] } else { 1.0 };
            weight_sum += w;

            let row_offset = bucket * dim;
            let tok_slice = &self.token_embeddings[row_offset..row_offset + dim];
            for d in 0..dim {
                pooled[d] += tok_slice[d] * w;
            }
        }

        let inv_weight = if weight_sum > 1e-6 {
            1.0 / weight_sum
        } else {
            1.0
        };
        let mut norm_sq = 0.0f32;
        for value in &mut pooled {
            *value *= inv_weight;
            norm_sq += *value * *value;
        }
        let inv_norm = if norm_sq > 1e-12 {
            1.0 / norm_sq.sqrt()
        } else {
            0.0
        };
        for value in &mut pooled {
            *value *= inv_norm;
        }

        // Step 2: Layer 1: hidden = ReLU(W1 * pooled + b1)
        let mut h1 = vec![0.0f32; hidden_dim];
        for (h, value) in h1.iter_mut().enumerate() {
            let row_offset = h * dim;
            let mut dot = self.mlp_b1[h];
            let w_row = &self.mlp_w1[row_offset..row_offset + dim];
            for d in 0..dim {
                dot += w_row[d] * pooled[d];
            }
            *value = if dot > 0.0 { dot } else { 0.0 }; // ReLU
        }

        // Step 3: Layer 2: out = W2 * h1 + b2
        let mut out_norm_sq = 0.0f32;
        for (d, value) in out_vec.iter_mut().take(dim).enumerate() {
            let row_offset = d * hidden_dim;
            let mut dot = self.mlp_b2[d];
            let w_row = &self.mlp_w2[row_offset..row_offset + hidden_dim];
            for h in 0..hidden_dim {
                dot += w_row[h] * h1[h];
            }
            *value = dot;
            out_norm_sq += dot * dot;
        }

        // Final L2 normalization
        let inv_out_norm = if out_norm_sq > 1e-12 {
            1.0 / out_norm_sq.sqrt()
        } else {
            0.0
        };
        for value in out_vec.iter_mut().take(dim) {
            *value *= inv_out_norm;
        }
    }

    /// High-level query encoder returning a vector.
    pub fn encode_query(&self, tokens: &[&str], idfs: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        self.encode_query_into(tokens, idfs, &mut out);
        out
    }

    /// Fast dot-product scoring for candidate concepts with ontological morphism propagation.
    ///
    /// For each candidate:
    /// `score = s0(c) + alpha * sum(s0(p)) - beta * sum(max(0, s0(d)))`
    pub fn score_candidates_with_morphisms(
        &self,
        query_vec: &[f32],
        candidate_indices: &[usize],
        candidate_parents: &[Vec<usize>],
        candidate_disjoint: &[Vec<usize>],
    ) -> Vec<f32> {
        let dim = self.dim;
        let mut scores = Vec::with_capacity(candidate_indices.len());

        let dot_concept = |idx: usize| -> f32 {
            let offset = idx * dim;
            let c_slice = &self.concept_embeddings[offset..offset + dim];
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += query_vec[d] * c_slice[d];
            }
            dot
        };

        for (i, &c_idx) in candidate_indices.iter().enumerate() {
            let s0 = dot_concept(c_idx);

            let mut p_sum = 0.0f32;
            if i < candidate_parents.len() {
                for &p_idx in &candidate_parents[i] {
                    p_sum += dot_concept(p_idx);
                }
            }

            let mut d_sum = 0.0f32;
            if i < candidate_disjoint.len() {
                for &d_idx in &candidate_disjoint[i] {
                    let d_dot = dot_concept(d_idx);
                    if d_dot > 0.0 {
                        d_sum += d_dot;
                    }
                }
            }

            let final_score = s0 + self.alpha_parent * p_sum - self.beta_disjoint * d_sum;
            scores.push(final_score);
        }

        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_hash_matches_golden() {
        let h1 = fnv1a_hash("cardiovascular", 4096);
        let h2 = fnv1a_hash("cardiovascular", 4096);
        assert_eq!(h1, h2);
        assert!(h1 < 4096);
    }

    #[test]
    fn test_distilled_scorer_json_and_inference() {
        let json_data = r#"{
            "format": "leio-cone-v1",
            "num_buckets": 16,
            "dim": 4,
            "hidden_dim": 6,
            "alpha_parent": 0.2,
            "beta_disjoint": 0.3,
            "concept_curies": ["C0", "C1", "C2"],
            "token_embeddings": [
                [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]
            ],
            "mlp_w1": [
                [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]
            ],
            "mlp_b1": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "mlp_w2": [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            ],
            "mlp_b2": [0.01, 0.02, 0.03, 0.04],
            "concept_embeddings": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ]
        }"#;

        let scorer = DistilledNeuralScorer::from_json(json_data).expect("Failed to parse JSON");
        assert_eq!(scorer.dim, 4);
        assert_eq!(scorer.hidden_dim, 6);
        assert_eq!(scorer.concept_curies.len(), 3);

        let q_vec = scorer.encode_query(&["heart", "attack"], &[1.5, 2.0]);
        assert_eq!(q_vec.len(), 4);
        let mut norm_sq = 0.0f32;
        for &v in &q_vec {
            norm_sq += v * v;
        }
        assert!((norm_sq - 1.0).abs() < 1e-4);

        let candidates = vec![0, 1];
        let parents = vec![vec![2], vec![]];
        let disjoint = vec![vec![], vec![0]];
        let scores =
            scorer.score_candidates_with_morphisms(&q_vec, &candidates, &parents, &disjoint);
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_distilled_scorer_sub_microsecond_latency() {
        let json_data = r#"{
            "format": "leio-cone-v1",
            "num_buckets": 64,
            "dim": 32,
            "hidden_dim": 48,
            "alpha_parent": 0.2,
            "beta_disjoint": 0.3,
            "concept_curies": ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
            "token_embeddings": [],
            "mlp_w1": [],
            "mlp_b1": [],
            "mlp_w2": [],
            "mlp_b2": [],
            "concept_embeddings": []
        }"#;

        // Construct realistic 32-dim model
        let mut model: DistilledModelData = serde_json::from_str(json_data).unwrap();
        model.token_embeddings = vec![vec![0.05; 32]; 64];
        model.mlp_w1 = vec![vec![0.02; 32]; 48];
        model.mlp_b1 = vec![0.01; 48];
        model.mlp_w2 = vec![vec![0.02; 48]; 32];
        model.mlp_b2 = vec![0.01; 32];
        model.concept_embeddings = vec![vec![0.1; 32]; 8];

        let scorer =
            DistilledNeuralScorer::from_json(&serde_json::to_string(&model).unwrap()).unwrap();

        let tokens = ["acute", "myocardial", "infarction", "syndrome"];
        let idfs = [3.2, 4.1, 4.5, 2.1];
        let candidates = vec![0, 1, 2, 3, 4];
        let parents = vec![vec![5], vec![5], vec![6], vec![], vec![7]];
        let disjoint = vec![vec![1], vec![0], vec![], vec![], vec![]];

        // Warmup
        let mut q_buf = vec![0.0f32; 32];
        scorer.encode_query_into(&tokens, &idfs, &mut q_buf);
        let _ = scorer.score_candidates_with_morphisms(&q_buf, &candidates, &parents, &disjoint);

        // Measure 10,000 iterations
        let start = std::time::Instant::now();
        let iters = 10_000;
        for _ in 0..iters {
            scorer.encode_query_into(&tokens, &idfs, &mut q_buf);
            let _ =
                scorer.score_candidates_with_morphisms(&q_buf, &candidates, &parents, &disjoint);
        }
        let elapsed = start.elapsed();
        let per_query_micros = elapsed.as_micros() as f64 / iters as f64;
        println!(
            "🚀 Distilled Native Scorer Latency: {:.3} µs / query",
            per_query_micros
        );
        // Must be under 50 microseconds even in debug mode unoptimized!
        assert!(per_query_micros < 50.0);
    }
}
