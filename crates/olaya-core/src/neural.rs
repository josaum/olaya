//! Neural cross-encoder scoring backend powered by ONNX Runtime and mycelia-ort-utils.

use anyhow::{Context, Result};
use mycelia_ort_utils::{build_session, ensure_ort_dylib_path, EpPref, SessionConfig};
use ort::{session::Session, value::Tensor};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Neural cross-encoder using an ONNX or ORT model (ModernBERT / MiniLM / BGE reranker).
pub struct NeuralScorer {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_length: usize,
    has_marker_inputs: bool,
    has_qtype_input: bool,
    special: SpecialIds,
}

/// Special token ids used to assemble the marker sequence (defaults: ModernBERT).
#[derive(Debug, Clone, Copy)]
struct SpecialIds {
    cls: u32,
    sep: u32,
    mask: u32,
}

/// Token budget for header + options, matching `head_max_len` in the Python sequence builder.
const HEAD_MAX_LEN: usize = 256;
/// Per-option token cap, matching the Python sequence builder.
const OPTION_MAX_TOKENS: usize = 48;

/// Marker-pooling input sequence: token ids plus the position of each candidate's `[MASK]`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MarkerSequence {
    pub input_ids: Vec<u32>,
    /// One entry per kept candidate, in candidate order.
    pub marker_positions: Vec<usize>,
}

/// Assembles `[CLS] concept question: <instr> [SEP] [MASK] opt_0 [MASK] opt_1 ... [SEP] <state> [SEP]`
/// exactly like `build_ontology_sequence` in `src/olaya/sequence.py`: the header is capped at half the
/// head budget, the first option is always kept, later options stop when the head budget is full,
/// and only markers inside the final `max_len` window are returned.
fn build_marker_sequence(
    header: &[u32],
    options: &[Vec<u32>],
    state: &[u32],
    special: SpecialIds,
    max_len: usize,
) -> MarkerSequence {
    let mut seq = vec![special.cls];
    seq.extend(header.iter().take(HEAD_MAX_LEN / 2));
    seq.push(special.sep);

    let mut marker_positions = Vec::with_capacity(options.len());
    for opt in options {
        let mut opt = &opt[..opt.len().min(OPTION_MAX_TOKENS)];
        if seq.len() + 1 + opt.len() > HEAD_MAX_LEN {
            if !marker_positions.is_empty() {
                break;
            }
            opt = &opt[..opt.len().min(HEAD_MAX_LEN.saturating_sub(seq.len() + 1))];
        }
        marker_positions.push(seq.len());
        seq.push(special.mask);
        seq.extend_from_slice(opt);
    }
    seq.push(special.sep);

    let room = max_len.saturating_sub(seq.len() + 1);
    seq.extend(state.iter().take(room));
    seq.push(special.sep);
    seq.truncate(max_len);

    marker_positions.retain(|&m| m < seq.len());
    MarkerSequence {
        input_ids: seq,
        marker_positions,
    }
}

impl NeuralScorer {
    /// Load an ONNX or ORT decision model with tokenizer.
    ///
    /// Transparently supports both `.onnx` and FlatBuffers `.ort` via `mycelia-ort-utils`.
    pub fn new<P: AsRef<Path>, T: AsRef<Path>>(
        model_path: P,
        tokenizer_path: T,
        pref: EpPref,
    ) -> Result<Self> {
        ensure_ort_dylib_path();

        let cfg = SessionConfig::default().with_ep(pref);
        let model_str = model_path.as_ref().to_string_lossy().to_string();
        let (session, info) = build_session(&model_str, &cfg)
            .map_err(|e| anyhow::anyhow!("Failed to build ORT session: {e}"))?;

        let has_marker_inputs = session.inputs().iter().any(|i| i.name() == "marker_pos");
        let has_qtype_input = session.inputs().iter().any(|i| i.name() == "qtype");

        tracing::info!(
            provider = info.provider,
            size_mb = info.size_mb,
            used_ort_format = info.used_ort_format,
            has_marker_inputs = has_marker_inputs,
            "NeuralScorer initialized"
        );

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let id = |names: &[&str], default: u32| {
            names
                .iter()
                .find_map(|n| tokenizer.token_to_id(n))
                .unwrap_or(default)
        };
        let special = SpecialIds {
            cls: id(&["[CLS]", "<s>"], 50281),
            sep: id(&["[SEP]", "</s>"], 50282),
            mask: id(&["[MASK]", "<mask>"], 50284),
        };

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_length: 512,
            has_marker_inputs,
            has_qtype_input,
            special,
        })
    }

    /// Score a state query against a list of candidate concept strings.
    /// Returns raw logits for each candidate slot.
    pub fn score_candidates(
        &self,
        state_text: &str,
        candidate_labels: &[String],
    ) -> Result<Vec<f32>> {
        if candidate_labels.is_empty() {
            return Ok(Vec::new());
        }

        let k = candidate_labels.len();

        // 1. Single forward pass non-autoregressive marker pooling (ModernBERT / O-Laya / Jev)
        if self.has_marker_inputs {
            let encode = |text: &str| -> Result<Vec<u32>> {
                Ok(self
                    .tokenizer
                    .encode(text, false)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?
                    .get_ids()
                    .to_vec())
            };
            let header = encode(
                "concept question: Classify the state into the most specific ontology concept.",
            )?;
            let options = candidate_labels
                .iter()
                .map(|c| encode(&format!(" {c}")))
                .collect::<Result<Vec<_>>>()?;
            let state = encode(state_text)?;
            let built =
                build_marker_sequence(&header, &options, &state, self.special, self.max_length);

            // Candidates whose marker did not fit get no score (filled with -inf below).
            let actual_k = built.marker_positions.len();
            let mut marker_pos_vec = vec![0i64; k];
            let mut marker_mask_vec = vec![false; k];
            for (i, &pos) in built.marker_positions.iter().enumerate() {
                marker_pos_vec[i] = pos as i64;
                marker_mask_vec[i] = true;
            }

            let seq_len = built.input_ids.len();
            let input_ids: Vec<i64> = built.input_ids.iter().map(|&t| t as i64).collect();
            let attention_mask = vec![1i64; seq_len];

            let input_shape = vec![1, seq_len];
            let marker_shape = vec![1, k];

            let input_ids_tensor = Tensor::from_array((input_shape.clone(), input_ids))
                .context("Failed to construct input_ids tensor")?;
            let attention_mask_tensor = Tensor::from_array((input_shape, attention_mask))
                .context("Failed to construct attention_mask tensor")?;
            let marker_pos_tensor = Tensor::from_array((marker_shape.clone(), marker_pos_vec))
                .context("Failed to construct marker_pos tensor")?;
            let marker_mask_tensor = Tensor::from_array((marker_shape, marker_mask_vec))
                .context("Failed to construct marker_mask tensor")?;

            let mut inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "marker_pos" => marker_pos_tensor,
                "marker_mask" => marker_mask_tensor,
            ];
            if self.has_qtype_input {
                // QTYPE_IDS["concept"] in the Python package.
                let qtype_tensor = Tensor::from_array((vec![1usize], vec![0i64]))
                    .context("Failed to construct qtype tensor")?;
                inputs.push(("qtype".into(), qtype_tensor.into()));
            }

            let mut session = self
                .session
                .lock()
                .map_err(|_| anyhow::anyhow!("Mutex poison"))?;
            let outputs = session.run(inputs).context("Inference execution failed")?;

            let output = outputs
                .get("logits")
                .or_else(|| outputs.get("output"))
                .context("No recognized output tensor in ONNX model")?;

            let (_, out_data) = output
                .try_extract_tensor::<f32>()
                .context("Failed to extract float tensor")?;

            let logits = (0..k)
                .map(|i| {
                    if i < actual_k {
                        out_data.get(i).copied().unwrap_or(0.0)
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect();
            return Ok(logits);
        }

        // 2. Pairwise cross-encoding fallback (BGE-reranker / MiniLM)
        let pairs: Vec<(&str, &str)> = candidate_labels
            .iter()
            .map(|lbl| (state_text, lbl.as_str()))
            .collect();

        let mut encodings = Vec::with_capacity(k);
        for (q, c) in &pairs {
            let enc = self
                .tokenizer
                .encode((*q, *c), true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
            encodings.push(enc);
        }

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(1)
            .min(self.max_length);

        let mut input_ids = vec![0i64; k * max_len];
        let mut attention_mask = vec![0i64; k * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let len = ids.len().min(max_len);
            for j in 0..len {
                input_ids[i * max_len + j] = ids[j] as i64;
                attention_mask[i * max_len + j] = mask[j] as i64;
            }
        }

        let shape = vec![k, max_len];
        let input_ids_tensor = Tensor::from_array((shape.clone(), input_ids))
            .context("Failed to construct input_ids tensor")?;
        let attention_mask_tensor = Tensor::from_array((shape, attention_mask))
            .context("Failed to construct attention_mask tensor")?;

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ];

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("Mutex poison"))?;
        let outputs = session.run(inputs).context("Inference execution failed")?;

        let output = outputs
            .get("logits")
            .or_else(|| outputs.get("output"))
            .or_else(|| outputs.get("last_hidden_state"))
            .context("No recognized output tensor in ONNX model")?;

        let (out_shape, out_data) = output
            .try_extract_tensor::<f32>()
            .context("Failed to extract float tensor")?;

        let mut logits = Vec::with_capacity(k);
        if out_shape.len() == 2 && out_shape[1] == 1 {
            logits.extend_from_slice(&out_data[..k]);
        } else if out_shape.len() == 2 && out_shape[1] >= 2 {
            for i in 0..k {
                let stride = out_shape[1] as usize;
                logits.push(out_data[i * stride + 1]);
            }
        } else {
            for i in 0..k {
                logits.push(out_data.get(i).copied().unwrap_or(0.0));
            }
        }

        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECIAL: SpecialIds = SpecialIds {
        cls: 1,
        sep: 2,
        mask: 3,
    };

    #[test]
    fn marker_sequence_matches_python_layout() {
        let built =
            build_marker_sequence(&[10, 11], &[vec![20], vec![30, 31]], &[40], SPECIAL, 512);
        assert_eq!(
            built.input_ids,
            vec![1, 10, 11, 2, 3, 20, 3, 30, 31, 2, 40, 2]
        );
        assert_eq!(built.marker_positions, vec![4, 6]);
    }

    #[test]
    fn markers_never_point_past_truncation() {
        let options: Vec<Vec<u32>> = (0..20).map(|_| vec![7; 48]).collect();
        let built = build_marker_sequence(&[9; 1000], &options, &[8; 1000], SPECIAL, 300);
        assert_eq!(built.input_ids.len(), 300);
        // Header capped at 128 tokens; two 48-token options fit in the 256-token head budget.
        assert_eq!(built.marker_positions, vec![130, 179]);
        assert!(built
            .marker_positions
            .iter()
            .all(|&m| built.input_ids[m] == SPECIAL.mask));
        assert_eq!(*built.input_ids.last().unwrap(), SPECIAL.sep);
    }
}
