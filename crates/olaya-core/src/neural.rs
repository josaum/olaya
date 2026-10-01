//! Neural cross-encoder scoring backend powered by ONNX Runtime and mycelia-ort-utils.

use std::path::Path;
use std::sync::Mutex;
use anyhow::{Context, Result};
use mycelia_ort_utils::{build_session, ensure_ort_dylib_path, EpPref, SessionConfig};
use ort::{session::Session, value::Tensor};
use tokenizers::Tokenizer;

/// Neural cross-encoder using an ONNX or ORT model (ModernBERT / MiniLM / BGE reranker).
pub struct NeuralScorer {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_length: usize,
    has_marker_inputs: bool,
    mask_token_id: u32,
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

        tracing::info!(
            provider = info.provider,
            size_mb = info.size_mb,
            used_ort_format = info.used_ort_format,
            has_marker_inputs = has_marker_inputs,
            "NeuralScorer initialized"
        );

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let mask_token_id = tokenizer
            .token_to_id("[MASK]")
            .or_else(|| tokenizer.token_to_id("<mask>"))
            .unwrap_or(50282);

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_length: 512,
            has_marker_inputs,
            mask_token_id,
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
            let mut seq_text = String::from("concept question: triage candidates:");
            for cand in candidate_labels {
                seq_text.push_str(" [MASK] ");
                seq_text.push_str(cand);
            }
            seq_text.push_str(" state: ");
            seq_text.push_str(state_text);

            let encoding = self
                .tokenizer
                .encode(seq_text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

            let token_ids = encoding.get_ids();
            let mut marker_positions = Vec::new();
            for (idx, &tid) in token_ids.iter().enumerate() {
                if tid == self.mask_token_id {
                    marker_positions.push(idx as i64);
                }
            }

            let actual_k = marker_positions.len().min(k);
            let mut marker_pos_vec = vec![0i64; k];
            let mut marker_mask_vec = vec![false; k];
            for i in 0..actual_k {
                marker_pos_vec[i] = marker_positions[i];
                marker_mask_vec[i] = true;
            }

            let seq_len = token_ids.len().min(self.max_length);
            let mut input_ids = vec![0i64; seq_len];
            let attention_mask = vec![1i64; seq_len];
            for i in 0..seq_len {
                input_ids[i] = token_ids[i] as i64;
            }

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

            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "marker_pos" => marker_pos_tensor,
                "marker_mask" => marker_mask_tensor,
            ];

            let mut session = self.session.lock().map_err(|_| anyhow::anyhow!("Mutex poison"))?;
            let outputs = session.run(inputs).context("Inference execution failed")?;

            let output = outputs
                .get("logits")
                .or_else(|| outputs.get("output"))
                .context("No recognized output tensor in ONNX model")?;

            let (_, out_data) = output
                .try_extract_tensor::<f32>()
                .context("Failed to extract float tensor")?;

            let mut logits = Vec::with_capacity(k);
            for i in 0..k {
                logits.push(out_data.get(i).copied().unwrap_or(0.0));
            }
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

        let mut session = self.session.lock().map_err(|_| anyhow::anyhow!("Mutex poison"))?;
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
            for i in 0..k {
                logits.push(out_data[i]);
            }
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
