//! Fast, in-place axiomatic consistency sweeps and calibration on flat memory buffers.

/// Computes calibrated temperature based on option cardinality.
pub fn temperature_by_cardinality(k: usize) -> f32 {
    if k <= 2 {
        0.85
    } else if k <= 5 {
        0.95
    } else if k <= 10 {
        1.10
    } else {
        1.25
    }
}

/// Computes normalized Shannon entropy confidence score in [0.0, 1.0].
///
/// 1.0 = absolute certainty (Dirac delta), 0.0 = uniform uncertainty.
pub fn normalized_entropy_confidence(probs: &[f32]) -> f32 {
    let k = probs.len();
    if k <= 1 {
        return 1.0;
    }

    let mut entropy = 0.0f32;
    for &p in probs {
        let p_clamped = p.max(1e-12);
        entropy -= p_clamped * p_clamped.ln();
    }

    let max_entropy = (k as f32).ln();
    (1.0 - (entropy / max_entropy)).clamp(0.0, 1.0)
}

/// Enforces subsumption monotonicity in-place: for every edge child -> parent,
/// guarantees that probs[parent] >= probs[child].
pub fn enforce_subsumption_monotonicity_inplace(
    probs: &mut [f32],
    parent_edges: &[(usize, usize)], // (child_slot, parent_slot)
) {
    let mut changed = true;
    let mut iter = 0;
    let max_iters = parent_edges.len() * 2;

    while changed && iter < max_iters {
        changed = false;
        iter += 1;
        for &(child_idx, parent_idx) in parent_edges {
            if child_idx < probs.len() && parent_idx < probs.len() {
                if probs[child_idx] > probs[parent_idx] {
                    probs[parent_idx] = probs[child_idx];
                    changed = true;
                }
            }
        }
    }
}

/// Enforces mutual exclusion: if probs[A] >= threshold, any B disjoint with A is set to 0.0.
pub fn enforce_disjointness_inplace(
    probs: &mut [f32],
    disjoint_edges: &[(usize, usize)], // (slot_a, slot_b)
    threshold: f32,
) {
    for &(a_idx, b_idx) in disjoint_edges {
        if a_idx < probs.len() && b_idx < probs.len() {
            if probs[a_idx] >= threshold {
                probs[b_idx] = 0.0;
            } else if probs[b_idx] >= threshold {
                probs[a_idx] = 0.0;
            }
        }
    }
}

/// Applies temperature-scaled softmax, axiom projections, and calculates confidence.
pub fn calibrate_and_project_inplace(
    logits: &mut [f32],
    parent_edges: &[(usize, usize)],
    disjoint_edges: &[(usize, usize)],
) -> f32 {
    let k = logits.len();
    if k == 0 {
        return 0.0;
    }

    let temp = temperature_by_cardinality(k);

    // 1. Softmax with temperature
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for l in logits.iter_mut() {
        *l = ((*l - max_logit) / temp).exp();
        sum_exp += *l;
    }

    if sum_exp > 0.0 {
        for l in logits.iter_mut() {
            *l /= sum_exp;
        }
    }

    // 2. Axiom enforcement
    enforce_subsumption_monotonicity_inplace(logits, parent_edges);
    enforce_disjointness_inplace(logits, disjoint_edges, 0.6);

    // 3. Compute normalized entropy confidence
    normalized_entropy_confidence(logits)
}
