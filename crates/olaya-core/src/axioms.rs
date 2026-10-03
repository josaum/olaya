//! Fast, in-place axiomatic consistency sweeps and calibration on flat memory buffers.
//!
//! Implements the shared kernel in `spec/AXIOMS.md`; `tests/test_olaya_core.rs` checks it against
//! `spec/axioms_golden.json`.

/// Disjointness threshold shared by all runtimes.
pub const DISJOINT_THRESHOLD: f32 = 0.6;

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
/// 1.0 = absolute certainty (Dirac delta), 0.0 = uniform uncertainty. The input is normalized
/// to sum to 1 first (it may not, after monotonicity lifting).
pub fn normalized_entropy_confidence(probs: &[f32]) -> f32 {
    let k = probs.len();
    if k <= 1 {
        return 1.0;
    }
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0f32;
    for &p in probs {
        let p = p / sum;
        if p > 1e-12 {
            entropy -= p * p.ln();
        }
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
            if child_idx < probs.len()
                && parent_idx < probs.len()
                && probs[child_idx] > probs[parent_idx]
            {
                probs[parent_idx] = probs[child_idx];
                changed = true;
            }
        }
    }
}

/// Enforces mutual exclusion: for each disjoint pair (in order), if the larger side is at least
/// `threshold`, the smaller side (on a tie, `slot_b`) and all of its descendants are set to 0.0,
/// so monotonicity still holds afterwards.
pub fn enforce_disjointness_inplace(
    probs: &mut [f32],
    parent_edges: &[(usize, usize)],   // (child_slot, parent_slot)
    disjoint_edges: &[(usize, usize)], // (slot_a, slot_b)
    threshold: f32,
) {
    let n = probs.len();
    for &(a_idx, b_idx) in disjoint_edges {
        if a_idx < n && b_idx < n && probs[a_idx].max(probs[b_idx]) >= threshold {
            let loser = if probs[a_idx] >= probs[b_idx] {
                b_idx
            } else {
                a_idx
            };
            zero_subtree(probs, parent_edges, loser);
        }
    }
}

/// Sets `root` and every slot below it (via `parent_edges`) to 0.0.
fn zero_subtree(probs: &mut [f32], parent_edges: &[(usize, usize)], root: usize) {
    let mut seen = vec![false; probs.len()];
    let mut stack = vec![root];
    seen[root] = true;
    while let Some(slot) = stack.pop() {
        probs[slot] = 0.0;
        for &(child, parent) in parent_edges {
            if parent == slot && child < probs.len() && !seen[child] {
                seen[child] = true;
                stack.push(child);
            }
        }
    }
}

/// Checks the projection's guarantees: `probs[parent] >= probs[child]` for every edge, and for
/// every disjoint pair either one side is zero or both are below `threshold`.
pub fn verify_axioms(
    probs: &[f32],
    parent_edges: &[(usize, usize)],
    disjoint_edges: &[(usize, usize)],
    threshold: f32,
) -> bool {
    const TOL: f32 = 1e-6;
    let get = |i: usize| probs.get(i).copied();
    let monotone = parent_edges.iter().all(|&(c, p)| match (get(c), get(p)) {
        (Some(pc), Some(pp)) => pp >= pc - TOL,
        _ => true,
    });
    let exclusive = disjoint_edges.iter().all(|&(a, b)| match (get(a), get(b)) {
        (Some(pa), Some(pb)) => pa.min(pb) <= TOL || pa.max(pb) < threshold,
        _ => true,
    });
    monotone && exclusive
}

/// Spec step 5: confidence of the *calibrated* (pre-lift) distribution, restricted to the slots
/// that survived disjointness exclusion.
///
/// `calibrated[i]` is slot `i`'s softmax value and `projected[i]` its value after monotonicity
/// lifting and disjointness. Lifting copies a child's mass onto every ancestor, so measuring entropy
/// on `projected` would make a confident deep decision look uniform (confidence 0); a slot that
/// disjointness zeroed no longer competes, so it is masked out here too.
pub fn projected_confidence(calibrated: &[f32], projected: &[f32]) -> f32 {
    let masked: Vec<f32> = calibrated
        .iter()
        .zip(projected)
        .map(|(&p, &q)| if q > 0.0 { p } else { 0.0 })
        .collect();
    normalized_entropy_confidence(&masked)
}

/// Applies temperature-scaled softmax, axiom projections, and calculates confidence.
///
/// On return `logits` holds the projected values (steps 2–4 of `spec/AXIOMS.md`) and the result is
/// the step-5 confidence, which is measured on the calibrated distribution before lifting.
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
    let calibrated = logits.to_vec();

    // 2. Axiom enforcement
    enforce_subsumption_monotonicity_inplace(logits, parent_edges);
    enforce_disjointness_inplace(logits, parent_edges, disjoint_edges, DISJOINT_THRESHOLD);

    // 3. Confidence of the calibrated distribution over the surviving candidates
    projected_confidence(&calibrated, logits)
}
