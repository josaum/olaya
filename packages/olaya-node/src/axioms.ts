/**
 * Ontological axiom enforcement and entropy calibration for O-Laya.
 */

export function softmax(logits: Float32Array, temperature = 1.0): Float32Array {
  const n = logits.length;
  if (n === 0) return new Float32Array(0);

  const t = Math.max(1e-4, temperature);
  let maxLogit = -Infinity;
  for (let i = 0; i < n; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }

  const exp = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = Math.exp((logits[i] - maxLogit) / t);
    exp[i] = v;
    sum += v;
  }

  const probs = new Float32Array(n);
  if (sum > 0) {
    for (let i = 0; i < n; i++) {
      probs[i] = exp[i] / sum;
    }
  }
  return probs;
}

/**
 * Enforces subsumption monotonicity: P(parent) >= P(child).
 * Performs iterative relaxation until convergence.
 */
export function enforceSubsumptionMonotonicityInplace(
  probs: Float32Array,
  parentEdges: [number, number][],
  maxPasses = 8
): void {
  for (let pass = 0; pass < maxPasses; pass++) {
    let changed = false;
    for (const [child, parent] of parentEdges) {
      if (child < probs.length && parent < probs.length) {
        if (probs[child] > probs[parent]) {
          probs[parent] = probs[child];
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
}

/**
 * Enforces disjointness: if P(a) > threshold, P(b) is suppressed to 0.0.
 */
export function enforceDisjointnessInplace(
  probs: Float32Array,
  disjointEdges: [number, number][],
  threshold = 0.5
): void {
  for (const [a, b] of disjointEdges) {
    if (a < probs.length && b < probs.length) {
      if (probs[a] > threshold && probs[a] > probs[b]) {
        probs[b] = 0.0;
      } else if (probs[b] > threshold && probs[b] > probs[a]) {
        probs[a] = 0.0;
      }
    }
  }
}

/**
 * Computes normalized Shannon entropy confidence: C = 1 - H(P) / ln(K).
 */
export function normalizedEntropyConfidence(probs: Float32Array): number {
  const k = probs.length;
  if (k <= 1) return 1.0;

  let sum = 0;
  for (let i = 0; i < k; i++) sum += probs[i];
  if (sum <= 0) return 0.0;

  let entropy = 0.0;
  for (let i = 0; i < k; i++) {
    const p = probs[i] / sum;
    if (p > 1e-12) {
      entropy -= p * Math.log(p);
    }
  }

  const maxEntropy = Math.log(k);
  const normalized = entropy / maxEntropy;
  return Math.max(0.0, Math.min(1.0, 1.0 - normalized));
}

/**
 * Full calibration and axiom projection pipeline.
 */
export function calibrateAndProject(
  rawLogits: Float32Array,
  parentEdges: [number, number][],
  disjointEdges: [number, number][],
  temperature = 1.0
): { probs: Float32Array; confidence: number } {
  const probs = softmax(rawLogits, temperature);
  enforceSubsumptionMonotonicityInplace(probs, parentEdges);
  enforceDisjointnessInplace(probs, disjointEdges);

  // Re-normalize probabilities
  let sum = 0;
  for (let i = 0; i < probs.length; i++) sum += probs[i];
  if (sum > 0) {
    for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  }

  const confidence = normalizedEntropyConfidence(probs);
  return { probs, confidence };
}
