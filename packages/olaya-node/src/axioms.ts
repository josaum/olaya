/**
 * Ontological axiom enforcement and entropy calibration for O-Laya.
 *
 * Implements the shared kernel in spec/AXIOMS.md; test/olaya.test.ts checks it against
 * spec/axioms_golden.json together with the Python and Rust runtimes.
 */

/** Disjointness threshold shared by all runtimes. */
export const DISJOINT_THRESHOLD = 0.6;

/** Calibrated softmax temperature by candidate cardinality. */
export function temperatureByCardinality(k: number): number {
  if (k <= 2) return 0.85;
  if (k <= 5) return 0.95;
  if (k <= 10) return 1.1;
  return 1.25;
}

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
 * Performs iterative relaxation until convergence (at most one pass per slot plus one, which
 * covers the longest possible chain).
 */
export function enforceSubsumptionMonotonicityInplace(
  probs: Float32Array,
  parentEdges: [number, number][],
  maxPasses = probs.length + 1
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
 * Enforces disjointness: for each pair (in order), if the larger side is at least `threshold`,
 * the smaller side (on a tie, `b`) and all of its descendants via `parentEdges` are set to 0.0,
 * so monotonicity still holds afterwards.
 */
export function enforceDisjointnessInplace(
  probs: Float32Array,
  disjointEdges: [number, number][],
  threshold = DISJOINT_THRESHOLD,
  parentEdges: [number, number][] = []
): void {
  for (const [a, b] of disjointEdges) {
    if (a < probs.length && b < probs.length && Math.max(probs[a], probs[b]) >= threshold) {
      zeroSubtree(probs, parentEdges, probs[a] >= probs[b] ? b : a);
    }
  }
}

function zeroSubtree(probs: Float32Array, parentEdges: [number, number][], root: number): void {
  const seen = new Set<number>([root]);
  const stack = [root];
  while (stack.length > 0) {
    const slot = stack.pop()!;
    probs[slot] = 0.0;
    for (const [child, parent] of parentEdges) {
      if (parent === slot && child < probs.length && !seen.has(child)) {
        seen.add(child);
        stack.push(child);
      }
    }
  }
}

/**
 * Checks the projection's guarantees: P(parent) >= P(child) for every edge, and for every disjoint
 * pair either one side is zero or both are below `threshold`.
 */
export function verifyAxioms(
  probs: Float32Array,
  parentEdges: [number, number][],
  disjointEdges: [number, number][],
  threshold = DISJOINT_THRESHOLD
): boolean {
  const tol = 1e-6;
  for (const [child, parent] of parentEdges) {
    if (child < probs.length && parent < probs.length && probs[parent] < probs[child] - tol) return false;
  }
  for (const [a, b] of disjointEdges) {
    if (a < probs.length && b < probs.length) {
      if (Math.min(probs[a], probs[b]) > tol && Math.max(probs[a], probs[b]) >= threshold) return false;
    }
  }
  return true;
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
 * Full calibration and axiom projection pipeline: cardinality-calibrated softmax, monotonicity
 * lifting, then disjointness. Probabilities are not renormalized afterwards (after lifting they are
 * hierarchical scores, not a simplex); the confidence normalizes internally.
 */
export function calibrateAndProject(
  rawLogits: Float32Array,
  parentEdges: [number, number][],
  disjointEdges: [number, number][],
  temperature = temperatureByCardinality(rawLogits.length)
): { probs: Float32Array; confidence: number } {
  const probs = softmax(rawLogits, temperature);
  enforceSubsumptionMonotonicityInplace(probs, parentEdges);
  enforceDisjointnessInplace(probs, disjointEdges, DISJOINT_THRESHOLD, parentEdges);

  const confidence = normalizedEntropyConfidence(probs);
  return { probs, confidence };
}
