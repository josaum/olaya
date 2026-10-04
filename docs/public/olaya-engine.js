// O-Laya Browser Engine — IDF-weighted BM25 candidate router
// Expanded/readable ESModule build (replaces minified bundle)

// ── OntologyDAG ──────────────────────────────────────────────────────────────

export class OntologyDAG {
  conceptsById   = new Map();
  conceptsByCurie = new Map();
  childrenMap    = new Map();
  ancestorsCache = new Map();

  constructor(concepts = []) {
    for (const c of concepts) this.addConcept(c);
  }

  addConcept(c) {
    this.conceptsById.set(c.id, c);
    this.conceptsByCurie.set(c.curie, c);
    if (!this.childrenMap.has(c.id)) this.childrenMap.set(c.id, new Set());
    for (const pid of c.parents) {
      if (!this.childrenMap.has(pid)) this.childrenMap.set(pid, new Set());
      this.childrenMap.get(pid).add(c.id);
    }
    this.ancestorsCache.clear();
  }

  getById(id)    { return this.conceptsById.get(id); }
  getByCurie(cu) { return this.conceptsByCurie.get(cu); }
  getAll()       { return Array.from(this.conceptsById.values()); }
  size()         { return this.conceptsById.size; }

  getAncestors(id) {
    if (this.ancestorsCache.has(id)) return this.ancestorsCache.get(id);
    const seen = new Set(), queue = [id];
    while (queue.length > 0) {
      const cur = queue.shift();
      const c   = this.conceptsById.get(cur);
      if (!c) continue;
      for (const pid of c.parents) {
        if (!seen.has(pid)) { seen.add(pid); queue.push(pid); }
      }
    }
    this.ancestorsCache.set(id, seen);
    return seen;
  }

  subsumes(ancestor, descendant) {
    if (ancestor === descendant) return true;
    return this.getAncestors(descendant).has(ancestor);
  }

  isDisjoint(a, b) {
    if (a === b) return false;
    const ca = this.conceptsById.get(a);
    if (ca && ca.disjointWith.includes(b)) return true;
    const cb = this.conceptsById.get(b);
    if (cb && cb.disjointWith.includes(a)) return true;
    const ancsA = this.getAncestors(a);
    const ancsB = this.getAncestors(b);
    for (const u of ancsA) {
      const cu = this.conceptsById.get(u);
      if (!cu) continue;
      if (cu.disjointWith.includes(b)) return true;
      for (const v of ancsB) if (cu.disjointWith.includes(v)) return true;
    }
    for (const u of ancsB) {
      const cu = this.conceptsById.get(u);
      if (cu && cu.disjointWith.includes(a)) return true;
    }
    return false;
  }

  static fromJSON(raw) {
    const arr = Array.isArray(raw) ? raw : (raw.concepts || []);
    const curieIdx = new Map();
    arr.forEach((c, i) => {
      const curie = c.curie || c.id || `concept_${i}`;
      curieIdx.set(curie, i);
      if (curie.includes(':')) curieIdx.set(curie.split(':')[1], i);
    });
    const concepts = arr.map((c, i) => {
      const curie = c.curie || c.id || `concept_${i}`;
      const resolve = (x) => typeof x === 'number' ? x : curieIdx.get(x);
      const parents     = (c.parents     || []).map(resolve).filter(x => typeof x === 'number');
      const disjointWith= (c.disjoint_with || c.disjointWith || []).map(resolve).filter(x => typeof x === 'number');
      return {
        id: i, curie,
        label:      c.label       || curie,
        description:c.description || '',
        parents, disjointWith,
        synonyms:   c.synonyms    || [],
      };
    });
    return new OntologyDAG(concepts);
  }
}

// ── Constants ─────────────────────────────────────────────────────────────────

export const DISJOINT_THRESHOLD = 0.6;
export const DEFAULT_K = 12;

// ── Utility: tokenize ─────────────────────────────────────────────────────────

const STOPWORDS = new Set([
  'a','an','the','and','or','but','in','on','at','to','for','with','by','about',
  'against','between','into','through','during','before','after','above','below',
  'from','up','down','out','off','over','under','again','further','then','once',
  'here','there','when','where','why','how','all','any','both','each','few',
  'more','most','other','some','such','no','nor','not','only','own','same','so',
  'than','too','very','can','will','just','don','should','now','is','are','was',
  'were','be','been','being','have','has','had','having','do','does','did',
  'doing','of','it','its','that','this','also','made'
]);

export function tokenize(text) {
  const split = text.replace(/([a-z])([A-Z])/g, '$1 $2');
  return split.toLowerCase().replace(/[^a-z0-9]+/g, ' ')
    .split(/\s+/).filter(t => t.length > 2 && !STOPWORDS.has(t));
}

// ── temperatureByCardinality ──────────────────────────────────────────────────

export function temperatureByCardinality(probs) {
  const n = probs.length;
  if (n <= 2)  return 0.85;
  if (n <= 5)  return 0.95;
  if (n <= 10) return 1.1;
  return 1.25;
}

// ── softmax ───────────────────────────────────────────────────────────────────

export function softmax(scores, temp = 1) {
  const n = scores.length;
  if (n === 0) return new Float32Array(0);
  const t    = Math.max(0.0001, temp);
  let   maxS = -Infinity;
  for (let i = 0; i < n; i++) if (scores[i] > maxS) maxS = scores[i];
  const out = new Float32Array(n);
  let   sum = 0;
  for (let i = 0; i < n; i++) { const e = Math.exp((scores[i] - maxS) / t); out[i] = e; sum += e; }
  const res = new Float32Array(n);
  if (sum > 0) for (let i = 0; i < n; i++) res[i] = out[i] / sum;
  return res;
}

// ── normalizedEntropyConfidence ───────────────────────────────────────────────

export function normalizedEntropyConfidence(probs) {
  const n = probs.length;
  if (n <= 1) return 1;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += probs[i];
  if (sum <= 0) return 0;
  let H = 0;
  for (let i = 0; i < n; i++) {
    const p = probs[i] / sum;
    if (p > 1e-12) H -= p * Math.log(p);
  }
  const Hmax = Math.log(n);
  return Math.max(0, Math.min(1, 1 - H / Hmax));
}

// ── enforceSubsumptionMonotonicityInplace ─────────────────────────────────────

export function enforceSubsumptionMonotonicityInplace(probs, parentEdges, maxIter = probs.length + 1) {
  for (let it = 0; it < maxIter; it++) {
    let changed = false;
    for (const [child, parent] of parentEdges) {
      if (child < probs.length && parent < probs.length) {
        if (probs[child] > probs[parent]) { probs[parent] = probs[child]; changed = true; }
      }
    }
    if (!changed) break;
  }
}

// ── enforceDisjointnessInplace ────────────────────────────────────────────────

export function enforceDisjointnessInplace(probs, disjointEdges, threshold = 0.6, parentEdges = []) {
  for (const [a, b] of disjointEdges) {
    if (a < probs.length && b < probs.length && Math.max(probs[a], probs[b]) >= threshold) {
      _zeroSubtree(probs, parentEdges, probs[a] >= probs[b] ? b : a);
    }
  }
}

function _zeroSubtree(probs, parentEdges, root) {
  const visited = new Set([root]);
  const stack   = [root];
  while (stack.length > 0) {
    const cur = stack.pop();
    probs[cur] = 0;
    for (const [child, parent] of parentEdges) {
      if (parent === cur && child < probs.length && !visited.has(child)) {
        visited.add(child); stack.push(child);
      }
    }
  }
}

// ── verifyAxioms ──────────────────────────────────────────────────────────────

export function verifyAxioms(probs, parentEdges, disjointEdges, threshold = 0.6) {
  for (const [child, parent] of parentEdges) {
    if (child < probs.length && parent < probs.length && probs[child] > probs[parent] + 1e-6) return false;
  }
  for (const [a, b] of disjointEdges) {
    if (a < probs.length && b < probs.length &&
        Math.min(probs[a], probs[b]) > 1e-6 &&
        Math.max(probs[a], probs[b]) >= threshold) return false;
  }
  return true;
}

// ── projectedConfidence ───────────────────────────────────────────────────────
// spec/AXIOMS.md step 5: confidence of the calibrated (pre-lift) distribution restricted to the
// slots that survived disjointness exclusion. Lifting copies a child's mass onto every ancestor,
// so measuring entropy after it would report confidence 0 for a certain deep decision.

export function projectedConfidence(calibrated, projected) {
  const masked = new Float32Array(calibrated.length);
  for (let i = 0; i < calibrated.length; i++) masked[i] = projected[i] > 0 ? calibrated[i] : 0;
  return normalizedEntropyConfidence(masked);
}

// ── calibrateAndProject ───────────────────────────────────────────────────────

export function calibrateAndProject(scores, parentEdges, disjointEdges, temp = temperatureByCardinality(scores)) {
  const probs      = softmax(scores, temp);
  const calibrated = Float32Array.from(probs);
  enforceSubsumptionMonotonicityInplace(probs, parentEdges);
  enforceDisjointnessInplace(probs, disjointEdges, 0.6, parentEdges);
  const confidence = projectedConfidence(calibrated, probs);
  return { probs, confidence };
}

// ── CandidateRouter (IDF-weighted BM25) ───────────────────────────────────────

export class CandidateRouter {
  invertedIndex = new Map(); // token → Set<conceptId>
  idfMap        = new Map(); // token → IDF weight

  constructor(ontology) {
    this.ontology = ontology;
    this.buildIndex();
  }

  /**
   * Build inverted index and compute smooth IDF for every token.
   * idf(t) = ln((N + 1) / (df + 1)) + 1    (add-1 / Laplace-smoothed BM25 IDF)
   */
  buildIndex() {
    const concepts = this.ontology.getAll();
    const N        = concepts.length;

    // OWL property expression patterns to strip from descriptions before indexing
    const OWL_PROPS_RE = /\b(derives_from|is_a|part_of|located_in|has_part|participates_in|preceded_by)\b/g;

    // Per-concept token sets (use distinct sets so each concept contributes df=1 per token)
    for (const concept of concepts) {
      const labelToks = tokenize(concept.label);
      const synToks   = concept.synonyms.flatMap(s => tokenize(s));
      // Strip OWL property expressions from description before indexing
      const cleanDesc = concept.description ? concept.description.replace(OWL_PROPS_RE, ' ') : '';
      const descToks  = cleanDesc ? tokenize(cleanDesc) : [];

      // Distinct tokens for this concept (for DF counting)
      const distinctToks = new Set([...labelToks, ...synToks, ...descToks]);

      for (const tok of distinctToks) {
        // Inverted index
        if (!this.invertedIndex.has(tok)) this.invertedIndex.set(tok, new Set());
        this.invertedIndex.get(tok).add(concept.id);
      }
    }

    // Compute IDF for every indexed token
    for (const [tok, conceptSet] of this.invertedIndex) {
      const df  = conceptSet.size;
      const idf = Math.log((N + 1) / (df + 1)) + 1;
      this.idfMap.set(tok, idf);
    }
  }

  /**
   * Route query to top-K candidate concepts and extract the closed active subgraph.
   * Scoring (IDF-weighted BM25):
   *   label match    → score += idf * 4
   *   synonym match  → score += idf * 2
   *   description match → score += idf * 1 (or 2x boost when label has zero query token overlap)
   */
  extractSubgraph(queryText, maxCandidates = 12) {
    const queryTokens = tokenize(queryText);
    const queryTokSet = new Set(queryTokens);
    const scoreMap    = new Map(); // conceptId → accumulated score

    for (const tok of queryTokens) {
      const matches = this.invertedIndex.get(tok);
      if (!matches) continue;

      const idf = this.idfMap.get(tok) || 1.0;

      for (const cid of matches) {
        const concept = this.ontology.getById(cid);
        if (!concept) continue;

        const labelTokens = tokenize(concept.label);
        let weight = 0;

        // Label match
        if (labelTokens.includes(tok)) {
          weight += idf * 4;
        }
        // Synonym match (only if not already counted as label)
        if (weight === 0 || !labelTokens.includes(tok)) {
          const inSyn = concept.synonyms.some(s => tokenize(s).includes(tok));
          if (inSyn) weight += idf * 2;
        }
        // Description match (additive) — with 2x boost when label has zero query token overlap
        if (concept.description && tokenize(concept.description).includes(tok)) {
          // Avoid double-counting if weight is already from label
          if (!labelTokens.includes(tok) &&
              !concept.synonyms.some(s => tokenize(s).includes(tok))) {
            // 2x boost when label has ZERO query token overlap (description is primary signal)
            const labelHit = labelTokens.some(t => queryTokSet.has(t));
            const descBoost = labelHit ? 1.0 : 2.0;
            weight += idf * descBoost;
          }
        }

        // Fallback: token in inverted index but matched via description only
        if (weight === 0) weight = idf * 1;

        scoreMap.set(cid, (scoreMap.get(cid) || 0) + weight);
      }
    }

    // Sort descending by score
    const sortedCandidates = Array.from(scoreMap.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([cid]) => cid);

    // Build active set: top-K candidates + their immediate parents
    const activeSet = new Set();
    for (const cid of sortedCandidates.slice(0, maxCandidates)) {
      activeSet.add(cid);
      const c = this.ontology.getById(cid);
      if (c) for (const pid of c.parents) activeSet.add(pid);
    }

    // Seed with roots if nothing matched
    if (activeSet.size === 0) {
      for (const c of this.ontology.getAll()) {
        if (c.parents.length === 0) {
          activeSet.add(c.id);
          if (activeSet.size >= maxCandidates) break;
        }
      }
    }

    // Fill remaining slots
    for (const c of this.ontology.getAll()) {
      if (activeSet.size >= maxCandidates) break;
      activeSet.add(c.id);
    }

    const concepts = Array.from(activeSet)
      .map(id => this.ontology.getById(id))
      .filter(Boolean);

    const slotMap = new Map();
    concepts.forEach((c, idx) => slotMap.set(c.id, idx));

    const parentEdges  = [];
    const disjointEdges = [];

    for (let i = 0; i < concepts.length; i++) {
      const c = concepts[i];
      for (const pid of c.parents) {
        if (slotMap.has(pid)) parentEdges.push([i, slotMap.get(pid)]);
      }
      for (const did of c.disjointWith) {
        if (slotMap.has(did)) {
          const j = slotMap.get(did);
          if (i < j) disjointEdges.push([i, j]);
        }
      }
    }

    return { concepts, parentEdges, disjointEdges };
  }
}

// ── OLaya Engine ─────────────────────────────────────────────────────────────

export class OLaya {
  constructor(ontology) {
    this.ontology = ontology;
    this.router   = new CandidateRouter(ontology);
  }

  switchOntology(ontology) {
    this.ontology = ontology;
    this.router   = new CandidateRouter(ontology);
  }

  async systemOne(queryText, k = 12) {
    const sub = this.router.extractSubgraph(queryText, k);
    const V   = sub.concepts.length;
    if (V === 0) throw new Error('Active subgraph is empty');

    const queryToks = new Set(tokenize(queryText));
    const rawScores = new Float32Array(V);
    rawScores.fill(1);

    for (let i = 0; i < V; i++) {
      const c = sub.concepts[i];
      const labelTokens = tokenize(c.label);
      const labelHit    = labelTokens.some(t => queryToks.has(t));

      // IDF-weighted label scoring
      for (const tok of labelTokens) {
        if (queryToks.has(tok)) {
          const idf = this.router.idfMap.get(tok) || 1.0;
          rawScores[i] += 6 * idf;
        }
      }

      // Per-synonym scoring: score each synonym as a unit to prevent
      // high-synonym-count concepts from accumulating unfair advantages.
      // A synonym scores: sum-of-IDF-of-matching-tokens × 5 × matchFraction
      // where matchFraction = matchingTokens / totalSynTokens (length normalisation).
      for (const syn of c.synonyms) {
        const synToks = tokenize(syn);
        if (synToks.length === 0) continue;
        let matchIdfSum = 0;
        let matchCount  = 0;
        for (const tok of synToks) {
          if (queryToks.has(tok)) {
            matchIdfSum += this.router.idfMap.get(tok) || 1.0;
            matchCount++;
          }
        }
        if (matchCount === 0) continue;
        // Length-normalised: multiply by fraction of syn tokens that match
        const matchFrac = matchCount / synToks.length;
        rawScores[i] += 5 * matchIdfSum * matchFrac;
      }

      // IDF-weighted description scoring — 2x boost when label has zero query overlap
      if (c.description) {
        const descBoost = labelHit ? 1.0 : 2.0;
        for (const tok of tokenize(c.description)) {
          if (queryToks.has(tok)) {
            const idf = this.router.idfMap.get(tok) || 1.0;
            rawScores[i] += descBoost * idf;
          }
        }
      }
    }

    const { probs, confidence } = calibrateAndProject(rawScores, sub.parentEdges, sub.disjointEdges);

    // Select best concept (prefer deeper descendant on tie)
    let bestIdx = 0, bestProb = -Infinity;
    for (let i = 0; i < V; i++) {
      const p = probs[i];
      if (p > bestProb) {
        bestProb = p; bestIdx = i;
      } else if (Math.abs(p - bestProb) < 1e-5) {
        const curId  = sub.concepts[i].id;
        const bestId = sub.concepts[bestIdx].id;
        if (this.ontology.subsumes(bestId, curId)) bestIdx = i;
      }
    }

    const probabilities = sub.concepts.map((c, i) => ({ curie: c.curie, label: c.label, prob: probs[i] }));

    return {
      selectedConcept:  sub.concepts[bestIdx],
      confidence,
      probabilities,
      axiomsVerified:   verifyAxioms(probs, sub.parentEdges, sub.disjointEdges),
    };
  }

  async subsumes(ancestorCurie, descendantCurie, queryText) {
    const a = this.ontology.getByCurie(ancestorCurie);
    const d = this.ontology.getByCurie(descendantCurie);
    if (!a || !d) return false;
    if (!this.ontology.subsumes(a.id, d.id)) return false;
    const result = await this.systemOne(queryText, 8);
    return result.selectedConcept.id === d.id || result.selectedConcept.id === a.id;
  }

  async choice(query, options, description) {
    const concepts = options.map((label, i) => ({
      id: i, curie: `OPT:${i}`, label,
      parents: [], disjointWith: [], synonyms: [], description: description || '',
    }));
    const combined = description ? `${query} ${description}` : query;
    const dag      = new OntologyDAG(concepts);
    const result   = await new OLaya(dag).systemOne(combined, options.length);
    const probs    = {};
    for (const p of result.probabilities) probs[p.label] = p.prob;
    return { choice: result.selectedConcept.label, confidence: result.confidence, probabilities: probs };
  }

  async score(query, criterion) {
    const qToks  = new Set(tokenize(query));
    const cToks  = tokenize(criterion);
    if (cToks.length === 0) return 0.5;
    let matches = 0;
    for (const t of cToks) if (qToks.has(t)) matches++;
    const logit = (matches / cToks.length - 0.3) * 6;
    return 1 / (1 + Math.exp(-logit));
  }

  async noul(query, options, threshold = 0.25) {
    const result = await this.choice(query, options);
    if (result.confidence < threshold) return { accepted: false, rejectionConfidence: 1 - result.confidence };
    return { accepted: true, selected: result.choice, rejectionConfidence: 1 - result.confidence };
  }
}
