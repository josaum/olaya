/**
 * Candidate routing and active subgraph extraction for O-Laya.
 *
 * Upgraded to use smooth IDF-weighted BM25 scoring with OWL property stripping,
 * description-first fallback, and default k=12.
 */

import { Concept, OntologyDAG } from "./ontology.js";

const STOPWORDS = new Set([
  "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
  "by", "about", "against", "between", "into", "through", "during", "before",
  "after", "above", "below", "from", "up", "down", "out", "off", "over", "under",
  "again", "further", "then", "once", "here", "there", "when", "where", "why",
  "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
  "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
  "can", "will", "just", "don", "should", "now", "is", "are", "was",
  "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
  "did", "doing", "of", "it", "its", "that", "this", "also", "made"
]);

/** OWL object property expressions to strip from descriptions before indexing */
const OWL_PROPS_RE = /\b(derives_from|is_a|part_of|located_in|has_part|participates_in|preceded_by|regulates|negatively_regulates|positively_regulates|has_quality|inheres_in|bearer_of|realized_in|causally_upstream_of|part_of_developmental_process)\b/g;

export function tokenize(text: string): string[] {
  // Split camelCase: "DatabaseDeadlock" -> "Database Deadlock"
  const splitCamel = text.replace(/([a-z])([A-Z])/g, "$1 $2");
  const clean = splitCamel.toLowerCase().replace(/[^a-z0-9]+/g, " ");
  return clean.split(/\s+/).filter((t) => t.length > 2 && !STOPWORDS.has(t));
}

function stripOwlProps(text: string): string {
  return text.replace(OWL_PROPS_RE, " ").replace(/\s+/g, " ").trim();
}

export interface ActiveSubgraph {
  concepts: Concept[];
  parentEdges: [number, number][]; // [childSlot, parentSlot]
  disjointEdges: [number, number][]; // [slotA, slotB]
}

export const DEFAULT_K = 12;

export class CandidateRouter {
  private invertedIndex = new Map<string, Set<number>>();
  /** Public IDF map — used by engine.ts for direct scoring */
  public idfMap = new Map<string, number>();
  /** Per-concept token sets by field (label, synonyms, description) */
  private conceptTokens = new Map<
    number,
    { label: Set<string>; synonyms: Set<string>; description: Set<string> }
  >();

  constructor(private ontology: OntologyDAG) {
    this.buildIndex();
  }

  private buildIndex(): void {
    const allConcepts = this.ontology.getAll();
    const N = allConcepts.length;

    // Pass 1: collect token sets per field and build inverted index
    const docFreq = new Map<string, number>();

    for (const concept of allConcepts) {
      const labelToks = new Set(tokenize(concept.label));
      const synToks = new Set(concept.synonyms.flatMap((s) => tokenize(s)));
      const descToks = new Set(
        tokenize(stripOwlProps(concept.description ?? ""))
      );

      this.conceptTokens.set(concept.id, {
        label: labelToks,
        synonyms: synToks,
        description: descToks,
      });

      // All distinct tokens for this concept (for DF counting and inverted index)
      const allToks = new Set([...labelToks, ...synToks, ...descToks]);
      for (const tok of allToks) {
        docFreq.set(tok, (docFreq.get(tok) ?? 0) + 1);
        if (!this.invertedIndex.has(tok)) {
          this.invertedIndex.set(tok, new Set());
        }
        this.invertedIndex.get(tok)!.add(concept.id);
      }
    }

    // Pass 2: compute smooth IDF for every indexed token
    for (const [tok, df] of docFreq) {
      this.idfMap.set(tok, Math.log((N + 1) / (df + 1)) + 1);
    }
  }

  /**
   * Routes query to top-K candidate concepts using IDF-weighted BM25 scoring.
   * Extracts the closed active subgraph including all ancestors.
   */
  public extractSubgraph(query: string, maxCandidates = DEFAULT_K): ActiveSubgraph {
    const queryTokens = tokenize(query);
    const queryTokSet = new Set(queryTokens);
    const scoreMap = new Map<number, number>();

    for (const tok of queryTokens) {
      const idf = this.idfMap.get(tok) ?? 1.0;
      const matches = this.invertedIndex.get(tok);
      if (!matches) continue;

      for (const cid of matches) {
        const fields = this.conceptTokens.get(cid);
        if (!fields) continue;

        const labelHit = fields.label.has(tok);
        const synHit = fields.synonyms.has(tok);
        const descHit = fields.description.has(tok);

        // Description boost: if label has no query token overlap, description becomes primary
        const labelOverlap = [...fields.label].some((t) => queryTokSet.has(t));
        const descBoost = labelOverlap ? 1.0 : 2.0;

        const score =
          (labelHit ? idf * 4.0 : 0) +
          (synHit ? idf * 2.0 : 0) +
          (descHit ? idf * 1.0 * descBoost : 0);

        if (score > 0) {
          scoreMap.set(cid, (scoreMap.get(cid) ?? 0) + score);
        }
      }
    }

    // Sort by score descending
    const sortedCandidates = Array.from(scoreMap.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([cid]) => cid);

    // Build active set: top-k leaves + all their ancestors
    const activeSet = new Set<number>();
    for (const cid of sortedCandidates.slice(0, maxCandidates)) {
      activeSet.add(cid);
      // Include full ancestor chain for monotonicity edges
      let queue = [cid];
      while (queue.length > 0) {
        const cur = queue.shift()!;
        const concept = this.ontology.getById(cur);
        if (concept) {
          for (const pid of concept.parents) {
            if (!activeSet.has(pid)) {
              activeSet.add(pid);
              queue.push(pid);
            }
          }
        }
      }
    }

    // Fallback: seed with root concepts if nothing matched
    if (activeSet.size === 0) {
      for (const c of this.ontology.getAll()) {
        if (c.parents.length === 0) {
          activeSet.add(c.id);
          if (activeSet.size >= maxCandidates) break;
        }
      }
    }

    const concepts = Array.from(activeSet)
      .map((id) => this.ontology.getById(id)!)
      .filter(Boolean);

    // Map conceptId -> slotIndex in subgraph
    const slotMap = new Map<number, number>();
    concepts.forEach((c, idx) => slotMap.set(c.id, idx));

    const parentEdges: [number, number][] = [];
    const disjointEdges: [number, number][] = [];

    for (let i = 0; i < concepts.length; i++) {
      const c = concepts[i];
      for (const pid of c.parents) {
        if (slotMap.has(pid)) {
          parentEdges.push([i, slotMap.get(pid)!]);
        }
      }
      for (const did of c.disjointWith) {
        if (slotMap.has(did)) {
          const j = slotMap.get(did)!;
          if (i < j) {
            disjointEdges.push([i, j]);
          }
        }
      }
    }

    return { concepts, parentEdges, disjointEdges };
  }
}
