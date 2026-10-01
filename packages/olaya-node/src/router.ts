/**
 * Candidate routing and active subgraph extraction for O-Laya.
 */

import { Concept, OntologyDAG } from "./ontology.js";

export function tokenize(text: string): string[] {
  // Split camelCase: "DatabaseDeadlock" -> "Database Deadlock"
  const splitCamel = text.replace(/([a-z])([A-Z])/g, "$1 $2");
  const clean = splitCamel.toLowerCase().replace(/[^a-z0-9]+/g, " ");
  return clean.split(/\s+/).filter((t) => t.length > 0);
}

export interface ActiveSubgraph {
  concepts: Concept[];
  parentEdges: [number, number][]; // [childSlot, parentSlot]
  disjointEdges: [number, number][]; // [slotA, slotB]
}

export class CandidateRouter {
  private invertedIndex = new Map<string, Set<number>>();

  constructor(private ontology: OntologyDAG) {
    this.buildIndex();
  }

  private buildIndex(): void {
    for (const concept of this.ontology.getAll()) {
      const tokens = new Set<string>([
        ...tokenize(concept.label),
        ...concept.synonyms.flatMap((s) => tokenize(s)),
        ...(concept.description ? tokenize(concept.description) : []),
      ]);

      for (const tok of tokens) {
        if (!this.invertedIndex.has(tok)) {
          this.invertedIndex.set(tok, new Set());
        }
        this.invertedIndex.get(tok)!.add(concept.id);
      }
    }
  }

  /**
   * Routes query to top-K candidate concepts and extracts the closed active subgraph.
   */
  public extractSubgraph(query: string, maxCandidates = 8): ActiveSubgraph {
    const queryTokens = tokenize(query);
    const scoreMap = new Map<number, number>();

    for (const tok of queryTokens) {
      const matches = this.invertedIndex.get(tok);
      if (matches) {
        for (const cid of matches) {
          scoreMap.set(cid, (scoreMap.get(cid) || 0) + 1);
        }
      }
    }

    // Sort by score descending
    const sortedCandidates = Array.from(scoreMap.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([cid]) => cid);

    const activeSet = new Set<number>();
    for (const cid of sortedCandidates.slice(0, maxCandidates)) {
      activeSet.add(cid);
      // Include immediate parents to ground subsumption
      const concept = this.ontology.getById(cid);
      if (concept) {
        for (const pid of concept.parents) {
          activeSet.add(pid);
        }
      }
    }

    // Fill remaining slots with peers / disjoint concepts up to maxCandidates
    for (const c of this.ontology.getAll()) {
      if (activeSet.size >= maxCandidates) break;
      activeSet.add(c.id);
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
