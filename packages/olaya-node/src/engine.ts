/**
 * High-performance System 1 Decision Engine for Node.js and TypeScript.
 * Fully compatible with Convai / Receptron's Laya and TypeSafe AI's Jev primitives.
 */

import { calibrateAndProject, verifyAxioms } from "./axioms.js";
import { Concept, OntologyDAG } from "./ontology.js";
import { CandidateRouter, tokenize } from "./router.js";

export interface DecisionResult {
  selectedConcept: Concept;
  confidence: number;
  probabilities: { curie: string; label: string; prob: number }[];
  axiomsVerified: boolean;
}

export interface ChoiceResult {
  choice: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface NoulResult {
  accepted: boolean;
  selected?: string;
  rejectionConfidence: number;
}

export class OLaya {
  private router: CandidateRouter;

  constructor(public ontology: OntologyDAG) {
    this.router = new CandidateRouter(ontology);
  }

  /**
   * Switch domain ontology instantaneously in-flight.
   */
  public switchOntology(newOntology: OntologyDAG): void {
    this.ontology = newOntology;
    this.router = new CandidateRouter(newOntology);
  }

  /**
   * System 1 decision evaluation: resolves query state to an ontology concept in < 1 ms.
   */
  public async systemOne(stateText: string, maxCandidates = 8): Promise<DecisionResult> {
    const subgraph = this.router.extractSubgraph(stateText, maxCandidates);
    const m = subgraph.concepts.length;
    if (m === 0) {
      throw new Error("Active subgraph is empty");
    }

    const stateTokens = new Set(tokenize(stateText));
    const logits = new Float32Array(m);
    logits.fill(1.0);

    for (let slot = 0; slot < m; slot++) {
      const concept = subgraph.concepts[slot];
      for (const tok of tokenize(concept.label)) {
        if (stateTokens.has(tok)) logits[slot] += 6.0;
      }
      for (const syn of concept.synonyms) {
        for (const tok of tokenize(syn)) {
          if (stateTokens.has(tok)) logits[slot] += 5.0;
        }
      }
      if (concept.description) {
        for (const tok of tokenize(concept.description)) {
          if (stateTokens.has(tok)) logits[slot] += 1.0;
        }
      }
    }

    const { probs, confidence } = calibrateAndProject(
      logits,
      subgraph.parentEdges,
      subgraph.disjointEdges
    );

    // Argmax selection breaking ties in favor of more specific leaf concepts
    let bestSlot = 0;
    let bestScore = -Infinity;

    for (let slot = 0; slot < m; slot++) {
      const p = probs[slot];
      if (p > bestScore) {
        bestScore = p;
        bestSlot = slot;
      } else if (Math.abs(p - bestScore) < 1e-5) {
        const currId = subgraph.concepts[slot].id;
        const bestId = subgraph.concepts[bestSlot].id;
        if (this.ontology.subsumes(bestId, currId)) {
          bestSlot = slot;
        }
      }
    }

    const probabilities = subgraph.concepts.map((c, idx) => ({
      curie: c.curie,
      label: c.label,
      prob: probs[idx],
    }));

    return {
      selectedConcept: subgraph.concepts[bestSlot],
      confidence,
      probabilities,
      axiomsVerified: verifyAxioms(probs, subgraph.parentEdges, subgraph.disjointEdges),
    };
  }

  /**
   * Evaluates if state entails child subsumption under parent.
   */
  public async subsumes(parentCurie: string, childCurie: string, stateText: string): Promise<boolean> {
    const parent = this.ontology.getByCurie(parentCurie);
    const child = this.ontology.getByCurie(childCurie);
    if (!parent || !child) return false;

    // Ontological axiom: child ⊑ parent must hold structurally
    if (!this.ontology.subsumes(parent.id, child.id)) return false;

    const res = await this.systemOne(stateText, 8);
    const selected = res.selectedConcept;
    return selected.id === child.id || selected.id === parent.id;
  }

  /**
   * Laya / Jev primitive: discrete choice over options.
   */
  public async choice(stateText: string, options: string[], instructions?: string): Promise<ChoiceResult> {
    const tempConcepts: Concept[] = options.map((opt, idx) => ({
      id: idx,
      curie: `OPT:${idx}`,
      label: opt,
      parents: [],
      disjointWith: [],
      synonyms: [],
      description: instructions,
    }));

    const query = instructions ? `${stateText} ${instructions}` : stateText;
    const tempDAG = new OntologyDAG(tempConcepts);
    const tempEngine = new OLaya(tempDAG);
    const res = await tempEngine.systemOne(query, options.length);

    const probabilities: Record<string, number> = {};
    for (const p of res.probabilities) {
      probabilities[p.label] = p.prob;
    }

    return {
      choice: res.selectedConcept.label,
      confidence: res.confidence,
      probabilities,
    };
  }

  /**
   * Laya / Jev primitive: scalar scoring against a criterion.
   */
  public async score(stateText: string, criterion: string): Promise<number> {
    const stateTokens = new Set(tokenize(stateText));
    const critTokens = tokenize(criterion);
    if (critTokens.length === 0) return 0.5;

    let matches = 0;
    for (const tok of critTokens) {
      if (stateTokens.has(tok)) matches++;
    }

    const matchRatio = matches / critTokens.length;
    const logit = (matchRatio - 0.3) * 6.0;
    return 1 / (1 + Math.exp(-logit));
  }

  /**
   * Laya / Jev primitive: null / out-of-domain rejection.
   */
  public async noul(stateText: string, options: string[], confidenceThreshold = 0.25): Promise<NoulResult> {
    const res = await this.choice(stateText, options);
    if (res.confidence < confidenceThreshold) {
      return {
        accepted: false,
        rejectionConfidence: 1.0 - res.confidence,
      };
    }
    return {
      accepted: true,
      selected: res.choice,
      rejectionConfidence: 1.0 - res.confidence,
    };
  }
}
