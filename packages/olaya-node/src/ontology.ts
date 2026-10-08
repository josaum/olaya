/**
 * Ontology DAG and concept definitions for O-Laya.
 */

export interface Concept {
  id: number;
  curie: string;
  label: string;
  description?: string;
  parents: number[];
  disjointWith: number[];
  synonyms: string[];
}

export class OntologyDAG {
  private conceptsById = new Map<number, Concept>();
  private conceptsByCurie = new Map<string, Concept>();
  private childrenMap = new Map<number, Set<number>>();
  private ancestorsCache = new Map<number, Set<number>>();

  constructor(concepts: Concept[] = []) {
    for (const c of concepts) {
      this.addConcept(c);
    }
  }

  public addConcept(concept: Concept): void {
    this.conceptsById.set(concept.id, concept);
    this.conceptsByCurie.set(concept.curie, concept);

    if (!this.childrenMap.has(concept.id)) {
      this.childrenMap.set(concept.id, new Set());
    }

    for (const parentId of concept.parents) {
      if (!this.childrenMap.has(parentId)) {
        this.childrenMap.set(parentId, new Set());
      }
      this.childrenMap.get(parentId)!.add(concept.id);
    }

    this.ancestorsCache.clear();
  }

  public getById(id: number): Concept | undefined {
    return this.conceptsById.get(id);
  }

  public getByCurie(curie: string): Concept | undefined {
    return this.conceptsByCurie.get(curie);
  }

  public getAll(): Concept[] {
    return Array.from(this.conceptsById.values());
  }

  public size(): number {
    return this.conceptsById.size;
  }

  /**
   * Returns all transitive ancestors of a given concept ID.
   */
  public getAncestors(id: number): Set<number> {
    if (this.ancestorsCache.has(id)) {
      return this.ancestorsCache.get(id)!;
    }

    const ancestors = new Set<number>();
    const queue: number[] = [id];

    while (queue.length > 0) {
      const curr = queue.shift()!;
      const concept = this.conceptsById.get(curr);
      if (!concept) continue;

      for (const p of concept.parents) {
        if (!ancestors.has(p)) {
          ancestors.add(p);
          queue.push(p);
        }
      }
    }

    this.ancestorsCache.set(id, ancestors);
    return ancestors;
  }

  /**
   * Transitive subsumption check: is child a subClassOf parent? (child ⊑ parent)
   */
  public subsumes(parentId: number, childId: number): boolean {
    if (parentId === childId) return true;
    return this.getAncestors(childId).has(parentId);
  }

  /**
   * Checks whether two concepts are declared disjoint.
   */
  public isDisjoint(aId: number, bId: number): boolean {
    if (aId === bId) return false;
    const a = this.conceptsById.get(aId);
    if (a && a.disjointWith.includes(bId)) return true;

    const b = this.conceptsById.get(bId);
    if (b && b.disjointWith.includes(aId)) return true;

    // Disjointness propagates down the hierarchy
    const aAncestors = this.getAncestors(aId);
    const bAncestors = this.getAncestors(bId);

    // Check direct disjointness with ancestors
    for (const ancA of aAncestors) {
      const ancConcept = this.conceptsById.get(ancA);
      if (ancConcept) {
        if (ancConcept.disjointWith.includes(bId)) return true;
        for (const ancB of bAncestors) {
          if (ancConcept.disjointWith.includes(ancB)) return true;
        }
      }
    }

    for (const ancB of bAncestors) {
      const ancConcept = this.conceptsById.get(ancB);
      if (ancConcept && ancConcept.disjointWith.includes(aId)) return true;
    }

    return false;
  }

  /**
   * Load any domain ontology from standard JSON format:
   * { name?: string, concepts: [{ curie, label, description?, parents?, disjoint_with?, synonyms? }] }
   */
  public static fromJSON(jsonData: any): OntologyDAG {
    const rawConcepts = Array.isArray(jsonData) ? jsonData : (jsonData.concepts || []);
    const curieToId = new Map<string, number>();

    // Local names are a convenience for single-prefix files. In a merged
    // ontology two prefixes can share a local name, so a stripped name is
    // only registered when exactly one concept carries it and it does not
    // collide with a full curie.
    const localCounts = new Map<string, number>();
    rawConcepts.forEach((c: any, idx: number) => {
      const curie = c.curie || c.id || `concept_${idx}`;
      curieToId.set(curie, idx);
      const colon = curie.indexOf(":");
      if (colon > 0) {
        const local = curie.slice(colon + 1);
        localCounts.set(local, (localCounts.get(local) ?? 0) + 1);
      }
    });
    rawConcepts.forEach((c: any, idx: number) => {
      const curie = c.curie || c.id || `concept_${idx}`;
      const colon = curie.indexOf(":");
      if (colon > 0) {
        const local = curie.slice(colon + 1);
        if (localCounts.get(local) === 1 && !curieToId.has(local)) {
          curieToId.set(local, idx);
        }
      }
    });

    const parsedConcepts: Concept[] = rawConcepts.map((c: any, idx: number) => {
      const curie = c.curie || c.id || `concept_${idx}`;
      const parentsRaw = c.parents || [];
      const parents: number[] = parentsRaw
        .map((p: any) => typeof p === "number" ? p : curieToId.get(p))
        .filter((p: any) => typeof p === "number");

      const disjointRaw = c.disjoint_with || c.disjointWith || [];
      const disjointWith: number[] = disjointRaw
        .map((d: any) => typeof d === "number" ? d : curieToId.get(d))
        .filter((d: any) => typeof d === "number");

      return {
        id: idx,
        curie,
        label: c.label || curie,
        description: c.description || "",
        parents,
        disjointWith,
        synonyms: c.synonyms || []
      };
    });

    return new OntologyDAG(parsedConcepts);
  }
}

