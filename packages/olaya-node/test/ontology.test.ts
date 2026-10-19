import { describe, expect, it } from "bun:test";
import { OntologyDAG } from "../src/index.js";

describe("OntologyDAG.fromJSON local-name resolution", () => {
  it("resolves parents given as bare local names when the name is unique", () => {
    const dag = OntologyDAG.fromJSON({
      concepts: [
        { curie: "ex:Root", label: "Root", parents: [] },
        { curie: "ex:Child", label: "Child", parents: ["Root"] },
      ],
    });
    expect(dag.getByCurie("ex:Child")?.parents).toEqual([dag.getByCurie("ex:Root")!.id]);
  });

  it("refuses to guess when two prefixes share a local name", () => {
    // Two modules both define `Brand`; a bare `Brand` reference must not
    // silently pick whichever came last.
    const dag = OntologyDAG.fromJSON({
      concepts: [
        { curie: "com:Brand", label: "Brand", parents: [] },
        { curie: "cmp:Brand", label: "Brand search", parents: [] },
        { curie: "ex:Line", label: "Line", parents: ["Brand"] },
        { curie: "ex:Search", label: "Search", parents: ["cmp:Brand"] },
      ],
    });
    expect(dag.getByCurie("ex:Line")?.parents).toEqual([]);
    expect(dag.getByCurie("ex:Search")?.parents).toEqual([dag.getByCurie("cmp:Brand")!.id]);
  });

  it("never lets a stripped local name shadow a full curie", () => {
    // `module:cb` is a real concept; `x:module:cb`-style stripping must not
    // overwrite its id with another concept's index.
    const dag = OntologyDAG.fromJSON({
      concepts: [
        { curie: "cb", label: "Company baseline module", parents: [] },
        { curie: "module:cb", label: "Module node", parents: [] },
        { curie: "ex:Leaf", label: "Leaf", parents: ["cb"] },
      ],
    });
    expect(dag.getByCurie("ex:Leaf")?.parents).toEqual([dag.getByCurie("cb")!.id]);
  });

  it("keeps disjointness references to full curies and drops unknown ones", () => {
    const dag = OntologyDAG.fromJSON({
      concepts: [
        { curie: "a:X", label: "X", parents: [], disjoint_with: ["a:Y", "nope:Z"] },
        { curie: "a:Y", label: "Y", parents: [], disjointWith: ["a:X"] },
      ],
    });
    expect(dag.isDisjoint(0, 1)).toBe(true);
    expect(dag.getByCurie("a:X")?.disjointWith).toEqual([1]);
  });
});
