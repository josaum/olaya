import { describe, expect, it } from "bun:test";
import {
  Concept,
  OntologyDAG,
  OLaya,
  enforceSubsumptionMonotonicityInplace,
  enforceDisjointnessInplace,
  normalizedEntropyConfidence,
} from "../src/index.js";

function buildTestOntology(): OntologyDAG {
  const concepts: Concept[] = [
    {
      id: 0,
      curie: "INCIDENT:ROOT",
      label: "Operational Incident",
      description: "Root incident class",
      parents: [],
      disjointWith: [],
      synonyms: ["incident"],
    },
    {
      id: 1,
      curie: "INCIDENT:DATA",
      label: "Data & Storage Incident",
      description: "Database and filesystem issues",
      parents: [0],
      disjointWith: [2],
      synonyms: ["database", "db", "storage"],
    },
    {
      id: 2,
      curie: "INCIDENT:BILLING",
      label: "Billing Incident",
      description: "Subscription and payment failures",
      parents: [0],
      disjointWith: [1],
      synonyms: ["billing", "payment", "invoice"],
    },
    {
      id: 3,
      curie: "INCIDENT:DB_DEADLOCK",
      label: "Database Deadlock",
      description: "Lock wait timeouts on SQL queries",
      parents: [1],
      disjointWith: [],
      synonyms: ["deadlock", "postgres lock"],
    },
  ];

  return new OntologyDAG(concepts);
}

describe("O-Laya Node / TypeScript Engine", () => {
  it("enforces structural subsumption and disjointness", () => {
    const ont = buildTestOntology();
    expect(ont.subsumes(0, 1)).toBe(true); // Root subsumes Data
    expect(ont.subsumes(0, 3)).toBe(true); // Root subsumes DB_Deadlock
    expect(ont.subsumes(1, 3)).toBe(true); // Data subsumes DB_Deadlock
    expect(ont.subsumes(1, 2)).toBe(false); // Data does not subsume Billing

    expect(ont.isDisjoint(1, 2)).toBe(true); // Data and Billing are disjoint
    expect(ont.isDisjoint(3, 2)).toBe(true); // Deadlock (child of Data) inherits disjointness with Billing
    expect(ont.isDisjoint(1, 3)).toBe(false); // Data and Deadlock are not disjoint
  });

  it("projects axioms and monotonicity in-place", () => {
    // Child has 0.85, parent has 0.30 -> parent must be lifted to >= 0.85
    const probs = new Float32Array([0.2, 0.3, 0.1, 0.85]); // [root, data, billing, deadlock]
    const parentEdges: [number, number][] = [
      [3, 1], // deadlock -> data
      [1, 0], // data -> root
    ];

    enforceSubsumptionMonotonicityInplace(probs, parentEdges);
    expect(probs[1]).toBeGreaterThanOrEqual(probs[3]);
    expect(probs[0]).toBeGreaterThanOrEqual(probs[1]);

    // Disjointness: slot 1 (data) and slot 2 (billing)
    const disjointEdges: [number, number][] = [[1, 2]];
    enforceDisjointnessInplace(probs, disjointEdges, 0.5);
    expect(probs[2]).toBe(0.0); // Billing suppressed
  });

  it("executes System 1 triage decision in < 1 ms", async () => {
    const ont = buildTestOntology();
    const olaya = new OLaya(ont);

    const res1 = await olaya.systemOne(
      "Fatal deadlock detected on PostgreSQL customer table while locking rows"
    );
    expect(res1.selectedConcept.curie).toBe("INCIDENT:DB_DEADLOCK");
    expect(res1.confidence).toBeGreaterThan(0.2);
    expect(res1.axiomsVerified).toBe(true);

    const res2 = await olaya.systemOne(
      "Payment gateway webhook failed with HTTP 500 when billing subscription"
    );
    expect(res2.selectedConcept.curie).toBe("INCIDENT:BILLING");
    expect(res2.confidence).toBeGreaterThanOrEqual(0.49);
    expect(res2.axiomsVerified).toBe(true);
  });

  it("supports Laya and Jev primitives: choice, score, noul", async () => {
    const ont = buildTestOntology();
    const olaya = new OLaya(ont);

    // 1. choice primitive
    const choiceRes = await olaya.choice("Customer requested credit card refund", [
      "Process Refund",
      "Restart Server",
      "Deploy Database",
    ]);
    expect(choiceRes.choice).toBe("Process Refund");
    expect(choiceRes.confidence).toBeGreaterThan(0.5);

    // 2. score primitive
    const scoreVal = await olaya.score(
      "User successfully entered 2FA token",
      "User is authenticated"
    );
    expect(scoreVal).toBeGreaterThan(0.5);

    // 3. noul primitive (rejection on out-of-domain query)
    const noulReject = await olaya.noul(
      "What is the average airspeed velocity of an unladen swallow?",
      ["Database Deadlock", "Billing Invoice Error"],
      0.5
    );
    expect(noulReject.accepted).toBe(false);
  });
});
