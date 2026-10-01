# @mycelia/olaya

Ontology-Grounded System 1 Decision Model runtime for Node.js and TypeScript.
Non-autoregressive, zero-copy inference engine fully compatible with Convai / Receptron's **Laya** and TypeSafe AI's **Jev** primitives.

## Features

- **System 1 Primitives**: `systemOne`, `concept`, `subsumes`, `choice`, `score`, `noul`
- **Ontological Consistency**: Transitive subsumption monotonicity ($A \sqsubseteq B$) and disjointness exclusion ($A \sqcap B = \bot$) projected in-place
- **Calibrated Confidence**: Normalized Shannon entropy confidence scoring
- **Zero Dependencies**: Pure TypeScript runtime with dual ESM + CommonJS exports

## Quick Start

```typescript
import { Concept, OntologyDAG, OLaya } from "@mycelia/olaya";

const ontology = new OntologyDAG([
  {
    id: 0,
    curie: "INCIDENT:DATA",
    label: "Data Incident",
    parents: [],
    disjointWith: [1],
    synonyms: ["database", "storage"],
  },
  {
    id: 1,
    curie: "INCIDENT:BILLING",
    label: "Billing Incident",
    parents: [],
    disjointWith: [0],
    synonyms: ["invoice", "payment"],
  },
]);

const olaya = new OLaya(ontology);

// System 1 decision in < 1 ms:
const decision = await olaya.systemOne("Payment webhook timeout on credit card charge");
console.log(decision.selectedConcept.label); // "Billing Incident"
console.log(decision.confidence);            // ~0.99
```

## Primitives

```typescript
// Discrete choice
const choice = await olaya.choice("Customer requested refund", ["Refund", "Restart", "Ignore"]);

// Criterion scoring
const score = await olaya.score("User entered 2FA token", "User is authenticated");

// Out-of-domain rejection
const { accepted, selected } = await olaya.noul("Out of domain prompt", ["Option A", "Option B"]);
```
