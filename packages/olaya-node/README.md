# @mycelia/olaya

Ontology-Grounded System 1 Decision Model runtime for Node.js and TypeScript.
Non-autoregressive lexical inference engine exposing structured decision primitives.
The package runtime and the HTTP edge demo have distinct response contracts.

## Features

- **System 1 Primitives**: `systemOne`, `concept`, `subsumes`, `choice`, `score`, `noul`
- **Ontological Consistency**: Transitive subsumption monotonicity ($A \sqsubseteq B$) and disjointness exclusion ($A \sqcap B = \bot$) projected in-place
- **Entropy Confidence**: Normalized Shannon entropy confidence scoring
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

## HTTP edge API boundary

The separately deployed [edge handler](../../docs/public/api/systemone.js) uses
`state` and a named `questions` object. Edge `choice` accepts `options`; edge
`score` accepts ordered `criteria`; edge `noul` takes a proposition in
`instructions`. Edge choices can return `choice: null` with `abstained: true` and
`reason` (`no_options`, `no_matching_evidence`, or `tied_options`). These fields
are not changes to this package's `OLaya.choice` return type.

The edge `noul` result is a lexical proposition score, unlike this package's
out-of-domain rejection primitive. Its English negation heuristic and abstention
contract are documented in [the edge specification](../../spec/systemone-reliability.md).
The edge handler does not run this package's ontology projection kernel.
Run its focused tests from the repository root with
`node --test tests/systemone-edge.test.mjs`. Local verification does not establish
that a published package or deployment contains the change.
