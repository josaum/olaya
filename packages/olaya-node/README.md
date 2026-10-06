# olaya

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
import { Concept, OntologyDAG, OLaya } from "olaya";

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

## Rating on a scale

```typescript
// Ordered level labels, lowest first. Each level's probability is its lexical overlap with the
// state; a state matching no label gives a uniform distribution and confidence 0.
const rating = await olaya.rate("Critical outage, customers cannot pay", ["low", "medium", "critical"]);
// rating.level         1-based index of the selected label (3 here)
// rating.score         expected position on the scale in [0, 1]
// rating.confidence    normalised-entropy confidence of the level distribution
// rating.probabilities probability per label, summing to 1
```

## MCP server and HTTP server

`npx olaya` starts a stdio Model Context Protocol server (protocol revisions
`2025-03-26` and `2024-11-05`) exposing four tools, `olaya_ground`, `olaya_subsumes`,
`olaya_choice` and `olaya_score`, and one `ontology://<domain>` resource per bundled ontology.
The bundled domains are `incident` and `gs1` (hand-written 5-class samples), `bfo`, `doid`,
`schemaorg` and `skos` (JSON under `src/ontologies/`, embedded into `dist/` at build time, so they
resolve the same way from any working directory); `domain` may also be a path to an ontology JSON
file. Unknown domains are an error, not a fallback.

`npx olaya serve --port 8080` additionally serves `POST /v1/systemone` (`choice`,
`score` with ordered `criteria`, and `noul` with a proposition in `instructions`), `POST /v1/mcp`
(MCP JSON-RPC over plain HTTP POST: one request, one JSON response, no SSE or sessions) and
`GET /v1/health`. The server has no authentication, metering or billing.

All decisions are lexical: token overlap between the state and class labels, synonyms and
descriptions, calibrated and projected by the kernel in `spec/AXIOMS.md`. Confidence is the
normalised-entropy confidence of that calibrated distribution; it says how peaked the lexical
evidence is, not how often the engine is right.

## HTTP edge API boundary

The edge `noul` contract now distinguishes supported/contradicted evidence from
`unmatched`, `conflicting`, `ambiguous_scope`, and `unsupported_question`.
Unknown cases return `abstained: true`, `noul: 0.5`, and confidence `0`.
Structured-state evidence uses separate string values, excluding keys and scalar
booleans/numbers, with no proposition matches spanning fields. This edge-only
behavior does not change the runtime/package/parser APIs described here.


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
