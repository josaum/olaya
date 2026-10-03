# System One edge reliability contract

Scope: `docs/public/api/systemone.js`, POST `/api/systemone`.

Choice results retain type, confidence, and probabilities, and add `abstained`.
A unique winner with label-token evidence returns its label and `abstained: false`.
Otherwise `choice` is null, `abstained` is true, and `reason` is one of
`no_options`, `no_matching_evidence`, or `tied_options`. Abstentions have
confidence 0; probabilities remain available as lexical weights.

Noul retains its numeric response and adds `abstained` and `evidence_status`.
Negation is assessed near matching proposition text, preserving negators
discarded by the matching tokenizer. The supported positive and directly
negated forms score above 0.9 and below 0.1 respectively. These are fixed lexical
scores, not empirically calibrated probabilities.

| `evidence_status` | Meaning | `abstained` |
|---|---|---|
| `supported` | A positive proposition mention matches | `false` |
| `contradicted` | A directly negated proposition mention matches | `false` |
| `unmatched` | No proposition mention matches | `true` |
| `conflicting` | Positive and negative mentions both match | `true` |
| `ambiguous_scope` | Negation and coordination make scope unsupported | `true` |
| `unsupported_question` | Proposition wording itself contains a recognized negator | `true` |

All abstentions return `noul: 0.5` and `confidence: 0`. An unmatched proposition
is unknown rather than false. Consumers must check `abstained` and
`evidence_status` before interpreting the numeric result.

For `noul`, structured state is traversed for string values only, including nested
objects and arrays. Object keys, booleans, numbers, and null are not proposition
evidence. Each string value is evaluated separately: tokens cannot form a
proposition across fields, and a negator in one field does not negate another.
Choice and score also use only string values, excluding keys and scalar
booleans/numbers. They pool tokens across values for lexical option matching;
`noul` requires a complete proposition mention within one string value.

Verification: `node --test tests/systemone-edge.test.mjs`.

## Request and response example

```json
{
  "state": "Office stationery delivery.",
  "questions": {
    "storage": {
      "type": "choice",
      "options": ["cold storage", "ambient storage"]
    }
  }
}
```

The `questions.storage` response is:

```json
{
  "type": "choice",
  "choice": null,
  "abstained": true,
  "reason": "no_matching_evidence",
  "confidence": 0,
  "probabilities": {"cold storage": 0.5, "ambient storage": 0.5}
}
```

Consumers must branch on `abstained` or a null `choice` before taking action.
This is a compatibility change for clients that assumed every nonempty option
list yielded a string. Probabilities are lexical weights, and a non-null answer
is not a correctness guarantee. Object-form options use their keys as labels;
values are not scored by this handler. `score` is the expected zero-based index
of its ordered criteria; it is not a normalized probability.

## Negation limits

Matching requires the proposition's content tokens in order within a clause.
The heuristic handles direct English prefixes such as `not`, `no`, `without`,
contractions ending in `n't`, and suffixes such as `is absent`. Unrelated
negators do not apply globally. This does not establish support for arbitrary
paraphrases, negated question wording, multilingual negation, or complex scope.
Coordinated negation such as “not chilled or frozen seafood” abstains rather
than choosing a scope. Negated instructions also abstain. This remains a bounded
English lexical heuristic and can miss paraphrases, double negation, and scope
that the recognized patterns do not detect. Forms such as “does not contain,” “no evidence of,” “may be absent,” and
“is not necessarily absent” also abstain where recognized, rather than asserting
a positive or negative interpretation. A supported result is not a general
truth or accuracy guarantee.

## Verification and release status

On 2026-10-02, the original handler failed 11 of the 15 focused regression tests;
the initial patch passed all 15. That historical baseline does not cover the
subsequent contract expansion. The current focused suite contains 34 tests,
including structured-state boundaries, unmatched evidence, unsupported question
wording, coordinated negation, internal negation, double negation, and
unsupported negative suffixes, verb-separated negation, and qualified absence. Passing this suite is not general reliability
or calibration evidence. Original coverage includes positive and negated propositions,
unrelated negation, contradictory mentions, empty and unsupported options,
ties, unique choices, and the existing score example.

Commit `7989f31` contains the source fix and tests. GitHub push and Vercel
deployment attempts were blocked by connectivity in that session. Confirm a
successful deployment and repeat the live cases before assuming the public
endpoint implements this contract. The scope is this edge handler; Python,
Rust, browser, and Node package primitives are not changed by this patch.

## Reproducible local handler benchmark

Run [bench_systemone_edge.mjs](../benchmarks/bench_systemone_edge.mjs) from the
repository root:

```bash
node benchmarks/bench_systemone_edge.mjs
```

Inspect its recorded Node version, workloads, warm-up, sample counts, and repeated
runs before comparing results. This measures local handler processing, including
request construction/parsing and response decoding, rather than Vercel or network
latency. The earlier 2,000-sample comparison was an ad hoc historical run against
the initial patch: original/patched median 0.0274/0.0317 ms and p95
0.0377/0.0420 ms. Those numbers do not describe the expanded handler.

### Subsequent production verification

On 2026-10-03 UTC, deployment `dpl_3yrTp5w3afrnmyamw7E8XFj8zqtK` reached READY,
with production alias `https://public-nine-jade-58.vercel.app`. All 34 focused
cases passed against that endpoint using:

```bash
SYSTEMONE_TEST_URL=https://public-nine-jade-58.vercel.app/api/systemone node --test tests/systemone-edge.test.mjs
```

This supersedes the earlier connectivity blocker. It verifies these cases only,
not arbitrary input semantics or model calibration. PR #2 remains unmerged with
passing edge/Node CI and failing Rust formatting/Python lint on unchanged source.
