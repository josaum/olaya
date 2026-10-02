# System One edge reliability contract

Scope: `docs/public/api/systemone.js`, POST `/api/systemone`.

Choice results retain type, confidence, and probabilities, and add `abstained`.
A unique winner with label-token evidence returns its label and `abstained: false`.
Otherwise `choice` is null, `abstained` is true, and `reason` is one of
`no_options`, `no_matching_evidence`, or `tied_options`. Abstentions have
confidence 0; probabilities remain available as lexical weights.

Noul retains its numeric response. Negation is assessed in the clause containing
proposition evidence, preserving negators discarded by the matching tokenizer.
A directly negated proposition must score below 0.1; its positive form must score
above 0.9. Unrelated negation in another clause or attached to another noun must
not invert the proposition. Contradictory positive and negative mentions return
0.5 with confidence 0. This is a bounded English lexical heuristic, not general
natural language reasoning or an empirically calibrated probability.

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
An unmatched proposition retains the conservative lexical NO default; lack of
matching text is not proof that a proposition is false.

## Verification and release status

On 2026-10-02, the original handler failed 11 of the 15 focused regression tests;
the patch passed all 15. Coverage includes positive and negated propositions,
unrelated negation, contradictory mentions, empty and unsupported options,
ties, unique choices, and the existing score example.

Commit `7989f31` contains the source fix and tests. GitHub push and Vercel
deployment attempts were blocked by connectivity in that session. Confirm a
successful deployment and repeat the live cases before assuming the public
endpoint implements this contract. The scope is this edge handler; Python,
Rust, browser, and Node package primitives are not changed by this patch.
