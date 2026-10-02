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
