# Axiom projection specification

This is the single definition of the calibrate-and-project kernel that every O-Laya runtime
implements. `axioms_golden.json` in this directory holds test vectors generated from the Python
reference (`scripts/gen_axiom_golden.py`); the Python, Rust and Node test suites each load that
file and must reproduce it within `tolerance`.

Implementations:

| Runtime | Kernel | Test |
|---|---|---|
| Python | `src/olaya/axioms.py` → `calibrate_and_project_distribution` | `tests/test_axiom_golden.py` |
| Rust | `crates/olaya-core/src/axioms.rs` → `calibrate_and_project_inplace` | `test_axiom_kernel_matches_shared_golden_vectors` |
| Rust (wasm) | `crates/olaya-wasm/src/lib.rs` → `calibrate_and_project` | `kernel_matches_shared_golden_vectors` (runs natively) |
| Node | `packages/olaya-node/src/axioms.ts` → `calibrateAndProject` | `"matches the shared axiom golden vectors"` |

`docs/public/olaya-engine.js` carries the same kernel for the demo pages but has no test harness.

## Scope boundary: HTTP edge demo

The edge `noul` contract now distinguishes supported/contradicted evidence from
`unmatched`, `conflicting`, `ambiguous_scope`, and `unsupported_question`.
Unknown cases return `abstained: true`, `noul: 0.5`, and confidence `0`.
Structured-state evidence uses separate string values, excluding keys and scalar
booleans/numbers, with no proposition matches spanning fields. This edge-only
behavior does not change the runtime/package/parser APIs described here.


`docs/public/api/systemone.js` does not implement the complete projection kernel
specified here. It has lexical flat scoring, entropy confidence, choice
abstention, and bounded English negation. Its temperature scaling is not evidence
of empirical calibration, and its outputs do not provide `axioms_verified`.
See [the edge API contract](systemone-reliability.md). The shared golden vectors
verify the listed kernel implementations, not the deployed HTTP demo.

## Inputs

- `logits[0..K)`: one raw score per candidate slot.
- `parent_edges`: pairs `(child_slot, parent_slot)` for every subsumption edge whose ends are
  both in the slot set.
- `disjoint_edges`: pairs `(slot_a, slot_b)` with `slot_a < slot_b`, in ascending slot order,
  for every `owl:disjointWith` assertion whose ends are both in the slot set.

## Steps

1. **Temperature by cardinality.** `T(K) = 0.85` for `K ≤ 2`, `0.95` for `K ≤ 5`, `1.10` for
   `K ≤ 10`, else `1.25`.
2. **Softmax.** `p_i = exp((l_i − max l) / T) / Σ_j exp((l_j − max l) / T)`.
3. **Subsumption monotonicity.** Repeat until no change: for each `(child, parent)` in
   `parent_edges`, if `p_child > p_parent` set `p_parent = p_child`. This only raises values, so
   it converges within `K` passes. After this step the values are hierarchical scores: they are
   no longer a simplex and are not renormalised.
4. **Disjointness.** With `threshold = 0.6`, for each `(a, b)` in `disjoint_edges` in order: if
   `max(p_a, p_b) ≥ threshold`, the smaller side loses (on a tie, `b` loses) and the loser and
   every slot below it via `parent_edges` are set to `0`. Zeroing the loser's subtree keeps step 3's
   guarantee intact. Because `threshold > 0.5` and the lifted values of two disjoint siblings cannot
   both reach it from one unit of mass unless lifting moved it, at most one side of a pair can win.
5. **Confidence.** Let `p` be the calibrated distribution from step 2 (before lifting) and `q` the
   projected values after steps 3–4. Build `p'_i = p_i` if `q_i > 0`, else `0` (slots that
   disjointness zeroed no longer compete), normalise `p'` to sum to 1 (for this calculation only),
   then `confidence = 1 − H(p') / ln K` with `H` the Shannon entropy in nats and `K` the number of
   slots, clamped to `[0, 1]`. `K ≤ 1` gives `1.0`; an all-zero `p'` gives `0.0`.

   Confidence is deliberately measured *before* monotonicity lifting: lifting copies a child's mass
   onto every ancestor, so a decision the model is certain about would otherwise look uniform over
   the ancestor chain and report confidence `0` (see the `deep_chain_of_ten` golden case, whose
   confidence is `≈ 0.79`, not `0`). It is measured *after* disjointness exclusion because an
   excluded sibling is no longer a competing hypothesis.

## Verification

`verify_axioms` (same name in each runtime) checks, with tolerance `1e-6`:

- for every `(child, parent)`: `p_parent ≥ p_child`;
- for every `(a, b)`: `min(p_a, p_b) = 0` or `max(p_a, p_b) < threshold`.

The engine's `axioms_verified` output is this function's result, not a constant.

## Concept questions in Python

For `concept` questions the Python engine does not use the flat kernel above. It scores only the
most specific routed concepts (the leaves of the active subgraph), applies steps 1–2 to them, and
derives each ancestor's marginal as the sum of the leaf mass beneath it
(`project_hierarchical_distribution`). Monotonicity then holds by construction and
`candidate_probabilities` stays a proper distribution. Disjointness there zeroes the leaf mass under
the losing side and renormalises, repeating until stable. The Rust and Node engines use the flat
kernel for all decisions.

## Decision parity across runtimes

The golden vectors pin the kernel; `decision_parity.json` in this directory ties the runtimes
together one level up. It holds a 10-class incident ontology and five states with the CURIE each
must select (and, where disjointness bites, the CURIEs that must be zeroed):

| Runtime | What is checked | Test |
|---|---|---|
| Rust | `OlayaEngine::system_one` selects `expected_curie` | `test_decisions_match_shared_parity_fixture` |
| Rust (wasm) | `WasmRouter::decide` selects `expected_curie` | `decisions_match_shared_parity_fixture` |
| Node | `OLaya.systemOne` selects `expected_curie` | `"selects the same concepts as the Rust, WASM and Python runtimes"` |
| Python | `CandidateRouter.route_candidates` ranks `expected_curie` first, untied | `tests/test_decision_parity.py` |

Python is checked at the router because its engine decisions come from a trained neural head, not
from lexical overlap.

### Tokenizers and scorers are not identical

The runtimes do not share one tokenizer or one scoring formula, and the fixture is written so that
the expected concept wins under all of them (several label/synonym tokens for the expected class,
at most one for any competitor):

- **Node and WASM** share one rule set: CamelCase split at a lower→upper boundary, lowercase,
  `[a-z0-9]` runs longer than two characters, a stopword list, smoothed IDF
  `ln((N + 1) / (df + 1)) + 1`, router weights 4/2/1 (label/synonym/description) and candidate
  logits `1 + 6·idf` per label token, `5·idf` per synonym token, `idf` per description token.
  `crates/olaya-wasm` is a port of `packages/olaya-node`, including argmax tie-breaking towards the
  most specific concept.
- **Rust core** (`crates/olaya-core/src/router.rs`) deliberately follows the Python package's
  `tokenize_normalize` instead: tokens longer than one character, no stopwords, acronym-aware case
  splitting (Rust additionally emits the whole lowercased word when it splits one), router IDF
  `1 / (1 + df)`, logits `1 + 3·label + 2.5·synonym + 1.5·description` overlaps. It is not ported to the Node rules
  because (a) its tokens are what the distilled neural head hashes (`fnv1a`) and weights by the
  smoothed IDF the head was trained with (`olaya.distilled_head`), so changing the tokenizer in
  Rust alone would desynchronise it from Python-trained weights; (b) Rust↔Python parity on tokens
  such as `db`, `500` or `postgresql` is covered by existing tests and would be lost (the Node rules
  drop every token of two characters or fewer and split `PostgreSQL` into `postgre`, `sql`); and (c)
  the Python side is the reference for that tokenizer and is maintained separately.

If a change to any scorer flips a fixture decision, that is a real cross-runtime divergence: fix
the scorer or change the fixture in the same commit and say why.

## Changing the spec

Edit the Python reference first, run `python scripts/gen_axiom_golden.py`, then bring the other
runtimes to green against the regenerated file.
