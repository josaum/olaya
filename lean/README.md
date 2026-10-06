# O-Laya axiom kernel, Lean 4 proof core

Core Lean only (no Mathlib, no Batteries, no package dependencies). `lean-toolchain` pins the
exact toolchain; bump it deliberately and commit `lake-manifest.json`.

```bash
lake build                         # library, oracle, and the #print axioms audit in Check.lean
echo '{"values":[70,20,60,10],"parent_edges":[[3,1],[1,0]],"disjoint_edges":[[1,2]],"threshold":60}' \
  | lake exe olaya-axioms-oracle   # {"projected":[70,0,60,0],"verified":true}
```

| Module | Spec | Theorems |
| --- | --- | --- |
| `Basic` | values, edges, `Monotone`, `Nonneg`, `Excluded` | `Excluded.of_le`, `at_most_one_reaches` |
| `Lift` | step 3 | `lift_monotone`, `lift_ge`, `lift_le` (minimality), `lift_nonneg`, `lift_of_monotone`, `liftPass_fixed` |
| `Descendants` | "every slot below it" | `descendants_closed`, `descendants_root`, `descendants_sound` |
| `Disjoint` | step 4 | `disjointAll_le`, `disjointAll_nonneg`, `disjointAll_monotone`, `disjointAll_excluded` |
| `Verify` | verification, steps 3–4 together | `verifyAxioms_iff`, `project_verified`, `project_nonneg`, `project_le_lift` |
| `Check` | | `#print axioms` for every theorem above; CI fails on anything but `propext`, `Quot.sound`, `Classical.choice` |

`tests/test_lean_oracle.py` runs the Python reference (`enforce_subsumption_monotonicity`,
`enforce_disjointness_exclusion`) and the verified oracle on the same integer inputs and asserts
equality, so the theorems are about the algorithm the runtimes implement.

Not formalised: softmax (steps 1–2), entropy confidence (step 5). Not proven: that `K` sweeps
always reach the fixed point; `lift` takes a fuel bound and returns `none` if it runs out.
