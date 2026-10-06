import OlayaAxioms.Basic

/-!
# Step 3: subsumption monotonicity

`liftPass` is one relaxation sweep in edge order: for each `(child, parent)`, if the child's
score exceeds the parent's, the parent is raised to it. `lift` repeats the sweep until the
monotonicity check passes, with a fuel bound, and returns `none` if the fuel runs out.

Proven: a returned result is monotone, never lower than the input, non-negative when the input
is, and is the least such result (any monotone `r` above the input is above it). The spec's
claim that `K` sweeps always suffice is **not** proven here; the runtimes observe it.
-/

namespace OlayaAxioms

/-- One edge of one sweep. Strict because `Values` is a structure (see `Values`). -/
def step (q : Values) (e : Edge) : Values :=
  if q e.2 < q e.1 then set q e.2 (q e.1) else q

/-- One sweep over all edges, in order. -/
def liftPass (edges : List Edge) (p : Values) : Values :=
  edges.foldl step p

/-- Decidable form of `Monotone`. -/
def monotoneB (edges : List Edge) (p : Values) : Bool :=
  edges.all fun e => decide (p e.1 ≤ p e.2)

theorem monotoneB_iff (edges : List Edge) (p : Values) :
    monotoneB edges p = true ↔ Monotone edges p := by
  simp [monotoneB, Monotone, List.all_eq_true]

/-- Spec step 3, "repeat until no change", with a fuel bound. -/
def lift : Nat → List Edge → Values → Option Values
  | 0, _, _ => none
  | fuel + 1, edges, p =>
    if monotoneB edges p then some p else lift fuel edges (liftPass edges p)

theorem step_ge (q : Values) (e : Edge) (i : Nat) : q i ≤ step q e i := by
  unfold step
  split
  · rename_i h
    by_cases hi : i = e.2
    · subst hi
      rw [set_self]
      exact Int.le_of_lt h
    · rw [set_other q _ hi]
      exact Int.le_refl _
  · exact Int.le_refl _

theorem step_edge (q : Values) (e : Edge) : q e.1 ≤ step q e e.2 := by
  unfold step
  split
  · rw [set_self]
    exact Int.le_refl _
  · rename_i h
    exact Int.not_lt.mp h

theorem step_le_of_monotone {r : Values} {q : Values} {e : Edge}
    (hr : r e.1 ≤ r e.2) (hq : ∀ i, q i ≤ r i) : ∀ i, step q e i ≤ r i := by
  intro i
  unfold step
  split
  · by_cases hi : i = e.2
    · subst hi
      rw [set_self]
      exact Int.le_trans (hq e.1) hr
    · rw [set_other q _ hi]
      exact hq i
  · exact hq i

theorem foldl_step_ge (es : List Edge) : ∀ (q : Values) (i : Nat), q i ≤ es.foldl step q i := by
  induction es with
  | nil => intro q i; simp only [List.foldl_nil]; exact Int.le_refl _
  | cons e es ih =>
    intro q i
    simp only [List.foldl_cons]
    exact Int.le_trans (step_ge q e i) (ih (step q e) i)

theorem foldl_step_edge (es : List Edge) :
    ∀ (q : Values), ∀ e ∈ es, q e.1 ≤ es.foldl step q e.2 := by
  induction es with
  | nil => intro q e he; simp at he
  | cons e' es ih =>
    intro q e he
    simp only [List.foldl_cons]
    rcases List.mem_cons.mp he with rfl | hmem
    · exact Int.le_trans (step_edge q e) (foldl_step_ge es (step q e) e.2)
    · exact Int.le_trans (step_ge q e' e.1) (ih (step q e') e hmem)

theorem foldl_step_le (es : List Edge) :
    ∀ (q r : Values), (∀ e ∈ es, r e.1 ≤ r e.2) → (∀ i, q i ≤ r i) → ∀ i, es.foldl step q i ≤ r i := by
  induction es with
  | nil => intro q r _ hq i; simpa [List.foldl] using hq i
  | cons e es ih =>
    intro q r hr hq i
    simp only [List.foldl_cons]
    have hr' := List.forall_mem_cons.mp hr
    exact ih (step q e) r hr'.2 (step_le_of_monotone hr'.1 hq) i

/-- A sweep never lowers a score. -/
theorem liftPass_ge (edges : List Edge) (p : Values) (i : Nat) : p i ≤ liftPass edges p i :=
  foldl_step_ge edges p i

/-- After a sweep, every parent is at least its child's score *before* the sweep. -/
theorem liftPass_edge (edges : List Edge) (p : Values) : ∀ e ∈ edges, p e.1 ≤ liftPass edges p e.2 :=
  foldl_step_edge edges p

/-- A sweep stays below any monotone upper bound of its input. -/
theorem liftPass_le (edges : List Edge) (p r : Values) (hr : Monotone edges r) (hp : ∀ i, p i ≤ r i) :
    ∀ i, liftPass edges p i ≤ r i :=
  foldl_step_le edges p r hr hp

/-- A monotone input is a fixed point of the sweep. -/
theorem liftPass_fixed (edges : List Edge) (p : Values) (hp : Monotone edges p) (i : Nat) :
    liftPass edges p i = p i :=
  Int.le_antisymm (liftPass_le edges p p hp (fun _ => Int.le_refl _) i) (liftPass_ge edges p i)

/-- **Soundness.** Whatever `lift` returns is monotone. -/
theorem lift_monotone (edges : List Edge) :
    ∀ (fuel : Nat) (p q : Values), lift fuel edges p = some q → Monotone edges q := by
  intro fuel
  induction fuel with
  | zero => intro p q h; simp [lift] at h
  | succ n ih =>
    intro p q h
    unfold lift at h
    split at h
    · rename_i hm
      cases h
      exact (monotoneB_iff edges _).mp hm
    · exact ih _ _ h

/-- **Only raises.** `lift` never lowers a score. -/
theorem lift_ge (edges : List Edge) :
    ∀ (fuel : Nat) (p q : Values), lift fuel edges p = some q → ∀ i, p i ≤ q i := by
  intro fuel
  induction fuel with
  | zero => intro p q h; simp [lift] at h
  | succ n ih =>
    intro p q h i
    unfold lift at h
    split at h
    · cases h; exact Int.le_refl _
    · exact Int.le_trans (liftPass_ge edges p i) (ih _ _ h i)

/-- **Minimality.** `lift` returns the least monotone function above its input, which is exactly
the fixed point the spec's "repeat until no change" describes. -/
theorem lift_le (edges : List Edge) :
    ∀ (fuel : Nat) (p q r : Values), lift fuel edges p = some q → Monotone edges r →
      (∀ i, p i ≤ r i) → ∀ i, q i ≤ r i := by
  intro fuel
  induction fuel with
  | zero => intro p q r h; simp [lift] at h
  | succ n ih =>
    intro p q r h hr hp i
    unfold lift at h
    split at h
    · cases h; exact hp i
    · exact ih _ _ r h hr (liftPass_le edges p r hr hp) i

theorem lift_nonneg (edges : List Edge) {fuel : Nat} {p q : Values}
    (h : lift fuel edges p = some q) (hp : Nonneg p) : Nonneg q :=
  fun i => Int.le_trans (hp i) (lift_ge edges fuel p q h i)

/-- `lift` returns its input unchanged whenever the input already satisfies the axiom. -/
theorem lift_of_monotone (edges : List Edge) (fuel : Nat) (p : Values) (hp : Monotone edges p) :
    lift (fuel + 1) edges p = some p := by
  unfold lift
  simp [(monotoneB_iff edges p).mpr hp]

end OlayaAxioms
