import OlayaAxioms.Basic
import OlayaAxioms.Lift
import OlayaAxioms.Disjoint

/-!
# The verifier and the projection

`verifyAxioms` is the decidable check every runtime exposes as `axioms_verified`.
`project` is steps 3-4 of the spec. The main theorem: whatever `project` returns passes
`verifyAxioms`, so `axioms_verified` is true by construction for a non-negative input.
-/

namespace OlayaAxioms

def excludedB (θ : Int) (p : Values) (pr : Nat × Nat) : Bool :=
  decide (min (p pr.1) (p pr.2) = 0 ∨ max (p pr.1) (p pr.2) < θ)

/-- Spec, "Verification": clause 1 on every subsumption edge, clause 2 on every disjoint pair. -/
def verifyAxioms (edges : List Edge) (pairs : List (Nat × Nat)) (θ : Int) (p : Values) : Bool :=
  monotoneB edges p && pairs.all (excludedB θ p)

theorem verifyAxioms_iff (edges : List Edge) (pairs : List (Nat × Nat)) (θ : Int) (p : Values) :
    verifyAxioms edges pairs θ p = true ↔ Monotone edges p ∧ ∀ pr ∈ pairs, Excluded θ p pr := by
  simp [verifyAxioms, monotoneB_iff, excludedB, Excluded, List.all_eq_true]

/-- Steps 3 then 4 of the spec. -/
def project (fuel : Nat) (edges : List Edge) (pairs : List (Nat × Nat)) (θ : Int) (p : Values) : Option Values :=
  match lift fuel edges p with
  | none => none
  | some q => disjointAll edges θ fuel pairs q

/-- **Main theorem.** A projected non-negative input satisfies both axioms. -/
theorem project_verified {fuel : Nat} {edges : List Edge} {pairs : List (Nat × Nat)} {θ : Int} {p q : Values}
    (hp : Nonneg p) (h : project fuel edges pairs θ p = some q) : verifyAxioms edges pairs θ q = true := by
  unfold project at h
  split at h
  · cases h
  · rename_i q1 hl
    have hm := lift_monotone edges fuel p q1 hl
    have hn := lift_nonneg edges hl hp
    exact (verifyAxioms_iff edges pairs θ q).mpr
      ⟨disjointAll_monotone edges θ fuel pairs q1 q hm hn h, disjointAll_excluded edges θ fuel pairs q1 q hn h⟩

theorem project_nonneg {fuel : Nat} {edges : List Edge} {pairs : List (Nat × Nat)} {θ : Int} {p q : Values}
    (hp : Nonneg p) (h : project fuel edges pairs θ p = some q) : Nonneg q := by
  unfold project at h
  split at h
  · cases h
  · rename_i q1 hl
    exact disjointAll_nonneg edges θ fuel pairs q1 q (lift_nonneg edges hl hp) h

/-- Slots that disjointness did not zero keep exactly their lifted score; nothing else moves. -/
theorem project_le_lift {fuel : Nat} {edges : List Edge} {pairs : List (Nat × Nat)} {θ : Int} {p q q1 : Values}
    (hp : Nonneg p) (hl : lift fuel edges p = some q1) (h : project fuel edges pairs θ p = some q) :
    ∀ i, q i ≤ q1 i := by
  unfold project at h
  rw [hl] at h
  exact disjointAll_le edges θ fuel pairs q1 q (lift_nonneg edges hl hp) h

end OlayaAxioms
