/-!
# Values, edges and the two axioms

The kernel in `spec/AXIOMS.md` works on one score per candidate slot. Slots are natural numbers
and scores are integers: a fixed-point view of the floating-point values every runtime uses, which
is enough for the two combinatorial steps formalised here (subsumption lifting and disjointness
exclusion). Softmax (steps 1-2) and the entropy confidence (step 5) are not formalised.
-/

namespace OlayaAxioms

/-- One score per slot. A total function avoids bounds proofs; slots outside the candidate set
are simply never mentioned by an edge.

It is wrapped in a two-field structure on purpose: a definition whose result type is a bare
function is eta-expanded by the compiler, so `if c then set q i v else q` would re-run the
comparison on every lookup, which is exponential on chains when the oracle executes it. A
structure result is computed once. `size` is informational (the slot count the oracle used). -/
structure Values where
  f : Nat → Int
  size : Nat := 0

instance : CoeFun Values (fun _ => Nat → Int) := ⟨Values.f⟩

/-- A subsumption edge `(child, parent)`. -/
abbrev Edge := Nat × Nat

/-- Pointwise update. -/
def set (p : Values) (i : Nat) (v : Int) : Values :=
  ⟨fun j => if j = i then v else p j, p.size⟩

theorem set_eq (p : Values) (i : Nat) (v : Int) (j : Nat) : set p i v j = if j = i then v else p j :=
  rfl

/-- Spec, verification clause 1: for every `(child, parent)`, `p parent ≥ p child`. -/
def Monotone (edges : List Edge) (p : Values) : Prop :=
  ∀ e ∈ edges, p e.1 ≤ p e.2

/-- Every score is non-negative (softmax output; preserved by every step below). -/
def Nonneg (p : Values) : Prop :=
  ∀ i, 0 ≤ p i

/-- Spec, verification clause 2 for one disjoint pair `(a, b)`:
`min(p a, p b) = 0` or `max(p a, p b) < threshold`. -/
def Excluded (threshold : Int) (p : Values) (pr : Nat × Nat) : Prop :=
  min (p pr.1) (p pr.2) = 0 ∨ max (p pr.1) (p pr.2) < threshold

theorem set_self (p : Values) (i : Nat) (v : Int) : set p i v i = v := by
  rw [set_eq]; simp

theorem set_other (p : Values) {i j : Nat} (v : Int) (h : j ≠ i) : set p i v j = p j := by
  rw [set_eq]; simp [h]

/-- Exclusion survives lowering scores, as long as they stay non-negative. -/
theorem Excluded.of_le {θ : Int} {p q : Values} {pr : Nat × Nat}
    (hle : ∀ i, q i ≤ p i) (hq : Nonneg q) (h : Excluded θ p pr) : Excluded θ q pr := by
  unfold Excluded at *
  have h1 := hle pr.1
  have h2 := hle pr.2
  have n1 := hq pr.1
  have n2 := hq pr.2
  omega

/-- Spec, step 4 remark: with `2 · threshold > total`, two scores that share at most `total` of
mass cannot both reach the threshold. -/
theorem at_most_one_reaches {a b total θ : Int} (hsum : a + b ≤ total) (hθ : total < 2 * θ) :
    ¬ (θ ≤ a ∧ θ ≤ b) := by
  omega

end OlayaAxioms
