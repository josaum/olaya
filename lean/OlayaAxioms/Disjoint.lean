import OlayaAxioms.Basic
import OlayaAxioms.Lift
import OlayaAxioms.Descendants

/-!
# Step 4: disjointness exclusion

For each disjoint pair `(a, b)` in order: if `max(p a, p b) ≥ threshold`, the smaller side loses
(`b` on a tie) and the loser together with every slot below it is set to `0`.

Proven: the result never raises a score, stays non-negative, stays monotone, and every pair is
excluded in the spec's sense (`min = 0` or `max < threshold`).
-/

namespace OlayaAxioms

/-- Spec: "the smaller side loses (on a tie, `b` loses)". -/
def loserOf (p : Values) (a b : Nat) : Nat :=
  if p b ≤ p a then b else a

def zeroAll (p : Values) (S : List Nat) : Values :=
  ⟨fun j => if j ∈ S then 0 else p j, p.size⟩

theorem zeroAll_eq (p : Values) (S : List Nat) (j : Nat) : zeroAll p S j = if j ∈ S then 0 else p j :=
  rfl

def disjointStep (edges : List Edge) (θ : Int) (fuel : Nat) (p : Values) (pr : Nat × Nat) : Option Values :=
  if θ ≤ max (p pr.1) (p pr.2) then
    match descendants fuel edges (loserOf p pr.1 pr.2) with
    | none => none
    | some S => some (zeroAll p S)
  else some p

def disjointAll (edges : List Edge) (θ : Int) (fuel : Nat) : List (Nat × Nat) → Values → Option Values
  | [], p => some p
  | pr :: rest, p =>
    match disjointStep edges θ fuel p pr with
    | none => none
    | some q => disjointAll edges θ fuel rest q

theorem zeroAll_le {p : Values} {S : List Nat} (hp : Nonneg p) (i : Nat) : zeroAll p S i ≤ p i := by
  rw [zeroAll_eq]
  split
  · exact hp i
  · exact Int.le_refl _

theorem zeroAll_nonneg {p : Values} {S : List Nat} (hp : Nonneg p) : Nonneg (zeroAll p S) := by
  intro i
  rw [zeroAll_eq]
  split
  · exact Int.le_refl _
  · exact hp i

theorem zeroAll_mem {p : Values} {S : List Nat} {i : Nat} (h : i ∈ S) : zeroAll p S i = 0 := by
  rw [zeroAll_eq]; simp [h]

/-- Zeroing a child-closed set keeps monotonicity. -/
theorem zeroAll_monotone {edges : List Edge} {p : Values} {S : List Nat}
    (hm : Monotone edges p) (hp : Nonneg p) (hS : ChildClosed edges S) : Monotone edges (zeroAll p S) := by
  intro e he
  rw [zeroAll_eq, zeroAll_eq]
  by_cases h2 : e.2 ∈ S
  · have h1 := hS e he h2
    simp [h1, h2]
  · simp only [h2, ↓reduceIte]
    split
    · exact hp e.2
    · exact hm e he

theorem disjointStep_le {edges : List Edge} {θ : Int} {fuel : Nat} {p q : Values} {pr : Nat × Nat}
    (hp : Nonneg p) (h : disjointStep edges θ fuel p pr = some q) : ∀ i, q i ≤ p i := by
  intro i
  unfold disjointStep at h
  split at h
  · split at h
    · cases h
    · cases h; exact zeroAll_le hp i
  · cases h; exact Int.le_refl _

theorem disjointStep_nonneg {edges : List Edge} {θ : Int} {fuel : Nat} {p q : Values} {pr : Nat × Nat}
    (hp : Nonneg p) (h : disjointStep edges θ fuel p pr = some q) : Nonneg q := by
  unfold disjointStep at h
  split at h
  · split at h
    · cases h
    · cases h; exact zeroAll_nonneg hp
  · cases h; exact hp

theorem disjointStep_monotone {edges : List Edge} {θ : Int} {fuel : Nat} {p q : Values} {pr : Nat × Nat}
    (hm : Monotone edges p) (hp : Nonneg p) (h : disjointStep edges θ fuel p pr = some q) : Monotone edges q := by
  unfold disjointStep at h
  split at h
  · split at h
    · cases h
    · rename_i hd
      cases h
      exact zeroAll_monotone hm hp (descendants_closed hd)
  · cases h; exact hm

/-- The processed pair is excluded. -/
theorem disjointStep_excluded {edges : List Edge} {θ : Int} {fuel : Nat} {p q : Values} {pr : Nat × Nat}
    (hp : Nonneg p) (h : disjointStep edges θ fuel p pr = some q) : Excluded θ q pr := by
  unfold disjointStep at h
  split at h
  · split at h
    · cases h
    · rename_i hd
      cases h
      have hroot := descendants_root hd
      have hz := zeroAll_mem (p := p) hroot
      have n1 := zeroAll_nonneg (S := ‹List Nat›) hp pr.1
      have n2 := zeroAll_nonneg (S := ‹List Nat›) hp pr.2
      unfold Excluded
      unfold loserOf at hz
      split at hz
      · left; omega
      · left; omega
  · rename_i hlt
    cases h
    unfold Excluded
    right
    exact Int.not_le.mp hlt

theorem disjointAll_le (edges : List Edge) (θ : Int) (fuel : Nat) :
    ∀ (pairs : List (Nat × Nat)) (p q : Values), Nonneg p →
      disjointAll edges θ fuel pairs p = some q → ∀ i, q i ≤ p i := by
  intro pairs
  induction pairs with
  | nil => intro p q _ h i; simp [disjointAll] at h; subst h; exact Int.le_refl _
  | cons pr rest ih =>
    intro p q hp h i
    unfold disjointAll at h
    split at h
    · cases h
    · rename_i q1 hs
      exact Int.le_trans (ih q1 q (disjointStep_nonneg hp hs) h i) (disjointStep_le hp hs i)

theorem disjointAll_nonneg (edges : List Edge) (θ : Int) (fuel : Nat) :
    ∀ (pairs : List (Nat × Nat)) (p q : Values), Nonneg p →
      disjointAll edges θ fuel pairs p = some q → Nonneg q := by
  intro pairs
  induction pairs with
  | nil => intro p q hp h; simp [disjointAll] at h; subst h; exact hp
  | cons pr rest ih =>
    intro p q hp h
    unfold disjointAll at h
    split at h
    · cases h
    · rename_i q1 hs
      exact ih q1 q (disjointStep_nonneg hp hs) h

theorem disjointAll_monotone (edges : List Edge) (θ : Int) (fuel : Nat) :
    ∀ (pairs : List (Nat × Nat)) (p q : Values), Monotone edges p → Nonneg p →
      disjointAll edges θ fuel pairs p = some q → Monotone edges q := by
  intro pairs
  induction pairs with
  | nil => intro p q hm _ h; simp [disjointAll] at h; subst h; exact hm
  | cons pr rest ih =>
    intro p q hm hp h
    unfold disjointAll at h
    split at h
    · cases h
    · rename_i q1 hs
      exact ih q1 q (disjointStep_monotone hm hp hs) (disjointStep_nonneg hp hs) h

/-- Every pair ends up excluded. -/
theorem disjointAll_excluded (edges : List Edge) (θ : Int) (fuel : Nat) :
    ∀ (pairs : List (Nat × Nat)) (p q : Values), Nonneg p →
      disjointAll edges θ fuel pairs p = some q → ∀ pr ∈ pairs, Excluded θ q pr := by
  intro pairs
  induction pairs with
  | nil => intro p q _ _ pr hpr; simp at hpr
  | cons pr0 rest ih =>
    intro p q hp h pr hpr
    unfold disjointAll at h
    split at h
    · cases h
    · rename_i q1 hs
      have hq1 := disjointStep_nonneg hp hs
      rcases List.mem_cons.mp hpr with rfl | hmem
      · exact (disjointStep_excluded hp hs).of_le (disjointAll_le edges θ fuel rest q1 q hq1 h)
          (disjointAll_nonneg edges θ fuel rest q1 q hq1 h)
      · exact ih q1 q hq1 h pr hmem

end OlayaAxioms
