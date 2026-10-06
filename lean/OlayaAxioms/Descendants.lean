import OlayaAxioms.Basic

/-!
# Descendants of a slot

Step 4 zeroes "the loser and every slot below it via `parent_edges`". `saturate` grows a set
from the loser by adding children of members until the set is closed under "child of a member",
with a fuel bound. A returned set is child-closed by construction, contains its root, and holds
only slots reachable from the root along child edges.
-/

namespace OlayaAxioms

/-- `root ⟶* x` along `(child, parent)` edges read downwards. -/
inductive Reach (edges : List Edge) (root : Nat) : Nat → Prop
  | refl : Reach edges root root
  | step {c pa : Nat} : Reach edges root pa → (c, pa) ∈ edges → Reach edges root c

def addChild (acc : List Nat) (e : Edge) : List Nat :=
  if e.2 ∈ acc ∧ e.1 ∉ acc then e.1 :: acc else acc

def descStep (edges : List Edge) (S : List Nat) : List Nat :=
  edges.foldl addChild S

/-- Spec, "every slot below it": the set is closed under taking children. -/
def ChildClosed (edges : List Edge) (S : List Nat) : Prop :=
  ∀ e ∈ edges, e.2 ∈ S → e.1 ∈ S

def childClosedB (edges : List Edge) (S : List Nat) : Bool :=
  edges.all fun e => decide (e.2 ∈ S → e.1 ∈ S)

theorem childClosedB_iff (edges : List Edge) (S : List Nat) :
    childClosedB edges S = true ↔ ChildClosed edges S := by
  unfold childClosedB ChildClosed
  rw [List.all_eq_true]
  constructor
  · intro h e he hb
    exact of_decide_eq_true (h e he) hb
  · intro h e he
    exact decide_eq_true (h e he)

def saturate : Nat → List Edge → List Nat → Option (List Nat)
  | 0, _, _ => none
  | fuel + 1, edges, S =>
    if childClosedB edges S then some S else saturate fuel edges (descStep edges S)

theorem mem_addChild_of_mem {acc : List Nat} {e : Edge} {x : Nat} (h : x ∈ acc) : x ∈ addChild acc e := by
  unfold addChild
  split
  · exact List.mem_cons_of_mem _ h
  · exact h

theorem mem_descStep_of_mem (es : List Edge) : ∀ (S : List Nat) {x : Nat}, x ∈ S → x ∈ es.foldl addChild S := by
  induction es with
  | nil => intro S x h; simpa [List.foldl] using h
  | cons e es ih => intro S x h; simp only [List.foldl_cons]; exact ih _ (mem_addChild_of_mem h)

theorem addChild_reach {edges : List Edge} {root : Nat} {acc : List Nat} {e : Edge} (he : e ∈ edges)
    (hacc : ∀ x ∈ acc, Reach edges root x) : ∀ x ∈ addChild acc e, Reach edges root x := by
  intro x hx
  unfold addChild at hx
  split at hx
  · rename_i hc
    rcases List.mem_cons.mp hx with rfl | hmem
    · exact Reach.step (hacc _ hc.1) he
    · exact hacc _ hmem
  · exact hacc _ hx

theorem foldl_addChild_reach (edges : List Edge) (root : Nat) (es : List Edge) :
    ∀ (S : List Nat), (∀ e ∈ es, e ∈ edges) → (∀ x ∈ S, Reach edges root x) →
      ∀ x ∈ es.foldl addChild S, Reach edges root x := by
  induction es with
  | nil => intro S _ hS; simpa [List.foldl] using hS
  | cons e es ih =>
    intro S hes hS
    simp only [List.foldl_cons]
    have hes' := List.forall_mem_cons.mp hes
    exact ih _ hes'.2 (addChild_reach hes'.1 hS)

/-- A returned set is child-closed. -/
theorem saturate_closed (edges : List Edge) :
    ∀ (fuel : Nat) (S T : List Nat), saturate fuel edges S = some T → ChildClosed edges T := by
  intro fuel
  induction fuel with
  | zero => intro S T h; simp [saturate] at h
  | succ n ih =>
    intro S T h
    unfold saturate at h
    split at h
    · rename_i hc; cases h; exact (childClosedB_iff edges _).mp hc
    · exact ih _ _ h

/-- A returned set contains the seed. -/
theorem saturate_supset (edges : List Edge) :
    ∀ (fuel : Nat) (S T : List Nat), saturate fuel edges S = some T → ∀ x ∈ S, x ∈ T := by
  intro fuel
  induction fuel with
  | zero => intro S T h; simp [saturate] at h
  | succ n ih =>
    intro S T h x hx
    unfold saturate at h
    split at h
    · cases h; exact hx
    · exact ih _ _ h x (mem_descStep_of_mem edges S hx)

/-- A returned set holds only slots reachable from the seed. -/
theorem saturate_sound (edges : List Edge) (root : Nat) :
    ∀ (fuel : Nat) (S T : List Nat), (∀ x ∈ S, Reach edges root x) →
      saturate fuel edges S = some T → ∀ x ∈ T, Reach edges root x := by
  intro fuel
  induction fuel with
  | zero => intro S T _ h; simp [saturate] at h
  | succ n ih =>
    intro S T hS h
    unfold saturate at h
    split at h
    · cases h; exact hS
    · exact ih _ _ (foldl_addChild_reach edges root edges S (fun _ he => he) hS) h

/-- Descendants of one slot, the loser of a disjoint pair. -/
def descendants (fuel : Nat) (edges : List Edge) (root : Nat) : Option (List Nat) :=
  saturate fuel edges [root]

theorem descendants_closed {fuel : Nat} {edges : List Edge} {root : Nat} {T : List Nat}
    (h : descendants fuel edges root = some T) : ChildClosed edges T :=
  saturate_closed edges fuel [root] T h

theorem descendants_root {fuel : Nat} {edges : List Edge} {root : Nat} {T : List Nat}
    (h : descendants fuel edges root = some T) : root ∈ T :=
  saturate_supset edges fuel [root] T h root (List.mem_cons_self ..)

theorem descendants_sound {fuel : Nat} {edges : List Edge} {root : Nat} {T : List Nat}
    (h : descendants fuel edges root = some T) : ∀ x ∈ T, Reach edges root x :=
  saturate_sound edges root fuel [root] T
    (fun x hx => by
      rcases List.mem_cons.mp hx with rfl | hx'
      · exact Reach.refl
      · simp at hx') h

end OlayaAxioms
