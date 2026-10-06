import OlayaAxioms.Verify

/-!
Axiom audit. CI greps the build log for these lines and fails if any theorem depends on
anything beyond `propext`, `Classical.choice` and `Quot.sound` (in particular on `sorryAx`).
-/

#print axioms OlayaAxioms.lift_monotone
#print axioms OlayaAxioms.lift_ge
#print axioms OlayaAxioms.lift_le
#print axioms OlayaAxioms.lift_of_monotone
#print axioms OlayaAxioms.liftPass_fixed
#print axioms OlayaAxioms.descendants_closed
#print axioms OlayaAxioms.descendants_root
#print axioms OlayaAxioms.descendants_sound
#print axioms OlayaAxioms.disjointAll_monotone
#print axioms OlayaAxioms.disjointAll_excluded
#print axioms OlayaAxioms.disjointAll_le
#print axioms OlayaAxioms.verifyAxioms_iff
#print axioms OlayaAxioms.project_verified
#print axioms OlayaAxioms.project_nonneg
#print axioms OlayaAxioms.project_le_lift
#print axioms OlayaAxioms.at_most_one_reaches
