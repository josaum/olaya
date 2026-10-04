# OWL 2 RL rule coverage

This file is the normative statement of which W3C OWL 2 RL rules `src/olaya/owl2_rl.py`
implements. Rule names and numbering follow the tables in
[OWL 2 Web Ontology Language Profiles, section 4.3](https://www.w3.org/TR/owl2-profiles/#Reasoning_in_OWL_2_RL_and_RDF_Graphs_using_Rules).
Tests: `tests/test_owl2_rl.py` (schema rules) and `tests/test_abox.py` (instance rules,
consistency, equality, serialisation, schema-closure regression hashes).

## Data model the rules run over

* **TBox**: `Concept` (subClassOf, equivalentClass, disjointWith, intersectionOf, unionOf,
  disjointUnionOf) and `ObjectProperty` (subPropertyOf, domain, range, inverseOf, symmetric,
  asymmetric, transitive, functional, inverse-functional, irreflexive, propertyDisjointWith,
  propertyChainAxiom). There is **no** representation of class expressions beyond named classes
  and the boolean operators above: no `owl:complementOf`, no `owl:oneOf`, no restrictions
  (`someValuesFrom`, `allValuesFrom`, `hasValue`, cardinality), no `owl:hasKey`, no datatypes.
* **ABox**: `Individual` with class assertions, object property assertions, `owl:sameAs`,
  `owl:differentFrom`. Only **object** property assertions between named individuals exist;
  there are no literals, data properties or negative property assertions.

Status legend: **schema** = applied to the class/property graph by `OWL2RLReasoner.compute_closure`;
**instance** = applied to individuals in the ABox fixpoint; **unsupported** = not applied, with
the reason. "Trivial" means the rule's conclusion is implied by the representation itself and no
firing is counted.

## Table 4: Semantics of Equality

| Rule | Status | Notes |
|---|---|---|
| eq-ref | trivial | `same_individuals(x)` always contains `x`; no explicit reflexive pair is stored. |
| eq-sym | instance | sameAs is a union-find partition; symmetric by construction. Counted once per merge. |
| eq-trans | instance | Transitivity is the partition itself; `eq-trans` counts the derived cross pairs. |
| eq-rep-s | instance | Types and subject positions are replicated over every member of a merged class. |
| eq-rep-p | unsupported | Properties are never individuals here, so `p sameAs q` cannot be stated. |
| eq-rep-o | instance | Object positions are replicated over every member of a merged class. |
| eq-diff1 | instance | Reported as `Inconsistency("eq-diff1", (x, y))`. |
| eq-diff2, eq-diff3 | unsupported | `owl:AllDifferent` lists are not parsed. Pairwise `assert_different_from` plus eq-diff1 gives the same consequences for explicitly enumerated pairs. |

## Table 5: Property Axioms

| Rule | Status | Notes |
|---|---|---|
| prp-ap | unsupported | Annotation properties are not modelled. |
| prp-dom | instance | Uses the schema-closed domain set (so scm-dom1/2 are honoured). |
| prp-rng | instance | Uses the schema-closed range set. |
| prp-fp | instance | Merges the objects into one sameAs class (then eq-rep-*). |
| prp-ifp | instance | Merges the subjects into one sameAs class. |
| prp-irp | instance | `is_irreflexive`; reported as inconsistency. |
| prp-symp | instance | `is_symmetric`. |
| prp-asyp | instance | `is_asymmetric`; reported as inconsistency. |
| prp-trp | instance | `is_transitive`. |
| prp-spo1 | instance | Uses the schema-closed subPropertyOf relation (scm-spo). |
| prp-spo2 | instance | `ObjectProperty.property_chains` / `Ontology.add_property_chain`; chains of any length ≥ 2. |
| prp-eqp1, prp-eqp2 | unsupported (representable) | There is no `equivalentProperty` field. Declaring mutual `subPropertyOf` gives identical consequences through prp-spo1. |
| prp-pdw | instance | `disjoint_with` on properties / `add_property_disjointness`; reported as inconsistency. |
| prp-adp | unsupported | `owl:AllDisjointProperties` lists are not parsed; use pairwise `add_property_disjointness`. |
| prp-inv1, prp-inv2 | instance | Both directions of `inverses`, counted separately. |
| prp-key | unsupported | `owl:hasKey` is not modelled. |
| prp-npa1, prp-npa2 | unsupported | Negative property assertions are not modelled. |

## Table 6: Class Axioms (the `cls-*` rules)

| Rule | Status | Notes |
|---|---|---|
| cls-thing | trivial | `owl:Thing` is not materialised as a type. |
| cls-nothing1 | trivial | `owl:Nothing` is not materialised as a class. |
| cls-nothing2 | schema + instance | Schema: a class disjoint with itself is `unsatisfiable`. Instance: an individual typed `owl:Nothing` or an unsatisfiable class is an inconsistency. |
| cls-int1 | instance | Membership in every operand of `intersections` ⇒ membership in the defined class. |
| cls-int2 | instance (via scm-int + cax-sco) | Not counted separately: `scm-int` makes the class a subclass of each operand, then `cax-sco` propagates. |
| cls-uni | instance (via scm-uni + cax-sco) | Same mechanism: each operand is a subclass of the union class. |
| cls-com | unsupported | `owl:complementOf` is not representable. |
| cls-svf1, cls-svf2 | unsupported | No `someValuesFrom` restrictions. |
| cls-avf | unsupported | No `allValuesFrom` restrictions. |
| cls-hv1, cls-hv2 | unsupported | No `hasValue` restrictions. |
| cls-maxc1, cls-maxc2 | unsupported | No cardinality restrictions. |
| cls-maxqc1 – cls-maxqc4 | unsupported | No qualified cardinality restrictions. |
| cls-oo | unsupported | No `owl:oneOf`. |

## Table 7: Class Axioms (the `cax-*` rules)

| Rule | Status | Notes |
|---|---|---|
| cax-sco | instance | Uses the schema-closed subClassOf relation (scm-sco), so multi-level inheritance is one rule per level. |
| cax-eqc1, cax-eqc2 | instance | One counter per direction (lexicographic order of the pair). Checked before cax-sco because `add_equivalence` also records mutual subsumption. |
| cax-dw | instance | Uses the schema-closed disjointness (scm-dco), so inherited disjointness is detected. Reported as inconsistency. |
| cax-adc | unsupported | `owl:AllDisjointClasses` lists are not parsed; `disjoint_union` and pairwise `add_disjointness` cover enumerated cases. |

## Table 8: Datatypes

| Rule | Status | Notes |
|---|---|---|
| dt-type1, dt-type2, dt-eq, dt-diff, dt-not-type | unsupported | No literals or datatypes in the model. |

## Table 9: Schema Vocabulary

| Rule | Status | Notes |
|---|---|---|
| scm-cls | schema | Reflexive `C ⊑ C` is recorded; `owl:Thing`/`owl:Nothing` edges are not materialised. |
| scm-sco | schema | Transitive closure of subClassOf. |
| scm-eqc1 | schema | `equivalent_to` ⇒ mutual subClassOf. |
| scm-eqc2 | schema | Mutual subClassOf ⇒ equivalence. |
| scm-op, scm-dp | trivial | Reflexive subPropertyOf is recorded in `property_subsumptions`. |
| scm-spo | schema | Transitive closure of subPropertyOf. |
| scm-eqp1, scm-eqp2 | unsupported (representable) | No `equivalentProperty` field; mutual subPropertyOf carries the same meaning. |
| scm-dom1, scm-dom2 | schema | Counted under `scm-dom1`: domain inherited down subPropertyOf. `scm-dom2` (domain widened up subClassOf) is applied implicitly, since instance rules run cax-sco after prp-dom. |
| scm-rng1, scm-rng2 | schema | Same as domains, counted under `scm-rng1`. |
| scm-hv, scm-svf1, scm-svf2, scm-avf1, scm-avf2 | unsupported | No restrictions. |
| scm-int | schema | `C ≡ C1 ⊓ … ⊓ Cn` ⇒ `C ⊑ Ci`. |
| scm-uni | schema | `C ≡ C1 ⊔ … ⊔ Cn` ⇒ `Ci ⊑ C`. |
| (`owl:disjointUnionOf`) | schema | Not an RL rule but accepted: members become subclasses and pairwise disjoint (`scm-dju-sub`, `scm-dju-disj`). |
| scm-dco (disjointness inheritance) | schema | `A ⊑ B, B ⊥ C ⇒ A ⊥ C`. Not in the W3C table (which derives it through cax-dw at instance level) but needed for the probability projection kernel. |

## Outputs

`OWL2RLClosure` carries the schema closure (`subclasses`, `equivalences`, `disjointnesses`,
`unsatisfiable_classes`, `property_*`) and the instance closure (`individual_types`,
`property_assertions`, `same_as`, `inconsistencies`). `closure.types_of(x)`,
`closure.same_individuals(x)`, `Ontology.instances_of(C, closure)` and
`abox_consistency_report(ontology)` are the query surface. Inconsistencies are returned as
`Inconsistency(rule, entities, message)` records; nothing raises. Every rule application
increments `closure.rule_firings[rule]`.

## Regenerating the regression hashes

`tests/test_abox.py::EXPECTED_SCHEMA_CLOSURE_HASHES` pins the schema closure of every
`src/olaya/data/*.json`. Regenerate after an intentional schema-rule change with
`.venv/bin/python -c "import tests.test_abox as t; t.print_schema_closure_hashes()"`.
