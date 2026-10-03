# OAEI Conference test fixtures

Three unmodified upstream files used by `tests/test_oaei_benchmark.py`, vendored
so the regression tests run offline in a clean checkout. This is one ontology
pair, not the full Conference benchmark.

Source: [OAEI 2025 Conference track](https://oaei.ontologymatching.org/2025/conference/index.html).
Downloaded 2026-10-03 UTC; all three files matched the existing local corpus byte
for byte. Ontologies originate from the OntoFarm project; upstream reference
alignments are the original ra1 set. Upstream authors retain their rights; this
fixture inclusion does not relicense their data under the repository code license.

| File | Source | SHA-256 |
|---|---|---|
| `ontologies/cmt.owl` | [Cmt](https://oaei.ontologymatching.org/2025/conference/data/cmt.owl) | `8bf45a9de7faf24bfee5a9d90cd95f744964f90bb0ad64ad80ec4335f6f5c1db` |
| `ontologies/ekaw.owl` | [Ekaw](https://oaei.ontologymatching.org/2025/conference/data/ekaw.owl) | `e4582518c2b2afe7eca9668c88e9d0f4ab34dcb7d169ebf6880c79dac5e6edaa` |
| `reference_alignments/cmt-ekaw.rdf` | `cmt-ekaw.rdf` from [reference-alignment.zip](https://oaei.ontologymatching.org/2025/conference/data/reference-alignment.zip) | `ab825abd429d435bbef672d4386ddef78e1d0f15e6e28cd16f67907831a8a42e` |

Upstream requested citation:
Ondřej Zamazal and Vojtěch Svátek. *The Ten-Year OntoFarm and its Fertilization
within the Onto-Sphere*. Web Semantics: Science, Services and Agents on the World
Wide Web, 43, 46–53, 2017.
