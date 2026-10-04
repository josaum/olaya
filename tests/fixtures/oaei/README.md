# OAEI Conference track fixtures (OntoFarm, ra1)

Unmodified upstream files from the OAEI Conference track, vendored so that
`scripts/benchmark_oaei.py` and `tests/test_oaei_benchmark.py` run offline in a
clean checkout.  They are the seven OntoFarm ontologies that have ra1 reference
alignments and the 21 ra1 reference alignments between them, i.e. the standard
"ra1-M1 / ra1-M2 / ra1-M3" evaluation set of the track.

Sources (downloaded 2026-10-03 UTC):

- `https://oaei.ontologymatching.org/2023/conference/data/conference.zip`
  (SHA-256 `78688cda05857b594be188db6831abc5d890d2e38d6602e0cd8d3c82ecb24546`);
  only the seven ontologies below were kept out of the 16 in the archive.
- `https://oaei.ontologymatching.org/2023/conference/data/reference-alignment.zip`
  (SHA-256 `782e0175253d817288e1f7ed29f738a7fa9e57a8b9c77bb3b11c12b80405d4bf`);
  all 21 files kept.

Every file is byte-for-byte identical to the archive member; nothing was
reformatted or edited.  The three files that were already vendored before
(`cmt.owl`, `ekaw.owl`, `cmt-ekaw.rdf`, fetched from the 2025 track pages) have
the same hashes as the 2023 archive members, so the dataset has not changed
between the 2023 and 2025 editions.  Ontologies originate from the OntoFarm
project.  Upstream authors retain their rights; this fixture inclusion does not
relicense their data under the repository code license.

Upstream requested citation:
Ondřej Zamazal and Vojtěch Svátek. *The Ten-Year OntoFarm and its Fertilization
within the Onto-Sphere*. Web Semantics: Science, Services and Agents on the World
Wide Web, 43, 46–53, 2017.

| File | SHA-256 |
|---|---|
| `ontologies/Conference.owl` | `9aa78f9673b776c26f615ec7946182c60a146015038ab9c3c105d903251ff1dd` |
| `ontologies/cmt.owl` | `8bf45a9de7faf24bfee5a9d90cd95f744964f90bb0ad64ad80ec4335f6f5c1db` |
| `ontologies/confOf.owl` | `e869c4e75c65c5fd26523a4e73b99a3b1916823a35708798e87c549a63242625` |
| `ontologies/edas.owl` | `6bf83df738c1c3cbf1321a679ee9867fe0ecf97d15a578d8483380e1f28408c3` |
| `ontologies/ekaw.owl` | `e4582518c2b2afe7eca9668c88e9d0f4ab34dcb7d169ebf6880c79dac5e6edaa` |
| `ontologies/iasted.owl` | `36fabbc70accc1260a3ab10fcba2bdf809edf4bb7f78030cc825a90c1f709f33` |
| `ontologies/sigkdd.owl` | `c0bcbb1376ce5d9e711e3d23d809c4e214416cdcfd5a204d80fde7140deaebf4` |
| `reference_alignments/cmt-confOf.rdf` | `e9c402455c838b75b74614a4019b9c1e3b63ba85ba2ab73ac104a5d059776b42` |
| `reference_alignments/cmt-conference.rdf` | `6ce52b6f2e9eca110f6d7fbf12be1475063e8556e7be89bef733b7defd28b0bf` |
| `reference_alignments/cmt-edas.rdf` | `d16347ed42a9f48b5695ed504d0714b627029a62848df77218b15b9de065f393` |
| `reference_alignments/cmt-ekaw.rdf` | `ab825abd429d435bbef672d4386ddef78e1d0f15e6e28cd16f67907831a8a42e` |
| `reference_alignments/cmt-iasted.rdf` | `cc67c2b41dc50deb8bee9c98a48007c698b825538bfc4820f1b28c41c884f187` |
| `reference_alignments/cmt-sigkdd.rdf` | `3c433c01260c329f07cdce5b8d01c353bc511fe36d5aac5571954d0820888e7f` |
| `reference_alignments/confOf-edas.rdf` | `f90c4427213cac1c6134d5d5ddf270268939a53c7605232209154a7b93735971` |
| `reference_alignments/confOf-ekaw.rdf` | `76e8935bb0976e48d3f9c97d0caef2fd1fbae292413c5a9e8da24c6d866e865c` |
| `reference_alignments/confOf-iasted.rdf` | `d32d37eb312c421ca40b7aceb7c8754af7ec418c0c4ca6b263614ecee8c9a1d6` |
| `reference_alignments/confOf-sigkdd.rdf` | `7fa8b226f53473067bf45d8318ef8630b16b32f3d0c0489e5792a716137cf351` |
| `reference_alignments/conference-confOf.rdf` | `acedbf5685196c14b3f8b2dba2b8ab93b9b448ba1a9252c81f30bb89086bf517` |
| `reference_alignments/conference-edas.rdf` | `8a0dfa0ddea8eb6f55c569eb898d505f93e1b8cfbf9c038d6909cd7021f23233` |
| `reference_alignments/conference-ekaw.rdf` | `23c0ded5a68600664ce0c669f3343e2cc6f077e028ef44d536a6aec370803d79` |
| `reference_alignments/conference-iasted.rdf` | `55765ad3b30b2b796234230c8ce33d56074e1b9752cee149b7343386778fba74` |
| `reference_alignments/conference-sigkdd.rdf` | `c90a348205b5e8447195aed1c722ee1086b03f0475a2055f3559e3d5499a59f1` |
| `reference_alignments/edas-ekaw.rdf` | `13ae6872b60976da8d32f33795776cabba1908a03e3035ca475835f98e425fd7` |
| `reference_alignments/edas-iasted.rdf` | `bfda638f4e16fa2e82d6bc33baf53f2a73bdb338977e71e84b66acf285bc17ff` |
| `reference_alignments/edas-sigkdd.rdf` | `7ae3b3440772687cfd4706b910dbe22a53fc45fa5255dbbd6d7bd438210d6f1b` |
| `reference_alignments/ekaw-iasted.rdf` | `20886cea3021f560b64e01917cb25c1bc2bd9d3b0b0c79ccf4cc441aefd5bfd8` |
| `reference_alignments/ekaw-sigkdd.rdf` | `080dd8d9089ee5d705f082c43ecc1160d94835f3bc9474f199bdeb4d50b41ab9` |
| `reference_alignments/iasted-sigkdd.rdf` | `69abc4a050b84cc8df98f4071aad382eae6ec172b64c20007f747770d4c2391a` |
