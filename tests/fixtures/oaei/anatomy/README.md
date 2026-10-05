# OAEI Anatomy track fixtures (mouse / human)

Unmodified upstream files from the OAEI Anatomy track, vendored so that
`scripts/train_reranker.py` (cross-encoder reranker training) runs offline in a
clean checkout.  They are the Adult Mouse Anatomy ontology, the anatomy fragment of
the NCI Thesaurus (human) and the reference alignment between them (1516
equivalence correspondences).  The track is used **only as training / validation
data** for the reranker evaluated on the Conference track
(`benchmarks/results/oaei-reranker.md`); the Conference track is never trained on.

Source (downloaded 2026-10-04 UTC):

- `https://oaei.ontologymatching.org/2025/anatomy/anatomy-dataset.zip`
  (SHA-256 `0ef7b817c612fb7503a481334b939a3cf6ef2595799810e11799cc04514900b6`, 329 860 bytes).
  The archive also contains `trivial.rdf` (the subset of the reference that a trivial
  string matcher finds) and `__MACOSX` resource forks; both were left out.
  The 2024 index page links to the same file name; the 2023 index page links to the
  2019 archive.  The dataset has not changed since the 2010 / 2011 file dates in the
  archive.

| file | SHA-256 | bytes |
|---|---|---|
| `mouse.owl` | `93756393d6306c8f332c884401aa447cc4b2557f0a6be3e0efd988a943cb68f8` | 1 402 405 |
| `human.owl` | `5c1ca432d9845f1abb36ecfce313df46d8c56d2fb4137d0590ec4e4a6a9b05bf` | 3 436 928 |
| `reference.rdf` | `b6a6b12f3e7a786e5b58c4898024f0f483426e9036bdc3939aa61b046dbc26c4` | 329 284 |

Every file is byte-for-byte identical to the archive member; nothing was
reformatted or edited.  Upstream authors retain their rights (the Adult Mouse
Anatomy ontology is from the Jackson Laboratory / GXD project; the NCI Thesaurus is
from the US National Cancer Institute); this fixture inclusion does not relicense
their data under the repository code license.

Upstream requested citation (track description):
Zlatan Dragisic, Valentina Ivanova, Huanyu Li and Patrick Lambrix. *Experiences
from the Anatomy track in the Ontology Alignment Evaluation Initiative*. Journal of
Biomedical Semantics 8:56, 2017.  See also the OAEI Anatomy track pages,
`https://oaei.ontologymatching.org/<year>/anatomy/`.
