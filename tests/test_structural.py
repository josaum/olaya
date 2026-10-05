"""Tests for the structural propagation matchers (stub encoders, no downloads)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from olaya.benchmarks.oaei import (
    HybridMatcher,
    HybridMatcherConfig,
    LexicalMatcher,
    OAEIConferenceBenchmark,
    select_mutual_best,
)
from olaya.benchmarks.structural import (
    RELATIONS,
    StructuralMatcher,
    StructuralMatcherConfig,
    build_graph,
    kind_mask,
    max_pooled_similarity,
    neighbour_aggregation,
    relative_similarity,
    similarity_flooding,
    smooth_embeddings,
)
from olaya.embeddings import HashEncoder
from olaya.ontology import Concept, ObjectProperty, Ontology

FIXTURES = Path(__file__).parent / "fixtures" / "oaei"


@pytest.fixture(scope="module")
def bench() -> OAEIConferenceBenchmark:
    return OAEIConferenceBenchmark(FIXTURES / "ontologies", FIXTURES / "reference_alignments")


def _onto(name: str, *, child: str, parent: str, sibling: str, prop: str) -> Ontology:
    """``child`` and ``sibling`` are subclasses of ``parent``; ``prop`` goes child -> parent."""
    o = Ontology(name)
    for c in (child, parent, sibling, "Unrelated"):
        o.add_concept(Concept(curie=c, label=c, metadata={"has_label": True}))
    o.add_subsumption(child, parent)
    o.add_subsumption(sibling, parent)
    o.add_disjointness(child, sibling)
    o.add_property(ObjectProperty(curie=prop, domains={child}, ranges={parent}))
    o.add_property(ObjectProperty(curie="other", domains={"Unrelated"}))
    return o


def _pair() -> tuple[Ontology, Ontology]:
    src = _onto("src", child="Paper", parent="Document", sibling="Review", prop="hasPart")
    tgt = _onto("tgt", child="Article", parent="Document", sibling="Critique", prop="contains")
    return src, tgt


# -- graph extraction ------------------------------------------------------------------


def test_build_graph_relations_and_order():
    src, _ = _pair()
    g = build_graph(src)
    assert g.names == ["Document", "Paper", "Review", "Unrelated", "hasPart", "other"]  # classes, then properties
    assert g.kinds == ["class"] * 4 + ["property"] * 2
    i = g.index
    assert g.relations["subClassOf"][i["Paper"], i["Document"]] == 1.0
    assert g.relations["subClassOf"][i["Document"], i["Paper"]] == 0.0  # directed
    assert (
        g.relations["disjointWith"][i["Paper"], i["Review"]]
        == g.relations["disjointWith"][i["Review"], i["Paper"]]
        == 1
    )
    assert g.relations["domain"][i["hasPart"], i["Paper"]] == 1.0
    assert g.relations["range"][i["hasPart"], i["Document"]] == 1.0
    assert set(g.relations) == set(RELATIONS)
    nb = g.neighbourhoods()
    assert nb["siblings"][i["Paper"], i["Review"]] and not nb["siblings"][i["Paper"], i["Paper"]]
    assert nb["domain_of"][i["Paper"], i["hasPart"]]
    und = g.undirected()
    assert und[i["Document"], i["hasPart"]] == 1.0 and np.allclose(und, und.T) and und.trace() == 0


def test_kind_mask_blocks_cross_kind_pairs():
    src, tgt = _pair()
    m = kind_mask(build_graph(src), build_graph(tgt))
    assert m.shape == (6, 6)
    assert m[:4, :4].all() and m[4:, 4:].all() and not m[:4, 4:].any()


# -- the three propagation rules ---------------------------------------------------------


def _seed() -> tuple[np.ndarray, object, object]:
    src, tgt = _pair()
    gs, gt = build_graph(src), build_graph(tgt)
    seed = np.zeros((gs.n, gt.n))
    seed[gs.index["Document"], gt.index["Document"]] = 1.0
    seed[gs.index["Paper"], gt.index["Article"]] = 0.3  # weak, but parents match and their props match
    seed[gs.index["Paper"], gt.index["Unrelated"]] = 0.3  # equally weak, no structural support
    seed[gs.index["hasPart"], gt.index["contains"]] = 0.2
    return seed, gs, gt


def test_similarity_flooding_prefers_structurally_supported_pair():
    seed, gs, gt = _seed()
    sigma, used = similarity_flooding(seed, gs, gt)
    assert 0 < used <= 50
    assert sigma.max() == pytest.approx(1.0)
    assert sigma[gs.index["Paper"], gt.index["Article"]] > sigma[gs.index["Paper"], gt.index["Unrelated"]]
    assert sigma[gs.index["hasPart"], gt.index["contains"]] > 0.0
    assert not sigma[:4, 4:].any()  # cross-kind pairs stay at zero
    # Deterministic.
    again, _ = similarity_flooding(seed, gs, gt)
    assert np.array_equal(sigma, again)


def test_relative_similarity_is_one_on_mutual_best_pairs():
    sigma = np.array([[0.8, 0.2], [0.1, 0.4]])
    rel = relative_similarity(sigma)
    assert rel[0, 0] == pytest.approx(1.0) and rel[1, 1] == pytest.approx(1.0)
    assert rel[0, 1] == pytest.approx(0.2 / 0.8) and rel[1, 0] == pytest.approx(0.1 / 0.8)
    assert relative_similarity(np.zeros((2, 2))).sum() == 0.0
    seed, gs, gt = _seed()
    sig_rel, _ = similarity_flooding(seed, gs, gt, normalise="relative")
    assert sig_rel[gs.index["Paper"], gt.index["Article"]] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        similarity_flooding(seed, gs, gt, normalise="bogus")


def test_max_pooled_similarity_matches_brute_force():
    rng = np.random.default_rng(0)
    sim = rng.random((5, 4))
    sa = rng.random((5, 5)) > 0.5
    ta = rng.random((4, 4)) > 0.5
    out = max_pooled_similarity(sim, sa, ta)
    for a in range(5):
        for b in range(4):
            na, nb = np.flatnonzero(sa[a]), np.flatnonzero(ta[b])
            expected = sim[np.ix_(na, nb)].max() if len(na) and len(nb) else -np.inf
            assert out[a, b] == expected


def test_neighbour_aggregation_is_a_convex_mix_and_converges():
    seed, gs, gt = _seed()
    sim, used = neighbour_aggregation(seed, gs, gt, weight=0.3, iterations=20)
    assert 0 < used < 20  # converged before the cap
    pa, pu = sim[gs.index["Paper"], gt.index["Article"]], sim[gs.index["Paper"], gt.index["Unrelated"]]
    assert pa > 0.3 > pu  # supported pair is lifted, unsupported pair (shared types, zero neighbours) is lowered
    assert sim.min() >= 0.0 and sim.max() <= 1.0
    assert not sim[:4, 4:].any()
    same, used0 = neighbour_aggregation(seed, gs, gt, weight=0.0)
    assert used0 == 0 and np.array_equal(same, np.where(kind_mask(gs, gt), seed, 0.0))


def test_smooth_embeddings_pulls_neighbours_together_and_keeps_unit_norm():
    src, _ = _pair()
    g = build_graph(src)
    rng = np.random.default_rng(1)
    e = rng.standard_normal((g.n, 8)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    s = smooth_embeddings(e, g.undirected(), beta=0.5, rounds=1)
    assert np.allclose(np.linalg.norm(s, axis=1), 1.0, atol=1e-5)
    i = g.index
    before = float(e[i["Paper"]] @ e[i["Document"]])
    after = float(s[i["Paper"]] @ s[i["Document"]])
    assert after > before
    assert np.array_equal(smooth_embeddings(e, g.undirected(), beta=0.0), e)  # no-op at beta 0


# -- the matcher ----------------------------------------------------------------------


def test_structural_matcher_none_reproduces_the_seed_matcher():
    src, tgt = _pair()
    enc = HashEncoder()
    hybrid = HybridMatcher(enc, HybridMatcherConfig(combine="cascade", structure_weight=0.0))
    base = {(c.entity1, c.entity2): c.measure for c in hybrid.score_all(src, tgt)}
    none = StructuralMatcher(enc, StructuralMatcherConfig(method="none"))
    out = {(c.entity1, c.entity2): c.measure for c in none.score_all(src, tgt)}
    assert out == base
    assert none.name.endswith("+structure[none]")
    with pytest.raises(ValueError):
        StructuralMatcher(enc, StructuralMatcherConfig(method="bogus"))


def test_structural_matcher_without_encoder_uses_the_lexical_seed():
    src, tgt = _pair()
    lex = {(c.entity1, c.entity2): c.measure for c in LexicalMatcher().score_all(src, tgt)}
    m = StructuralMatcher(None, StructuralMatcherConfig(method="neighbour", weight=0.3))
    out = {(c.entity1, c.entity2): c.measure for c in m.score_all(src, tgt)}
    assert lex[("Document", "Document")] == 1.0 and lex.get(("Paper", "Article"), 0.0) < 0.1
    # The convex mix keeps an exact match above (1 - w) and lifts a pair that has almost no
    # lexical evidence but whose parents / properties match.
    assert 0.7 <= out[("Document", "Document")] <= 1.0
    assert out[("Paper", "Article")] > lex.get(("Paper", "Article"), 0.0) + 0.05
    assert m.name.startswith("olaya-lexical+structure[neighbour")
    assert m.last_iterations[("src", "tgt")] > 0


@pytest.mark.parametrize("method", ["flooding", "flooding_relative", "neighbour", "smooth"])
def test_structural_matcher_on_real_pair_with_stub_encoder(bench: OAEIConferenceBenchmark, method: str):
    cache: dict = {}
    cfg = StructuralMatcherConfig(method=method, weight=0.3, iterations=10 if method != "smooth" else 1)
    m = StructuralMatcher(HashEncoder(), cfg, seed_cache=cache)
    res = bench.evaluate_pair(m, "cmt", "ekaw", 0.5)
    assert res.results["M1"].f1 >= 0.6
    assert res.system_size <= 30
    alignment = bench.align(m, "cmt", "ekaw", 0.5)
    assert len({c.entity1 for c in alignment}) == len(alignment) == len({c.entity2 for c in alignment})
    if method != "smooth":
        assert cache  # the seed was memoised for reuse by other structural configs


def test_seed_cache_is_shared_between_structural_configs():
    src, tgt = _pair()
    cache: dict = {}
    a = StructuralMatcher(None, StructuralMatcherConfig(method="neighbour", weight=0.1), seed_cache=cache)
    b = StructuralMatcher(None, StructuralMatcherConfig(method="flooding", weight=0.5), seed_cache=cache)
    a.score_all(src, tgt)
    assert len(cache) == 1
    b.score_all(src, tgt)
    assert len(cache) == 1  # same lexical seed, reused
    assert select_mutual_best(b.score_all(src, tgt))  # and still produces an alignment
