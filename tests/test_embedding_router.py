"""Embedding / hybrid routers and the engine's router + scorer injection, on a tiny synthetic ontology.

Everything runs on :class:`olaya.embeddings.HashEncoder` (token hashing, no model, no network) and the
bundled local tokenizer, so the tests never download anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from olaya.candidate_router import CandidateRouter
from olaya.embedding_router import (
    EmbeddingRouter,
    EmbeddingScorer,
    HybridRouter,
    build_class_texts,
    camel_to_words,
    class_context_text,
    lexical_rank,
    root_path,
    sibling_curies,
)
from olaya.embeddings import HashEncoder
from olaya.engine import INJECTED_MODEL_ID, UNTRAINED_MODEL_ID, OLaya
from olaya.ontology import Concept, Ontology


def tiny_ontology() -> Ontology:
    onto = Ontology("tiny")
    onto.add_concept(Concept("T:ROOT", "Thing", "Anything at all"))
    onto.add_concept(Concept("T:ANIMAL", "Animal", "A living creature", parents={"T:ROOT"}))
    onto.add_concept(Concept("T:VEHICLE", "Vehicle", "A machine for transport", parents={"T:ROOT"}))
    onto.add_concept(
        Concept("T:CAT", "Cat", "Small domestic feline that purrs", synonyms=["kitty", "feline"], parents={"T:ANIMAL"})
    )
    onto.add_concept(Concept("T:DOG", "Dog", "Domestic canine that barks", parents={"T:ANIMAL"}))
    onto.add_concept(Concept("T:TRUCK", "Truck", "Heavy road vehicle hauling cargo", parents={"T:VEHICLE"}))
    onto.add_concept(Concept("T:BICYCLE", "Bicycle", "Two wheeled pedal vehicle", parents={"T:VEHICLE"}))
    for child, parent in [
        ("T:ANIMAL", "T:ROOT"),
        ("T:VEHICLE", "T:ROOT"),
        ("T:CAT", "T:ANIMAL"),
        ("T:DOG", "T:ANIMAL"),
        ("T:TRUCK", "T:VEHICLE"),
        ("T:BICYCLE", "T:VEHICLE"),
    ]:
        onto.add_subsumption(child, parent)
    onto.add_disjointness("T:ANIMAL", "T:VEHICLE")
    return onto


@pytest.fixture
def onto() -> Ontology:
    return tiny_ontology()


@pytest.fixture
def encoder() -> HashEncoder:
    return HashEncoder(dim=128)


@pytest.fixture
def router(onto, encoder) -> EmbeddingRouter:
    return EmbeddingRouter(onto, encoder)


# --- class texts ------------------------------------------------------------------------------


def test_class_context_text_has_label_description_path_and_synonyms(onto):
    text = class_context_text(onto, "T:CAT")
    assert text.startswith("Cat.")
    assert "Small domestic feline that purrs." in text
    assert "Path: Thing > Animal > Cat." in text
    assert "Also called feline, kitty." in text
    assert "Dog" not in text  # siblings are deliberately excluded
    assert root_path(onto, "T:ROOT") == ["Thing"]
    assert "Path:" not in class_context_text(onto, "T:ROOT")
    assert camel_to_words("EX:DatabaseOutage_v2") == "Database Outage v2"
    texts = build_class_texts(onto)
    assert list(texts) == sorted(onto.concepts)


# --- embedding router ---------------------------------------------------------------------------


def test_router_ranks_by_shared_tokens_and_restricts_to_root(router):
    ranked = router.rank("the kitty purrs on the sofa")
    assert [c for c, _ in ranked][0] == "T:CAT"
    assert len(ranked) == 7
    assert ranked == sorted(ranked, key=lambda x: (-x[1], x[0]))
    under_vehicle = router.rank("the kitty purrs", root_curie="T:VEHICLE")
    assert {c for c, _ in under_vehicle} == {"T:VEHICLE", "T:TRUCK", "T:BICYCLE"}
    with pytest.raises(ValueError):
        router.rank("x", root_curie="T:NOPE")
    top = router.route_candidates("heavy cargo hauling truck", top_k=2)
    assert [c.curie for c, _ in top][0] == "T:TRUCK" and len(top) == 2
    sub = router.extract_decision_subgraph("heavy cargo hauling truck", max_candidates=1)
    assert set(sub.concepts) == {"T:TRUCK", "T:VEHICLE", "T:ROOT"}


def test_score_candidates_matches_similarities_and_floors_unknowns(router):
    sims = router.similarities("domestic canine barks")
    scores = router.score_candidates("domestic canine barks", ["T:DOG", "T:CAT", "T:UNKNOWN"])
    assert scores[0] == pytest.approx(float(sims[router.index["T:DOG"]]))
    assert scores[1] == pytest.approx(float(sims[router.index["T:CAT"]]))
    assert scores[2] == pytest.approx(float(sims.min()))
    assert scores[0] > scores[1]


def test_csls_router_demotes_hubs_but_keeps_scores_finite(onto, encoder):
    plain = EmbeddingRouter(onto, encoder)
    csls = EmbeddingRouter(onto, encoder, csls_k=3)
    assert csls.class_background is not None and csls.class_background.shape == (7,)
    assert np.all(np.isfinite(csls.similarities("two wheeled pedal")))
    assert csls.name.endswith("+csls3") and plain.name == f"embedding:{encoder.name}"
    # ``Vehicle`` shares its label token with both of its children ("... pedal vehicle", "... road
    # vehicle") and their path texts, so it is the hub: the largest class-side background. CSLS is
    # ``2 cos - r_query - r_class``, so relative to any other class the hub loses ``r_hub - r_other``.
    hub = int(np.argmax(csls.class_background))
    assert csls.curies[hub] == "T:VEHICLE"
    query = "two wheeled pedal vehicle"
    cos = plain.similarities(query)
    adj = csls.similarities(query)
    other = csls.index["T:CAT"]
    expected_gap = 2.0 * (cos[hub] - cos[other]) - (csls.class_background[hub] - csls.class_background[other])
    assert adj[hub] - adj[other] == pytest.approx(expected_gap, abs=1e-5)
    assert (adj[hub] - adj[other]) < 2.0 * (cos[hub] - cos[other])
    with pytest.raises(ValueError):
        EmbeddingRouter(onto, encoder, csls_k=0)
    with pytest.raises(ValueError):
        EmbeddingRouter(Ontology("empty"), encoder)


# --- hybrid router ------------------------------------------------------------------------------


def test_lexical_rank_is_empty_without_evidence(onto):
    lexical = CandidateRouter(onto)
    assert lexical_rank(lexical, "zzzz qqqq") == []
    ranked = lexical_rank(lexical, "truck cargo")
    assert ranked and ranked[0][0] == "T:TRUCK"


def test_hybrid_union_interleaves_and_rrf_fuses(onto, router):
    lexical = CandidateRouter(onto)
    union = HybridRouter(lexical, router, mode="union")
    rrf = HybridRouter(lexical, router, mode="rrf")
    query = "kitty purrs"  # lexical: only the synonym hit (Cat); embedding: Cat first too
    u = union.rank(query)
    assert u[0][0] == "T:CAT" and len(u) == 7 and len({c for c, _ in u}) == 7
    assert [s for _, s in u] == [1.0 / (1 + i) for i in range(7)]
    r = rrf.rank(query)
    assert r[0][0] == "T:CAT"
    # Cat is ranked first by both routers: lexical term + embedding term.
    assert r[0][1] == pytest.approx(0.5 / 61 + 0.5 / 61)
    # A class with no lexical evidence only receives the embedding term.
    emb_only = dict(r)["T:TRUCK"]
    emb_pos = [c for c, _ in router.rank(query)].index("T:TRUCK")
    assert emb_only == pytest.approx(0.5 / (60 + emb_pos + 1))
    # Without lexical evidence the hybrid degenerates to the embedding ranking.
    assert [c for c, _ in union.rank("zzzz")] == [c for c, _ in router.rank("zzzz")]
    assert [c for c, _ in rrf.rank("zzzz")] == [c for c, _ in router.rank("zzzz")]
    assert union.route_candidates(query, top_k=3)[0][0].curie == "T:CAT"
    assert "T:CAT" in union.extract_decision_subgraph(query, max_candidates=2).concepts
    np.testing.assert_allclose(rrf.score_candidates(query, ["T:CAT"]), router.score_candidates(query, ["T:CAT"]))
    with pytest.raises(ValueError):
        HybridRouter(lexical, router, mode="nope")
    with pytest.raises(ValueError):
        HybridRouter(CandidateRouter(Ontology("other")), router)


# --- scorer + engine injection -------------------------------------------------------------------


def test_scorer_validates_inputs(router):
    scorer = EmbeddingScorer(router, temperature=0.1)
    assert scorer.name == f"scorer({router.name})"
    assert scorer.score("kitty", ["T:CAT", "T:DOG"]).shape == (2,)
    with pytest.raises(ValueError):
        EmbeddingScorer(router, temperature=0.0)
    with pytest.raises(TypeError):
        EmbeddingScorer(object())


def test_engine_uses_injected_router_and_scorer(onto, router, tokenizer):
    scorer = EmbeddingScorer(router, temperature=0.05)
    engine = OLaya(ontology=onto, tokenizer=tokenizer, device="cpu", weights_path=False, router=router, scorer=scorer)
    assert engine.router is router and engine.scorer is scorer
    assert engine.model_id == f"{UNTRAINED_MODEL_ID}+scorer:{scorer.name}"
    result = engine.system_one("the kitty purrs on the sofa", {"q": {"type": "concept", "root": "T:ROOT"}})
    answer = result.answers["q"]
    assert result.model == engine.model_id
    assert answer.curie == "T:CAT"
    assert answer.lineage == ["T:ROOT", "T:ANIMAL", "T:CAT"]
    assert answer.axioms_verified is True
    assert sum(answer.candidate_probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    # Marginals: the ancestors carry the summed mass of their descendants.
    assert answer.probabilities["T:ANIMAL"] >= answer.probabilities["T:CAT"]
    assert answer.probabilities["T:ROOT"] == pytest.approx(1.0, abs=1e-3)
    # The scored leaves are Cat, Dog, Truck, Bicycle. Animal collects > 0.5 of the mass, so the
    # disjointness axiom (Animal vs Vehicle) zeroes both vehicles and the engine renormalises: the
    # survivors carry exactly softmax(similarity / temperature) restricted to the surviving leaves.
    probs = answer.candidate_probabilities
    assert set(probs) == {"T:CAT", "T:DOG", "T:TRUCK", "T:BICYCLE"}
    assert probs["T:TRUCK"] == 0.0 and probs["T:BICYCLE"] == 0.0
    survivors = ["T:CAT", "T:DOG"]
    sims = router.score_candidates("the kitty purrs on the sofa", survivors) / 0.05
    expected = np.exp(sims - sims.max())
    expected /= expected.sum()
    np.testing.assert_allclose([probs[c] for c in survivors], expected, atol=2e-4)
    # Other question types still run through the model (no scorer involvement, still valid).
    flat = engine.system_one("kitty", {"n": {"type": "noul"}, "c": {"type": "choice", "criteria": ["a", "b"]}})
    assert 0.0 <= flat.answers["n"].noul <= 1.0 and flat.answers["c"].choice in {"a", "b"}


def test_engine_without_injection_is_unchanged(onto, tokenizer):
    engine = OLaya(ontology=onto, tokenizer=tokenizer, device="cpu", weights_path=False)
    assert isinstance(engine.router, CandidateRouter) and engine.scorer is None
    assert engine.model_id == UNTRAINED_MODEL_ID
    result = engine.system_one("the kitty purrs", {"q": {"type": "concept"}})
    assert result.answers["q"].axioms_verified is True
    assert result.model == UNTRAINED_MODEL_ID
    injected = OLaya(ontology=onto, tokenizer=tokenizer, device="cpu", model=engine.model)
    assert injected.model_id == INJECTED_MODEL_ID


def test_engine_rejects_bad_router_or_scorer(onto, router, tokenizer):
    other = EmbeddingRouter(tiny_ontology(), HashEncoder(dim=16))
    with pytest.raises(ValueError):
        OLaya(ontology=onto, tokenizer=tokenizer, device="cpu", weights_path=False, router=other)
    with pytest.raises(TypeError):
        OLaya(ontology=onto, tokenizer=tokenizer, device="cpu", weights_path=False, scorer=object())

    class BadScorer:
        name = "bad"
        temperature = 0.1

        def score(self, state_text, candidate_curies):
            return np.zeros(len(candidate_curies) + 1)

    engine = OLaya(ontology=onto, tokenizer=tokenizer, device="cpu", weights_path=False, scorer=BadScorer())
    with pytest.raises(ValueError, match="scorer returned"):
        engine.system_one("kitty", {"q": {"type": "concept"}})


def test_sibling_curies(onto):
    assert sibling_curies(onto, "T:CAT") == ["T:DOG"]
    assert sibling_curies(onto, "T:ROOT") == []


def test_engine_routed_candidate_policy_scores_non_leaf_routed_concepts(onto, router, tokenizer):
    scorer = EmbeddingScorer(router, temperature=0.05)
    leaves = OLaya(ontology=onto, tokenizer=tokenizer, device="cpu", weights_path=False, router=router, scorer=scorer)
    routed = OLaya(
        ontology=onto,
        tokenizer=tokenizer,
        device="cpu",
        weights_path=False,
        router=router,
        scorer=scorer,
        candidate_policy="routed",
    )
    assert routed.model_id.endswith("+candidates:routed")
    q = {"q": {"type": "concept", "root": "T:ROOT"}}
    a_leaves = leaves.system_one("the kitty purrs on the sofa", q).answers["q"]
    a_routed = routed.system_one("the kitty purrs on the sofa", q).answers["q"]
    routed_curies = [
        c.curie for c, _ in router.route_candidates("the kitty purrs on the sofa", root_curie="T:ROOT", top_k=10)
    ]
    # Every routed concept competes, including ones that have a routed descendant.
    assert set(a_routed.candidate_probabilities) == set(routed_curies)
    # The leaf policy's candidates are the subset of routed concepts without routed children.
    assert set(a_leaves.candidate_probabilities) < set(a_routed.candidate_probabilities)
    assert any(onto.concepts[c].children for c in a_routed.candidate_probabilities)
    assert abs(sum(a_routed.candidate_probabilities.values()) - 1.0) < 1e-6
    # Marginals stay monotone under subsumption.
    for c, p in a_routed.probabilities.items():
        for parent in onto.concepts[c].parents:
            if parent in a_routed.probabilities:
                assert a_routed.probabilities[parent] >= p - 1e-9
    assert a_routed.axioms_verified
    with pytest.raises(ValueError):
        OLaya(ontology=onto, tokenizer=tokenizer, device="cpu", weights_path=False, candidate_policy="all")


def test_mined_hard_negatives_rank_non_gold_classes_deterministically():
    from olaya.embedding_router import mined_hard_negatives

    classes = ["A", "B", "C", "D"]
    c = np.eye(4, dtype=np.float32)
    q = np.array([[0.9, 0.8, 0.1, 0.0], [0.0, 0.2, 0.3, 0.9]], dtype=np.float32)
    mined = mined_hard_negatives(q, c, classes, ["A", "D"], per_query=2)
    assert mined == [["B", "C"], ["C", "B"]]
    assert mined_hard_negatives(q, c, classes, ["A", "D"], per_query=0) == [[], []]
    # A gold class is never its own negative even when it scores highest.
    assert "A" not in mined[0] and "D" not in mined[1]
