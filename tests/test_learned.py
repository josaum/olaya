"""Tests for the learned OAEI matcher (features, training / decoding math, protocol hygiene).

Everything runs on tiny synthetic ontologies with the dependency-free ``HashEncoder``:
no model download, no fixture ontologies, a few seconds in total.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from olaya.benchmarks.features import (
    FEATURE_VERSION,
    PairFeatures,
    compute_pair_features,
    feature_groups,
    load_or_compute_pair_features,
    pair_cache_key,
)
from olaya.benchmarks.learned import (
    DECODERS,
    ModelSpec,
    PairDataset,
    Standardiser,
    best_operating_point,
    cross_validate_model,
    decode,
    decode_all,
    hard_negative_rows,
    hungarian_max,
    inner_select,
    inner_splits,
    select_features,
    train_model,
)
from olaya.benchmarks.oaei import OAEIAlignment, leave_one_pair_out_folds
from olaya.embeddings import HashEncoder
from olaya.ontology import Concept, ObjectProperty, Ontology

# -- synthetic data --------------------------------------------------------------------


def _onto(
    name: str, classes: dict[str, str | None], props: dict[str, tuple[str, str]], *, comment: str = ""
) -> Ontology:
    """``classes``: name -> parent (or None); ``props``: name -> (domain, range)."""
    o = Ontology(name)
    for c in classes:
        o.add_concept(
            Concept(curie=c, label=c, description=comment if c.endswith("Paper") else "", metadata={"has_label": True})
        )
    for c, p in classes.items():
        if p:
            o.add_subsumption(c, p)
    for p, (d, r) in props.items():
        o.add_property(ObjectProperty(curie=p, domains={d}, ranges={r}))
    return o


def _pair(seed_name: str = "") -> tuple[Ontology, Ontology, list[OAEIAlignment]]:
    src = _onto(
        "src" + seed_name,
        {
            "Document": None,
            "Paper": "Document",
            "Review": "Document",
            "Person": None,
            "Author": "Person",
            "Venue": None,
        },
        {"hasAuthor": ("Paper", "Author"), "writesReview": ("Person", "Review")},
        comment="A scientific paper submitted to a conference",
    )
    tgt = _onto(
        "tgt" + seed_name,
        {
            "Document": None,
            "Article": "Document",
            "Review": "Document",
            "Person": None,
            "Writer": "Person",
            "Location": None,
        },
        {"authoredBy": ("Article", "Writer"), "reviews": ("Person", "Review")},
    )
    ref = [
        OAEIAlignment("Document", "Document"),
        OAEIAlignment("Paper", "Article"),
        OAEIAlignment("Review", "Review"),
        OAEIAlignment("Person", "Person"),
        OAEIAlignment("Author", "Writer"),
        OAEIAlignment("hasAuthor", "authoredBy"),
    ]
    return src, tgt, ref


@pytest.fixture(scope="module")
def features() -> PairFeatures:
    src, tgt, _ = _pair()
    return compute_pair_features(src, tgt, {"hash": HashEncoder()})


# -- features ------------------------------------------------------------------------


def test_feature_matrix_covers_same_kind_pairs_only(features: PairFeatures):
    pf = features
    assert pf.shape == (8, 8)  # 6 classes + 2 properties on each side
    assert pf.n_pairs == 6 * 6 + 2 * 2
    kinds = pf.pair_kinds()
    assert kinds.sum() == 36 and (~kinds).sum() == 4
    for i, j in zip(pf.rows, pf.cols, strict=True):
        assert pf.src_kinds[i] == pf.tgt_kinds[j]
    assert pf.X.shape == (pf.n_pairs, len(pf.feature_names))
    assert np.isfinite(pf.X).all()
    groups = feature_groups(pf.feature_names)
    assert set(groups) == {"lex", "emb", "hybrid", "rank", "struct", "meta"}
    assert "emb:hash:label_csls" in pf.feature_names and "struct:parents" in pf.feature_names
    assert "hybrid:cascade" in pf.feature_names and "hybrid:cascade_alpha02" in pf.feature_names


def test_lexical_features_have_expected_values(features: PairFeatures):
    pf = features
    names = dict(zip(pf.pair_names(), range(pf.n_pairs), strict=True))

    def f(a: str, b: str, feat: str) -> float:
        return float(pf.X[names[(a, b)], pf.feature_names.index(feat)])

    assert f("Document", "Document", "lex:exact") == 1.0
    assert f("Document", "Document", "lex:levenshtein") == pytest.approx(1.0, abs=1e-5)
    assert f("Paper", "Article", "lex:exact") == 0.0
    assert f("Review", "Review", "lex:head_equal") == 1.0
    assert f("Document", "Document", "rank:lex:mutual") == 1.0
    assert f("Document", "Document", "rank:lex:row_rank") == 0.0
    assert f("Paper", "Location", "rank:lex:row_margin") < 0.0 < f("Document", "Document", "rank:lex:row_margin")
    # Paper and Article share the parent Document, which matches exactly: parent similarity is 1 and present.
    assert f("Paper", "Article", "struct:parents") == pytest.approx(1.0, abs=1e-5)
    assert f("Paper", "Article", "struct:parents_present") == 1.0
    assert f("Document", "Document", "struct:parents_present") == 0.0  # roots have no parents
    assert f("Document", "Document", "struct:depth_diff") == 0.0 and f("Paper", "Document", "struct:depth_diff") == 1.0
    assert f("Paper", "Article", "meta:is_class") == 1.0 and f("hasAuthor", "authoredBy", "meta:is_class") == 0.0
    assert f("Paper", "Article", "meta:comment_either") == 1.0 and f("Review", "Review", "meta:comment_either") == 0.0
    # Properties: domains Paper / Article and ranges Author / Writer are the structural neighbourhoods.
    assert (
        f("hasAuthor", "authoredBy", "struct:domains_present") == 1.0
        and f("hasAuthor", "authoredBy", "struct:parents_present") == 0.0
    )


def test_labels_and_matrix_round_trip(features: PairFeatures):
    _, _, ref = _pair()
    y = features.labels(ref)
    assert y.sum() == len(ref)
    m = features.to_matrix(y, fill=-1.0)
    i, j = features.src_names.index("Paper"), features.tgt_names.index("Article")
    assert m[i, j] == 1.0
    assert m[features.src_names.index("Paper"), features.tgt_names.index("authoredBy")] == -1.0  # cross kind


def test_feature_cache_round_trip_and_key(tmp_path: Path):
    src, tgt, _ = _pair()
    encs = {"hash": HashEncoder()}
    pf, cached = load_or_compute_pair_features(src, tgt, encs, cache_dir=tmp_path)
    assert not cached
    pf2, cached2 = load_or_compute_pair_features(src, tgt, encs, cache_dir=tmp_path)
    assert cached2
    assert pf2.feature_names == pf.feature_names and np.array_equal(pf2.X, pf.X)
    assert np.array_equal(pf2.src_emb_label, pf.src_emb_label) and np.array_equal(pf2.tgt_adj, pf.tgt_adj)
    # The key depends on the ontology content and the encoder set, not on object identity.
    k1 = pair_cache_key(src, tgt, ["hash"], "hash")
    assert k1 == pair_cache_key(src, tgt, ["hash"], "hash")
    assert k1 != pair_cache_key(src, tgt, ["hash", "other"], "hash")
    src.add_concept(Concept(curie="Extra", label="Extra", metadata={"has_label": True}))
    assert k1 != pair_cache_key(src, tgt, ["hash"], "hash")
    assert FEATURE_VERSION >= 1


# -- decoding ------------------------------------------------------------------------


def test_hungarian_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(60):
        n, m = (int(x) for x in rng.integers(1, 6, size=2))
        s = rng.random((n, m))
        got = hungarian_max(s)
        k = min(n, m)
        best = max(
            sum(s[r, c] for r, c in zip(rs, cs, strict=True))
            for rs in itertools.permutations(range(n), k)
            for cs in itertools.permutations(range(m), k)
        )
        assert sum(s[r, c] for r, c in got) == pytest.approx(best)
        assert len({r for r, _ in got}) == len(got) == len({c for _, c in got})
    assert hungarian_max(np.zeros((0, 3))) == []


def test_decoders_are_one_to_one_and_threshold_consistent(features: PairFeatures):
    pf = features
    rng = np.random.default_rng(1)
    probs = rng.random(pf.n_pairs)
    for dec in DECODERS:
        full = decode_all(pf, probs, dec)
        assert len({c.entity1 for c in full}) == len(full) == len({c.entity2 for c in full})
        assert [c.measure for c in full] == sorted((c.measure for c in full), reverse=True)
        cut = decode(pf, probs, dec, 0.5)
        assert cut == [c for c in full if c.measure >= 0.5]
        for c in full:  # same kind only
            assert (c.entity1 in pf.src_names[:6]) == (c.entity2 in pf.tgt_names[:6])


def test_decoders_recover_a_perfect_score_matrix(features: PairFeatures):
    _, _, ref = _pair()
    pf = features
    probs = pf.labels(ref) * 0.9 + 0.05
    for dec in DECODERS:
        assert {(c.entity1, c.entity2) for c in decode(pf, probs, dec, 0.5)} == {(c.entity1, c.entity2) for c in ref}


# -- training math -----------------------------------------------------------------------


def test_standardiser_and_feature_selection():
    X = np.array([[1.0, 2.0, 5.0], [3.0, 2.0, 7.0]], dtype=np.float32)
    std = Standardiser.fit(X)
    Z = std(X)
    assert np.allclose(Z.mean(axis=0), 0.0) and np.allclose(Z[:, 1], 0.0)  # constant column is left at 0
    names = ["lex:score", "emb:hash:label_cos", "rank:hash:row_rank", "struct:parents"]
    assert select_features(names, keep_prefixes=("lex",)) == [0]
    assert select_features(names, drop_substrings=("hash",)) == [0, 3]
    with pytest.raises(ValueError):
        select_features(names, keep_prefixes=("nothing",))


def _dataset(n: int = 6) -> dict[str, PairDataset]:
    enc = HashEncoder()
    data = {}
    for k in range(n):
        src, tgt, ref = _pair(str(k))
        pf = compute_pair_features(src, tgt, {"hash": enc})
        data[f"p{k}"] = PairDataset(f"p{k}", src, tgt, pf, ref, pf.labels(ref))
    return data


def test_hard_negatives_keep_every_positive():
    data = _dataset(1)
    pd = data["p0"]
    rows = hard_negative_rows(pd.features, pd.y, top_k=1, n_random=2, seed=0)
    assert set(np.flatnonzero(pd.y > 0)) <= set(rows)
    assert len(rows) < pd.features.n_pairs


@pytest.mark.parametrize("kind", ["logreg", "mlp", "gbt", "emb_mlp", "gcn_mlp"])
def test_models_learn_the_synthetic_alignment(kind: str):
    data = _dataset(3)
    spec = ModelSpec(kind, kind, epochs=150, lr=0.05, negatives="hard" if kind == "gbt" else "all", n_trees=20)
    scorer = train_model(spec, [data["p0"], data["p1"]])
    probs = scorer.score(data["p2"])
    assert probs.shape == (data["p2"].features.n_pairs,) and ((probs >= 0) & (probs <= 1)).all()
    y = data["p2"].y
    assert probs[y > 0].min() > probs[y == 0].mean()
    assert probs[y > 0].mean() > 0.5
    dec, th, f1 = best_operating_point([(data["p2"], probs)])
    assert f1 >= 0.8
    imp = scorer.importances()
    if kind in ("logreg", "gbt"):
        assert imp is not None and set(imp) == set(data["p2"].features.feature_names)
    else:
        assert imp is None


def test_training_is_deterministic():
    data = _dataset(2)
    a = train_model(ModelSpec("m", "mlp", epochs=30), [data["p0"]]).score(data["p1"])
    b = train_model(ModelSpec("m", "mlp", epochs=30), [data["p0"]]).score(data["p1"])
    assert np.array_equal(a, b)


# -- protocol: the held-out pair is never seen by the inner selection ------------------------


def test_inner_splits_partition_the_training_pairs():
    pairs = [f"p{k}" for k in range(7)]
    splits = inner_splits(pairs, 3)
    assert len(splits) == 3
    held = [h for _, h in splits]
    assert sorted(sum(held, [])) == sorted(pairs)  # every training pair is inner-held-out exactly once
    for tr, h in splits:
        assert not set(tr) & set(h) and set(tr) | set(h) == set(pairs)


def test_inner_selection_never_touches_the_held_out_pair(monkeypatch: pytest.MonkeyPatch):
    import olaya.benchmarks.learned as learned

    data = _dataset(4)
    folds = leave_one_pair_out_folds(list(data))
    seen: list[set[str]] = []
    original = learned.train_model

    def spy(spec: ModelSpec, train: list[PairDataset]) -> learned.Scorer:
        seen.append({pd.pair for pd in train})
        return original(spec, train)

    monkeypatch.setattr(learned, "train_model", spy)
    held = "p0"
    train, test = folds[held]
    spec = ModelSpec("lr", "logreg", epochs=30)
    inner_select(spec, data, train, n_splits=3)
    assert seen, "inner selection trained nothing"
    for s in seen:
        assert held not in s
        assert s < set(train)  # strict subset: each inner model also leaves inner-held-out pairs out
    # And the full nested CV: every model trained for a fold excludes that fold's test pairs.
    seen.clear()
    res = cross_validate_model(spec, data, folds, inner_splits_n=3)
    assert res.folds == 4 and len(res.pairs) == 4
    assert set(res.chosen) == set(folds)
    for fold in folds:
        assert res.chosen[fold]["decoder"] in DECODERS
    # The spy cannot attribute a training set to a fold directly, but no training set may ever contain
    # all four pairs (every fold leaves at least its own test pair out).
    assert all(len(s) < len(data) for s in seen)
    assert res.micro("M3")["f1"] >= 0.8
    d = res.to_dict()
    assert d["model"] == "lr" and d["pairs"][0]["pair"] in data
