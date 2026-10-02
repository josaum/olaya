"""
Tests for Distilled Native Neural Head.
"""

import json
import pytest
import torch

from olaya.distilled_head import (
    DistilledNeuralHead,
    DistilledWeights,
    fnv1a_hash,
    train_distilled_head_for_ontology,
)
from olaya.ontology import Concept, Ontology


def test_fnv1a_hash_deterministic():
    h1 = fnv1a_hash("cardiovascular", 4096)
    h2 = fnv1a_hash("cardiovascular", 4096)
    assert h1 == h2
    assert 0 <= h1 < 4096

    h3 = fnv1a_hash("diabetes", 4096)
    assert h1 != h3


def test_distilled_head_forward_and_export():
    concepts = ["C1", "C2", "C3", "C4"]
    head = DistilledNeuralHead(concept_curies=concepts, dim=32, hidden_dim=48, num_buckets=512)

    # Encode query
    bucket_ids = [fnv1a_hash("heart", 512), fnv1a_hash("failure", 512)]
    idfs = [2.5, 3.1]
    q_vec = head.encode_tokens_and_idfs(bucket_ids, idfs)
    assert q_vec.shape == (1, 32)
    assert torch.isclose(torch.norm(q_vec, p=2), torch.tensor(1.0), atol=1e-4)

    # Score all concepts
    logits = head(q_vec)
    assert logits.shape == (1, 4)

    # Test morphism scoring
    parents_map = {0: [1]}  # C1 -> C2
    disjoint_map = {0: [3]} # C1 disjoint C4
    scores = head.score_with_morphisms(q_vec, [0, 1, 2], parents_map, disjoint_map)
    assert len(scores) == 3

    # Test export to JSON and binary
    weights = head.export_weights()
    json_str = weights.to_json()
    assert "leio-cone-v1" in json_str
    parsed = json.loads(json_str)
    assert parsed["dim"] == 32
    assert len(parsed["concept_curies"]) == 4

    bin_bytes = weights.to_binary()
    assert bin_bytes[:4] == b"LEIO"
    assert len(bin_bytes) > 1000

    # Roundtrip from weights
    head_loaded = DistilledNeuralHead.from_weights(weights)
    q_vec_loaded = head_loaded.encode_tokens_and_idfs(bucket_ids, idfs)
    assert torch.allclose(q_vec, q_vec_loaded, atol=1e-5)


def test_train_distilled_head_on_ontology():
    ont = Ontology(name="test")
    ont.add_concept(Concept("C_ROOT", "Root Concept", "top level", parents=set()))
    ont.add_concept(Concept("C_HEART", "Heart Disease", "cardiac condition", parents={"C_ROOT"}, synonyms=["cardiac arrest"]))
    ont.add_concept(Concept("C_LUNG", "Lung Disease", "pulmonary condition", parents={"C_ROOT"}, disjoint_with={"C_HEART"}))

    samples = [
        ("cardiac arrest severe chest pain", "C_HEART"),
        ("heart failure coronary artery disease", "C_HEART"),
        ("pulmonary condition shortness of breath asthma", "C_LUNG"),
    ]

    head = train_distilled_head_for_ontology(
        ont,
        samples,
        epochs=5,
        dim=32,
        hidden_dim=48,
        num_buckets=256,
    )
    assert head is not None

    b, idf = [fnv1a_hash("cardiac", 256)], [2.0]
    q_vec = head.encode_tokens_and_idfs(b, idf)
    logits = head(q_vec).squeeze(0)
    heart_idx = head.curie_to_idx["C_HEART"]
    lung_idx = head.curie_to_idx["C_LUNG"]
    # Heart score should be higher than lung score for cardiac query
    assert logits[heart_idx] > logits[lung_idx]


def test_create_zero_shot_head_for_ontology():
    from olaya.distilled_head import create_zero_shot_head_for_ontology

    ont = Ontology(name="zero_shot_test")
    ont.add_concept(Concept("ANIMAL", "Animal", "living organism", parents=set()))
    ont.add_concept(Concept("MAMMAL", "Mammal", "warm blooded animal", parents={"ANIMAL"}))
    ont.add_concept(Concept("DOG", "Canine Dog", "domesticated dog puppy", parents={"MAMMAL"}))

    # Instant zero-shot initialization - ZERO samples, ZERO training epochs
    head = create_zero_shot_head_for_ontology(ont, dim=32, hidden_dim=48, num_buckets=256)
    assert head.concept_embeddings.shape == (3, 32)
    # Vectors should be non-zero and unit normalized
    norms = torch.norm(head.concept_embeddings, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-3)

    # Canine puppy query should match DOG highest zero-shot
    b, idf = [fnv1a_hash("puppy", 256), fnv1a_hash("dog", 256)], [2.0, 2.0]
    q_vec = head.encode_tokens_and_idfs(b, idf)
    logits = head(q_vec).squeeze(0)
    dog_idx = head.curie_to_idx["DOG"]
    mammal_idx = head.curie_to_idx["MAMMAL"]
    animal_idx = head.curie_to_idx["ANIMAL"]
    assert logits[dog_idx] >= logits[mammal_idx]
    assert logits[dog_idx] >= logits[animal_idx]
