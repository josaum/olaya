"""Unit tests for Category-Theoretic Neural Architecture, Loss Functions, and Morphism Propagation."""

import pytest
import torch
import torch.nn as nn

from olaya.category_theory import (
    CategoricalDisjointnessLoss,
    CategoricalMorphismLayer,
    FunctorMonotonicityLoss,
    OrderConeEmbeddingLoss,
    extract_morphism_matrices,
)
from olaya.model import OntologyDecisionModel
from olaya.ontology import Concept, Ontology


@pytest.fixture
def sample_poset_ontology() -> Ontology:
    """Category poset:
            ROOT (top)
           /    \
      VEHICLE   ANIMAL  (disjoint)
        |         |
       CAR       DOG
    """
    onto = Ontology("test_poset")
    onto.add_concept(Concept("ROOT", "Root", "Top object"))
    onto.add_concept(Concept("VEHICLE", "Vehicle", "Transport device"))
    onto.add_concept(Concept("ANIMAL", "Animal", "Living organism"))
    onto.add_concept(Concept("CAR", "Car", "Four wheeled automobile"))
    onto.add_concept(Concept("DOG", "Dog", "Canine pet"))

    # Subsumptions
    onto.add_subsumption("VEHICLE", "ROOT")
    onto.add_subsumption("ANIMAL", "ROOT")
    onto.add_subsumption("CAR", "VEHICLE")
    onto.add_subsumption("DOG", "ANIMAL")

    # Disjointness
    onto.add_disjointness("VEHICLE", "ANIMAL")
    return onto


def test_extract_morphism_matrices(sample_poset_ontology):
    onto = sample_poset_ontology
    curies = ["CAR", "VEHICLE", "DOG", "ANIMAL", "ROOT"]
    sub_m, p_m, c_m, disj_m = extract_morphism_matrices(curies, onto)

    assert sub_m.shape == (5, 5)
    assert p_m.shape == (5, 5)
    assert c_m.shape == (5, 5)
    assert disj_m.shape == (5, 5)

    idx = {c: i for i, c in enumerate(curies)}

    # Subsumptions: CAR <= VEHICLE, CAR <= ROOT, VEHICLE <= ROOT
    assert sub_m[idx["CAR"], idx["VEHICLE"]] == 1.0
    assert sub_m[idx["CAR"], idx["ROOT"]] == 1.0
    assert sub_m[idx["VEHICLE"], idx["ROOT"]] == 1.0
    # No reverse morphism: ROOT is NOT <= CAR
    assert sub_m[idx["ROOT"], idx["CAR"]] == 0.0

    # Direct parent: parent of CAR is VEHICLE
    assert p_m[idx["CAR"], idx["VEHICLE"]] == 1.0
    assert p_m[idx["CAR"], idx["ROOT"]] == 0.0  # Not direct parent

    # Direct child: child of VEHICLE is CAR
    assert c_m[idx["VEHICLE"], idx["CAR"]] == 1.0

    # Disjointness: VEHICLE disjoint with ANIMAL, and CAR inherited disjoint with DOG
    assert disj_m[idx["VEHICLE"], idx["ANIMAL"]] == 1.0
    assert disj_m[idx["ANIMAL"], idx["VEHICLE"]] == 1.0
    assert disj_m[idx["CAR"], idx["DOG"]] == 1.0
    assert disj_m[idx["CAR"], idx["CAR"]] == 0.0


def test_functor_monotonicity_loss_zero_on_monotone():
    loss_fn = FunctorMonotonicityLoss()
    # Batch size 1, 3 candidates: [CHILD, PARENT, UNRELATED]
    # Parent logit = 5.0 (high prob), Child logit = 1.0 (lower prob) -> Monotone!
    logits = torch.tensor([[1.0, 5.0, 0.0]], requires_grad=True)
    subsumption_mask = torch.zeros((1, 3, 3))
    subsumption_mask[0, 0, 1] = 1.0  # child=0, parent=1

    loss, viol = loss_fn(logits, subsumption_mask)
    assert loss.item() == 0.0
    assert viol.item() == 0.0


def test_functor_monotonicity_loss_positive_on_violation():
    loss_fn = FunctorMonotonicityLoss()
    # Child logit = 5.0 (high prob), Parent logit = 1.0 (lower prob) -> Violation!
    logits = torch.tensor([[5.0, 1.0, 0.0]], requires_grad=True)
    subsumption_mask = torch.zeros((1, 3, 3))
    subsumption_mask[0, 0, 1] = 1.0  # child=0, parent=1

    loss, viol = loss_fn(logits, subsumption_mask)
    assert loss.item() > 0.0
    assert viol.item() == pytest.approx(1.0, abs=1e-4)

    # Backward pass gradients check
    loss.backward()
    assert logits.grad is not None
    # Child logit gradient should be positive (wants to decrease child probability)
    assert logits.grad[0, 0] > 0.0
    # Parent logit gradient should be negative (wants to increase parent probability)
    assert logits.grad[0, 1] < 0.0


def test_categorical_disjointness_loss():
    loss_fn = CategoricalDisjointnessLoss()
    disjoint_mask = torch.zeros((1, 2, 2))
    disjoint_mask[0, 0, 1] = 1.0
    disjoint_mask[0, 1, 0] = 1.0

    # Case 1: Both disjoint candidates have high probability (violation)
    logits_viol = torch.tensor([[5.0, 5.0]], requires_grad=True)
    loss_viol, coll_viol = loss_fn(logits_viol, disjoint_mask)
    # Both probs are 0.5, prod is 0.25
    assert pytest.approx(loss_viol.item(), abs=1e-3) == 0.25
    assert coll_viol.item() == pytest.approx(1.0, abs=1e-4)

    loss_viol.backward()
    assert logits_viol.grad is not None

    # Case 2: One candidate dominates (P(A) -> 1, P(B) -> 0)
    logits_clean = torch.tensor([[10.0, -10.0]])
    loss_clean, coll_clean = loss_fn(logits_clean, disjoint_mask)
    assert loss_clean.item() < 1e-4
    assert coll_clean.item() == 0.0


def test_order_cone_embedding_loss():
    loss_fn = OrderConeEmbeddingLoss()
    # Batch size 1, 2 candidates: [CHILD, PARENT], hidden size 4
    subsumption_mask = torch.zeros((1, 2, 2))
    subsumption_mask[0, 0, 1] = 1.0  # child=0, parent=1

    # Case 1: Child is within cone (elementwise <= parent)
    h_child = torch.tensor([[1.0, 2.0, 0.5, 3.0]])
    h_parent = torch.tensor([[2.0, 3.0, 1.0, 4.0]])
    states_clean = torch.stack([h_child, h_parent], dim=1)  # [1, 2, 4]
    loss_clean = loss_fn(states_clean, subsumption_mask)
    assert loss_clean.item() == 0.0

    # Case 2: Child breaks cone (elementwise > parent in first dimension)
    h_viol_child = torch.tensor([[5.0, 2.0, 0.5, 3.0]])
    states_viol = torch.stack([h_viol_child, h_parent], dim=1)
    loss_viol = loss_fn(states_viol, subsumption_mask)
    assert loss_viol.item() > 0.0


def test_categorical_morphism_layer_identity_at_init():
    layer = CategoricalMorphismLayer(hidden_size=16)
    x = torch.randn(2, 4, 16)

    # 1. Forward without morphism matrices -> exactly identity
    out_no_mats = layer(x)
    assert torch.allclose(out_no_mats, x, atol=1e-6)

    # 2. Forward with zero morphism matrices -> exactly identity
    zeros_adj = torch.zeros(2, 4, 4)
    out_zeros = layer(x, parent_adj=zeros_adj, child_adj=zeros_adj, disjoint_adj=zeros_adj)
    assert torch.allclose(out_zeros, x, atol=1e-6)


def test_categorical_morphism_layer_propagation():
    layer = CategoricalMorphismLayer(hidden_size=8)
    # Force out_proj to non-zero to test message routing
    nn.init.eye_(layer.out_proj.weight)
    x = torch.randn(1, 3, 8)

    parent_adj = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    out = layer(x, parent_adj=parent_adj)
    assert out.shape == (1, 3, 8)
    # Gradients flow through layer
    loss = out.sum()
    loss.backward()
    assert layer.w_parent.weight.grad is not None


def test_decision_model_forward_compatibility():
    """Verify OntologyDecisionModel runs identically with and without morphism matrices."""
    from transformers import BertConfig, BertModel

    cfg = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    encoder = BertModel(cfg)
    model = OntologyDecisionModel(hidden_size=64, custom_encoder=encoder)

    B, L, K = 2, 32, 4
    input_ids = torch.randint(0, 1000, (B, L))
    attention_mask = torch.ones((B, L), dtype=torch.long)
    marker_pos = torch.tensor([[5, 10, 15, 20], [6, 12, 18, 24]], dtype=torch.long)
    marker_mask = torch.ones((B, K), dtype=torch.bool)
    qtype = torch.tensor([0, 2], dtype=torch.long)

    # Vanilla forward (used by ONNX export and standard runtime)
    logits_1, act_1 = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    assert logits_1.shape == (B, K)
    assert act_1.shape == (B, 2)

    # Category morphism forward
    p_adj = torch.zeros((B, K, K))
    p_adj[:, 0, 1] = 1.0
    logits_2, act_2, marker_states = model(
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        parent_adj=p_adj,
        return_marker_states=True,
    )
    assert logits_2.shape == (B, K)
    assert marker_states.shape == (B, K, 64)
