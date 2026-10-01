"""Unit tests for Real ABox evidence ingestion and Temperature/ECE Calibration."""

import numpy as np
import pytest
import torch

from olaya.calibration import CalibrationMetrics, TemperatureScaler, compute_ece, fit_temperature
from olaya.engine import OLaya
from olaya.model import OntologyDecisionModel
from olaya.ontology import create_incident_ontology
from olaya.real_abox import find_atlas_root, load_atlas_real_abox
from olaya.train import create_trainable_model


def test_real_abox_harvesting():
    root = find_atlas_root()
    if root is None:
        pytest.skip("Atlas repository root not found in standard paths.")

    facts = load_atlas_real_abox(root)
    assert len(facts) >= 500, f"Expected at least 500 real facts, got {len(facts)}"

    # Check key types are present
    source_types = {f.source_type for f in facts}
    assert "nylabone_sku" in source_types
    assert "nylabone_model" in source_types
    assert "sharepoint_fact" in source_types or "sharepoint_page" in source_types
    assert "web_fact" in source_types

    # Check target CURIEs
    target_curies = {f.target_curie for f in facts}
    assert "ManufacturerSKU" in target_curies
    assert "VariantListing" in target_curies
    assert any(c in target_curies for c in ["OralCareProduct", "ProductModel"])


def test_compute_ece_perfect_calibration():
    # Synthetic perfectly calibrated predictions
    probs = np.array([
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1],
        [0.1, 0.9],
    ])
    # 9 out of 10 match class 0 -> accuracy in bin [0.8-1.0] is 90%, confidence is 90%
    targets = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    metrics = compute_ece(probs, targets, num_bins=10)

    assert metrics.accuracy == 0.9
    assert metrics.ece < 0.05
    assert len(metrics.bin_accuracies) == 10
    assert len(metrics.bin_counts) == 10


def test_temperature_scaler_optimization():
    torch.manual_seed(42)
    # Overconfident uncalibrated logits
    targets = torch.randint(0, 4, (500,))
    logits = torch.randn(500, 4)
    logits[torch.arange(500), targets] += 1.5
    logits = logits * 3.0  # overconfident

    opt_temp, uncal, cal = fit_temperature(logits, targets)

    # Temperature should be > 1.0 to soften overconfident logits
    assert opt_temp > 1.0
    # ECE and NLL should improve
    assert cal.ece < uncal.ece
    assert cal.nll < uncal.nll


def test_create_trainable_model_backbone_switching():
    # Custom lightweight BERT
    m_light = create_trainable_model(vocab_size=1000, hidden_size=128, backbone="custom_bert")
    assert m_light.hidden_size == 128
    assert isinstance(m_light, OntologyDecisionModel)

    # Pretrained ModernBERT
    m_modern = create_trainable_model(backbone="answerdotai/ModernBERT-base")
    assert m_modern.hidden_size == 768
    assert isinstance(m_modern, OntologyDecisionModel)


def test_engine_loads_calibrated_temperature(tmp_path):
    onto = create_incident_ontology()
    ckpt_path = tmp_path / "test_model.pt"

    # Save a mock checkpoint with temperature
    model = create_trainable_model(vocab_size=1000, hidden_size=256)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"vocab_size": 1000, "hidden_size": 256, "backbone": "custom_bert"},
            "temperature": 1.15,
            "ece": 0.042,
        },
        str(ckpt_path),
    )

    engine = OLaya(ontology=onto, weights_path=ckpt_path)
    assert engine.calibrated_temperature == pytest.approx(1.15, rel=1e-3)


def test_gs1_ontology_loading_and_routing():
    from pathlib import Path
    from olaya.candidate_router import CandidateRouter
    from olaya.real_ontology_train import load_ontology_file

    gs1_path = Path("data/gs1_ontology.json")
    if not gs1_path.exists():
        pytest.skip("GS1 ontology json file not found.")

    onto = load_ontology_file(gs1_path)
    assert len(onto.concepts) == 55
    assert "Beverage" in onto.concepts
    assert "FoodBeverageTobaccoProduct" in onto.concepts
    assert "Product" in onto.concepts

    # Test subsumption lineage in GS1
    ancestors = onto.get_ancestors("Beverage")
    assert "FoodBeverageTobaccoProduct" in ancestors
    assert "Product" in ancestors
    assert onto.is_subsumed_by("Beverage", "Product")

    # Test candidate routing
    router = CandidateRouter(onto, max_candidates=5)
    candidates = router.route_candidates("Barcode GTIN scan for refrigerated beverage", top_k=3)
    curies = [c.curie for c, _ in candidates]
    assert "Beverage" in curies

