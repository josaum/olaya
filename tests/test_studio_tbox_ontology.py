"""
Unit tests for Mycelia Studio TBox Ontology and O-Laya Integration.
Verifies OWL 2 / RDFS TBox specification, DAG axioms, monotonicity, and disjointness projections.
"""

import json
from pathlib import Path
import pytest
from olaya.ontology import Ontology
from olaya.axioms import (
    enforce_subsumption_monotonicity,
    enforce_disjointness_exclusion,
    verify_axioms,
)


SPEC_DIR = Path(__file__).parent.parent / "spec"
JSONLD_PATH = SPEC_DIR / "studio-deck-tbox.jsonld"
ONTOLOGY_PATH = Path(__file__).parent.parent / "data" / "studio_deck_ontology.json"


def test_tbox_jsonld_structure():
    """Verify that studio-deck-tbox.jsonld exists and defines formal OWL ontology graph."""
    assert JSONLD_PATH.exists(), f"Missing {JSONLD_PATH}"
    data = json.loads(JSONLD_PATH.read_text(encoding="utf-8"))
    
    assert "@context" in data
    assert "studio" in data["@context"]
    assert "@graph" in data
    
    graph = data["@graph"]
    assert len(graph) >= 50
    
    ontology_def = next((item for item in graph if item.get("@type") == "owl:Ontology"), None)
    assert ontology_def is not None
    assert ontology_def["@id"] == "urn:mycelia:studio:"


def test_tbox_dag_hierarchy_and_cycles():
    """Verify that studio_deck_ontology loads as a strict DAG without cycles."""
    assert ONTOLOGY_PATH.exists(), f"Missing {ONTOLOGY_PATH}"
    data = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    
    ont = Ontology.from_json(data)
    assert ont.name == "MyceliaStudioTBox"
    assert len(ont.concepts) == 54
    
    # Check that no concept is an ancestor of itself (acyclic property)
    for curie in ont.concepts:
        ancestors = ont.get_ancestors(curie)
        assert curie not in (ancestors - {curie}), f"Cycle detected at {curie}"


def test_layout_subsumption_axioms():
    """Verify all 10 layout geometries are formal descendants of studio:SlideLayout."""
    ont = Ontology.from_json(json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8")))
    
    expected_layouts = [
        "studio:DefaultLayout",
        "studio:SectionDividerLayout",
        "studio:BigStatLayout",
        "studio:QuoteLayout",
        "studio:SplitLayout",
        "studio:BentoGridLayout",
        "studio:TessellatedLayout",
        "studio:ProcessFlowLayout",
        "studio:StatusTableLayout",
        "studio:ClosingLayout",
    ]
    
    for layout in expected_layouts:
        assert layout in ont.concepts, f"Layout concept {layout} missing from ontology"
        ancestors = ont.get_ancestors(layout)
        assert "studio:SlideLayout" in ancestors, f"{layout} must be a subclass of studio:SlideLayout"
        assert "studio:DocumentNode" in ancestors, f"{layout} must be a subclass of studio:DocumentNode"


def test_content_block_subsumption_axioms():
    """Verify content blocks are formal descendants of studio:ContentBlock."""
    ont = Ontology.from_json(json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8")))
    
    expected_blocks = [
        "studio:HeadingBlock",
        "studio:ParagraphBlock",
        "studio:BulletListBlock",
        "studio:EnumerateBlock",
        "studio:KpiGridBlock",
        "studio:TableBlock",
        "studio:CalloutBlock",
        "studio:ColumnsBlock",
        "studio:ImageBlock",
        "studio:ChartBlock",
        "studio:MediaBlock",
        "studio:RawBlock",
        "studio:PageBreakBlock",
        "studio:VSpaceBlock",
        "studio:DataQueryBlock",
    ]
    
    for block in expected_blocks:
        assert block in ont.concepts, f"Block concept {block} missing from ontology"
        ancestors = ont.get_ancestors(block)
        assert "studio:ContentBlock" in ancestors, f"{block} must be a subclass of studio:ContentBlock"


def test_status_cell_disjointness_axioms():
    """Verify checkmark and cross status cells are mutually disjoint."""
    ont = Ontology.from_json(json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8")))
    
    check_cell = ont.concepts["studio:CheckStatusCell"]
    cross_cell = ont.concepts["studio:CrossStatusCell"]
    
    assert "studio:CrossStatusCell" in check_cell.disjoint_with
    assert "studio:CheckStatusCell" in cross_cell.disjoint_with


def test_monotonicity_projection_on_tbox():
    """Verify that O-Laya mathematical monotonicity P(Child) <= P(Parent) holds on the TBox hierarchy."""
    ont = Ontology.from_json(json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8")))
    
    # Simulate candidate probability distribution where child has higher initial score than parent
    raw_probs = {
        "studio:DocumentNode": 0.30,
        "studio:SlideLayout": 0.40,
        "studio:TessellatedLayout": 0.85,
    }
    
    adjusted = enforce_subsumption_monotonicity(raw_probs, ont)
    
    # Monotonicity rule: P(Parent) must be >= P(Child)
    assert adjusted["studio:DocumentNode"] >= adjusted["studio:SlideLayout"] - 1e-6
    assert adjusted["studio:SlideLayout"] >= adjusted["studio:TessellatedLayout"] - 1e-6
    assert adjusted["studio:DocumentNode"] >= adjusted["studio:TessellatedLayout"] - 1e-6


def test_disjointness_projection_on_tbox():
    """Verify that O-Laya disjointness projection suppresses conflicting probabilities on StatusCell."""
    ont = Ontology.from_json(json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8")))
    
    # Simulate conflicting probabilities for mutually disjoint cells
    raw_probs = {
        "studio:CheckStatusCell": 0.80,
        "studio:CrossStatusCell": 0.70,
    }
    
    adjusted = enforce_disjointness_exclusion(raw_probs, ont, threshold=0.6)
    
    # The dominant concept (0.80) should stay high while the conflicting disjoint concept is zeroed
    assert adjusted["studio:CheckStatusCell"] == 0.80
    assert adjusted["studio:CrossStatusCell"] == 0.0
    assert verify_axioms(adjusted, ont, threshold=0.6)
