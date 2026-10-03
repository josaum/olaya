#!/usr/bin/env python3
"""
enrich_ontologies.py — Enrich DOID and GS1 ontology JSON files with
domain-specific synonyms derived from descriptions and hard-coded enrichments.
"""

import json
import re
from pathlib import Path

# ── Hard-coded DOID synonym enrichments ──────────────────────────────────────

DOID_SYNONYM_ENRICHMENTS = {
    "DOID:0001816": ["vascular cancer", "blood vessel cancer", "endothelial sarcoma"],  # angiosarcoma
    "DOID:162": ["malignant neoplasm", "malignant tumor", "cancer"],  # cancer
    "DOID:3571": ["liver cancer", "hepatic malignancy"],  # liver disease
    "DOID:9351": ["diabetes", "high blood sugar", "insulin resistance", "glucose intolerance"],  # diabetes
    "DOID:10652": [
        "Alzheimer",
        "dementia",
        "amyloid plaque",
        "cognitive decline",
        "neurofibrillary tangles",
    ],  # Alzheimers
    "DOID:1909": ["melanoma", "skin cancer", "BRAF mutation", "UV radiation cancer"],  # melanoma
    "DOID:3908": ["lung cancer", "pulmonary carcinoma", "bronchogenic carcinoma"],  # NSCLC
    "DOID:4": ["disease", "illness", "disorder", "condition", "pathology"],  # disease root
    "DOID:14566": ["infectious disease", "infection", "pathogen", "bacterial", "viral"],  # infectious agent
    "DOID:2351": ["COPD", "emphysema", "chronic bronchitis", "airflow obstruction", "pulmonary disease"],  # COPD
    "DOID:8778": ["Crohn disease", "inflammatory bowel disease", "IBD"],  # Crohn
    "DOID:0050589": ["inflammatory bowel disease", "IBD", "colitis"],  # IBD
    "DOID:2394": ["ovarian cancer", "ovarian carcinoma"],
    "DOID:1612": ["breast cancer", "breast carcinoma", "mammary tumor"],
    "DOID:4159": ["skin disease", "dermatological condition", "dermatosis"],
    "DOID:0060903": ["sepsis", "septicemia", "bloodstream infection", "bacteremia", "systemic infection"],
    "DOID:1686": ["hypertension", "high blood pressure", "arterial hypertension"],
    "DOID:3610": ["autoimmune disease", "autoimmunity", "self-reactive immune response"],
    "DOID:11044": ["peripheral artery disease", "vascular insufficiency", "limb ischemia"],
    "DOID:1579": ["respiratory disease", "lung disease", "pulmonary condition", "breathing disorder"],
}

# ── Hard-coded GS1 synonym enrichments ───────────────────────────────────────

GS1_SYNONYM_ENRICHMENTS = {
    "MilkButterCreamYogurtCheeseEggsSubstitutes": [
        "dairy",
        "dairy product",
        "perishable dairy",
        "refrigerated dairy",
        "milk product",
        "cheese product",
        "yogurt",
    ],
    "FreshFruitVegetables": ["produce", "fresh produce", "fresh food", "vegetables", "fruit", "perishable produce"],
    "FrozenFood": ["frozen", "frozen product", "deep freeze", "blast frozen", "IQF"],
    "MeatPoultryFish": ["meat", "poultry", "protein", "animal protein", "beef", "chicken", "fish", "seafood"],
    "BatteryProduct": ["battery", "lithium", "lithium ion", "rechargeable", "UN3480", "energy storage"],
    "HazardousMaterial": ["hazmat", "dangerous goods", "hazardous", "UN number", "MSDS", "SDS", "chemical"],
    "PharmaProduct": ["pharmaceutical", "drug", "medication", "medicine", "API", "active ingredient"],
    "MedicalDevice": ["medical device", "medical equipment", "hospital supply", "clinical device", "sterile"],
    "FoodBeverageTobaccoProduct": ["food", "beverage", "grocery", "consumable", "edible", "FMCG"],
    "RetailProductPackaging": ["packaging", "retail packaging", "consumer packaging", "label", "UPC"],
}

# ── OWL property patterns to strip before noun phrase extraction ──────────────

OWL_PROPS_RE = re.compile(
    r"\b(derives_from|is_a|part_of|located_in|has_part|participates_in|preceded_by)\b", re.IGNORECASE
)

# Simple domain-relevant terms for DOID noun-phrase extraction
DOID_DOMAIN_TERMS = {
    "disease",
    "cancer",
    "tumor",
    "carcinoma",
    "sarcoma",
    "lymphoma",
    "leukemia",
    "disorder",
    "syndrome",
    "infection",
    "inflammation",
    "deficiency",
    "failure",
    "lesion",
    "malignant",
    "benign",
    "acute",
    "chronic",
    "neoplasm",
    "neoplasia",
    "metastasis",
    "mutation",
    "receptor",
    "protein",
    "enzyme",
    "pathway",
    "immune",
    "autoimmune",
    "inflammatory",
    "vascular",
    "neurological",
    "cardiac",
    "pulmonary",
    "renal",
    "hepatic",
    "pancreatic",
    "endocrine",
    "metabolic",
    "bacterial",
    "viral",
    "fungal",
    "parasitic",
    "genetic",
    "hereditary",
    "congenital",
    "acquired",
    "progressive",
    "degenerative",
    "fibrosis",
}


def clean_description(desc: str) -> str:
    """Strip OWL property expressions from a description string."""
    return OWL_PROPS_RE.sub(" ", desc).strip()


# Ultra-generic phrases to skip (they appear in hundreds of disease descriptions)
GENERIC_PHRASES = {
    "malignant tumor",
    "malignant neoplasm",
    "benign tumor",
    "benign neoplasm",
    "tumor cells",
    "cancer cells",
    "malignant cells",
    "neoplastic cells",
    "cell tumor",
    "cell cancer",
    "cell carcinoma",
}


def extract_noun_phrases(desc: str) -> list[str]:
    """
    Simple heuristic: extract 2-3 word noun phrases from a description
    that include at least one SPECIFIC domain-relevant term (not just
    generic 'malignant', 'tumor', 'cancer').
    Returns up to 3 candidate phrases.
    """
    cleaned = clean_description(desc)
    # Tokenize into words (lowercase, letters only)
    words = re.findall(r"[a-z][a-z'-]+", cleaned.lower())
    phrases = []
    seen = set()

    for i in range(len(words)):
        for length in (3, 2):
            if i + length > len(words):
                continue
            chunk = words[i : i + length]
            # Must contain at least one domain term
            if not any(w in DOID_DOMAIN_TERMS for w in chunk):
                continue
            phrase = " ".join(chunk)
            # Skip ultra-generic phrases
            if phrase in GENERIC_PHRASES:
                continue
            # Skip phrases consisting entirely of generic high-freq words
            generic_only = {"malignant", "tumor", "neoplasm", "cancer", "benign", "cells", "cell"}
            if all(w in generic_only for w in chunk):
                continue
            if phrase not in seen and len(phrase) > 5:
                seen.add(phrase)
                phrases.append(phrase)
        if len(phrases) >= 3:
            break

    return phrases[:3]


def enrich_doid(path: Path) -> None:
    """Enrich DOID ontology with hard-coded and description-derived synonyms."""
    print(f"Reading DOID ontology from {path}...")
    data = json.loads(path.read_text(encoding="utf-8"))

    concepts = data["concepts"] if isinstance(data, dict) else data
    enriched = 0
    desc_derived = 0

    for concept in concepts:
        curie = concept.get("curie", "")
        existing_syns = set(s.lower() for s in concept.get("synonyms", []))
        new_syns = list(concept.get("synonyms", []))

        # Apply hard-coded enrichments
        if curie in DOID_SYNONYM_ENRICHMENTS:
            for syn in DOID_SYNONYM_ENRICHMENTS[curie]:
                if syn.lower() not in existing_syns:
                    new_syns.append(syn)
                    existing_syns.add(syn.lower())

        # Auto-derive from description if fewer than 3 synonyms
        desc = concept.get("description", "")
        if desc and len(new_syns) < 3:
            derived = extract_noun_phrases(desc)
            for phrase in derived:
                if phrase not in existing_syns and len(new_syns) < 5:
                    new_syns.append(phrase)
                    existing_syns.add(phrase)
                    desc_derived += 1

        if len(new_syns) != len(concept.get("synonyms", [])):
            enriched += 1
        concept["synonyms"] = new_syns

    print(f"  Enriched {enriched} DOID concepts ({desc_derived} description-derived phrases added)")

    # Write back
    if isinstance(data, dict):
        data["concepts"] = concepts
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved to {path}")


def enrich_gs1(path: Path) -> None:
    """Enrich GS1 ontology with hard-coded industry synonyms."""
    print(f"Reading GS1 ontology from {path}...")
    data = json.loads(path.read_text(encoding="utf-8"))

    concepts = data["concepts"] if isinstance(data, dict) else data
    enriched = 0

    for concept in concepts:
        curie = concept.get("curie", "")
        existing_syns = set(s.lower() for s in concept.get("synonyms", []))
        new_syns = list(concept.get("synonyms", []))

        # Apply hard-coded enrichments
        if curie in GS1_SYNONYM_ENRICHMENTS:
            for syn in GS1_SYNONYM_ENRICHMENTS[curie]:
                if syn.lower() not in existing_syns:
                    new_syns.append(syn)
                    existing_syns.add(syn.lower())
            if len(new_syns) != len(concept.get("synonyms", [])):
                enriched += 1

        concept["synonyms"] = new_syns

    print(f"  Enriched {enriched} GS1 concepts")

    # Write back
    if isinstance(data, dict):
        data["concepts"] = concepts
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved to {path}")


def verify_angiosarcoma(path: Path) -> None:
    """Verify angiosarcoma has 4+ synonyms including 'vascular cancer'."""
    data = json.loads(path.read_text(encoding="utf-8"))
    concepts = data["concepts"] if isinstance(data, dict) else data
    for c in concepts:
        if c.get("curie") == "DOID:0001816":
            syns = c.get("synonyms", [])
            has_vascular = any("vascular cancer" in s.lower() for s in syns)
            print("\nVerification — angiosarcoma (DOID:0001816):")
            print(f"  Synonyms ({len(syns)}): {syns}")
            print(f"  ✓ has 'vascular cancer': {has_vascular}")
            print(f"  ✓ >= 4 synonyms: {len(syns) >= 4}")
            return
    print("WARNING: DOID:0001816 not found!")


def main():
    base = Path(__file__).parent.parent / "docs" / "public"
    doid_path = base / "ontologies" / "doid_ontology.json"
    gs1_path = base / "gs1_ontology.json"

    enrich_doid(doid_path)
    enrich_gs1(gs1_path)
    verify_angiosarcoma(doid_path)
    print("\nDone. Both ontologies enriched successfully.")


if __name__ == "__main__":
    main()
