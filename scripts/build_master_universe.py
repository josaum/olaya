"""
Leio Universal Master Ontology Universe Builder.

Unifies the world's formal ontologies into a single zero-copy Arrow IPC table:
1. Upper-Level Foundations: BFO (ISO/IEC 21838-2), DOLCE UltraLite (DUL), gUFO
2. W3C Reference Standards: SKOS, PROV-O, Org, DCAT, Dublin Core (DCMI), FOAF, SIOC
3. Healthcare & Genomics: MONDO (19,000+ unified diseases), DOID (730), Gene Ontology (GO Slim)
4. Finance & Business: FIBO (17,100+ concepts), Schema.org (985), GS1 (1,907)
5. Manufacturing & Engineering: Industrial Ontologies Foundry (IOF Core)
6. Brazilian Standards: OntoPix (Pix payment system), APTO (B3 / CEPEA Agribusiness)
7. NATO & Tactical Defense: C2SIM & JC3IEDM Battlefield Operations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from olaya.ontology import Concept, Ontology

MODELS_DIR = Path("models")
DATA_DIR = Path("data")


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_universe() -> Ontology:
    print("🌌 Assembling the Leio Universal Master Ontology Universe...")
    master = Ontology(name="LeioUniversalMasterUniverse")

    json_sources = [
        # (filepath, domain_prefix, bridge_parent)
        (MODELS_DIR / "universe_bfo.json", "bfo", None),
        (MODELS_DIR / "universe_gufo.json", "gufo", "bfo:BFO_0000001"),
        (MODELS_DIR / "universe_dul.json", "dul", "bfo:BFO_0000001"),
        (MODELS_DIR / "universe_w3c.json", "w3c", "bfo:BFO_0000001"),
        (MODELS_DIR / "schemaorg.json", "schema", "bfo:BFO_0000001"),
        (MODELS_DIR / "gs1.json", "gs1", "bfo:BFO_0000040"),
        (MODELS_DIR / "universe_go_slim.json", "go", "bfo:BFO_0000003"),
        (MODELS_DIR / "doid.json", "doid", "bfo:BFO_0000016"),
        (MODELS_DIR / "universe_iof.json", "iof", "bfo:BFO_0000040"),
        (MODELS_DIR / "universe_ontopix.json", "pix", "gufo:Event"),
        (MODELS_DIR / "universe_apto.json", "apto", "bfo:BFO_0000040"),
        (MODELS_DIR / "universe_c2sim.json", "c2sim", "bfo:BFO_0000001"),
        (MODELS_DIR / "universe_fibo.json", "fibo", "bfo:BFO_0000001"),
        (MODELS_DIR / "universe_mondo.json", "mondo", "bfo:BFO_0000016"),
        (MODELS_DIR / "universe_d3fend.json", "d3fend", "bfo:BFO_0000003"),
        (MODELS_DIR / "universe_hs_codes.json", "hs", "bfo:BFO_0000040"),
        (MODELS_DIR / "universe_lkif.json", "lkif", "bfo:BFO_0000001"),
        (MODELS_DIR / "universe_sweet.json", "sweet", "bfo:BFO_0000001"),
    ]

    total_added = 0
    for path, domain, bridge_parent in json_sources:
        if not path.exists():
            print(f"  ⚠️ Skipping missing file: {path}")
            continue

        data = load_json(path)
        concepts = data.get("concepts", [])
        print(f"  * Integrating {domain.upper()} ({len(concepts)} concepts)...")

        for c in concepts:
            curie = c["curie"]
            # Deduplicate CURIEs if already present
            if curie in master.concepts:
                continue

            parents = set(c.get("parents", []))
            if not parents and bridge_parent and bridge_parent in master.concepts:
                parents.add(bridge_parent)

            master.add_concept(
                Concept(
                    curie=curie,
                    label=c.get("label", curie),
                    description=c.get("description", ""),
                    synonyms=c.get("synonyms", []),
                    parents=parents,
                    disjoint_with=set(c.get("disjoint_with", [])),
                )
            )
            total_added += 1

    print(f"\n🎉 Master Universe successfully assembled: {len(master.concepts)} total concepts!")
    return master


def export_master(master: Ontology):
    out_json = MODELS_DIR / "master_ontology.json"
    print(f"💾 Serializing master JSON to {out_json}...")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(master.to_dict(), f)
    print(f"  -> JSON size: {out_json.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    universe = build_universe()
    export_master(universe)
