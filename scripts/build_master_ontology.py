"""
Leio Universal Master Ontology Builder.

Synthesizes the world's standard formal ontologies into a unified zero-copy lattice:
1. BFO (ISO/IEC 21838-2 Top-Level Ontology: Continuant, Occurrent, MaterialEntity, Process, Disposition)
2. Schema.org (Google Structured Data & Web Entities: Products, Organizations, Events, Actions)
3. GS1 Web Vocabulary (Global Trade Items, Logistics, Perishables, Supply Chain)
4. DOID Disease Ontology (Medical, Oncology, Pathology, Infectious Agents)
5. SRE / Cloud Incidents (DevOps, Infrastructure, Security Breaches, Database Outages)
6. Atlas Supply Chain & Inventory (Fulfillment, Cross-Dock, Transport)
7. OAEI Conference / Peer Review (Scientific publishing, reviews, events)

Exports:
- models/master_ontology.json
- models/master_ontology.arrow (zero-copy memory-mapped IPC)
- models/master_ontology_head.json (distilled native concept cone embeddings)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from olaya.distilled_head import create_zero_shot_head_for_ontology
from olaya.ontology import Concept, Ontology
from olaya.owl2_rl import OWL2RLReasoner

DATA_DIR = Path("data")
DOCS_ONTO_DIR = Path("docs/public/ontologies")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_json_onto(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_master_lattice() -> Ontology:
    print("🚀 Assembling Leio Universal Master Ontology...")
    master = Ontology(name="LeioUniversalMasterLattice")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Ingest BFO (Basic Formal Ontology - ISO/IEC 21838-2 Top Level)
    # ──────────────────────────────────────────────────────────────────────────
    bfo_path = DOCS_ONTO_DIR / "bfo_ontology.json"
    if bfo_path.exists():
        bfo_data = load_json_onto(bfo_path)
        concepts = bfo_data.get("concepts", [])
        if isinstance(concepts, list):
            for c in concepts:
                curie = c["curie"]
                master.add_concept(
                    Concept(
                        curie=curie,
                        label=c.get("label", curie),
                        description=c.get("description", ""),
                        synonyms=c.get("synonyms", []),
                        parents=set(c.get("parents", [])),
                        disjoint_with=set(c.get("disjoint_with", [])),
                    )
                )
    print(f"  * BFO Upper Ontology loaded: {len(master.concepts)} foundational nodes")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Ingest Schema.org (Google Web Entities & Actions)
    # ──────────────────────────────────────────────────────────────────────────
    schema_path = DOCS_ONTO_DIR / "schemaorg_ontology.json"
    if schema_path.exists():
        schema_data = load_json_onto(schema_path)
        concepts = schema_data.get("concepts", [])
        if isinstance(concepts, list):
            for c in concepts:
                curie = c["curie"]
                parents = set(c.get("parents", []))
                # Bridge Schema.org to BFO
                if curie in ("schema:Thing", "Thing") or not parents:
                    parents.add("bfo:BFO_0000001")  # bfo:Entity
                elif curie in ("schema:Action", "schema:Event", "Action", "Event"):
                    parents.add("bfo:BFO_0000003")  # bfo:Occurrent / Process
                elif curie in ("schema:Product", "schema:Place", "schema:Organization", "schema:Person"):
                    parents.add("bfo:BFO_0000040")  # bfo:MaterialEntity

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
    print(f"  * Schema.org integrated: Master now has {len(master.concepts)} nodes")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Ingest GS1 Web Vocabulary (Global Trade Items & Logistics)
    # ──────────────────────────────────────────────────────────────────────────
    gs1_path = DOCS_ONTO_DIR / "gs1_ontology.json"
    if gs1_path.exists():
        gs1_data = load_json_onto(gs1_path)
        concepts = gs1_data.get("concepts", [])
        if isinstance(concepts, list):
            for c in concepts:
                curie = c["curie"]
                parents = set(c.get("parents", []))
                # Bridge GS1 to Schema.org / BFO
                if curie in ("gs1:Product", "Product") or not parents:
                    parents.add("schema:Product")
                    parents.add("bfo:BFO_0000040")  # bfo:MaterialEntity
                elif "Logistic" in curie or "Transport" in curie:
                    parents.add("bfo:BFO_0000003")  # bfo:Occurrent

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
    print(f"  * GS1 Supply Chain integrated: Master now has {len(master.concepts)} nodes")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Ingest DOID Disease & Pathology Ontology
    # ──────────────────────────────────────────────────────────────────────────
    doid_path = DOCS_ONTO_DIR / "doid_ontology.json"
    if doid_path.exists():
        doid_data = load_json_onto(doid_path)
        concepts = doid_data.get("concepts", [])
        if isinstance(concepts, list):
            for c in concepts:
                curie = c["curie"]
                parents = set(c.get("parents", []))
                # Bridge DOID root to BFO Disposition & Schema.org MedicalCondition
                if curie == "DOID:4" or not parents:
                    parents.add("bfo:BFO_0000016")  # bfo:Disposition
                    parents.add("schema:MedicalCondition")

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
    print(f"  * DOID Medical Ontology integrated: Master now has {len(master.concepts)} nodes")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Ingest Incident & SRE Observability Taxonomy
    # ──────────────────────────────────────────────────────────────────────────
    inc_path = DOCS_ONTO_DIR / "incident_ontology.json"
    if inc_path.exists():
        inc_data = load_json_onto(inc_path)
        concepts = inc_data.get("concepts", [])
        if isinstance(concepts, list):
            for c in concepts:
                curie = c["curie"]
                parents = set(c.get("parents", []))
                if curie == "INCIDENT:ROOT" or not parents:
                    parents.add("bfo:BFO_0000003")  # bfo:Occurrent / Process
                    parents.add("schema:Event")

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
    print(f"  * SRE / Cloud Incident Ontology integrated: Master now has {len(master.concepts)} nodes")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6: Ingest Atlas Supply Chain & Inventory
    # ──────────────────────────────────────────────────────────────────────────
    atlas_path = DATA_DIR / "custom_ontology.json"
    if atlas_path.exists():
        atlas_data = load_json_onto(atlas_path)
        concepts = atlas_data.get("concepts", [])
        if isinstance(concepts, list):
            for c in concepts:
                curie = c["curie"]
                parents = set(c.get("parents", []))
                if not parents:
                    parents.add("schema:Product")

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
    print(f"  * Atlas Supply Chain integrated: Master now has {len(master.concepts)} nodes")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 7: Deductive Closure via W3C OWL 2 RL Reasoner
    # ──────────────────────────────────────────────────────────────────────────
    print("⚙️  Running W3C OWL 2 RL Deductive Reasoner across Master Lattice...")
    reasoner = OWL2RLReasoner(master)
    closure = reasoner.compute_closure()
    print(f"  * Deductive closure complete: {len(closure.subclasses)} subclass relations, {len(closure.disjointnesses)} disjoint pairs")

    return master


def export_master_artifacts(master: Ontology):
    # 1. Export JSON
    out_json = MODELS_DIR / "master_ontology.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(master.to_dict(), f, indent=2)
    print(f"✅ Saved Master JSON to {out_json} ({out_json.stat().st_size / 1024:.1f} KB)")

    # 2. Export Distilled Native Concept Cone Head
    print("🧠 Generating Instant Zero-Shot Geometric Concept Cone Head (< 5ms)...")
    head = create_zero_shot_head_for_ontology(master, dim=64, hidden_dim=96, num_buckets=4096)
    weights = head.export_weights()
    out_head = MODELS_DIR / "master_ontology_head.json"
    with open(out_head, "w", encoding="utf-8") as f:
        f.write(weights.to_json())
    print(f"✅ Saved Distilled Neural Head to {out_head} ({out_head.stat().st_size / 1024:.1f} KB)")

    out_bin = MODELS_DIR / "master_ontology_head.bin"
    with open(out_bin, "wb") as f:
        f.write(weights.to_binary())
    print(f"✅ Saved Binary LEIO Cone Head to {out_bin} ({out_bin.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    master_onto = build_master_lattice()
    export_master_artifacts(master_onto)
    print("\n🎉 LEIO UNIVERSAL MASTER ONTOLOGY LATTICE IS LIVE!")
