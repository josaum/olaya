"""
Convert WCO Harmonized System (HS Codes) CSV to Leio JSON & Arrow IPC.
"""

import csv
import json
from pathlib import Path

CSV_PATH = Path("data/universe/wave2/hs_codes.csv")
OUT_JSON = Path("models/universe_hs_codes.json")

def convert():
    print(f"Reading HS Codes from {CSV_PATH}...")
    concepts = []
    
    # Root concept
    concepts.append({
        "curie": "HS:TOTAL",
        "label": "Harmonized System Commodities Root",
        "description": "World Customs Organization Harmonized Tariff and Commodity Description System",
        "parents": ["bfo:BFO_0000040"], # MaterialEntity
        "children": [],
        "disjoint_with": [],
        "synonyms": ["HS Code", "Commodity Code", "Tariff Classification", "Customs Code"]
    })

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hscode = row["hscode"].strip()
            desc = row["description"].strip()
            parent = row["parent"].strip()
            section = row["section"].strip()
            level = row["level"].strip()

            curie = f"HS:{hscode}"
            parent_curie = f"HS:{parent}" if parent and parent != "TOTAL" else "HS:TOTAL"

            synonyms = [f"HS {hscode}"]
            if level == "2":
                label = f"Chapter {hscode}: {desc}"
            elif level == "4":
                label = f"Heading {hscode}: {desc}"
            else:
                label = f"Subheading {hscode}: {desc}"

            concepts.append({
                "curie": curie,
                "label": label,
                "description": f"Section {section}, Level {level} commodity: {desc}",
                "parents": [parent_curie],
                "children": [],
                "disjoint_with": [],
                "synonyms": synonyms
            })

    output_data = {
        "name": "HarmonizedCommodityDescriptionAndCodingSystem",
        "concepts": concepts
    }

    print(f"Exporting {len(concepts)} HS Code concepts to {OUT_JSON}...")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Saved HS Codes JSON ({OUT_JSON.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    convert()
