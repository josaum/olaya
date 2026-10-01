"""Real ABox evidence harvester and grounding bridge for O-Laya.

Parses real enterprise ABox facts from Atlas:
- Central Garden & Pet SharePoint facts (central-sharepoint-evidence.ttl)
- Central Garden & Pet Web crawl facts (central-web-evidence.ttl)
- Nylabone Manufacturer Catalog SKUs & Products (atlas_nylabone_catalog.ttl)
- Operational Metric claims & ROAS draft claims (metric-claims-draft.ttl)

Maps real entity facts directly to Atlas ontology classes for decision grounding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ATLAS_DEFAULT_LOCATIONS = [
    Path("/Users/josaum/projects/atlas"),
    Path("../atlas"),
]


@dataclass
class RealABoxFact:
    """An empirical enterprise observation or fact grounded in an ontology concept."""
    text: str
    target_curie: str
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


def find_atlas_root() -> Path | None:
    """Find the root path of the Atlas repository if available."""
    for loc in ATLAS_DEFAULT_LOCATIONS:
        p = loc.resolve()
        if p.exists() and (p / "cartridges").exists():
            return p
    return None


def harvest_sharepoint_evidence(sp_path: Path) -> list[RealABoxFact]:
    """Harvest real facts from SharePoint crawl evidence."""
    if not sp_path.exists():
        return []
    content = sp_path.read_text(encoding="utf-8")
    facts: list[RealABoxFact] = []

    # 1. Harvest pages
    page_blocks = re.findall(r'(atlas:central_sp_page_\w+.*?)\s*\.\s*(?=atlas:|\Z)', content, re.DOTALL)
    for b in page_blocks:
        title_m = re.search(r'atlas:pageTitle\s+"([^"]+)"', b)
        url_m = re.search(r'atlas:sourceUrl\s+"([^"]+)"', b)
        if title_m and url_m:
            title = title_m.group(1).strip()
            url = url_m.group(1).strip()
            facts.append(
                RealABoxFact(
                    text=f"SharePoint internal portal: {title} ({url})",
                    target_curie="CompletenessEvidence",
                    source_type="sharepoint_page",
                    metadata={"url": url, "title": title},
                )
            )

    # 2. Harvest facts
    fact_blocks = re.findall(r'(atlas:central_sp_fact_\w+.*?)\s*\.\s*(?=atlas:|\Z)', content, re.DOTALL)
    for b in fact_blocks:
        text_m = re.search(r'atlas:factText\s+"([^"]+)"', b)
        url_m = re.search(r'atlas:sourceUrl\s+"([^"]+)"', b)
        if text_m:
            clean = text_m.group(1).strip()
            url = url_m.group(1).strip() if url_m else ""
            if len(clean) < 3:
                continue
            curie = "CurrencyEvidence" if any(w in clean.lower() for w in ["accounting", "finance", "billing", "invoice"]) else "EvidenceAssessment"
            facts.append(
                RealABoxFact(
                    text=f"SharePoint enterprise evidence: {clean} [portal: {url}]",
                    target_curie=curie,
                    source_type="sharepoint_fact",
                    metadata={"url": url, "fact": clean},
                )
            )

    return facts


def harvest_web_evidence(web_path: Path) -> list[RealABoxFact]:
    """Harvest real facts from external web crawl evidence."""
    if not web_path.exists():
        return []
    content = web_path.read_text(encoding="utf-8")
    facts: list[RealABoxFact] = []

    fact_blocks = re.findall(r'(atlas:central_fact_\w+.*?)\s*\.\s*(?=atlas:|\Z)', content, re.DOTALL)
    for b in fact_blocks:
        text_m = re.search(r'atlas:factText\s+"([^"]+)"', b)
        url_m = re.search(r'atlas:sourceUrl\s+"([^"]+)"', b)
        if text_m:
            clean = text_m.group(1).strip()
            url = url_m.group(1).strip() if url_m else ""
            if len(clean) < 3:
                continue
            curie = "Brand" if any(w in clean.lower() for w in ["brand", "company", "who we are", "about", "mission"]) else "EvidenceAssessment"
            facts.append(
                RealABoxFact(
                    text=f"Public web evidence: {clean} [crawl: {url}]",
                    target_curie=curie,
                    source_type="web_fact",
                    metadata={"url": url, "fact": clean},
                )
            )

    return facts


def harvest_nylabone_catalog(catalog_path: Path) -> list[RealABoxFact]:
    """Harvest real manufacturer SKUs, product models, and variant listings."""
    if not catalog_path.exists():
        return []
    content = catalog_path.read_text(encoding="utf-8")
    facts: list[RealABoxFact] = []

    # 1. Product Models
    model_matches = re.findall(
        r'nyla:Product_([^\s]+)\s+a\s+nyla:ProductModel\s*;\s*rdfs:label\s+"([^"]+)"',
        content,
    )
    for model_id, label in model_matches:
        is_oral = "dental" in label.lower() or "oral" in label.lower() or "chew" in label.lower()
        target_curie = "OralCareProduct" if is_oral else "ProductModel"
        facts.append(
            RealABoxFact(
                text=f"Nylabone commercial catalog product: {label} (model: {model_id})",
                target_curie=target_curie,
                source_type="nylabone_model",
                metadata={"model_id": model_id, "label": label},
            )
        )

    # 2. Manufacturer SKUs
    sku_matches = re.findall(
        r'nyla:SKU_(\w+)\s+a\s+nyla:ManufacturerSKU\s*;\s*rdfs:label\s+"([^"]+)"\s*;\s*nyla:skuCode\s+"([^"]+)"',
        content,
    )
    for sku_id, label, sku_code in sku_matches:
        facts.append(
            RealABoxFact(
                text=f"Manufacturer inventory SKU: {sku_code} ({label})",
                target_curie="ManufacturerSKU",
                source_type="nylabone_sku",
                metadata={"sku": sku_code, "label": label},
            )
        )

    # 3. Variant Listings
    listing_matches = re.findall(
        r'rdfs:label\s+"([^"]+)"\s*;\s*nyla:listedSKU\s+nyla:SKU_(\w+)',
        content,
    )
    for listing_label, sku_code in listing_matches:
        facts.append(
            RealABoxFact(
                text=f"Active commercial variant listing: {listing_label} [SKU {sku_code}]",
                target_curie="VariantListing",
                source_type="nylabone_listing",
                metadata={"listing": listing_label, "sku": sku_code},
            )
        )

    return facts


def harvest_metric_claims(metric_path: Path) -> list[RealABoxFact]:
    """Harvest real metric observations and ROAS claims."""
    if not metric_path.exists():
        return []
    content = metric_path.read_text(encoding="utf-8")
    facts: list[RealABoxFact] = []

    claim_blocks = re.findall(r'(cgp:Demo_\w+.*?)\s*\.\s*(?=cgp:|\Z)', content, re.DOTALL)
    for b in claim_blocks:
        name_m = re.search(r'cgp:Demo_(\w+)', b)
        val_m = re.search(r'cgp:claimValue\s+([0-9.]+)', b)
        comm_m = re.search(r'rdfs:comment\s+"([^"]+)"', b)
        claim_name = name_m.group(1) if name_m else "claim"
        val_str = f" val={val_m.group(1)}" if val_m else ""
        comm_str = f" ({comm_m.group(1)})" if comm_m else ""
        target = "MetricObservation" if "nyla:MetricROAS" in b or "evaluates" in b else "MetricDefinition"
        facts.append(
            RealABoxFact(
                text=f"Operational telemetry metric observation {claim_name}{val_str}{comm_str}",
                target_curie=target,
                source_type="metric_claim",
                metadata={"claim": claim_name, "raw": b.strip()},
            )
        )

    return facts


def load_atlas_real_abox(atlas_root: Path | None = None) -> list[RealABoxFact]:
    """Load all real ABox evidence facts available in the Atlas project."""
    root = atlas_root or find_atlas_root()
    if root is None:
        return []

    facts: list[RealABoxFact] = []

    # 1. SharePoint
    sp_file = root / "app" / "cartridge" / "data" / "central-sharepoint-evidence.ttl"
    facts.extend(harvest_sharepoint_evidence(sp_file))

    # 2. Web evidence
    web_file = root / "app" / "cartridge" / "data" / "central-web-evidence.ttl"
    facts.extend(harvest_web_evidence(web_file))

    # 3. Nylabone catalog ABox
    cat_file = root / "cartridges" / "atlas" / "ontologies" / "abox" / "atlas_nylabone_catalog.ttl"
    facts.extend(harvest_nylabone_catalog(cat_file))

    # 4. Metric claims draft
    metric_file = root / "docs" / "atlas" / "analysis" / "nylabone-2026-09-16" / "metric-claims-draft.ttl"
    facts.extend(harvest_metric_claims(metric_file))

    return facts
