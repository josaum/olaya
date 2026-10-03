"""
Universal Standard Ontologies Compiler for O-Laya Decision Engine.
Compiles:
1. Schema.org (Google Structured Data / SEO specification: Product, Article, FAQPage, BreadcrumbList, Event, LocalBusiness, JobPosting, Organization, SoftwareApplication, Recipe, VideoObject, HowTo, Course, etc.)
2. BFO (Basic Formal Ontology: ISO/IEC 21838-2 Top-Level Ontology: Continuant, Occurrent, Spatial/Temporal Region, RealizableEntity, Role, Disposition, Process, etc.)
3. SKOS (Simple Knowledge Organization System: W3C Standard: Concept, ConceptScheme, Collection, OrderedCollection)
4. GS1 Web Vocabulary (Supply chain, Logistics, Perishables, Traceability, GPC)
5. Enterprise Incident & Observability (IT, Cloud, SRE, Triage)
"""

import gzip
import json
import urllib.request
from pathlib import Path

from rdflib import OWL, RDF, RDFS, Graph, URIRef

OUTPUT_DIR = Path("/Users/josaum/projects/decision-model/docs/public/ontologies")
SRC_DATA_DIR = Path("/Users/josaum/projects/decision-model/src/olaya/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SRC_DATA_DIR.mkdir(parents=True, exist_ok=True)

GOOGLE_SEO_TYPES = {
    "Article",
    "NewsArticle",
    "BlogPosting",
    "TechArticle",
    "Product",
    "ProductModel",
    "AggregateOffer",
    "Offer",
    "Brand",
    "LocalBusiness",
    "Store",
    "Restaurant",
    "AutomotiveBusiness",
    "MedicalBusiness",
    "LodgingBusiness",
    "Organization",
    "Corporation",
    "EducationalOrganization",
    "GovernmentOrganization",
    "NGO",
    "BreadcrumbList",
    "ListItem",
    "FAQPage",
    "Question",
    "Answer",
    "QAPage",
    "HowTo",
    "HowToStep",
    "HowToSection",
    "HowToSupply",
    "HowToTool",
    "Recipe",
    "Course",
    "CourseInstance",
    "SpecialAnnouncement",
    "JobPosting",
    "OccupationalExperienceRequirements",
    "Event",
    "BusinessEvent",
    "ChildrensEvent",
    "MusicEvent",
    "SportsEvent",
    "SaleEvent",
    "ScreeningEvent",
    "VideoObject",
    "AudioObject",
    "MediaObject",
    "ImageObject",
    "Review",
    "AggregateRating",
    "Rating",
    "CriticReview",
    "EmployerReview",
    "SoftwareApplication",
    "WebApplication",
    "MobileApplication",
    "VideoGame",
    "WebPage",
    "AboutPage",
    "ContactPage",
    "CollectionPage",
    "ItemPage",
    "ProfilePage",
    "SearchAction",
    "EntryPoint",
    "PostalAddress",
    "GeoCoordinates",
    "OpeningHoursSpecification",
    "Person",
    "ContactPoint",
    "MonetaryAmount",
    "QuantitativeValue",
    "MerchantReturnPolicy",
}


def clean_curie(uri: str) -> str:
    prefixes = [
        ("http://schema.org/", "schema:"),
        ("https://schema.org/", "schema:"),
        ("http://www.w3.org/2004/02/skos/core#", "skos:"),
        ("http://purl.obolibrary.org/obo/BFO_", "bfo:BFO_"),
        ("http://purl.obolibrary.org/obo/bfo#", "bfo:"),
        ("https://gs1.org/voc/", "gs1:"),
    ]
    for pfx, rep in prefixes:
        if uri.startswith(pfx):
            return rep + uri[len(pfx) :]
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]


def build_schemaorg():
    print("\n--- Compiling Schema.org (Google SEO & Core Vocab) ---")
    url = "https://schema.org/version/latest/schemaorg-current-https.jsonld"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
        if resp.info().get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        data = json.loads(raw)

    graph_nodes = data.get("@graph", [])
    classes = {}

    # First pass: collect classes
    for item in graph_nodes:
        types = item.get("@type", [])
        if isinstance(types, str):
            types = [types]
        if "rdfs:Class" in types or "owl:Class" in types:
            curie_raw = item.get("@id", "")
            local_id = (
                curie_raw.replace("schema:", "").replace("https://schema.org/", "").replace("http://schema.org/", "")
            )

            # Label
            label_field = item.get("rdfs:label", "")
            if isinstance(label_field, dict):
                label_text = label_field.get("@value", local_id)
            elif isinstance(label_field, str):
                label_text = label_field
            else:
                label_text = local_id

            # Comment / Description
            comment_field = item.get("rdfs:comment", "")
            if isinstance(comment_field, dict):
                desc_text = comment_field.get("@value", "")
            elif isinstance(comment_field, str):
                desc_text = comment_field
            else:
                desc_text = ""

            curie = f"schema:{local_id}"

            # Parents
            subclass_field = item.get("rdfs:subClassOf", [])
            if isinstance(subclass_field, dict):
                subclass_field = [subclass_field]
            elif isinstance(subclass_field, str):
                subclass_field = [{"@id": subclass_field}]

            parents = []
            for p in subclass_field:
                pid = p.get("@id", "")
                p_local = (
                    pid.replace("schema:", "").replace("https://schema.org/", "").replace("http://schema.org/", "")
                )
                if p_local and p_local != local_id:
                    parents.append(f"schema:{p_local}")

            classes[curie] = {
                "curie": curie,
                "label": label_text,
                "description": desc_text,
                "parents": parents,
                "children": [],
                "disjoint_with": [],
                "synonyms": [label_text.lower(), local_id.lower()],
                "is_google_seo": local_id in GOOGLE_SEO_TYPES,
            }

    # Filter to SEO types + their ancestral closures + high-value core commerce/media
    needed_curies = set()
    for local_seo in GOOGLE_SEO_TYPES:
        c = f"schema:{local_seo}"
        if c in classes:
            needed_curies.add(c)
            # Add ancestors
            q = [c]
            while q:
                curr = q.pop()
                for p in classes[curr]["parents"]:
                    if p in classes and p not in needed_curies:
                        needed_curies.add(p)
                        q.append(p)

    # Also keep direct children of essential SEO types (e.g. Product subclasses, Article subclasses)
    for c, cinfo in classes.items():
        if any(p in needed_curies for p in cinfo["parents"]):
            needed_curies.add(c)

    filtered_concepts = []
    for curie in sorted(needed_curies):
        info = classes[curie]
        # Keep parents that are in needed_curies
        info["parents"] = [p for p in info["parents"] if p in needed_curies]
        filtered_concepts.append(info)

    schema_payload = {
        "name": "SchemaOrg_GoogleSEO",
        "description": "Schema.org Ontology covering full Google Structured Metadata, Rich Snippets, and Knowledge Graph types.",
        "concepts": filtered_concepts,
    }

    out_public = OUTPUT_DIR / "schemaorg_ontology.json"
    out_public.write_text(json.dumps(schema_payload, indent=2))
    (SRC_DATA_DIR / "schemaorg_ontology.json").write_text(json.dumps(schema_payload, indent=2))
    print(f"Generated {out_public} with {len(filtered_concepts)} SEO & Schema.org classes.")


def build_bfo():
    print("\n--- Compiling BFO (Basic Formal Ontology ISO/IEC 21838-2) ---")
    g = Graph()
    g.parse("http://purl.obolibrary.org/obo/bfo.owl", format="xml")

    concepts = {}
    for s in g.subjects(RDF.type, OWL.Class):
        if not isinstance(s, URIRef):
            continue
        uri = str(s)
        if "BFO_" not in uri:
            continue

        bfo_num = uri.split("BFO_")[-1]
        curie = f"bfo:BFO_{bfo_num}"

        # Labels
        labels = list(g.objects(s, RDFS.label))
        label_text = str(labels[0]) if labels else f"BFO_{bfo_num}"

        # Definitions
        # IAO_0000115 is standard definition predicate in OBO
        def_uri = URIRef("http://purl.obolibrary.org/obo/IAO_0000115")
        defs = list(g.objects(s, def_uri)) or list(g.objects(s, RDFS.comment))
        desc_text = str(defs[0]) if defs else ""

        # Parents
        parents = []
        for p in g.objects(s, RDFS.subClassOf):
            if isinstance(p, URIRef) and "BFO_" in str(p):
                p_num = str(p).split("BFO_")[-1]
                parents.append(f"bfo:BFO_{p_num}")

        # Disjointness
        disjoints = []
        for d in g.objects(s, OWL.disjointWith):
            if isinstance(d, URIRef) and "BFO_" in str(d):
                d_num = str(d).split("BFO_")[-1]
                disjoints.append(f"bfo:BFO_{d_num}")

        synonyms = [label_text.lower(), f"bfo_{bfo_num}"]

        concepts[curie] = {
            "curie": curie,
            "label": label_text,
            "description": desc_text,
            "parents": parents,
            "children": [],
            "disjoint_with": disjoints,
            "synonyms": synonyms,
        }

    bfo_payload = {
        "name": "BasicFormalOntology_ISO21838",
        "description": "BFO Top-Level Upper Ontology: Continuant, Occurrent, Spatial/Temporal Regions, and Realizable Entities.",
        "concepts": list(concepts.values()),
    }

    out_public = OUTPUT_DIR / "bfo_ontology.json"
    out_public.write_text(json.dumps(bfo_payload, indent=2))
    (SRC_DATA_DIR / "bfo_ontology.json").write_text(json.dumps(bfo_payload, indent=2))
    print(f"Generated {out_public} with {len(concepts)} formal BFO classes.")


def build_skos():
    print("\n--- Compiling SKOS (Simple Knowledge Organization System) ---")
    g = Graph()
    g.parse("http://www.w3.org/2004/02/skos/core", format="xml")

    concepts = {}
    for s in g.subjects(RDF.type, OWL.Class):
        if not isinstance(s, URIRef):
            continue
        curie = clean_curie(str(s))
        if not curie.startswith("skos:"):
            continue

        labels = list(g.objects(s, RDFS.label))
        label_text = str(labels[0]) if labels else curie.split(":")[-1]

        defs = list(g.objects(s, URIRef("http://www.w3.org/2004/02/skos/core#definition"))) or list(
            g.objects(s, RDFS.comment)
        )
        desc_text = str(defs[0]) if defs else ""

        parents = []
        for p in g.objects(s, RDFS.subClassOf):
            if isinstance(p, URIRef):
                p_curie = clean_curie(str(p))
                if p_curie.startswith("skos:") and p_curie != curie:
                    parents.append(p_curie)

        disjoints = []
        for d in g.objects(s, OWL.disjointWith):
            if isinstance(d, URIRef):
                d_curie = clean_curie(str(d))
                if d_curie.startswith("skos:"):
                    disjoints.append(d_curie)

        synonyms = [label_text.lower(), curie.split(":")[-1].lower()]

        concepts[curie] = {
            "curie": curie,
            "label": label_text,
            "description": desc_text,
            "parents": parents,
            "children": [],
            "disjoint_with": disjoints,
            "synonyms": synonyms,
        }

    # Add core properties as relation nodes if useful
    skos_payload = {
        "name": "SKOS_SimpleKnowledgeOrganizationSystem",
        "description": "W3C SKOS Recommendation for thesauri, classification schemes, subject heading lists, and taxonomies.",
        "concepts": list(concepts.values()),
    }

    out_public = OUTPUT_DIR / "skos_ontology.json"
    out_public.write_text(json.dumps(skos_payload, indent=2))
    (SRC_DATA_DIR / "skos_ontology.json").write_text(json.dumps(skos_payload, indent=2))
    print(f"Generated {out_public} with {len(concepts)} SKOS classes.")


if __name__ == "__main__":
    build_schemaorg()
    build_bfo()
    build_skos()
