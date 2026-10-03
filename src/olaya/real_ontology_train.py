"""Multi-domain training pipeline for real-world ontologies in O-Laya.

Supports training the ModernBERT / BERT decision head on real ontologies:
- Atlas Brain Ontology (commerce, inventory, supply chain, execution)
- Human Disease Ontology (DOID - clinical diagnosis, oncology, pathology)
- Schema.org (web entities, products, actions, organizations, media)
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any

from olaya.engine import _flat_subgraph
from olaya.hard_negative_sampler import HardNegativeSampler
from olaya.ontology import Concept, Ontology
from olaya.real_abox import load_atlas_real_abox
from olaya.sequence import QTYPE_IDS, build_ontology_sequence
from olaya.train import TrainingSample

# Domain query templates to produce diverse, realistic state queries.
# Templates marked with {term} use the concept label (with optional synonym augmentation).
# Property-based templates (no {term}) describe observable characteristics directly,
# forcing the model to learn discriminative features rather than label memorisation.
DOMAIN_QUERY_TEMPLATES: dict[str, list[str]] = {
    "atlas": [
        "Inventory snapshot observed for SKU with {term} status at central fulfillment facility.",
        "Operational telemetry alert: {term} detected in Databricks commerce streaming pipeline.",
        "Catalog record update: manufacturer published new {term} specifications for retail distribution.",
        "Fulfillment orchestration: carrier dispatched order matching {term} requirements.",
        "Evidence analysis: verified {term} claim from SharePoint documentation and commercial feeds.",
        "Commerce event stream recorded state transition consistent with {term} entity classification.",
        "Master data registry flagged product record as conforming to {term} trade item category.",
        "Warehouse management system generated pick-list referencing {term} location assignment.",
        "Cross-dock routing decision triggered by detection of {term} classification on inbound shipment.",
        "Supplier portal sync: vendor catalogue entry categorised under {term} attribute hierarchy.",
    ],
    "doid": [
        "Patient biopsy and histopathology report confirms clinical diagnosis of {term}.",
        "Diagnostic oncology screening indicates malignant progression consistent with {term}.",
        "Clinical summary: patient presents with symptoms and laboratory markers of {term}.",
        "Pathology requisition: specimen analysis requested for suspected {term}.",
        "Electronic health record: ICD-10 crosswalk mapped to clinical disease entity {term}.",
        # Property-based templates (no {term}) — describe by observable clinical characteristics
        "Malignant neoplasm arising from epithelial tissue lining, rapid local invasion, lymph node metastasis staging.",
        "Autoimmune condition, elevated inflammatory markers CRP/ESR, joint synovitis, morning stiffness characteristic.",
        "Infectious agent transmitted by respiratory droplets, incubation 2-14 days, fever and myalgia prodrome.",
        "Genetic chromosomal disorder, trisomy pattern, developmental delay, characteristic dysmorphic features.",
        "Neurodegenerative condition, progressive cognitive decline, amyloid plaques confirmed on PET imaging.",
        "Metabolic dysfunction, insulin resistance pathway, fasting glucose elevated, HbA1c above 6.5%.",
        "Cardiac muscle disease, ejection fraction reduced, breathlessness on exertion, peripheral edema.",
        "Primary CNS tumor, WHO grade classification, contrast enhancement on MRI, neurological deficits.",
        "Chronic autoimmune hepatitis, elevated transaminases, interface hepatitis on biopsy, IgG elevation.",
        "Viral hemorrhagic fever, thrombocytopenia, coagulation cascade disruption, hemorrhagic manifestations.",
        "Haematological malignancy, bone marrow infiltration by blast cells, pancytopenia on CBC.",
        "Lysosomal storage disorder, enzymatic deficiency confirmed, substrate accumulation in macrophages.",
    ],
    "schemaorg": [
        "Structured data payload parsed from webpage json-ld metadata representing {term}.",
        "Search engine entity extraction identified knowledge graph node typed as {term}.",
        "E-commerce catalog index entry conforming to standard schema specification for {term}.",
        "API schema definition validating web application resource of type {term}.",
        "Rich snippet preview generated for online published content classified under {term}.",
        # Property-based templates — describe by observable web-entity characteristics
        "Step-by-step instructions with numbered list format, requires specific equipment, user-submitted tutorial.",
        "Business listing with hours, address, phone number, accepts online reservations and walk-ins.",
        "Academic research publication, peer-reviewed journal, DOI identifier, abstract and methodology sections.",
        "E-commerce product listing with price, availability, seller rating, delivery estimate and return policy.",
        "Employment opportunity posting, full-time position, location and remote options, salary range disclosed.",
        "User-submitted question awaiting expert answers, vote counts, accepted answer marked, tagged with topics.",
        "Audio recording file, podcast episode, MP3 format, duration listed, hosted on distribution platform.",
        "Geographic location coordinates, administrative region boundary, points of interest density.",
        "Video content with runtime, director credit, genre tags, age rating, streaming availability noted.",
        "News article with byline, publication date, headline, lead paragraph summary and related topics.",
        "Medical drug entity: active ingredient, dosage form, prescribing indications, contraindications listed.",
        "Real estate listing: square footage, bedrooms, neighbourhood, asking price, open house schedule.",
        "Sports event: competing teams, venue, start time, current score or final result, league classification.",
        "Review entity with star rating, reviewer name, product target, pros and cons sections.",
    ],
    "gs1": [
        "Trade item scan: barcode GTIN payload classified under GS1 standard {term}.",
        "Global Data Synchronization Network (GDSN) attribute update for {term}.",
        "Supply chain logistics event: Electronic Product Code Information Services (EPCIS) record for {term}.",
        "Retail master data management: packaging and trade specification registered for {term}.",
        "Digital Link resolution: product packaging QR code redirected to verified {term} descriptor.",
        # Property-based templates — describe by physical/logistical observable attributes
        "Perishable dairy product, refrigerated at 34-38°F, tamper-evident induction seal, GTIN prefix 0614141.",
        "Frozen poultry protein, blast-frozen below -18°C, vacuum sealed corrugated master case, cross-dock routing required.",
        "Class 9 dangerous goods UN3480 placard, lithium-ion chemistry, watt-hour rating exceeds 100Wh per battery.",
        "Non-perishable consumer product, ambient storage conditions, retail barcode GS1-128 compliant, standard shelf life.",
        "Medical device sterile packaging, single-use disposable, class II FDA classification, hospital supply chain.",
        "Returnable transport packaging, RFID tag enabled, tare weight documented, deposit credit system eligible.",
        "High-value electronics, serialized inventory, SGTIN encoding, anti-tamper label with EPC memory bank.",
        "Chemical compound, GHS hazard pictograms, SDS documentation required, segregation from food products.",
        "Fresh produce with variable weight, price-embedded barcode, traceability lot code, cold chain monitored.",
        "Apparel item, size and colour variant matrix, UPC-A scan on hang tag, retailer-specific PLU code.",
        "Pharmaceutical blister pack, NDC number encoded, expiry date attribute mandatory, unit-dose dispensing.",
        "Bulk commodity, net weight declared in kg, loose or bagged format, pallet quantity in SSCC label.",
        "Beverage can, volume 330ml, aluminium substrate, deposit-refund scheme eligible, multipak shrink-wrapped.",
        "Luxury fragrance bottle, serialised item-level RFID, brand authentication seal, duty-free eligible flag.",
    ],
}


def load_ontology_file(path_or_onto: str | Path | Ontology, name: str | None = None) -> Ontology:
    """Load an ontology from JSON, Turtle, or OWL using the Rust converter if needed."""
    if isinstance(path_or_onto, Ontology):
        return path_or_onto

    path = Path(path_or_onto)
    if not path.exists():
        raise FileNotFoundError(f"Ontology file not found: {path}")

    # If already a JSON ontology
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return Ontology.from_json(data)

    # If Turtle or OWL, convert using Rust convert_ontology binary
    json_out = path.with_suffix(".json")
    conv_cmd = [
        "cargo",
        "run",
        "--release",
        "-p",
        "olaya-core",
        "--bin",
        "convert_ontology",
        "--",
        str(path),
        str(json_out),
    ]
    if name:
        conv_cmd.append(name)

    print(f"Running owl-fast-core converter on {path}...")
    subprocess.check_call(conv_cmd)
    data = json.loads(json_out.read_text(encoding="utf-8"))
    return Ontology.from_json(data)


def _generate_tasks_for_concept_query(
    ontology: Ontology,
    target_curie: str,
    query_text: str,
    domain: str,
    pool: list[str],
    tokenizer_encode: Any,
    special_ids: dict[str, int],
    rng: random.Random,
    sampler: HardNegativeSampler | None = None,
) -> list[TrainingSample]:
    """Generate all 5 typed operational decision tasks for a single concept query.

    Args:
        sampler: Optional :class:`HardNegativeSampler` for structured hard-negative
            distractor selection.  When *None* the legacy random sampling is used.
    """
    concept = ontology.concepts[target_curie]
    ancestors = ontology.get_ancestors(target_curie)
    state = {
        "domain": domain,
        "query": query_text,
        "curie": target_curie,
    }
    samples: list[TrainingSample] = []

    # 1. Concept triage task: pick target from candidate set (target + distractors)
    if sampler is not None:
        chosen_distractors = sampler.sample_distractors(target_curie, k=5, rng=rng)
    else:
        distractors = [c for c in pool if c != target_curie and c not in ancestors]
        k_distractors = min(5, len(distractors))
        chosen_distractors = rng.sample(distractors, k_distractors) if distractors else []
    competing = chosen_distractors + [target_curie]
    rng.shuffle(competing)
    target_idx = competing.index(target_curie)

    subgraph_concepts = [ontology.concepts[c] for c in competing]
    bundle_concept = build_ontology_sequence(
        tokenizer_encode=tokenizer_encode,
        special_ids=special_ids,
        state=state,
        instructions=f"Classify the entity into the {ontology.name} domain hierarchy.",
        active_concepts=subgraph_concepts,
        ontology=ontology,
        qtype_id=QTYPE_IDS["concept"],
        qtype_name="concept",
    )
    if target_curie in bundle_concept.candidate_curies:
        target_idx = bundle_concept.candidate_curies.index(target_curie)
        samples.append(
            TrainingSample(
                bundle=bundle_concept,
                target_idx=target_idx,
                act_target=1 if len(ancestors) > 1 else 0,
                qtype_name="concept",
                ontology=ontology,
            )
        )

    # 2. Subsumes task: test membership in down-set \downarrow C
    is_positive = rng.random() < 0.5
    if is_positive:
        query_curie = rng.choice(sorted(ancestors))
        expected_label = 1  # true: target_curie in \downarrow query_curie
    else:
        non_ancestors = [c for c in pool if c not in ancestors]
        query_curie = rng.choice(non_ancestors) if non_ancestors else target_curie
        expected_label = 0 if query_curie not in ancestors else 1

    query_concept = ontology.concepts[query_curie]
    q_descendants = sorted(ontology.get_descendants(query_curie) - {query_curie})
    q_desc_labels = [ontology.concepts[d].label for d in q_descendants if d in ontology.concepts]
    desc_str = f" or subtypes ({', '.join(q_desc_labels[:5])})" if q_desc_labels else ""
    query_desc_extra = f" - {query_concept.description}" if query_concept.description else ""

    false_desc = f"Entity is not an instance of {query_concept.label} ({query_curie})"
    true_desc = f"Entity is an instance of {query_concept.label} ({query_curie}){desc_str}{query_desc_extra}"
    subsumes_options = [
        Concept(curie="false", label="false", description=false_desc),
        Concept(curie="true", label="true", description=true_desc),
    ]
    subsumes_subgraph = _flat_subgraph(f"real_subsumes_{query_curie}", subsumes_options)
    query_desc_hint = f" ({query_concept.description})" if query_concept.description else ""
    instructions_sub = (
        f"Does entity satisfy {query_concept.label} ({query_curie}){query_desc_hint} or any of its subtypes?"
        if q_desc_labels
        else f"Does entity satisfy {query_concept.label} ({query_curie}){query_desc_hint}?"
    )
    bundle_subsumes = build_ontology_sequence(
        tokenizer_encode=tokenizer_encode,
        special_ids=special_ids,
        state=state,
        instructions=instructions_sub,
        active_concepts=subsumes_options,
        ontology=subsumes_subgraph,
        qtype_id=QTYPE_IDS["subsumes"],
        qtype_name="subsumes",
    )
    samples.append(
        TrainingSample(
            bundle=bundle_subsumes,
            target_idx=expected_label,
            act_target=expected_label,
            qtype_name="subsumes",
        )
    )

    # 3. Choice task: select best match among domain options with rich descriptions
    if sampler is not None:
        choice_distractors = sampler.sample_distractors(target_curie, k=3, rng=rng)
    else:
        _choice_pool = [c for c in pool if c != target_curie and c not in ancestors]
        choice_distractors = rng.sample(_choice_pool, min(3, len(_choice_pool))) if _choice_pool else []
    choice_curies = [target_curie] + choice_distractors
    rng.shuffle(choice_curies)
    choice_idx = choice_curies.index(target_curie)
    choice_concepts = [ontology.concepts[c] for c in choice_curies]
    bundle_choice = build_ontology_sequence(
        tokenizer_encode=tokenizer_encode,
        special_ids=special_ids,
        state=state,
        instructions=f"Identify the correct semantic classification in {ontology.name} for this observation.",
        active_concepts=choice_concepts,
        ontology=ontology,
        qtype_id=QTYPE_IDS["choice"],
        qtype_name="choice",
    )
    if target_curie in bundle_choice.candidate_curies:
        choice_idx = bundle_choice.candidate_curies.index(target_curie)
        samples.append(
            TrainingSample(
                bundle=bundle_choice,
                target_idx=choice_idx,
                act_target=1,
                qtype_name="choice",
                ontology=ontology,
            )
        )

    # 4. Score task: relevance / hierarchy depth level (0..3)
    score_level = min(3, max(0, len(ancestors) - 1))
    score_criteria = [
        "high-level category: top domain partition",
        "domain branch: broad conceptual family",
        "specialized class: domain-specific typed entity",
        "leaf instance: concrete operational or clinical concept",
    ]
    score_concepts = [Concept(curie=str(i), label=str(i), description=c) for i, c in enumerate(score_criteria)]
    score_subgraph = _flat_subgraph("real_score", score_concepts)
    bundle_score = build_ontology_sequence(
        tokenizer_encode=tokenizer_encode,
        special_ids=special_ids,
        state=state,
        instructions="Determine the ontology specificity depth level for this entity.",
        active_concepts=score_concepts,
        ontology=score_subgraph,
        qtype_id=QTYPE_IDS["score"],
        qtype_name="score",
    )
    samples.append(
        TrainingSample(
            bundle=bundle_score,
            target_idx=score_level,
            act_target=1 if score_level >= 2 else 0,
            qtype_name="score",
        )
    )

    # 5. Noul task: boolean assertion holds
    is_valid_assertion = rng.random() < 0.5
    noul_options = [
        Concept(curie="false", label="false", description="Assertion does not hold"),
        Concept(curie="true", label="true", description="Assertion holds"),
    ]
    noul_subgraph = _flat_subgraph("real_noul", noul_options)
    c_desc = f" ({concept.description})" if concept.description else ""
    noul_text = (
        f"Is this entity accurately classified as {concept.label} ({concept.curie}){c_desc}?"
        if is_valid_assertion
        else f"Is this entity completely unrelated to {concept.label} ({concept.curie})?"
    )
    bundle_noul = build_ontology_sequence(
        tokenizer_encode=tokenizer_encode,
        special_ids=special_ids,
        state=state,
        instructions=noul_text,
        active_concepts=noul_options,
        ontology=noul_subgraph,
        qtype_id=QTYPE_IDS["noul"],
        qtype_name="noul",
    )
    samples.append(
        TrainingSample(
            bundle=bundle_noul,
            target_idx=1 if is_valid_assertion else 0,
            act_target=1 if is_valid_assertion else 0,
            qtype_name="noul",
        )
    )

    return samples


def generate_real_ontology_samples(
    ontology: Ontology,
    domain: str = "atlas",
    num_samples: int = 1500,
    tokenizer_encode: Any = None,
    special_ids: dict[str, int] | None = None,
    use_real_abox: bool = True,
    seed: int = 42,
    synonym_aug_prob: float = 0.3,
) -> list[TrainingSample]:
    """Generate diverse training samples across all 5 decision tasks for any real ontology.

    Args:
        synonym_aug_prob: Probability (0–1) of substituting a concept's canonical label with
            one of its synonyms when building a ``{term}``-template query.  Set to 0 to disable.
    """
    rng = random.Random(seed)
    special = special_ids or {"cls": 50281, "sep": 50282, "mask": 50284, "pad": 50283}
    templates = DOMAIN_QUERY_TEMPLATES.get(domain, DOMAIN_QUERY_TEMPLATES["atlas"])

    # Eligible concepts with valid labels (filter out numeric identifiers like SNOMED codes)
    all_curies = [
        c
        for c, concept in ontology.concepts.items()
        if concept.label and len(concept.label.strip()) > 1 and not concept.label.strip().isdigit()
    ]
    if not all_curies:
        raise ValueError(f"Ontology {ontology.name} has no concepts with labels.")

    # Select candidate pool (prioritize concepts with parents/children)
    hierarchy_curies = [c for c in all_curies if ontology.concepts[c].parents or ontology.concepts[c].children]
    pool = hierarchy_curies if len(hierarchy_curies) >= 8 else all_curies

    # Build hard-negative sampler once so the children map is shared across all calls
    sampler = HardNegativeSampler(ontology)

    samples: list[TrainingSample] = []

    # 1. Ingest real ABox facts when domain is atlas and use_real_abox is enabled
    if domain == "atlas" and use_real_abox:
        abox_facts = load_atlas_real_abox()
        valid_abox = [f for f in abox_facts if f.target_curie in ontology.concepts]
        print(f"Ingested {len(valid_abox)} real enterprise ABox facts for Atlas grounding.")
        rng.shuffle(valid_abox)
        for fact in valid_abox:
            generated = _generate_tasks_for_concept_query(
                ontology=ontology,
                target_curie=fact.target_curie,
                query_text=fact.text,
                domain=domain,
                pool=pool,
                tokenizer_encode=tokenizer_encode,
                special_ids=special,
                rng=rng,
                sampler=sampler,
            )
            samples.extend(generated)
            if len(samples) >= num_samples:
                break

    # 2. Complete remaining required sample volume via domain templates across full pool
    per_concept = max(1, (num_samples - len(samples)) // len(pool) + 1)
    for target_curie in pool:
        if len(samples) >= num_samples * 2:
            break
        concept = ontology.concepts[target_curie]

        for _ in range(per_concept):
            tmpl = rng.choice(templates)

            # Build query text: property-based templates have no {term} placeholder
            if "{term}" in tmpl:
                # Synonym augmentation: with some probability use a synonym instead of the label
                if synonym_aug_prob > 0 and concept.synonyms and rng.random() < synonym_aug_prob:
                    term = rng.choice(concept.synonyms)
                else:
                    term = concept.label
                state_text = tmpl.format(term=term)
            else:
                # Property-based template — use directly; it already encodes observable attributes
                state_text = tmpl

            generated = _generate_tasks_for_concept_query(
                ontology=ontology,
                target_curie=target_curie,
                query_text=state_text,
                domain=domain,
                pool=pool,
                tokenizer_encode=tokenizer_encode,
                special_ids=special,
                rng=rng,
                sampler=sampler,
            )
            samples.extend(generated)

    rng.shuffle(samples)
    return samples[:num_samples]
