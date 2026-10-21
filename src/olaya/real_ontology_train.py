"""Multi-domain training pipeline for real-world ontologies in O-Laya.

Supports training the ModernBERT / BERT decision head on real ontologies:
- any user-supplied ontology (``domain="generic"``, ontology-agnostic templates)
- Human Disease Ontology (DOID - clinical diagnosis, oncology, pathology)
- Schema.org (web entities, products, actions, organizations, media)
"""

from __future__ import annotations

import json
import random
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from olaya.engine import _flat_subgraph
from olaya.evaluation import SourceObservation, normalized_text, stable_id
from olaya.hard_negative_sampler import HardNegativeSampler
from olaya.ontology import Concept, Ontology
from olaya.sequence import QTYPE_IDS, build_ontology_sequence
from olaya.train import TrainingSample

# Domain query templates to produce diverse, realistic state queries.
# Templates marked with {term} use the concept label (with optional synonym augmentation).
# Property-based templates (no {term}) describe observable characteristics directly,
# forcing the model to learn discriminative features rather than label memorisation.
DOMAIN_QUERY_TEMPLATES: dict[str, list[str]] = {
    # Ontology-agnostic templates for any user-supplied ontology (the default domain).
    "generic": [
        "Observation classified as {term}.",
        "Record received matching the definition of {term}.",
        "Event log entry describing an instance of {term}.",
        "Data payload whose attributes conform to {term}.",
        "Operational report: the observed entity is a {term}.",
        "Analyst note: evidence points to {term}.",
        "Catalogue entry categorised under {term}.",
        "Inbound message referencing a {term}.",
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
    """Load an ontology from O-Laya JSON, Turtle, OWL/RDF or JSON-LD (see :mod:`olaya.ontology_io`)."""
    if isinstance(path_or_onto, Ontology):
        return path_or_onto

    path = Path(path_or_onto)
    if not path.exists():
        raise FileNotFoundError(f"Ontology file not found: {path}")

    # If already a JSON ontology
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return Ontology.from_json(data)

    # Turtle / OWL / RDF / JSON-LD: pure-Python extraction with the Rust converter's CURIE scheme,
    # so files converted earlier with ``convert_ontology`` keep the same CURIEs.
    from olaya.ontology_io import load_ontology

    return load_ontology(path, name=name, curie_style="local")


def _specificity_depth(ontology: Ontology, target_curie: str) -> int:
    """Longest parent path depth, capped at three; cycles fail closed."""
    visiting: set[str] = set()
    memo: dict[str, int] = {}

    def depth(curie: str) -> int:
        if curie in visiting:
            raise ValueError("Ontology parent cycle prevents specificity scoring")
        if curie in memo:
            return memo[curie]
        visiting.add(curie)
        parents = sorted(ontology.concepts[curie].parents)
        if any(parent not in ontology.concepts for parent in parents):
            raise ValueError("Ontology parent is absent from specificity graph")
        value = 1 + max((depth(parent) for parent in parents), default=-1)
        visiting.remove(curie)
        memo[curie] = value
        return value

    return min(3, depth(target_curie))


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
    ancestors = ontology.get_ancestors(target_curie)
    state = {
        "domain": domain,
        "query": query_text,
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
    score_level = _specificity_depth(ontology, target_curie)
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
    negative_queries = sorted(c for c in pool if c not in ancestors)
    is_valid_assertion = rng.random() < 0.5 or not negative_queries
    assertion_curie = target_curie if is_valid_assertion else rng.choice(negative_queries)
    assertion_concept = ontology.concepts[assertion_curie]
    noul_options = [
        Concept(curie="false", label="false", description="Assertion does not hold"),
        Concept(curie="true", label="true", description="Assertion holds"),
    ]
    noul_subgraph = _flat_subgraph("real_noul", noul_options)
    c_desc = f" ({assertion_concept.description})" if assertion_concept.description else ""
    noul_text = f"Is this entity accurately classified as {assertion_concept.label} ({assertion_curie}){c_desc}?"
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


def _eligible_pool(ontology: Ontology) -> list[str]:
    labelled = sorted(
        c
        for c, concept in ontology.concepts.items()
        if concept.label and len(concept.label.strip()) > 1 and not concept.label.strip().isdigit()
    )
    if not labelled:
        raise ValueError(f"Ontology {ontology.name} has no concepts with labels.")
    hierarchical = [c for c in labelled if ontology.concepts[c].parents or ontology.concepts[c].children]
    return hierarchical if len(hierarchical) >= 8 else labelled


def _evidence_pairs(evidence: Iterable[Any] | bool | None) -> list[tuple[str, str]]:
    """Normalise caller-supplied evidence to ``(target_curie, text)`` pairs.

    Accepts ``(curie, text)`` tuples or objects with ``target_curie`` and ``text`` attributes (an
    ABox harvester in a private domain pack, for example). ``True`` / ``False`` are the legacy
    ``use_real_abox`` flag: the host-specific harvester it enabled no longer ships with O-Laya.
    """
    if evidence is None or evidence is False:
        return []
    if evidence is True:
        warnings.warn(
            "use_real_abox=True is deprecated and ignored: pass evidence=[(curie, text), ...] from your domain pack",
            DeprecationWarning,
            stacklevel=3,
        )
        return []
    pairs = []
    for item in evidence:
        if isinstance(item, tuple) and len(item) == 2:
            pairs.append((str(item[0]), str(item[1])))
        else:
            pairs.append((str(item.target_curie), str(item.text)))
    return pairs


def generate_real_ontology_observations(
    ontology: Ontology,
    domain: str = "generic",
    num_samples: int = 1500,
    evidence: Iterable[Any] | bool | None = None,
    seed: int = 42,
    synonym_aug_prob: float = 0.3,
    *,
    use_real_abox: bool | None = None,
) -> list[SourceObservation]:
    """Generate source records before task expansion or dataset partitioning.

    Synthetic template families stay together across targets and synonyms.
    Unlabelled property templates are ineligible: they cannot justify arbitrary targets.
    ``evidence`` (real ``(target_curie, text)`` observations) fills up to a fifth of the budget
    before the templates; targets absent from ``ontology`` are skipped.
    """
    if use_real_abox is not None:
        evidence = use_real_abox if evidence is None else evidence
    rng = random.Random(seed)
    pool = _eligible_pool(ontology)
    templates = [t for t in DOMAIN_QUERY_TEMPLATES.get(domain, DOMAIN_QUERY_TEMPLATES["generic"]) if "{term}" in t]
    budget = max(0, num_samples // 5)
    rows: list[SourceObservation] = []
    seen: set[str] = set()
    facts = [(curie, text) for curie, text in _evidence_pairs(evidence) if curie in ontology.concepts]
    if facts:
        rng.shuffle(facts)
        for curie, text in facts:
            source = stable_id(domain, curie, normalized_text(text))
            if source not in seen and len(rows) < budget:
                rows.append(
                    SourceObservation(source, stable_id("abox", normalized_text(text)), curie, text, "real_abox")
                )
                seen.add(source)
    # Bounded attempts avoid infinite loops when the finite template space is exhausted.
    rng.shuffle(pool)
    rng.shuffle(templates)
    candidates = [(target, template) for target in pool for template in templates]
    for target, template in candidates:
        if len(rows) >= budget:
            break
        concept = ontology.concepts[target]
        term = rng.choice(concept.synonyms) if concept.synonyms and rng.random() < synonym_aug_prob else concept.label
        query = template.format(term=term)
        source = stable_id(domain, target, normalized_text(query))
        if source not in seen:
            rows.append(
                SourceObservation(
                    source, stable_id("synthetic_template", domain, template), target, query, "synthetic_template"
                )
            )
            seen.add(source)
    return rows


def expand_observations(
    ontology: Ontology,
    observations: list[SourceObservation],
    domain: str,
    tokenizer_encode: Any,
    special_ids: dict[str, int],
    seed: int = 42,
) -> list[TrainingSample]:
    """Expand only the already selected source partition into five derived tasks."""
    pool = _eligible_pool(ontology)
    sampler = HardNegativeSampler(ontology)
    samples: list[TrainingSample] = []
    for row in observations:
        if row.target_curie not in ontology.concepts:
            raise ValueError("Observation target is absent from ontology")
        rng = random.Random(stable_id(str(seed), row.source_id))
        derived = _generate_tasks_for_concept_query(
            ontology, row.target_curie, row.query_text, domain, pool, tokenizer_encode, special_ids, rng, sampler
        )
        for sample in derived:
            sample.source_id = row.source_id
            sample.group_id = row.group_id
            sample.state_text = row.query_text
            sample.source_kind = row.source_kind
        samples.extend(derived)
    return samples


def generate_real_ontology_samples(
    ontology: Ontology,
    domain: str = "generic",
    num_samples: int = 1500,
    tokenizer_encode: Any = None,
    special_ids: dict[str, int] | None = None,
    evidence: Iterable[Any] | bool | None = None,
    seed: int = 42,
    synonym_aug_prob: float = 0.3,
    *,
    use_real_abox: bool | None = None,
) -> list[TrainingSample]:
    """Legacy unsplit expansion; evaluation callers must split observations first."""
    if use_real_abox is not None and evidence is None:
        evidence = use_real_abox
    observations = generate_real_ontology_observations(ontology, domain, num_samples, evidence, seed, synonym_aug_prob)
    special = special_ids or {"cls": 50281, "sep": 50282, "mask": 50284, "pad": 50283}
    return expand_observations(ontology, observations, domain, tokenizer_encode, special, seed)[:num_samples]
