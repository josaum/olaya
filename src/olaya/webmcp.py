"""Model Context Protocol (MCP) and WebMCP Decision Server for O-Laya.

Exposes O-Laya's ontology-grounded decision engine as standardized MCP tools,
prompts, and resources compliant with the MCP 2024-11-05 specification.
Supports grounding live web payloads, JSON-LD, GTIN barcode data, and text
to GS1, Schema.org, DOID, and Incident ontologies.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from olaya.engine import OLaya
from olaya.ontology import Ontology
from olaya.real_ontology_train import load_ontology_file

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
ROOT = PACKAGE_DIR.parent.parent

ONTOLOGY_CACHE: dict[str, Ontology] = {}
ENGINE_CACHE: dict[str, OLaya] = {}

# Stated on every tool and resource: confidence is normalized-entropy confidence of the projected
# distribution, not an empirically calibrated probability unless the loaded checkpoint carries a
# fitted temperature (see ``olaya.engine``).
CONFIDENCE_CAVEAT = "entropy-based confidence (not empirically calibrated unless a fitted temperature is loaded)"


def bundled_class_count(domain: str) -> int | None:
    """Number of classes in the bundled ontology JSON for ``domain`` (None when not bundled)."""
    path = find_ontology_file(domain)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    concepts = data.get("concepts")
    return len(concepts) if isinstance(concepts, list) else None


def _class_count_text(domain: str) -> str:
    count = bundled_class_count(domain)
    return f"{count} classes" if count is not None else "class count unavailable"


def find_ontology_file(domain: str) -> Path | None:
    """Resolve an ontology JSON file from package data or local repository."""
    domain = domain.lower()
    # 1. Check packaged open data
    bundled = DATA_DIR / f"{domain}_ontology.json"
    if bundled.is_file():
        return bundled
    # 2. Check local repo data/ folder if running from checkout
    local_data = ROOT / "data" / f"{domain}_ontology.json"
    if local_data.is_file():
        return local_data
    # 3. Check alias names
    aliases = {
        "incident": DATA_DIR / "incident_ontology.json",
        "gs1voc": DATA_DIR / "gs1_ontology.json",
        "schema": DATA_DIR / "schemaorg_ontology.json",
        "leio": DATA_DIR / "leio_ontology.json",
        "product": DATA_DIR / "leio_ontology.json",
        "model": DATA_DIR / "leio_ontology.json",
        "pricing": DATA_DIR / "leio_ontology.json",
    }
    if domain in aliases and aliases[domain].is_file():
        return aliases[domain]
    return None


def get_cached_engine(domain: str = "gs1", mode: str = "auto") -> OLaya:
    """Retrieve or lazily initialize an OLaya engine for a domain."""
    domain = domain.lower()
    cache_key = f"{domain}_{mode}"
    if cache_key in ENGINE_CACHE:
        return ENGINE_CACHE[cache_key]

    model_paths = {
        "gs1": ROOT / "models" / "olaya_gs1.pt",
        "schemaorg": ROOT / "models" / "olaya_schemaorg.pt",
        "doid": ROOT / "models" / "olaya_doid.pt",
        "incident": ROOT / "models" / "olaya_decision_model.pt",
    }

    onto_file = find_ontology_file(domain)
    if onto_file is None:
        engine = OLaya()  # fallback to incident ontology
    else:
        onto = load_ontology_file(onto_file)
        ONTOLOGY_CACHE[domain] = onto
        weights = model_paths.get(domain)
        weights_path = str(weights) if (mode != "fast" and weights and weights.exists()) else False
        engine = OLaya(ontology=onto, weights_path=weights_path)

    ENGINE_CACHE[cache_key] = engine
    return engine


def extract_jsonld_or_microdata(html_or_text: str) -> dict[str, Any]:
    """Extract structured JSON-LD, GTINs, or key-value attributes from raw web text."""
    extracted: dict[str, Any] = {
        "jsonld": [],
        "gtins": [],
        "key_values": {},
        "raw_text_clean": "",
    }

    # 1. Search for JSON-LD blocks
    jsonld_matches = re.findall(
        r'<script[^>]*type=[\'"]application/ld\+json[\'"][^>]*>(.*?)</script>',
        html_or_text,
        re.DOTALL | re.IGNORECASE,
    )
    for block in jsonld_matches:
        try:
            parsed = json.loads(block.strip())
            extracted["jsonld"].append(parsed)
        except Exception:
            pass

    # 2. Search for GTIN / barcode numbers
    gtin_match = re.search(r"\b(gtin(?:13|14|8|12)?|upc|ean)[\s:=]+([0-9]{8,14})\b", html_or_text, re.IGNORECASE)
    gtin_val = gtin_match.group(2) if gtin_match else None
    if gtin_val:
        extracted["gtins"].append(gtin_val)

    # Standalone 12-14 digit number heuristic (standard GTIN-12 / GTIN-13 / GTIN-14)
    if not extracted["gtins"]:
        barcode_cands = re.findall(r"\b([0-9]{12,14})\b", html_or_text)
        for b in barcode_cands:
            if not b.startswith("000000"):
                extracted["gtins"].append(b)

    # 3. Clean raw HTML tags
    clean_text = re.sub(r"<[^>]+>", " ", html_or_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    extracted["raw_text_clean"] = clean_text[:1000]

    # Flatten the first JSON-LD entity into the root dict for convenient direct access
    # (multi-block pages keep the full list in extracted["jsonld"])
    if extracted["jsonld"] and isinstance(extracted["jsonld"][0], dict):
        first = extracted["jsonld"][0]
        for k, v in first.items():
            if k not in extracted:
                extracted[k] = v

    return extracted


TOOLS = [
    {
        "name": "olaya_ground",
        "description": f"Ground raw text, web observations, or logs into a formal ontology class (GS1, Schema.org, DOID, or Incident) with {CONFIDENCE_CAVEAT} and axiom verification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "The observation or text to ground (e.g. barcode scan, product description, error log).",
                },
                "domain": {
                    "type": "string",
                    "enum": ["gs1", "schemaorg", "doid", "incident", "leio"],
                    "default": "gs1",
                    "description": "Target ontology domain.",
                },
                "instructions": {
                    "type": "string",
                    "description": "Optional custom classification instructions.",
                },
            },
            "required": ["state"],
        },
    },
    {
        "name": "olaya_subsumes",
        "description": f"Test whether an observation is an instance of a target ontology class or any of its subclasses (principal down-set test). Returns P(state in downset C) with {CONFIDENCE_CAVEAT}.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "The observation text or entity descriptor.",
                },
                "target_concept": {
                    "type": "string",
                    "description": "CURIE or label of the target ontology concept (e.g. 'PackagingDetails', 'gs1:PackagingMarking', 'INCIDENT:DATA').",
                },
                "domain": {
                    "type": "string",
                    "enum": ["gs1", "schemaorg", "doid", "incident", "leio"],
                    "default": "gs1",
                    "description": "Ontology domain.",
                },
            },
            "required": ["state", "target_concept"],
        },
    },
    {
        "name": "olaya_choice",
        "description": f"Multi-criteria decision ranking across user-specified discrete alternatives with normalized probabilities and {CONFIDENCE_CAVEAT}.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "The context or observation state.",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of candidate option strings or labels.",
                },
                "instructions": {
                    "type": "string",
                    "description": "Decision instructions.",
                },
            },
            "required": ["state", "options"],
        },
    },
    {
        "name": "olaya_score",
        "description": f"Evaluate an observation on an ordered rating/severity scale (expected level plus per-level probabilities) with {CONFIDENCE_CAVEAT}.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "Observation or context state.",
                },
                "criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered scale levels, e.g. ['low', 'medium', 'urgent', 'critical'].",
                },
                "instructions": {
                    "type": "string",
                    "description": "Rating instructions.",
                },
            },
            "required": ["state", "criteria"],
        },
    },
    {
        "name": "leio_inspect_ecosystem",
        "description": f"Classify a question or requirement against the bundled Leio product ontology ({_class_count_text('leio')}: product components, model axiom classes and billing classes). Returns the grounded class with {CONFIDENCE_CAVEAT}.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Question, requirement, or scenario text to classify.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "olaya_extract_and_ground",
        "description": "Parse raw web HTML, JSON-LD, or GTIN product text, extract microdata, and ground simultaneously across GS1 and Schema.org.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "web_payload": {
                    "type": "string",
                    "description": "Raw HTML webpage snippet, JSON-LD string, or product specification.",
                }
            },
            "required": ["web_payload"],
        },
    },
    {
        "name": "olaya_obfuscate_pii",
        "description": "Air-gapped, sovereign PII/PHI/MNPI obfuscation on demand. Sanitizes names, emails, phones, SSNs, credit cards, IPs, street addresses, and deal code names before data leaves your boundary or enters reasoning models.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Raw sensitive text, deal memo, transcript, or customer message.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["pseudonymize", "redact", "mask", "hash"],
                    "default": "pseudonymize",
                    "description": "Obfuscation mode: 'pseudonymize' (referential surrogates [PERSON_1]), 'redact' ([REDACTED_EMAIL]), 'mask' (j***@acme.com), or 'hash'.",
                },
                "custom_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional custom company names, executive names, or target assets to protect.",
                },
                "deal_codenames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of sensitive project code names (e.g. ['Project Helios', 'Project Titan']).",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "olaya_deanonymize_pii",
        "description": "Reverses pseudonymized tokens in sanitized text back to their original entities using a private local de-anonymization vault.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sanitized_text": {
                    "type": "string",
                    "description": "Sanitized text containing pseudonym tokens like [PERSON_1], [EMAIL_1].",
                },
                "vault": {
                    "type": "object",
                    "description": "Local de-anonymization vault dictionary returned by olaya_obfuscate_pii.",
                },
            },
            "required": ["sanitized_text", "vault"],
        },
    },
    {
        "name": "olaya_sovereign_reason",
        "description": "One-shot atomic sovereign privacy reasoning loop: automatically obfuscates all PII/MNPI into an in-memory vault (<100 µs), sends strictly sanitized surrogate tokens to the local LLM (Qwen 2.5 7B, 0 data egress), and restores real identities on-premises from the vault. Ensures zero data leakage to model weights or external networks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Prompt or raw memo containing confidential entities, names, deals, or numbers.",
                },
                "system": {
                    "type": "string",
                    "description": "Optional system prompt instruction.",
                },
                "model": {
                    "type": "string",
                    "description": "Local model name (defaults to 'qwen2.5:7b').",
                    "default": "qwen2.5:7b",
                },
                "custom_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target company names, executive names, or banks to vault.",
                },
                "deal_codenames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Deal or project code names to vault.",
                },
            },
            "required": ["prompt"],
        },
    },
]

RESOURCE_SUMMARY_NOTE = (
    "resources/read returns the bundled class list summary (name, class count, sample classes) and the "
    f"weight provenance of the engine serving it; tool outputs carry {CONFIDENCE_CAVEAT}."
)

RESOURCES = [
    {
        "uri": "ontology://gs1",
        "name": "GS1 Web Vocabulary subset",
        "description": f"Bundled subset of the GS1 Web Vocabulary class hierarchy ({_class_count_text('gs1')}) for trade items, packaging and supply chain. {RESOURCE_SUMMARY_NOTE}",
        "mimeType": "application/json",
    },
    {
        "uri": "ontology://schemaorg",
        "name": "Schema.org subset",
        "description": f"Bundled subset of Schema.org web entity, product and e-commerce classes ({_class_count_text('schemaorg')}). {RESOURCE_SUMMARY_NOTE}",
        "mimeType": "application/json",
    },
    {
        "uri": "ontology://doid",
        "name": "Human Disease Ontology (DOID) subset",
        "description": f"Bundled subset of the Human Disease Ontology ({_class_count_text('doid')}). {RESOURCE_SUMMARY_NOTE}",
        "mimeType": "application/json",
    },
    {
        "uri": "ontology://incident",
        "name": "Operational Incident Ontology",
        "description": f"Bundled operational incident triage taxonomy ({_class_count_text('incident')}). {RESOURCE_SUMMARY_NOTE}",
        "mimeType": "application/json",
    },
    {
        "uri": "ontology://leio",
        "name": "Leio Product Ontology",
        "description": f"Bundled ontology of Leio product components, model axiom classes and billing classes ({_class_count_text('leio')}). {RESOURCE_SUMMARY_NOTE}",
        "mimeType": "application/json",
    },
]

PROMPTS = [
    {
        "name": "ground_product_claim",
        "description": "Verify product packaging, barcodes, or e-commerce claims against GS1 and Schema.org standards.",
        "arguments": [
            {
                "name": "product_text",
                "description": "Raw text or barcode description of the product.",
                "required": True,
            }
        ],
    },
    {
        "name": "triage_operational_incident",
        "description": "Triage an infrastructure, database, or software incident into a formal incident taxonomy.",
        "arguments": [
            {
                "name": "log_payload",
                "description": "Error stack trace or incident telemetry summary.",
                "required": True,
            }
        ],
    },
    {
        "name": "classify_leio_requirement",
        "description": "Classify a requirement against the bundled Leio product ontology using the leio_inspect_ecosystem tool.",
        "arguments": [
            {
                "name": "customer_requirement",
                "description": "Description of the use case or requirement to classify.",
                "required": True,
            }
        ],
    },
]


class OLayaMCPServer:
    """Model Context Protocol (MCP) server for O-Laya decision services."""

    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self.server_info = {
            "name": "leio",
            "version": "0.4.0",
            "protocolVersion": "2024-11-05",
        }

    def execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a decision tool and return structured results."""
        if name == "olaya_ground":
            state = args.get("state", "")
            domain = args.get("domain", "gs1")
            instr = args.get("instructions", f"Classify the entity into the {domain} hierarchy.")
            engine = get_cached_engine(domain, mode=self.mode)

            res = engine.system_one(
                state=state,
                questions={
                    "grounding": {
                        "type": "concept",
                        "instructions": instr,
                    }
                },
            )
            ans = res.answers["grounding"]
            return {
                "domain": domain,
                "curie": ans.curie,
                "label": ans.label,
                "confidence": ans.confidence,
                "lineage": ans.lineage,
                "probabilities": {k: round(v, 4) for k, v in ans.candidate_probabilities.items()},
                "axioms_verified": ans.axioms_verified,
                "model": res.model,
                "latency_ms": res.latency_ms,
            }

        elif name == "olaya_subsumes":
            state = args.get("state", "")
            target_concept = args.get("target_concept", "")
            domain = args.get("domain", "gs1")
            engine = get_cached_engine(domain, mode=self.mode)

            matched_curie = target_concept
            for c_id, c in engine.ontology.concepts.items():
                if c_id.lower() == target_concept.lower() or c.label.lower() == target_concept.lower():
                    matched_curie = c_id
                    break

            res = engine.system_one(
                state=state,
                questions={
                    "subsumes_check": {
                        "type": "subsumes",
                        "concept": matched_curie,
                    }
                },
            )
            ans = res.answers["subsumes_check"]
            return {
                "domain": domain,
                "target_concept": matched_curie,
                "subsumes": ans.subsumes,
                "probability": ans.subsumes,  # alias for compatibility
                "confidence": ans.confidence,
                "axioms_verified": ans.axioms_verified,
                "model": res.model,
                "latency_ms": res.latency_ms,
            }

        elif name == "olaya_choice":
            state = args.get("state", "")
            options = args.get("options", [])
            instr = args.get("instructions", "Select the most appropriate option for the context.")
            engine = get_cached_engine("incident", mode=self.mode)

            res = engine.system_one(
                state=state,
                questions={
                    "choice_task": {
                        "type": "choice",
                        "instructions": instr,
                        "criteria": options,  # Python engine uses 'criteria', not 'options'
                    }
                },
            )
            ans = res.answers["choice_task"]
            # ChoiceAnswer has .choice (the winning label)
            return {
                "selected_choice": ans.choice,
                "confidence": ans.confidence,
                "probabilities": {k: round(v, 4) for k, v in ans.probabilities.items()},
                "axioms_verified": ans.axioms_verified,
                "model": res.model,
                "latency_ms": res.latency_ms,
            }

        elif name == "olaya_score":
            state = args.get("state", "")
            criteria = args.get("criteria", ["low", "medium", "urgent", "critical"])
            instr = args.get("instructions", "Rate the observation on the ordered criteria scale.")
            engine = get_cached_engine("incident", mode=self.mode)

            res = engine.system_one(
                state=state,
                questions={
                    "score_check": {
                        "type": "score",
                        "instructions": instr,
                        "criteria": criteria,
                    }
                },
            )
            ans = res.answers["score_check"]
            return {
                "score": ans.score,
                "confidence": ans.confidence,
                "probabilities": {k: round(v, 4) for k, v in ans.probabilities.items()},
                "axioms_verified": ans.axioms_verified,
                "model": res.model,
                "latency_ms": res.latency_ms,
            }

        elif name == "olaya_extract_and_ground":
            payload = args.get("web_payload", "")
            extracted = extract_jsonld_or_microdata(payload)

            text_to_ground = extracted["raw_text_clean"]
            if extracted["gtins"]:
                text_to_ground = f"Barcode GTIN: {extracted['gtins'][0]}. {text_to_ground}"

            gs1_engine = get_cached_engine("gs1", mode=self.mode)
            schema_engine = get_cached_engine("schemaorg", mode=self.mode)

            res_gs1 = gs1_engine.system_one(
                state=text_to_ground,
                questions={
                    "gs1_ground": {
                        "type": "concept",
                        "instructions": "Classify product and packaging under GS1 standards.",
                    }
                },
            )
            res_schema = schema_engine.system_one(
                state=text_to_ground,
                questions={
                    "schema_ground": {
                        "type": "concept",
                        "instructions": "Classify entity under Schema.org vocabulary.",
                    }
                },
            )

            ans_gs1 = res_gs1.answers["gs1_ground"]
            ans_schema = res_schema.answers["schema_ground"]

            return {
                "extracted_metadata": {
                    "gtins": extracted["gtins"],
                    "jsonld_count": len(extracted["jsonld"]),
                },
                "gs1_grounding": {
                    "curie": ans_gs1.curie,
                    "label": ans_gs1.label,
                    "confidence": ans_gs1.confidence,
                    "lineage": ans_gs1.lineage,
                    "axioms_verified": ans_gs1.axioms_verified,
                },
                "schemaorg_grounding": {
                    "curie": ans_schema.curie,
                    "label": ans_schema.label,
                    "confidence": ans_schema.confidence,
                    "lineage": ans_schema.lineage,
                    "axioms_verified": ans_schema.axioms_verified,
                },
                "models": {"gs1": res_gs1.model, "schemaorg": res_schema.model},
                "total_latency_ms": res_gs1.latency_ms + res_schema.latency_ms,
            }

        elif name == "leio_inspect_ecosystem":
            query = args.get("query", "")
            engine = get_cached_engine("leio", mode=self.mode)
            res = engine.system_one(
                state=query,
                questions={
                    "ecosystem_ground": {
                        "type": "concept",
                        "instructions": "Classify the requirement under the Leio product ontology classes.",
                    }
                },
            )
            ans = res.answers["ecosystem_ground"]
            return {
                "domain": "leio",
                "curie": ans.curie,
                "label": ans.label,
                "confidence": ans.confidence,
                "lineage": ans.lineage,
                "probabilities": {k: round(v, 4) for k, v in ans.candidate_probabilities.items()},
                "axioms_verified": ans.axioms_verified,
                "model": res.model,
                "latency_ms": res.latency_ms,
            }

        if name == "olaya_obfuscate_pii":
            from olaya.pii import PIIObfuscator

            obfuscator = PIIObfuscator()
            res = obfuscator.obfuscate(
                text=args["text"],
                mode=args.get("mode", "pseudonymize"),
                custom_entities=args.get("custom_entities"),
                deal_codenames=args.get("deal_codenames"),
            )
            return res.to_dict()

        if name == "olaya_deanonymize_pii":
            from olaya.pii import PIIObfuscator

            obfuscator = PIIObfuscator()
            restored = obfuscator.deanonymize(
                sanitized_text=args["sanitized_text"],
                vault=args["vault"],
            )
            return {"restored_text": restored}

        if name == "olaya_sovereign_reason":
            from olaya.pii import PIIObfuscator

            obfuscator = PIIObfuscator()
            res = obfuscator.sovereign_reason(
                prompt=args["prompt"],
                system=args.get("system"),
                model=args.get("model", "qwen2.5:7b"),
                custom_entities=args.get("custom_entities"),
                deal_codenames=args.get("deal_codenames"),
            )
            return res.to_dict()

        raise ValueError(f"Unknown tool: {name}")

    def handle_message(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle a JSON-RPC 2.0 message from an MCP client."""
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        # Protocol handshake & negotiation
        if method == "initialize":
            client_ver = params.get("protocolVersion", "2024-11-05")
            # Support 2024-11-05 and 2026 specifications
            protocol_version = client_ver if str(client_ver).startswith(("2024", "2026")) else "2024-11-05"
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": protocol_version,
                    "serverInfo": self.server_info,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                        "prompts": {"listChanged": False},
                        "logging": {},
                    },
                },
            }

        elif method in ("notifications/initialized", "initialized"):
            # Client handshake confirmation (notification, no response)
            return None

        elif method in ("notifications/cancelled", "cancelled"):
            return None

        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {},
            }

        elif method == "logging/setLevel":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {},
            }

        elif method in ("resources/templates/list", "resourceTemplates/list"):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"resourceTemplates": []},
            }

        elif method == "completion/complete":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"completion": {"values": [], "hasMore": False}},
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS},
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                result_data = self.execute_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result_data, indent=2),
                            }
                        ],
                        "isError": False,
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {e}"}],
                        "isError": True,
                    },
                }

        elif method == "prompts/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"prompts": PROMPTS},
            }

        elif method == "prompts/get":
            prompt_name = params.get("name")
            p_args = params.get("arguments", {})
            if prompt_name == "ground_product_claim":
                pt = p_args.get("product_text", "")
                text = f"Please analyze and ground the following product observation using the GS1 and Schema.org tools:\n\n{pt}"
            elif prompt_name == "triage_operational_incident":
                lp = p_args.get("log_payload", "")
                text = f"Please classify the following operational incident telemetry using O-Laya tools:\n\n{lp}"
            elif prompt_name == "classify_leio_requirement":
                cr = p_args.get("customer_requirement", "")
                text = f"Please classify the following requirement against the bundled Leio product ontology using the leio_inspect_ecosystem tool:\n\n{cr}"
            else:
                text = f"Prompt: {prompt_name}"

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "messages": [
                        {
                            "role": "user",
                            "content": {"type": "text", "text": text},
                        }
                    ]
                },
            }

        elif method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"resources": RESOURCES},
            }

        elif method == "resources/read":
            uri = params.get("uri", "")
            domain = uri.replace("ontology://", "").lower()
            engine = get_cached_engine(domain, mode=self.mode)
            summary = {
                "uri": uri,
                "domain": domain,
                "name": engine.ontology.name,
                "num_concepts": len(engine.ontology.concepts),
                "model": engine.model_id,
                "confidence": CONFIDENCE_CAVEAT,
                "sample_concepts": [
                    {"curie": c.curie, "label": c.label, "description": c.description}
                    for c in list(engine.ontology.concepts.values())[:15]
                ],
            }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(summary, indent=2),
                        }
                    ]
                },
            }

        # Notifications without response
        if req_id is None:
            return None

        # Method not found
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def run_stdio(self) -> None:
        """Run the MCP server reading JSON-RPC from stdin and replying to stdout."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_message(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()

    def run_sse(self, port: int = 8080) -> None:
        """Run standard WebMCP HTTP/SSE transport for browser and cloud agents."""
        server_inst = self

        class MCPHandler(http.server.BaseHTTPRequestHandler):
            def send_cors_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
                self.send_header(
                    "Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key, mcp-session-id"
                )

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_cors_headers()
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                try:
                    req = json.loads(body.decode("utf-8"))
                    resp = server_inst.handle_message(req)
                    self.send_response(200 if resp is not None else 204)
                    self.send_cors_headers()
                    if resp is not None:
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps(resp).encode("utf-8"))
                    else:
                        self.end_headers()
                except Exception as e:
                    self.send_response(400)
                    self.send_cors_headers()
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {e}"}}
                        ).encode("utf-8")
                    )

            def do_GET(self):
                if self.path in ("/sse", "/events"):
                    self.send_response(200)
                    self.send_cors_headers()
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    init_event = f"event: endpoint\ndata: http://localhost:{port}/mcp\n\n: keep-alive\n\n"
                    self.wfile.write(init_event.encode("utf-8"))
                    self.wfile.flush()
                else:
                    self.send_response(200)
                    self.send_cors_headers()
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    status_info = {
                        "name": "leio",
                        "version": "0.4.0",
                        "protocolVersion": "2024-11-05",
                        "status": "ready",
                        "supportedDomains": ["gs1", "schemaorg", "doid", "incident", "leio"],
                        "capabilities": {"tools": True, "resources": True, "prompts": True},
                        "confidence": CONFIDENCE_CAVEAT,
                    }
                    self.wfile.write(json.dumps(status_info, indent=2).encode("utf-8"))

            def log_message(self, format, *args):
                pass  # suppress standard access logging

        httpd = http.server.HTTPServer(("0.0.0.0", port), MCPHandler)
        print(f"O-Laya WebMCP HTTP/SSE server listening on http://0.0.0.0:{port}...")
        httpd.serve_forever()


def install_mcp_config(client: str = "claude", dry_run: bool = False) -> str:
    """Generate or update Claude Desktop or Cursor configuration for olaya-mcp."""
    home = Path.home()
    if client == "claude":
        if sys.platform == "darwin":
            config_path = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        elif sys.platform == "win32":
            appdata = Path(os.environ.get("APPDATA", str(home / "AppData" / "Roaming")))
            config_path = appdata / "Claude" / "claude_desktop_config.json"
        else:
            config_path = home / ".config" / "Claude" / "claude_desktop_config.json"
    elif client == "cursor":
        config_path = Path(".cursor") / "mcp.json"
    else:
        raise ValueError(f"Unknown client: {client}. Supported: 'claude', 'cursor'")

    config_data: dict[str, Any] = {}
    if config_path.is_file():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config_data = {}

    if "mcpServers" not in config_data:
        config_data["mcpServers"] = {}

    config_data["mcpServers"]["leio"] = {
        "command": "uvx",
        "args": ["leio"],
    }

    out_json = json.dumps(config_data, indent=2)
    if dry_run:
        return f"[Dry-Run] Target: {config_path}\n{out_json}"

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(out_json, encoding="utf-8")
    return f"Successfully installed O-Laya MCP configuration to: {config_path}"


def main() -> None:
    """CLI entrypoint for running the O-Laya MCP server."""
    parser = argparse.ArgumentParser(description="O-Laya Model Context Protocol (MCP) Decision Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: stdio (default) or sse (HTTP/SSE server)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "fast", "neural"],
        default="auto",
        help="Scoring mode: auto (default), fast (pure lexical, <1ms), neural (cross-encoder)",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port for SSE transport (default: 8080)")

    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommands")

    # Install subcommand
    install_parser = subparsers.add_parser("install", help="Install O-Laya MCP config into agent clients")
    install_parser.add_argument(
        "--client",
        choices=["claude", "cursor"],
        default="claude",
        help="Client application (claude or cursor)",
    )
    install_parser.add_argument("--dry-run", action="store_true", help="Print config without writing file")

    # Ground subcommand (instant CLI one-liner)
    ground_parser = subparsers.add_parser("ground", help="Ground text directly from CLI without MCP server")
    ground_parser.add_argument("text", help="Text or barcode to ground")
    ground_parser.add_argument(
        "--domain",
        choices=["gs1", "schemaorg", "doid", "incident"],
        default="gs1",
        help="Target ontology domain",
    )
    ground_parser.add_argument(
        "--mode",
        choices=["auto", "fast", "neural"],
        default=None,
        help="Scoring mode: auto (default), fast (pure lexical, <1ms), neural",
    )

    args = parser.parse_args()

    if args.subcommand == "install":
        msg = install_mcp_config(client=args.client, dry_run=args.dry_run)
        print(msg)
        return

    if args.subcommand == "ground":
        mode = args.mode if args.mode is not None else "auto"
        server = OLayaMCPServer(mode=mode)
        res = server.execute_tool("olaya_ground", {"state": args.text, "domain": args.domain})
        print(json.dumps(res, indent=2))
        return

    # Default: Run MCP server
    server = OLayaMCPServer(mode=args.mode)
    if args.transport == "sse":
        server.run_sse(port=args.port)
    else:
        server.run_stdio()


if __name__ == "__main__":
    main()
