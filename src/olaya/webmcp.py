"""Model Context Protocol (MCP) and WebMCP Decision Server for O-Laya.

Exposes O-Laya's ontology-grounded decision engine as standardized MCP tools
and resources compliant with the MCP 2024-11-05 specification.
Supports grounding live web payloads, JSON-LD, GTIN barcode data, and clinical
text to GS1, Schema.org, Atlas Brain, and DOID ontologies.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from olaya.engine import OLaya
from olaya.ontology import Ontology
from olaya.real_ontology_train import load_ontology_file

ROOT = Path(__file__).resolve().parent.parent.parent

ONTOLOGY_CACHE: dict[str, Ontology] = {}
ENGINE_CACHE: dict[str, OLaya] = {}


def get_cached_engine(domain: str = "gs1") -> OLaya:
    """Retrieve or lazily initialize an OLaya engine for a domain."""
    domain = domain.lower()
    if domain in ENGINE_CACHE:
        return ENGINE_CACHE[domain]

    data_paths = {
        "gs1": ROOT / "data" / "gs1_ontology.json",
        "schemaorg": ROOT / "data" / "schemaorg_ontology.json",
        "atlas": ROOT / "data" / "custom_ontology.json",
        "doid": ROOT / "data" / "doid_ontology.json",
    }
    model_paths = {
        "gs1": ROOT / "models" / "olaya_gs1.pt",
        "schemaorg": ROOT / "models" / "olaya_schemaorg.pt",
        "atlas": ROOT / "models" / "olaya_atlas.pt",
        "doid": ROOT / "models" / "olaya_doid.pt",
    }

    if domain not in data_paths or not data_paths[domain].exists():
        engine = OLaya()  # fallback to incident ontology
    else:
        onto = load_ontology_file(data_paths[domain])
        ONTOLOGY_CACHE[domain] = onto
        weights = model_paths.get(domain)
        weights_path = str(weights) if weights and weights.exists() else None
        engine = OLaya(ontology=onto, weights_path=weights_path)

    ENGINE_CACHE[domain] = engine
    return engine


def extract_jsonld_or_microdata(html_or_text: str) -> dict[str, Any]:
    """Extract structured JSON-LD, GTINs, or key-value attributes from raw web text."""
    # 1. Search for JSON-LD scripts
    json_ld_matches = re.findall(
        r'<script[^>]*type=[\'"]application/ld\+json[\'"][^>]*>(.*?)</script>',
        html_or_text,
        re.DOTALL | re.IGNORECASE,
    )
    for match in json_ld_matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except Exception:
            continue

    # 2. Search for GTIN / barcode numbers
    gtin_match = re.search(r'\b(gtin(?:13|14|8|12)?|upc|ean)[\s:=]+([0-9]{8,14})\b', html_or_text, re.IGNORECASE)
    gtin_val = gtin_match.group(2) if gtin_match else None

    # 3. Fallback: package text into state structure
    return {
        "raw_text": html_or_text[:1000],
        "gtin": gtin_val,
        "tokens": html_or_text.split()[:50],
    }


TOOLS = [
    {
        "name": "olaya_ground",
        "description": "Ground an input state, product payload, or web observation into a verified ontology concept (GS1, Schema.org, Atlas, DOID) with hierarchical lineage and calibrated confidence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "Input text, JSON payload, or web observation to classify.",
                },
                "domain": {
                    "type": "string",
                    "enum": ["gs1", "schemaorg", "atlas", "doid", "incident"],
                    "default": "gs1",
                    "description": "Target ontology domain.",
                },
                "instructions": {
                    "type": "string",
                    "description": "Optional classification instruction or task description.",
                },
            },
            "required": ["state"],
        },
    },
    {
        "name": "olaya_subsumes",
        "description": "Test whether an input entity satisfies an ontology concept or any of its subtypes (down-set subsumption check) with calibrated probability P(true).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "The input state or entity text.",
                },
                "target_concept": {
                    "type": "string",
                    "description": "Concept CURIE to test (e.g. 'Beverage', 'Product', 'MedicalCondition').",
                },
                "domain": {
                    "type": "string",
                    "enum": ["gs1", "schemaorg", "atlas", "doid"],
                    "default": "gs1",
                },
            },
            "required": ["state", "target_concept"],
        },
    },
    {
        "name": "olaya_choice",
        "description": "Select the best match from a discrete set of candidate options with calibrated probabilities and confidence.",
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
]

RESOURCES = [
    {
        "uri": "ontology://gs1",
        "name": "GS1 Web Vocabulary",
        "description": "Official GS1 Web standard class hierarchy for trade items, packaging, and supply chain.",
        "mimeType": "application/json",
    },
    {
        "uri": "ontology://schemaorg",
        "name": "Schema.org Ontology",
        "description": "Schema.org web entity and e-commerce vocabulary.",
        "mimeType": "application/json",
    },
    {
        "uri": "ontology://atlas",
        "name": "Atlas Brain Commerce Ontology",
        "description": "Enterprise commerce, inventory, catalog, and advertising performance ontology.",
        "mimeType": "application/json",
    },
]


class OLayaMCPServer:
    """Model Context Protocol (MCP) server for O-Laya decision services."""

    def __init__(self):
        self.server_info = {
            "name": "olaya-webmcp",
            "version": "0.2.0",
            "protocolVersion": "2024-11-05",
        }

    def execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a decision tool and return structured results."""
        if name == "olaya_ground":
            state = args.get("state", "")
            domain = args.get("domain", "gs1")
            instr = args.get("instructions", f"Classify the entity into the {domain} hierarchy.")
            engine = get_cached_engine(domain)

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
                "latency_ms": res.latency_ms,
            }

        elif name == "olaya_subsumes":
            state = args.get("state", "")
            target_concept = args.get("target_concept", "")
            domain = args.get("domain", "gs1")
            engine = get_cached_engine(domain)

            # Match concept in ontology
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
                "is_instance": ans.subsumes >= 0.5,
                "confidence": ans.confidence,
                "latency_ms": res.latency_ms,
            }

        elif name == "olaya_choice":
            state = args.get("state", "")
            options = args.get("options", [])
            instr = args.get("instructions", "Select the optimal candidate option.")
            engine = get_cached_engine("gs1")

            res = engine.system_one(
                state=state,
                questions={
                    "choice_decision": {
                        "type": "choice",
                        "criteria": options,
                        "instructions": instr,
                    }
                },
            )
            ans = res.answers["choice_decision"]
            return {
                "selected_choice": ans.choice,
                "confidence": ans.confidence,
                "probabilities": ans.probabilities,
                "latency_ms": res.latency_ms,
            }

        elif name == "olaya_extract_and_ground":
            payload = args.get("web_payload", "")
            extracted = extract_jsonld_or_microdata(payload)

            gs1_engine = get_cached_engine("gs1")
            schema_engine = get_cached_engine("schemaorg")

            # Ground to GS1
            gs1_res = gs1_engine.system_one(
                state=extracted,
                questions={"gs1": {"type": "concept", "instructions": "Classify GS1 trade item."}},
            )
            gs1_ans = gs1_res.answers["gs1"]

            # Ground to Schema.org
            schema_res = schema_engine.system_one(
                state=extracted,
                questions={"schema": {"type": "concept", "instructions": "Classify Schema.org entity."}},
            )
            schema_ans = schema_res.answers["schema"]

            return {
                "extracted_microdata": extracted,
                "gs1_grounding": {
                    "curie": gs1_ans.curie,
                    "label": gs1_ans.label,
                    "confidence": gs1_ans.confidence,
                    "lineage": gs1_ans.lineage,
                },
                "schemaorg_grounding": {
                    "curie": schema_ans.curie,
                    "label": schema_ans.label,
                    "confidence": schema_ans.confidence,
                    "lineage": schema_ans.lineage,
                },
            }

        raise ValueError(f"Unknown tool: {name}")

    def handle_message(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle a JSON-RPC 2.0 message from an MCP client."""
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": self.server_info,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                    },
                },
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
                                "text": json.dumps(result_data, indent=2, ensure_ascii=False),
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
                        "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                        "isError": True,
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
            engine = get_cached_engine(domain)
            summary = {
                "name": engine.ontology.name,
                "num_concepts": len(engine.ontology.concepts),
                "sample_concepts": [
                    {"curie": c.curie, "label": c.label, "description": c.description}
                    for c in list(engine.ontology.concepts.values())[:10]
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


def main() -> None:
    """CLI entrypoint for running the O-Laya MCP server."""
    server = OLayaMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
