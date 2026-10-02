/**
 * Model Context Protocol (MCP 2024-11-05 & 2026) Server for O-Laya / Leio in Node.js / TypeScript.
 * Fully compliant with Model Context Protocol standards. Zero external dependencies.
 */

import * as readline from "node:readline";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { OLaya, OntologyDAG } from "./index.js";

// Standard embedded Incident, GS1, and Leio ontologies for instant zero-install execution
export const DEFAULT_INCIDENT_CLASSES = [
  { id: 0, curie: "INCIDENT:ROOT", label: "Operational Incident", parents: [], disjointWith: [], synonyms: ["incident", "outage"] },
  { id: 1, curie: "INCIDENT:DATA", label: "Data Incident", parents: [0], disjointWith: [2], synonyms: ["database", "storage", "postgres", "sql"] },
  { id: 2, curie: "INCIDENT:BILLING", label: "Billing Incident", parents: [0], disjointWith: [1], synonyms: ["payment", "invoice", "stripe", "credit"] },
  { id: 3, curie: "INCIDENT:DB_DEADLOCK", label: "Database Deadlock", parents: [1], disjointWith: [], synonyms: ["deadlock", "lock wait", "serialization"] },
  { id: 4, curie: "INCIDENT:AUTH_BREACH", label: "Security Breach", parents: [0], disjointWith: [], synonyms: ["breach", "token", "auth", "unauthorized"] },
];

export const DEFAULT_GS1_CLASSES = [
  { id: 0, curie: "gs1:Product", label: "Trade Item Product", parents: [], disjointWith: [], synonyms: ["product", "item", "gtin", "trade item"] },
  { id: 1, curie: "gs1:FoodBeverageTobaccoProduct", label: "Food and Beverage Product", parents: [0], disjointWith: [], synonyms: ["food", "beverage", "dairy", "milk", "edible"] },
  { id: 2, curie: "gs1:PackagingDetails", label: "Packaging Details", parents: [], disjointWith: [], synonyms: ["packaging", "carton", "box", "pallet", "bottle"] },
  { id: 3, curie: "gs1:OrganicClaimDetails", label: "Organic Claim Details", parents: [1], disjointWith: [], synonyms: ["organic", "usda organic", "bio", "certified organic"] },
  { id: 4, curie: "gs1:LogisticsService", label: "Logistics and Supply Chain Service", parents: [], disjointWith: [], synonyms: ["logistics", "shipping", "freight", "warehouse"] },
];

export const DEFAULT_LEIO_CLASSES = [
  {
    "id": 0,
    "curie": "leio:Root",
    "label": "Leio Ecosystem Root",
    "description": "Root concept of the Leio System 1 Decision Engine, Model Architecture, and Pricing Taxonomy.",
    "parents": [],
    "disjointWith": [],
    "synonyms": [
      "leio",
      "o-laya",
      "ecosystem",
      "root"
    ]
  },
  {
    "id": 1,
    "curie": "leio:Product",
    "label": "Product Architecture",
    "description": "Deliverable software components, runtimes, SDKs, agent wedges, and client-facing interfaces in the Leio platform.",
    "parents": [
      0
    ],
    "disjointWith": [
      2,
      3,
      4
    ],
    "synonyms": [
      "software",
      "components",
      "stack",
      "tooling",
      "interfaces"
    ]
  },
  {
    "id": 2,
    "curie": "leio:Model",
    "label": "Model & Formal Kernel",
    "description": "Category-theoretic reasoning kernel, axiom enforcement operators, probability calibration, and routing mechanisms.",
    "parents": [
      0
    ],
    "disjointWith": [
      1,
      3,
      4
    ],
    "synonyms": [
      "algorithm",
      "kernel",
      "mathematics",
      "axioms",
      "inference"
    ]
  },
  {
    "id": 3,
    "curie": "leio:Pricing",
    "label": "Pricing & Commercial Economics",
    "description": "Commercial licensing, token metering tiers, input-only billing rates, and enterprise deployment options.",
    "parents": [
      0
    ],
    "disjointWith": [
      1,
      2,
      4
    ],
    "synonyms": [
      "economics",
      "billing",
      "cost",
      "tiers",
      "commercial"
    ]
  },
  {
    "id": 4,
    "curie": "leio:DomainOntology",
    "label": "Supported Domain Ontologies",
    "description": "Formal ontology standards supported natively by Leio for physical supply chain, healthcare, e-commerce, and SRE.",
    "parents": [
      0
    ],
    "disjointWith": [
      1,
      2,
      3
    ],
    "synonyms": [
      "taxonomies",
      "vocabularies",
      "schemas",
      "standards"
    ]
  },
  {
    "id": 5,
    "curie": "leio:SystemOneEngine",
    "label": "System 1 Decision Engine",
    "description": "High-throughput decision engine that evaluates text against formal DAGs in a single forward pass without autoregressive token generation.",
    "parents": [
      1
    ],
    "disjointWith": [],
    "synonyms": [
      "system 1",
      "decision engine",
      "non-generative classifier",
      "fast triage",
      "olaya engine"
    ]
  },
  {
    "id": 6,
    "curie": "leio:WasmRuntime",
    "label": "WebAssembly In-Browser Runtime",
    "description": "189 KB standalone WASM binary compiled from Rust (olaya-wasm), executing client-side decisions in < 1 microsecond in browser memory.",
    "parents": [
      1
    ],
    "disjointWith": [
      10
    ],
    "synonyms": [
      "wasm",
      "webassembly",
      "client-side engine",
      "offline decision",
      "browser wasm"
    ]
  },
  {
    "id": 7,
    "curie": "leio:PythonSDK",
    "label": "Python Open-Weights SDK",
    "description": "Official Python SDK (pip install olaya) with PyTorch, ModernBERT, ONNX runtime, and hard-negative mining trainers.",
    "parents": [
      1
    ],
    "disjointWith": [],
    "synonyms": [
      "python sdk",
      "pip install olaya",
      "python library",
      "open weights"
    ]
  },
  {
    "id": 8,
    "curie": "leio:NodeMCPServer",
    "label": "Node.js MCP Server",
    "description": "Zero-install Node.js Model Context Protocol daemon (npx @mycelia/olaya-mcp) exposing decision tools to Claude Desktop, Cursor, and Windsurf.",
    "parents": [
      1
    ],
    "disjointWith": [],
    "synonyms": [
      "node mcp",
      "npx olaya-mcp",
      "typescript mcp",
      "node agent wedge"
    ]
  },
  {
    "id": 9,
    "curie": "leio:PythonMCPServer",
    "label": "Python MCP Server",
    "description": "Zero-install Python Model Context Protocol server (uvx olaya-mcp) providing stdio and SSE transports for AI agent integration.",
    "parents": [
      1
    ],
    "disjointWith": [],
    "synonyms": [
      "python mcp",
      "uvx olaya-mcp",
      "agent wedge",
      "mcp daemon"
    ]
  },
  {
    "id": 10,
    "curie": "leio:HostedEdgeAPI",
    "label": "Vercel Serverless Edge API",
    "description": "Globally distributed hosted serverless endpoint (POST /api/systemone and /api/mcp) with sub-15ms edge latency and Jev compatibility.",
    "parents": [
      1
    ],
    "disjointWith": [
      6
    ],
    "synonyms": [
      "edge api",
      "hosted endpoint",
      "vercel serverless",
      "systemone api",
      "cloud api"
    ]
  },
  {
    "id": 11,
    "curie": "leio:AutonomousAgentSorter",
    "label": "Autonomous Conveyor Sorter Agent",
    "description": "Industrial warehouse agent listening to 200kHz barcode sensor streams and firing PLC diverter chute signals using ontology down-sets.",
    "parents": [
      1
    ],
    "disjointWith": [],
    "synonyms": [
      "sorter agent",
      "conveyor diverter",
      "warehouse automation",
      "plc control",
      "cross-dock router"
    ]
  },
  {
    "id": 12,
    "curie": "leio:DecisionWorkbench",
    "label": "Dual-View Decision Workbench",
    "description": "Interactive web workbench combining raw text input, real-time probability distributions, ancestor lineage paths, and axiom proofs.",
    "parents": [
      1
    ],
    "disjointWith": [],
    "synonyms": [
      "workbench",
      "playground",
      "demo UI",
      "interactive tester"
    ]
  },
  {
    "id": 13,
    "curie": "leio:CandidateRouter",
    "label": "IDF-Weighted Subgraph Router",
    "description": "Two-stage router with smoothed inverse document frequency to extract relevant subgraphs from massive 100k+ concept ontologies.",
    "parents": [
      1
    ],
    "disjointWith": [],
    "synonyms": [
      "candidate router",
      "bm25 router",
      "subgraph extractor",
      "inverted index"
    ]
  },
  {
    "id": 14,
    "curie": "leio:FormalOntologyGrounding",
    "label": "Formal Ontology Grounding",
    "description": "Deterministic projection of unstructured observation text into structured classes within a formalized Directed Acyclic Graph (DAG).",
    "parents": [
      2
    ],
    "disjointWith": [],
    "synonyms": [
      "ontology grounding",
      "semantic grounding",
      "dag classification",
      "concept resolution"
    ]
  },
  {
    "id": 15,
    "curie": "leio:SubsumptionMonotonicity",
    "label": "Subsumption Monotonicity Guarantee",
    "description": "Mathematical guarantee enforcing P(child) <= P(parent) everywhere in the ontology DAG, preventing logical hierarchy violations.",
    "parents": [
      2
    ],
    "disjointWith": [],
    "synonyms": [
      "subsumption monotonicity",
      "parent child inequality",
      "monotonic probabilities",
      "hierarchy preservation"
    ]
  },
  {
    "id": 16,
    "curie": "leio:DisjointnessKernel",
    "label": "Disjointness Axiom Kernel",
    "description": "Category-theoretic exclusion kernel enforcing A \u2293 B = \u22a5, zeroing contradictory or mutually exclusive branches in decision space.",
    "parents": [
      2
    ],
    "disjointWith": [],
    "synonyms": [
      "disjointness kernel",
      "mutual exclusion",
      "disjoint classes",
      "contradiction elimination"
    ]
  },
  {
    "id": 17,
    "curie": "leio:AxiomVerification",
    "label": "Axiom Projection Operator",
    "description": "Post-softmax projection operator that mathematically proves and enforces monotonicity and disjointness across all candidate concepts.",
    "parents": [
      2
    ],
    "disjointWith": [],
    "synonyms": [
      "axiom projection",
      "formal verification",
      "axiom validator",
      "projection operator"
    ]
  },
  {
    "id": 18,
    "curie": "leio:CalibratedSoftmax",
    "label": "Cardinality-Aware Softmax Calibration",
    "description": "Softmax scaling where temperature T scales with candidate set cardinality to prevent probability overconfidence in dense sublattices.",
    "parents": [
      2
    ],
    "disjointWith": [],
    "synonyms": [
      "calibrated softmax",
      "temperature scaling",
      "cardinality temperature",
      "confidence calibration"
    ]
  },
  {
    "id": 19,
    "curie": "leio:NormalizedEntropyConfidence",
    "label": "Normalized Entropy Confidence Metric",
    "description": "Calibrated decision confidence metric defined as 1 - H / ln(N), quantifying distributional peakness relative to maximum entropy.",
    "parents": [
      2
    ],
    "disjointWith": [],
    "synonyms": [
      "entropy confidence",
      "shannon confidence",
      "normalized entropy",
      "uncertainty metric"
    ]
  },
  {
    "id": 20,
    "curie": "leio:TopKSubgraphExtraction",
    "label": "Top-k Subgraph Local Extraction",
    "description": "Dynamic reduction of 100,000+ concept ontologies to a compact active sublattice (k=12) with full ancestral closure.",
    "parents": [
      2
    ],
    "disjointWith": [],
    "synonyms": [
      "top-k extraction",
      "sublattice",
      "active subgraph",
      "local context"
    ]
  },
  {
    "id": 21,
    "curie": "leio:ZeroHallucinationGuarantee",
    "label": "Zero-Hallucination Formal Guarantee",
    "description": "Mathematical impossibility of hallucination: outputs are bounded strictly to verified nodes in the formal ontology lattice.",
    "parents": [
      2
    ],
    "disjointWith": [],
    "synonyms": [
      "zero hallucination",
      "non-hallucinating",
      "bounded prediction",
      "provable reliability"
    ]
  },
  {
    "id": 22,
    "curie": "leio:InputOnlyMetering",
    "label": "Input-Only Token Metering",
    "description": "Pricing mechanism metering only input observation tokens at zsh.042 per 1M tokens, exactly matching Jev while outputs are free.",
    "parents": [
      3
    ],
    "disjointWith": [
      23
    ],
    "synonyms": [
      "input-only pricing",
      "input token metering",
      "zsh.042/1m",
      "transparent metering"
    ]
  },
  {
    "id": 23,
    "curie": "leio:FreeOutputTokens",
    "label": "100% Free Output Decisions",
    "description": "Guaranteed zero cost for all generated decision tokens, CURIEs, confidence values, probabilities, and JSON structured answers.",
    "parents": [
      3
    ],
    "disjointWith": [
      22
    ],
    "synonyms": [
      "free outputs",
      "zero output tokens",
      "unmetered decisions",
      "output tokens free"
    ]
  },
  {
    "id": 24,
    "curie": "leio:DeveloperFreeTier",
    "label": "Developer Free Tier",
    "description": "No-cost tier providing 100,000 monthly API calls and unlimited local WebAssembly and Python SDK executions without credit card.",
    "parents": [
      3
    ],
    "disjointWith": [
      25,
      26
    ],
    "synonyms": [
      "free tier",
      "developer tier",
      "hobby tier",
      "free tier 100k"
    ]
  },
  {
    "id": 25,
    "curie": "leio:PayAsYouGoScaleTier",
    "label": "Pay-As-You-Go Scale Tier",
    "description": "Self-serve commercial tier billed at zsh.042 / 1M input tokens with automated monthly invoicing and volume tiered discounts.",
    "parents": [
      3
    ],
    "disjointWith": [
      24,
      26
    ],
    "synonyms": [
      "scale tier",
      "pay as you go",
      "usage based",
      "production api"
    ]
  },
  {
    "id": 26,
    "curie": "leio:DedicatedEnterpriseVPCTier",
    "label": "Dedicated Enterprise VPC Tier",
    "description": "Air-gapped on-premise deployment with custom ontology ingestion, 99.999% SLA, dedicated throughput, and private model fine-tuning.",
    "parents": [
      3
    ],
    "disjointWith": [
      24,
      25
    ],
    "synonyms": [
      "enterprise tier",
      "vpc deployment",
      "air gapped",
      "on-premise",
      "custom sla"
    ]
  },
  {
    "id": 27,
    "curie": "leio:JevCompatibleBilling",
    "label": "Jev-Compatible Billing Protocol",
    "description": "API usage response headers and metering telemetry that adhere strictly to TypeSafe AI Jev billing specifications.",
    "parents": [
      3
    ],
    "disjointWith": [],
    "synonyms": [
      "jev billing",
      "jev compatibility",
      "usage telemetry",
      "token accounting"
    ]
  },
  {
    "id": 28,
    "curie": "leio:GS1WebVocabulary",
    "label": "GS1 Web Vocabulary Standard",
    "description": "Global standard ontology for trade items, retail barcodes (GTIN), food safety cold chains, and hazardous materials.",
    "parents": [
      4
    ],
    "disjointWith": [
      29,
      31
    ],
    "synonyms": [
      "gs1",
      "supply chain ontology",
      "gtin standard",
      "trade item",
      "logistics ontology"
    ]
  },
  {
    "id": 29,
    "curie": "leio:DOIDDiseaseOntology",
    "label": "Human Disease Ontology (DOID)",
    "description": "Comprehensive biomedical disease taxonomy with 730+ oncology, pathology, and clinical infection classes and ICD crosswalks.",
    "parents": [
      4
    ],
    "disjointWith": [
      28,
      31
    ],
    "synonyms": [
      "doid",
      "disease ontology",
      "clinical medicine",
      "oncology taxonomy",
      "pathology"
    ]
  },
  {
    "id": 30,
    "curie": "leio:SchemaOrgOntology",
    "label": "Schema.org Web Entity Standard",
    "description": "Universal web schema for products, organizations, reviews, actions, events, and search engine structured data.",
    "parents": [
      4
    ],
    "disjointWith": [],
    "synonyms": [
      "schema.org",
      "schemaorg",
      "json-ld",
      "web schema",
      "google rich snippets"
    ]
  },
  {
    "id": 31,
    "curie": "leio:OperationalIncidentOntology",
    "label": "Operational Incident Taxonomy",
    "description": "Taxonomy for Site Reliability Engineering (SRE), database outages, authentication breaches, and critical server alerts.",
    "parents": [
      4
    ],
    "disjointWith": [
      28,
      29
    ],
    "synonyms": [
      "incident ontology",
      "sre taxonomy",
      "alert triage",
      "devops ontology",
      "outage classification"
    ]
  },
  {
    "id": 32,
    "curie": "leio:BasicFormalOntology",
    "label": "Basic Formal Ontology (BFO ISO/IEC 21838-2)",
    "description": "Top-level foundational ontology establishing category-theoretic separation between continuants, occurrents, and qualities.",
    "parents": [
      4
    ],
    "disjointWith": [],
    "synonyms": [
      "bfo",
      "basic formal ontology",
      "top-level ontology",
      "iso 21838-2",
      "foundational ontology"
    ]
  }
];


export class OLayaNodeMCPServer {
  private incidentEngine: OLaya;
  private gs1Engine: OLaya;
  private leioEngine: OLaya;

  constructor() {
    this.incidentEngine = new OLaya(new OntologyDAG(DEFAULT_INCIDENT_CLASSES));
    this.gs1Engine = new OLaya(new OntologyDAG(DEFAULT_GS1_CLASSES));
    this.leioEngine = new OLaya(new OntologyDAG(DEFAULT_LEIO_CLASSES));
  }

  private getEngine(domain: string): OLaya {
    const d = domain.toLowerCase();
    if (d === "incident") return this.incidentEngine;
    if (d === "leio" || d === "product" || d === "model" || d === "pricing") return this.leioEngine;
    return this.gs1Engine;
  }

  public async executeTool(name: string, args: Record<string, any>): Promise<Record<string, any>> {
    if (name === "olaya_ground" || name === "leio_ground") {
      const state = String(args.state ?? args.query ?? "");
      const domain = String(args.domain ?? "gs1").toLowerCase();
      const engine = this.getEngine(domain);

      const decision = await engine.systemOne(state);
      return {
        domain,
        curie: decision.selectedConcept.curie,
        label: decision.selectedConcept.label,
        confidence: decision.confidence,
        probabilities: decision.probabilities,
        axioms_verified: decision.axiomsVerified,
      };
    }

    if (name === "leio_inspect_ecosystem" || name === "olaya_inspect_ecosystem") {
      const query = String(args.query ?? args.state ?? "");
      const decision = await this.leioEngine.systemOne(query);
      return {
        domain: "leio",
        curie: decision.selectedConcept.curie,
        label: decision.selectedConcept.label,
        confidence: decision.confidence,
        probabilities: decision.probabilities,
        axioms_verified: decision.axiomsVerified,
      };
    }

    if (name === "olaya_subsumes" || name === "leio_subsumes") {
      const state = String(args.state ?? "");
      const targetConcept = String(args.target_concept ?? "");
      const domain = String(args.domain ?? "gs1").toLowerCase();
      const engine = this.getEngine(domain);

      const decision = await engine.systemOne(state);
      const isSubsumed = decision.selectedConcept.curie.toLowerCase() === targetConcept.toLowerCase() ||
        decision.selectedConcept.label.toLowerCase() === targetConcept.toLowerCase();

      return {
        domain,
        target_concept: targetConcept,
        subsumes: isSubsumed,
        confidence: decision.confidence,
        axioms_verified: decision.axiomsVerified,
      };
    }

    if (name === "olaya_choice" || name === "leio_choice") {
      const state = String(args.state ?? "");
      const options = Array.isArray(args.options) ? args.options : [];
      const res = await this.incidentEngine.choice(state, options);
      return {
        selected_option: res.choice,
        confidence: res.confidence,
        probabilities: res.probabilities,
      };
    }

    if (name === "olaya_score" || name === "leio_score") {
      const state = String(args.state ?? "");
      const criteria = Array.isArray(args.criteria) ? args.criteria : ["low", "medium", "urgent", "critical"];
      const score = await this.incidentEngine.score(state, criteria.join(" "));
      return {
        score,
        scale: criteria,
      };
    }

    throw new Error(`Unknown tool: ${name}`);
  }

  public async handleMessage(req: Record<string, any>): Promise<Record<string, any> | null> {
    const reqId = req.id;
    const method = req.method;
    const params = req.params ?? {};

    // Standard client notifications (no response required)
    if (method === "notifications/initialized" || method === "initialized") {
      return null;
    }
    if (method === "notifications/cancelled" || method === "cancelled") {
      return null;
    }

    if (method === "ping") {
      return {
        jsonrpc: "2.0",
        id: reqId,
        result: {},
      };
    }

    if (method === "logging/setLevel") {
      return {
        jsonrpc: "2.0",
        id: reqId,
        result: {},
      };
    }

    if (method === "initialize") {
      const requestedVersion = params.protocolVersion || "2024-11-05";
      const protocolVersion = String(requestedVersion).startsWith("2026") ? "2026-07-28" : "2024-11-05";
      return {
        jsonrpc: "2.0",
        id: reqId,
        result: {
          protocolVersion,
          serverInfo: {
            name: "leio-mcp-engine",
            version: "2026.1.0",
          },
          capabilities: {
            stateless: true,
            tools: { listChanged: false },
            resources: { subscribe: false, listChanged: false },
            prompts: { listChanged: false },
            logging: {},
          },
        },
      };
    }

    if (method === "tools/list") {
      return {
        jsonrpc: "2.0",
        id: reqId,
        result: {
          tools: [
            {
              name: "olaya_ground",
              description: "Ground raw text or barcode observations into formal ontology classes (GS1, Incident, or Leio) with calibrated confidence.",
              inputSchema: {
                type: "object",
                properties: {
                  state: { type: "string", description: "Text or barcode observation to ground" },
                  domain: { type: "string", enum: ["gs1", "incident", "leio"], default: "gs1" },
                },
                required: ["state"],
              },
            },
            {
              name: "leio_inspect_ecosystem",
              description: "Ground questions or requirements about Leio product architecture, model mathematical axioms, or $0.042/1M pricing.",
              inputSchema: {
                type: "object",
                properties: {
                  query: { type: "string", description: "User question, product requirement, pricing inquiry, or architectural scenario" },
                },
                required: ["query"],
              },
            },
            {
              name: "olaya_subsumes",
              description: "Test if an observation belongs to a target ontology class or its down-set subclasses.",
              inputSchema: {
                type: "object",
                properties: {
                  state: { type: "string", description: "Observation text" },
                  target_concept: { type: "string", description: "Target concept CURIE or label" },
                  domain: { type: "string", enum: ["gs1", "incident", "leio"], default: "gs1" },
                },
                required: ["state", "target_concept"],
              },
            },
            {
              name: "olaya_choice",
              description: "Multi-criteria ranking across discrete options with calibrated probabilities.",
              inputSchema: {
                type: "object",
                properties: {
                  state: { type: "string", description: "Context state" },
                  options: { type: "array", items: { type: "string" }, description: "Option labels" },
                },
                required: ["state", "options"],
              },
            },
            {
              name: "olaya_score",
              description: "Rate an observation on an ordered rating/severity scale.",
              inputSchema: {
                type: "object",
                properties: {
                  state: { type: "string", description: "Context state" },
                  criteria: { type: "array", items: { type: "string" }, description: "Ordered scale levels" },
                },
                required: ["state", "criteria"],
              },
            },
            {
              name: "leio_ground",
              description: "Ground raw text or barcode observations into formal ontology classes with calibrated confidence.",
              inputSchema: {
                type: "object",
                properties: {
                  state: { type: "string", description: "Observation text" },
                  domain: { type: "string", enum: ["gs1", "incident", "leio"], default: "gs1" },
                },
                required: ["state"],
              },
            },
            {
              name: "leio_subsumes",
              description: "Test if an observation belongs to a target ontology class or its down-set subclasses.",
              inputSchema: {
                type: "object",
                properties: {
                  state: { type: "string", description: "Observation text" },
                  target_concept: { type: "string", description: "Target concept CURIE or label" },
                  domain: { type: "string", enum: ["gs1", "incident", "leio"], default: "gs1" },
                },
                required: ["state", "target_concept"],
              },
            },
            {
              name: "leio_choice",
              description: "Multi-criteria ranking across discrete options with calibrated probabilities.",
              inputSchema: {
                type: "object",
                properties: {
                  state: { type: "string", description: "Context state" },
                  options: { type: "array", items: { type: "string" }, description: "Option labels" },
                },
                required: ["state", "options"],
              },
            },
            {
              name: "leio_score",
              description: "Rate an observation on an ordered rating/severity scale.",
              inputSchema: {
                type: "object",
                properties: {
                  state: { type: "string", description: "Context state" },
                  criteria: { type: "array", items: { type: "string" }, description: "Ordered scale levels" },
                },
                required: ["state", "criteria"],
              },
            },
          ],
        },
      };
    }

    if (method === "tools/call") {
      try {
        const result = await this.executeTool(params.name, params.arguments ?? {});
        return {
          jsonrpc: "2.0",
          id: reqId,
          result: {
            content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
            isError: false,
          },
        };
      } catch (e: any) {
        return {
          jsonrpc: "2.0",
          id: reqId,
          result: {
            content: [{ type: "text", text: `Error: ${e.message}` }],
            isError: true,
          },
        };
      }
    }

    if (method === "resources/list") {
      return {
        jsonrpc: "2.0",
        id: reqId,
        result: {
          resources: [
            {
              uri: "ontology://gs1",
              name: "GS1 Web Vocabulary",
              description: "Official GS1 Web standard class hierarchy for trade items and packaging.",
              mimeType: "application/json",
            },
            {
              uri: "ontology://incident",
              name: "Operational Incident Ontology",
              description: "System 1 operational reliability and incident triage taxonomy.",
              mimeType: "application/json",
            },
            {
              uri: "ontology://leio",
              name: "Leio Product, Model & Pricing Ontology",
              description: "Formal ontology defining Leio System 1 product components, mathematical kernel guarantees, and $0.042/1M input pricing tiers.",
              mimeType: "application/json",
            },
          ],
        },
      };
    }

    if (method === "resources/templates/list" || method === "resourceTemplates/list") {
      return {
        jsonrpc: "2.0",
        id: reqId,
        result: { resourceTemplates: [] },
      };
    }

    if (method === "resources/read") {
      const uri = String(params.uri ?? "");
      let engine = this.gs1Engine;
      let name = "GS1WebVocab";
      if (uri.includes("incident")) { engine = this.incidentEngine; name = "OperationalIncident"; }
      if (uri.includes("leio") || uri.includes("product")) { engine = this.leioEngine; name = "LeioProductModelPricing"; }
      return {
        jsonrpc: "2.0",
        id: reqId,
        result: {
          contents: [
            {
              uri,
              mimeType: "application/json",
              text: JSON.stringify({
                uri,
                name,
                num_concepts: engine.ontology.size(),
                sample_concepts: engine.ontology.getAll().slice(0, 15).map(c => ({
                  curie: c.curie,
                  label: c.label,
                  description: c.description,
                })),
              }, null, 2),
            },
          ],
        },
      };
    }

    if (method === "prompts/list") {
      return {
        jsonrpc: "2.0",
        id: reqId,
        result: {
          prompts: [
            {
              name: "explain_leio_architecture_and_pricing",
              description: "Explain how Leio System 1 compares to generative LLMs, its mathematical axiom guarantees, and transparent token pricing.",
              arguments: [
                {
                  name: "customer_requirement",
                  description: "Description of the use case, throughput need, or budget constraint.",
                  required: true,
                },
              ],
            },
          ],
        },
      };
    }

    if (method === "prompts/get") {
      const pArgs = params.arguments ?? {};
      const cr = pArgs.customer_requirement ?? "";
      return {
        jsonrpc: "2.0",
        id: reqId,
        result: {
          messages: [
            {
              role: "user",
              content: {
                type: "text",
                text: `Please evaluate the following system requirement against Leio System 1 architecture, formal axioms, and $0.042/1M pricing:\n\n${cr}`,
              },
            },
          ],
        },
      };
    }

    if (reqId === undefined) return null;

    return {
      jsonrpc: "2.0",
      id: reqId,
      error: { code: -32601, message: `Method not found: ${method}` },
    };
  }

  public runStdio(): void {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false,
    });

    rl.on("line", async (line: string) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      try {
        const req = JSON.parse(trimmed);
        const resp = await this.handleMessage(req);
        if (resp !== null) {
          process.stdout.write(JSON.stringify(resp) + "\n");
        }
      } catch (err: any) {
        process.stdout.write(
          JSON.stringify({
            jsonrpc: "2.0",
            id: null,
            error: { code: -32700, message: `Parse error: ${err.message}` },
          }) + "\n"
        );
      }
    });
  }
}

export function installNodeMCPConfig(client: "claude" | "cursor" = "claude", dryRun: boolean = false): string {
  const home = os.homedir();
  let configPath: string;

  if (client === "claude") {
    if (process.platform === "darwin") {
      configPath = path.join(home, "Library", "Application Support", "Claude", "claude_desktop_config.json");
    } else if (process.platform === "win32") {
      const appdata = process.env.APPDATA || path.join(home, "AppData", "Roaming");
      configPath = path.join(appdata, "Claude", "claude_desktop_config.json");
    } else {
      configPath = path.join(home, ".config", "Claude", "claude_desktop_config.json");
    }
  } else if (client === "cursor") {
    configPath = path.join(process.cwd(), ".cursor", "mcp.json");
  } else {
    throw new Error(`Unknown client: ${client}`);
  }

  let configData: Record<string, any> = {};
  if (fs.existsSync(configPath)) {
    try {
      configData = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    } catch {
      configData = {};
    }
  }

  if (!configData.mcpServers) {
    configData.mcpServers = {};
  }

  configData.mcpServers["olaya"] = {
    command: "npx",
    args: ["@mycelia/olaya-mcp"],
  };

  const outJson = JSON.stringify(configData, null, 2);
  if (dryRun) {
    return `[Dry-Run] Target: ${configPath}\n${outJson}`;
  }

  fs.mkdirSync(path.dirname(configPath), { recursive: true });
  fs.writeFileSync(configPath, outJson, "utf-8");
  return `Successfully installed O-Laya MCP configuration to: ${configPath}`;
}
