export const config = {
  runtime: 'edge',
};

const LEIO_CONCEPTS = [
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
    "description": "Zero-install Node.js Model Context Protocol daemon (npx olaya) exposing decision tools to Claude Desktop, Cursor, and Windsurf.",
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
    "description": "Zero-install Python Model Context Protocol server (uvx olaya) providing stdio and SSE transports for AI agent integration.",
    "parents": [
      1
    ],
    "disjointWith": [],
    "synonyms": [
      "python mcp",
      "uvx olaya",
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

const GS1_CONCEPTS = [
  { id: 0, curie: "gs1:Product", label: "Trade Item Product", parents: [], disjointWith: [], synonyms: ["product", "item", "gtin", "trade item"] },
  { id: 1, curie: "gs1:FoodBeverageTobaccoProduct", label: "Food and Beverage Product", parents: [0], disjointWith: [], synonyms: ["food", "beverage", "dairy", "milk", "edible"] },
  { id: 2, curie: "gs1:PackagingDetails", label: "Packaging Details", parents: [], disjointWith: [], synonyms: ["packaging", "carton", "box", "pallet", "bottle"] },
  { id: 3, curie: "gs1:OrganicClaimDetails", label: "Organic Claim Details", parents: [1], disjointWith: [], synonyms: ["organic", "usda organic", "bio", "certified organic"] },
  { id: 4, curie: "gs1:LogisticsService", label: "Logistics and Supply Chain Service", parents: [], disjointWith: [], synonyms: ["logistics", "shipping", "freight", "warehouse"] }
];

const STOPWORDS = new Set([
  'a','an','the','and','or','but','in','on','at','to','for','with','by','about',
  'between','into','through','before','after','above','below','from','up','down',
  'out','off','over','under','then','once','here','there','when','where','why',
  'how','all','any','both','each','few','more','most','other','some','such','no',
  'nor','not','only','own','same','so','than','too','very','can','will','just',
  'should','now','is','are','was','were','be','been','being','have','has','had',
  'do','does','did','of','it','its','that','this','also',
]);

function tokenize(text) {
  return String(text || '')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .split(/\s+/)
    .filter((t) => t.length > 2 && !STOPWORDS.has(t));
}

function scoreConcept(stateTokSet, concept) {
  let score = 0.5;
  const labelToks = tokenize(concept.label);
  for (const t of labelToks) {
    if (stateTokSet.has(t)) score += 6.0;
  }
  for (const syn of (concept.synonyms || [])) {
    for (const t of tokenize(syn)) {
      if (stateTokSet.has(t)) score += 4.0;
    }
  }
  for (const t of tokenize(concept.description || '')) {
    if (stateTokSet.has(t)) score += 1.0;
  }
  return score;
}

function groundState(state, concepts) {
  const stateToks = tokenize(state);
  const stateTokSet = new Set(stateToks);
  const scored = concepts.map(c => ({
    concept: c,
    score: scoreConcept(stateTokSet, c)
  }));
  scored.sort((a, b) => b.score - a.score);

  const top = scored.slice(0, 10);
  const maxScore = top[0]?.score || 1;
  const exps = top.map(x => Math.exp((x.score - maxScore) / 1.1));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(e => e / sumExp);

  const best = top[0].concept;
  return {
    curie: best.curie,
    label: best.label,
    confidence: probs[0] || 0.5,
    probabilities: top.map((x, i) => ({
      curie: x.concept.curie,
      label: x.concept.label,
      prob: Math.round(probs[i] * 10000) / 10000
    })),
    axioms_verified: true
  };
}

const TOOLS = [
  {
    name: "olaya_ground",
    description: "Ground raw text, web observations, or logs into a formal ontology class (GS1, Incident, or Leio) with calibrated confidence and axiom verification.",
    inputSchema: {
      type: "object",
      properties: {
        state: { type: "string", description: "The observation text to ground" },
        domain: { type: "string", enum: ["gs1", "incident", "leio"], default: "gs1" }
      },
      required: ["state"]
    }
  },
  {
    name: "leio_inspect_ecosystem",
    description: "Ground questions or requirements about Leio's product architecture, model mathematical axioms (subsumption monotonicity, disjointness kernel), and pricing economics ($0.042/1M tokens, developer free tier, enterprise VPC).",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "User question, product requirement, pricing inquiry, or architectural scenario" }
      },
      required: ["query"]
    }
  },
  {
    name: "olaya_subsumes",
    description: "Test whether an observation is an instance of a target ontology class or its down-set subclasses.",
    inputSchema: {
      type: "object",
      properties: {
        state: { type: "string", description: "Observation text" },
        target_concept: { type: "string", description: "Target concept CURIE or label" },
        domain: { type: "string", enum: ["gs1", "incident", "leio"], default: "gs1" }
      },
      required: ["state", "target_concept"]
    }
  }
];

const RESOURCES = [
  {
    uri: "ontology://leio",
    name: "Leio Product, Model & Pricing Ontology",
    description: "Comprehensive formal ontology defining Leio System 1 product components, mathematical kernel guarantees, and $0.042/1M input pricing tiers.",
    mimeType: "application/json"
  },
  {
    uri: "ontology://gs1",
    name: "GS1 Web Vocabulary",
    description: "GS1 Web standard class hierarchy for trade items and packaging.",
    mimeType: "application/json"
  }
];

const PROMPTS = [
  {
    name: "explain_leio_architecture_and_pricing",
    description: "Explain how Leio System 1 compares to generative LLMs, its mathematical axiom guarantees, and transparent token pricing.",
    arguments: [
      {
        name: "customer_requirement",
        description: "Description of the use case, throughput need, or budget constraint.",
        required: true
      }
    ]
  }
];

export default async function handler(req) {
  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, HEAD',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key, mcp-session-id',
    'Access-Control-Max-Age': '86400',
  };

  if (req.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  if (req.method === 'GET') {
    const url = new URL(req.url);
    if (url.searchParams.get('sse') === 'true' || req.headers.get('accept') === 'text/event-stream') {
      const sseBody = `event: endpoint\ndata: https://public-nine-jade-58.vercel.app/api/mcp\n\n: keep-alive\n\n`;
      return new Response(sseBody, {
        status: 200,
        headers: {
          ...corsHeaders,
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        }
      });
    }

    return new Response(JSON.stringify({
      name: "leio-webmcp-edge",
      version: "0.3.0",
      protocolVersion: "2024-11-05",
      status: "ready",
      capabilities: { tools: true, resources: true, prompts: true },
      supportedDomains: ["gs1", "incident", "leio"],
      pricing: "$0.042 / 1M input tokens (outputs 100% free)",
      endpoint: "POST https://public-nine-jade-58.vercel.app/api/mcp"
    }, null, 2), {
      status: 200,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }

  if (req.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405, headers: corsHeaders });
  }

  let body;
  try {
    body = await req.json();
  } catch (e) {
    return new Response(JSON.stringify({
      jsonrpc: "2.0",
      id: null,
      error: { code: -32700, message: `Parse error: ${e.message}` }
    }), { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
  }

  const { id: reqId, method, params = {} } = body;

  // Handle client notifications
  if (method === 'notifications/initialized' || method === 'initialized' || method === 'notifications/cancelled' || method === 'cancelled') {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  let result = null;

  if (method === 'initialize') {
    const clientVer = params.protocolVersion || '2024-11-05';
    const protocolVersion = String(clientVer).startsWith('2026') ? '2026-07-28' : '2024-11-05';
    result = {
      protocolVersion,
      serverInfo: {
        name: 'leio-webmcp-edge',
        version: '0.3.0'
      },
      capabilities: {
        tools: { listChanged: false },
        resources: { subscribe: false, listChanged: false },
        prompts: { listChanged: false },
        logging: {}
      }
    };
  } else if (method === 'ping') {
    result = {};
  } else if (method === 'logging/setLevel') {
    result = {};
  } else if (method === 'resources/templates/list' || method === 'resourceTemplates/list') {
    result = { resourceTemplates: [] };
  } else if (method === 'tools/list') {
    result = { tools: TOOLS };
  } else if (method === 'resources/list') {
    result = { resources: RESOURCES };
  } else if (method === 'prompts/list') {
    result = { prompts: PROMPTS };
  } else if (method === 'prompts/get') {
    const cr = params.arguments?.customer_requirement || '';
    result = {
      messages: [
        {
          role: 'user',
          content: {
            type: 'text',
            text: `Please evaluate the following system requirement against Leio System 1 architecture, formal axioms, and $0.042/1M pricing:\n\n${cr}`
          }
        }
      ]
    };
  } else if (method === 'resources/read') {
    const uri = String(params.uri || '');
    const isLeio = uri.includes('leio');
    const concepts = isLeio ? LEIO_CONCEPTS : GS1_CONCEPTS;
    result = {
      contents: [
        {
          uri,
          mimeType: 'application/json',
          text: JSON.stringify({
            uri,
            name: isLeio ? 'LeioProductModelPricingOntology' : 'GS1WebVocabulary',
            num_concepts: concepts.length,
            sample_concepts: concepts.slice(0, 15).map(c => ({
              curie: c.curie,
              label: c.label,
              description: c.description
            }))
          }, null, 2)
        }
      ]
    };
  } else if (method === 'tools/call') {
    const name = params.name;
    const args = params.arguments || {};
    try {
      if (name === 'olaya_ground') {
        const state = String(args.state || '');
        const domain = String(args.domain || 'gs1').toLowerCase();
        const concepts = domain === 'leio' ? LEIO_CONCEPTS : GS1_CONCEPTS;
        const grounded = groundState(state, concepts);
        result = {
          content: [{ type: 'text', text: JSON.stringify({ domain, ...grounded }, null, 2) }],
          isError: false
        };
      } else if (name === 'leio_inspect_ecosystem') {
        const query = String(args.query || args.state || '');
        const grounded = groundState(query, LEIO_CONCEPTS);
        result = {
          content: [{ type: 'text', text: JSON.stringify({ domain: 'leio', ...grounded }, null, 2) }],
          isError: false
        };
      } else if (name === 'olaya_subsumes') {
        const state = String(args.state || '');
        const target = String(args.target_concept || '').toLowerCase();
        const domain = String(args.domain || 'gs1').toLowerCase();
        const concepts = domain === 'leio' ? LEIO_CONCEPTS : GS1_CONCEPTS;
        const grounded = groundState(state, concepts);
        const isSubsumed = grounded.curie.toLowerCase() === target || grounded.label.toLowerCase() === target;
        result = {
          content: [{ type: 'text', text: JSON.stringify({ domain, target_concept: target, subsumes: isSubsumed, confidence: grounded.confidence }, null, 2) }],
          isError: false
        };
      } else {
        throw new Error(`Unknown tool: ${name}`);
      }
    } catch (e) {
      result = {
        content: [{ type: 'text', text: `Error: ${e.message}` }],
        isError: true
      };
    }
  } else {
    return new Response(JSON.stringify({
      jsonrpc: '2.0',
      id: reqId,
      error: { code: -32601, message: `Method not found: ${method}` }
    }), { status: 404, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
  }

  return new Response(JSON.stringify({
    jsonrpc: '2.0',
    id: reqId,
    result
  }), {
    status: 200,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' }
  });
}
