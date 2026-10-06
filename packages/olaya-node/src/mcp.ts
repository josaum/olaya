/**
 * Model Context Protocol (MCP) server for @josaum/leio over stdio. Zero external dependencies.
 *
 * Implements the JSON-RPC 2.0 base lifecycle (initialize, ping, notifications), `tools/*` and
 * `resources/*` of the 2024-11-05 and 2025-03-26 protocol revisions. No prompts, sampling,
 * completions, subscriptions or list-change notifications are offered, and none are advertised.
 *
 * The ontologies are embedded in the package at build time (the JSON files under
 * `src/ontologies/` are bundled into `dist/`), so discovery does not depend on the process
 * working directory and `npx` users get them without extra files.
 */

import * as readline from "node:readline";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { OLaya } from "./engine.js";
import { Concept, OntologyDAG } from "./ontology.js";
import bfoOntology from "./ontologies/bfo_ontology.json";
import doidOntology from "./ontologies/doid_ontology.json";
import schemaorgOntology from "./ontologies/schemaorg_ontology.json";
import skosOntology from "./ontologies/skos_ontology.json";

/** Protocol revisions this server implements, latest first. */
export const SUPPORTED_PROTOCOL_VERSIONS: readonly string[] = ["2025-03-26", "2024-11-05"];
export const LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0];

/**
 * MCP version negotiation: echo the client's version when supported, otherwise answer with the
 * latest version this server supports (the client then decides whether to continue).
 */
export function negotiateProtocolVersion(requested: unknown): string {
  return typeof requested === "string" && SUPPORTED_PROTOCOL_VERSIONS.includes(requested)
    ? requested
    : LATEST_PROTOCOL_VERSION;
}

export const SERVER_INFO = { name: "olaya-node", version: "0.1.0" } as const;

// Small embedded sample ontologies (5 classes each) for instant zero-install execution.
export const DEFAULT_INCIDENT_CLASSES: Concept[] = [
  { id: 0, curie: "INCIDENT:ROOT", label: "Operational Incident", parents: [], disjointWith: [], synonyms: ["incident", "outage"] },
  { id: 1, curie: "INCIDENT:DATA", label: "Data Incident", parents: [0], disjointWith: [2], synonyms: ["database", "storage", "postgres", "sql"] },
  { id: 2, curie: "INCIDENT:BILLING", label: "Billing Incident", parents: [0], disjointWith: [1], synonyms: ["payment", "invoice", "stripe", "credit"] },
  { id: 3, curie: "INCIDENT:DB_DEADLOCK", label: "Database Deadlock", parents: [1], disjointWith: [], synonyms: ["deadlock", "lock wait", "serialization"] },
  { id: 4, curie: "INCIDENT:AUTH_BREACH", label: "Security Breach", parents: [0], disjointWith: [], synonyms: ["breach", "token", "auth", "unauthorized"] },
];

export const DEFAULT_GS1_CLASSES: Concept[] = [
  { id: 0, curie: "gs1:Product", label: "Trade Item Product", parents: [], disjointWith: [], synonyms: ["product", "item", "gtin", "trade item"] },
  { id: 1, curie: "gs1:FoodBeverageTobaccoProduct", label: "Food and Beverage Product", parents: [0], disjointWith: [], synonyms: ["food", "beverage", "dairy", "milk", "edible"] },
  { id: 2, curie: "gs1:PackagingDetails", label: "Packaging Details", parents: [], disjointWith: [], synonyms: ["packaging", "carton", "box", "pallet", "bottle"] },
  { id: 3, curie: "gs1:OrganicClaimDetails", label: "Organic Claim Details", parents: [1], disjointWith: [], synonyms: ["organic", "usda organic", "bio", "certified organic"] },
  { id: 4, curie: "gs1:LogisticsService", label: "Logistics and Supply Chain Service", parents: [], disjointWith: [], synonyms: ["logistics", "shipping", "freight", "warehouse"] },
];

interface CatalogEntry {
  title: string;
  summary: string;
  load: () => OntologyDAG;
}

/** Domains available without any files on disk. Keys are the `domain` values the tools accept. */
export const ONTOLOGY_CATALOG: Record<string, CatalogEntry> = {
  incident: {
    title: "Operational incident sample",
    summary: "A hand-written sample taxonomy for incident triage (data, billing, deadlock, breach).",
    load: () => new OntologyDAG(DEFAULT_INCIDENT_CLASSES),
  },
  gs1: {
    title: "GS1 Web Vocabulary sample",
    summary: "A hand-written sample of GS1 Web Vocabulary classes, not the full vocabulary.",
    load: () => new OntologyDAG(DEFAULT_GS1_CLASSES),
  },
  bfo: {
    title: "Basic Formal Ontology classes",
    summary: "BFO class hierarchy with its disjointness axioms.",
    load: () => OntologyDAG.fromJSON(bfoOntology),
  },
  doid: {
    title: "Human Disease Ontology subset",
    summary: "A subset of DOID classes with labels, definitions and subclass axioms.",
    load: () => OntologyDAG.fromJSON(doidOntology),
  },
  schemaorg: {
    title: "Schema.org types subset",
    summary: "A subset of Schema.org types with descriptions and subtype axioms.",
    load: () => OntologyDAG.fromJSON(schemaorgOntology),
  },
  skos: {
    title: "SKOS core classes",
    summary: "The SKOS core class vocabulary.",
    load: () => OntologyDAG.fromJSON(skosOntology),
  },
};

const DOMAIN_ALIASES: Record<string, string> = {
  supply_chain: "gs1",
  disease: "doid",
  medical: "doid",
  healthcare: "doid",
  clinical: "doid",
  schema: "schemaorg",
  web: "schemaorg",
};

export const DEFAULT_DOMAIN = "gs1";
const DEFAULT_TOP_K = 12;

function textContent(value: unknown) {
  return { content: [{ type: "text", text: JSON.stringify(value, null, 2) }], isError: false };
}

export class OLayaNodeMCPServer {
  private engines = new Map<string, OLaya>();

  /** Domains the server can ground into: the catalog keys. */
  public availableDomains(): string[] {
    return Object.keys(ONTOLOGY_CATALOG);
  }

  /**
   * Resolves a domain name (catalog key or alias) or an explicit path to an ontology JSON file.
   * Unknown domains are an error rather than a silent fallback to another ontology.
   */
  public getEngine(domain: string = DEFAULT_DOMAIN): OLaya {
    const raw = String(domain).trim();
    const key = DOMAIN_ALIASES[raw.toLowerCase()] ?? raw.toLowerCase();

    const cached = this.engines.get(key);
    if (cached) return cached;

    const entry = ONTOLOGY_CATALOG[key];
    if (entry) {
      const engine = new OLaya(entry.load());
      this.engines.set(key, engine);
      return engine;
    }

    // An explicit file path given by the caller (absolute, or relative to their working directory).
    if (raw.endsWith(".json") && fs.existsSync(raw) && fs.statSync(raw).isFile()) {
      const engine = new OLaya(OntologyDAG.fromJSON(JSON.parse(fs.readFileSync(raw, "utf-8"))));
      this.engines.set(key, engine);
      return engine;
    }

    throw new Error(
      `Unknown domain "${domain}". Available: ${this.availableDomains().join(", ")}, or a path to an ontology JSON file.`
    );
  }

  public describeDomain(key: string): { uri: string; name: string; description: string; mimeType: string } {
    const entry = ONTOLOGY_CATALOG[key];
    const size = this.getEngine(key).ontology.size();
    return {
      uri: `ontology://${key}`,
      name: entry.title,
      description: `${entry.summary} ${size} classes, bundled with the package.`,
      mimeType: "application/json",
    };
  }

  public toolDefinitions() {
    const domains = this.availableDomains();
    const domainProperty = {
      type: "string",
      enum: domains,
      default: DEFAULT_DOMAIN,
      description: "Bundled ontology to ground into, or a path to an ontology JSON file.",
    };
    return [
      {
        name: "olaya_ground",
        description:
          "Ground an observation into the single best-matching class of an ontology by lexical overlap " +
          "(labels, synonyms, descriptions), with the subsumption/disjointness projection and a " +
          "normalised-entropy confidence. Returns the probabilities of every candidate it considered.",
        inputSchema: {
          type: "object",
          properties: {
            state: { type: "string", description: "Observation text to ground" },
            domain: domainProperty,
            top_k: { type: "integer", description: `Candidates to route (default ${DEFAULT_TOP_K})`, minimum: 1 },
          },
          required: ["state"],
        },
      },
      {
        name: "olaya_subsumes",
        description:
          "Ground an observation, then report whether the selected class is the target class or one of " +
          "its subclasses (down-set membership).",
        inputSchema: {
          type: "object",
          properties: {
            state: { type: "string", description: "Observation text" },
            target_concept: { type: "string", description: "Target class CURIE or label" },
            domain: domainProperty,
          },
          required: ["state", "target_concept"],
        },
      },
      {
        name: "olaya_choice",
        description:
          "Pick one of the given option labels by lexical overlap with the state, with calibrated " +
          "probabilities and confidence (no ontology involved).",
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
        description:
          "Rate a state on an ordered scale of level labels (lowest first). Each level's probability is " +
          "its lexical overlap with the state, so a state matching no label yields a uniform " +
          "distribution and confidence 0.",
        inputSchema: {
          type: "object",
          properties: {
            state: { type: "string", description: "Context state" },
            criteria: { type: "array", items: { type: "string" }, description: "Ordered level labels, lowest first" },
          },
          required: ["state", "criteria"],
        },
      },
    ];
  }

  public async executeTool(name: string, args: Record<string, any>): Promise<Record<string, any>> {
    if (name === "olaya_ground") {
      const state = String(args.state ?? args.query ?? "");
      const domain = String(args.domain ?? DEFAULT_DOMAIN);
      const topK = typeof args.top_k === "number" ? args.top_k : typeof args.k === "number" ? args.k : DEFAULT_TOP_K;
      const decision = await this.getEngine(domain).systemOne(state, topK);
      return {
        domain,
        curie: decision.selectedConcept.curie,
        label: decision.selectedConcept.label,
        confidence: decision.confidence,
        probabilities: decision.probabilities,
        axioms_verified: decision.axiomsVerified,
      };
    }

    if (name === "olaya_subsumes") {
      const state = String(args.state ?? "");
      const targetConcept = String(args.target_concept ?? "");
      const domain = String(args.domain ?? DEFAULT_DOMAIN);
      const engine = this.getEngine(domain);
      const wanted = targetConcept.toLowerCase();
      const target =
        engine.ontology.getByCurie(targetConcept) ??
        engine.ontology.getAll().find((c) => c.curie.toLowerCase() === wanted || c.label.toLowerCase() === wanted);
      if (!target) {
        throw new Error(`Unknown target concept "${targetConcept}" in domain "${domain}"`);
      }
      const decision = await engine.systemOne(state);
      return {
        domain,
        target_concept: target.curie,
        selected_concept: decision.selectedConcept.curie,
        subsumes: engine.ontology.subsumes(target.id, decision.selectedConcept.id),
        confidence: decision.confidence,
        axioms_verified: decision.axiomsVerified,
      };
    }

    if (name === "olaya_choice") {
      const state = String(args.state ?? "");
      const options = Array.isArray(args.options) ? args.options.map(String) : [];
      if (options.length === 0) throw new Error("olaya_choice needs a non-empty `options` array");
      const res = await this.getEngine("incident").choice(state, options);
      return { selected_option: res.choice, confidence: res.confidence, probabilities: res.probabilities };
    }

    if (name === "olaya_score") {
      const state = String(args.state ?? "");
      const criteria = Array.isArray(args.criteria) ? args.criteria.map(String) : [];
      if (criteria.length === 0) throw new Error("olaya_score needs a non-empty `criteria` array (ordered level labels)");
      const res = await this.getEngine("incident").rate(state, criteria);
      return { ...res, scale: criteria };
    }

    throw new Error(`Unknown tool: ${name}`);
  }

  public async handleMessage(req: Record<string, any>): Promise<Record<string, any> | null> {
    const reqId = req.id;
    const method = req.method;
    const params = req.params ?? {};

    // Client notifications carry no id and get no response.
    if (method === "notifications/initialized" || method === "notifications/cancelled") {
      return null;
    }

    if (method === "ping") {
      return { jsonrpc: "2.0", id: reqId, result: {} };
    }

    if (method === "initialize") {
      return {
        jsonrpc: "2.0",
        id: reqId,
        result: {
          protocolVersion: negotiateProtocolVersion(params.protocolVersion),
          serverInfo: SERVER_INFO,
          capabilities: {
            tools: { listChanged: false },
            resources: { subscribe: false, listChanged: false },
          },
        },
      };
    }

    if (method === "tools/list") {
      return { jsonrpc: "2.0", id: reqId, result: { tools: this.toolDefinitions() } };
    }

    if (method === "tools/call") {
      try {
        const result = await this.executeTool(params.name, params.arguments ?? {});
        return { jsonrpc: "2.0", id: reqId, result: textContent(result) };
      } catch (e: any) {
        return {
          jsonrpc: "2.0",
          id: reqId,
          result: { content: [{ type: "text", text: `Error: ${e.message}` }], isError: true },
        };
      }
    }

    if (method === "resources/list") {
      return {
        jsonrpc: "2.0",
        id: reqId,
        result: { resources: this.availableDomains().map((key) => this.describeDomain(key)) },
      };
    }

    if (method === "resources/templates/list") {
      return { jsonrpc: "2.0", id: reqId, result: { resourceTemplates: [] } };
    }

    if (method === "resources/read") {
      const uri = String(params.uri ?? "");
      const key = uri.startsWith("ontology://") ? uri.slice("ontology://".length) : "";
      if (!ONTOLOGY_CATALOG[key]) {
        return { jsonrpc: "2.0", id: reqId, error: { code: -32002, message: `Resource not found: ${uri}` } };
      }
      const ontology = this.getEngine(key).ontology;
      const byId = (id: number) => ontology.getById(id)?.curie ?? String(id);
      return {
        jsonrpc: "2.0",
        id: reqId,
        result: {
          contents: [
            {
              uri,
              mimeType: "application/json",
              text: JSON.stringify(
                {
                  uri,
                  name: ONTOLOGY_CATALOG[key].title,
                  num_concepts: ontology.size(),
                  concepts: ontology.getAll().map((c) => ({
                    curie: c.curie,
                    label: c.label,
                    description: c.description ?? "",
                    parents: c.parents.map(byId),
                    disjoint_with: c.disjointWith.map(byId),
                  })),
                },
                null,
                2
              ),
            },
          ],
        },
      };
    }

    if (reqId === undefined) return null;

    return { jsonrpc: "2.0", id: reqId, error: { code: -32601, message: `Method not found: ${method}` } };
  }

  public runStdio(): void {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout, terminal: false });

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
          JSON.stringify({ jsonrpc: "2.0", id: null, error: { code: -32700, message: `Parse error: ${err.message}` } }) + "\n"
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
    args: ["@josaum/leio"],
  };

  const outJson = JSON.stringify(configData, null, 2);
  if (dryRun) {
    return `[Dry-Run] Target: ${configPath}\n${outJson}`;
  }

  fs.mkdirSync(path.dirname(configPath), { recursive: true });
  fs.writeFileSync(configPath, outJson, "utf-8");
  return `Successfully installed O-Laya MCP configuration to: ${configPath}`;
}
