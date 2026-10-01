/**
 * Model Context Protocol (MCP 2024-11-05) Server for O-Laya in Node.js / TypeScript.
 * Zero external dependencies.
 */

import * as readline from "node:readline";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { OLaya, OntologyDAG } from "./index.js";

// Standard embedded GS1 & Incident ontologies for instant zero-install execution
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

export class OLayaNodeMCPServer {
  private incidentEngine: OLaya;
  private gs1Engine: OLaya;

  constructor() {
    this.incidentEngine = new OLaya(new OntologyDAG(DEFAULT_INCIDENT_CLASSES));
    this.gs1Engine = new OLaya(new OntologyDAG(DEFAULT_GS1_CLASSES));
  }

  public async executeTool(name: string, args: Record<string, any>): Promise<Record<string, any>> {
    if (name === "olaya_ground" || name === "leio_ground") {
      const state = String(args.state ?? "");
      const domain = String(args.domain ?? "gs1").toLowerCase();
      const engine = domain === "incident" ? this.incidentEngine : this.gs1Engine;

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

    if (name === "olaya_subsumes" || name === "leio_subsumes") {
      const state = String(args.state ?? "");
      const targetConcept = String(args.target_concept ?? "");
      const domain = String(args.domain ?? "gs1").toLowerCase();
      const engine = domain === "incident" ? this.incidentEngine : this.gs1Engine;

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

    if (method === "initialize") {
      const requestedVersion = params.protocolVersion || "2026-07-28";
      // Support 2026-07-28 latest spec while keeping backward compatibility with 2024-11-05
      const protocolVersion = requestedVersion.startsWith("2024") ? requestedVersion : "2026-07-28";
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
            tasks: { enabled: true },
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
              description: "Ground raw text or barcode observations into formal ontology classes (GS1 or Incident) with calibrated confidence.",
              inputSchema: {
                type: "object",
                properties: {
                  state: { type: "string", description: "Text or barcode observation to ground" },
                  domain: { type: "string", enum: ["gs1", "incident"], default: "gs1" },
                },
                required: ["state"],
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
                  domain: { type: "string", enum: ["gs1", "incident"], default: "gs1" },
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
                  domain: { type: "string", enum: ["gs1", "incident"], default: "gs1" },
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
                  domain: { type: "string", enum: ["gs1", "incident"], default: "gs1" },
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
  } else {
    configPath = path.join(process.cwd(), ".cursor", "mcp.json");
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
    args: ["-y", "@mycelia/olaya-mcp"],
  };

  const outJson = JSON.stringify(configData, null, 2);
  if (dryRun) {
    return `[Dry-Run] Target: ${configPath}\n${outJson}`;
  }

  fs.mkdirSync(path.dirname(configPath), { recursive: true });
  fs.writeFileSync(configPath, outJson, "utf-8");
  return `Successfully installed O-Laya MCP configuration to: ${configPath}`;
}
