/**
 * Leio Server: Compatible with TypeSafe AI's Jev (api.typesafe.ai/v1/systemone)
 * and Model Context Protocol (MCP 2026-07-28 stateless HTTP + stdio).
 *
 * Implements:
 * 1. POST /v1/systemone (Exact Jev Request/Response spec, $0.042/1M input pricing)
 * 2. POST /v1/mcp (MCP 2026-07-28 Stateless HTTP JSON-RPC endpoint)
 * 3. GET /v1/billing/usage (Live API token metering and invoice estimator)
 * 4. GET /v1/health (Sub-millisecond latency heartbeat)
 */

import * as http from "node:http";
import * as crypto from "node:crypto";
import { OLaya } from "./engine.js";
import { OntologyDAG } from "./ontology.js";
import { DEFAULT_GS1_CLASSES, DEFAULT_INCIDENT_CLASSES } from "./mcp.js";

export type OutputFormat = "jev" | "json-rpc" | "json-schema" | "json-ld";

export interface JevSystemOneRequest {
  id?: string | number;
  model?: string;
  state: string | Record<string, any> | any[];
  questions: Record<string, JevQuestion>;
  format?: OutputFormat;
}

export type JevQuestion =
  | { type: "choice"; instructions?: string; options: string[] }
  | { type: "score"; instructions?: string; levels?: number; criteria?: string[] }
  | { type: "noul"; instructions?: string };

export interface JevChoiceResponse {
  type: "choice";
  choice: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface JevScoreResponse {
  type: "score";
  score: number;
  level: number;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface JevNoulResponse {
  type: "noul";
  noul: number;
}

export type JevQuestionResponse = JevChoiceResponse | JevScoreResponse | JevNoulResponse;

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
}

// Global in-memory billing ledger (mock/production ready with API key attribution)
export interface ApiKeyUsage {
  apiKey: string;
  totalRequests: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCostUSD: number;
  lastActive: string;
}

export const USAGE_LEDGER = new Map<string, ApiKeyUsage>();

const RATE_PER_MILLION_INPUT = 0.042; // $0.042 per 1,000,000 tokens (Jev exact rate)

export function estimateTokens(input: any): number {
  const str = typeof input === "string" ? input : JSON.stringify(input);
  // Accurate heuristic: average 1 token per 3.8 characters for structured payloads
  return Math.max(1, Math.ceil(str.length / 3.8));
}

export function recordTokenUsage(apiKey: string, inputTokens: number, outputTokens: number = 0): TokenUsage {
  const costUsd = Number(((inputTokens / 1_000_000) * RATE_PER_MILLION_INPUT).toFixed(8));
  let ledger = USAGE_LEDGER.get(apiKey);
  if (!ledger) {
    ledger = {
      apiKey,
      totalRequests: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      totalCostUSD: 0,
      lastActive: new Date().toISOString(),
    };
    USAGE_LEDGER.set(apiKey, ledger);
  }
  ledger.totalRequests++;
  ledger.totalInputTokens += inputTokens;
  ledger.totalOutputTokens += outputTokens;
  ledger.totalCostUSD += costUsd;
  ledger.lastActive = new Date().toISOString();

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens,
    cost_usd: costUsd,
  };
}

export function getApiKeyUsage(apiKey: string) {
  const ledger = USAGE_LEDGER.get(apiKey) || {
    apiKey,
    totalRequests: 0,
    totalInputTokens: 0,
    totalOutputTokens: 0,
    totalCostUSD: 0,
    lastActive: new Date().toISOString(),
  };

  return {
    ...ledger,
    ratePerMillionInput: "$0.042",
    outputPricing: "$0.00 (Free)",
    invoiceEstimateUSD: `$${ledger.totalCostUSD.toFixed(4)}`,
  };
}

export function computeInputTokens(req: JevSystemOneRequest): number {
  const rawState = typeof req.state === "string" ? req.state : JSON.stringify(req.state);
  return estimateTokens(rawState) + estimateTokens(req.questions || {});
}

export class LeioServer {
  private engine: OLaya;
  private port: number;
  private server: http.Server | null = null;

  constructor(port: number = 8080, ontology?: OntologyDAG) {
    this.port = port;
    this.engine = new OLaya(ontology || new OntologyDAG([...DEFAULT_GS1_CLASSES, ...DEFAULT_INCIDENT_CLASSES]));
  }

  public async start(): Promise<http.Server> {
    return new Promise((resolve) => {
      this.server = this.listen(this.port);
      this.server.once("listening", () => resolve(this.server!));
    });
  }

  public async stop(): Promise<void> {
    return new Promise((resolve) => {
      if (this.server) {
        this.server.close(() => {
          this.server = null;
          resolve();
        });
      } else {
        resolve();
      }
    });
  }

  /**
   * Process TypeSafe AI Jev-compatible /v1/systemone payload
   */
  public async processSystemOne(
    body: JevSystemOneRequest,
    apiKey: string = "anon_key"
  ): Promise<{ result: Record<string, JevQuestionResponse>; usage: TokenUsage }> {
    const rawState = typeof body.state === "string" ? body.state : JSON.stringify(body.state);
    const questions = body.questions || {};
    const result: Record<string, JevQuestionResponse> = {};

    const inputTokens = estimateTokens(rawState) + estimateTokens(questions);

    for (const [qKey, qVal] of Object.entries(questions)) {
      if (qVal.type === "choice") {
        const options = qVal.options || [];
        const choiceRes = await this.engine.choice(rawState, options, qVal.instructions);
        result[qKey] = {
          type: "choice",
          choice: choiceRes.choice,
          confidence: Number(choiceRes.confidence.toFixed(4)),
          probabilities: Object.fromEntries(
            Object.entries(choiceRes.probabilities).map(([k, v]) => [k, Number(v.toFixed(4))])
          ),
        };
      } else if (qVal.type === "score") {
        const levels = qVal.levels || (qVal.criteria ? qVal.criteria.length : 5);
        const criteria = qVal.criteria || Array.from({ length: levels }, (_, i) => `level_${i + 1}`);
        const rawScore = await this.engine.score(rawState, criteria.join(" "));
        const assignedLevel = Math.min(levels, Math.max(1, Math.ceil(rawScore * levels)));

        // Compute softmax spread across levels
        const levelProbs: Record<string, number> = {};
        for (let l = 1; l <= levels; l++) {
          const dist = Math.abs(l - (rawScore * (levels - 1) + 1));
          levelProbs[String(l)] = Number(Math.exp(-dist * 1.5).toFixed(4));
        }
        const sum = Object.values(levelProbs).reduce((a, b) => a + b, 0);
        for (const k of Object.keys(levelProbs)) {
          levelProbs[k] = Number((levelProbs[k] / sum).toFixed(4));
        }

        result[qKey] = {
          type: "score",
          score: Number(rawScore.toFixed(4)),
          level: assignedLevel,
          confidence: Number((levelProbs[String(assignedLevel)] || 0.8).toFixed(4)),
          probabilities: levelProbs,
        };
      } else if (qVal.type === "noul") {
        const instructions = qVal.instructions || "Is this hypothesis true?";
        const noulScore = await this.engine.score(rawState, instructions);
        result[qKey] = {
          type: "noul",
          noul: Number(noulScore.toFixed(4)),
        };
      }
    }

    // Free output tokens per Jev pricing architecture
    const outputTokens = 0;
    const costUsd = Number(((inputTokens / 1_000_000) * RATE_PER_MILLION_INPUT).toFixed(8));

    // Update billing ledger
    let ledger = USAGE_LEDGER.get(apiKey);
    if (!ledger) {
      ledger = {
        apiKey,
        totalRequests: 0,
        totalInputTokens: 0,
        totalOutputTokens: 0,
        totalCostUSD: 0,
        lastActive: new Date().toISOString(),
      };
      USAGE_LEDGER.set(apiKey, ledger);
    }
    ledger.totalRequests++;
    ledger.totalInputTokens += inputTokens;
    ledger.totalCostUSD += costUsd;
    ledger.lastActive = new Date().toISOString();

    const usage: TokenUsage = {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      total_tokens: inputTokens,
      cost_usd: costUsd,
    };

    return { result, usage };
  }

  /**
   * Process MCP 2026-07-28 Stateless HTTP JSON-RPC Request
   */
  public async processMcp(body: Record<string, any>): Promise<Record<string, any>> {
    const { id, method, params } = body;

    if (method === "initialize") {
      return {
        jsonrpc: "2.0",
        id,
        result: {
          protocolVersion: "2026-07-28",
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
        id,
        result: {
          protocolVersion: "2026-07-28",
          tools: [
            {
              name: "leio_systemone",
              description: "Jev-compatible System 1 decision evaluation (choice, score, noul) in <1ms.",
              inputSchema: {
                type: "object",
                properties: {
                  state: { type: "string", description: "State observation or event context" },
                  questions: { type: "object", description: "Jev decision questions" },
                },
                required: ["state", "questions"],
              },
            },
            {
              name: "leio_ground",
              description: "Ground state into formal GS1, Schema.org, or Incident ontology classes with axiom guarantees.",
              inputSchema: {
                type: "object",
                properties: {
                  state: { type: "string", description: "Observation text" },
                  domain: { type: "string", enum: ["gs1", "schemaorg", "bfo", "skos", "incident"], default: "gs1" },
                },
                required: ["state"],
              },
            },
          ],
        },
      };
    }

    if (method === "tools/call") {
      const name = params?.name;
      const args = params?.arguments || {};

      if (name === "leio_systemone") {
        const { result, usage } = await this.processSystemOne({
          state: args.state,
          questions: args.questions,
        });
        return {
          jsonrpc: "2.0",
          id,
          result: {
            content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
            telemetry: {
              protocol: "2026-07-28",
              pricing: "$0.042 / 1M tokens",
              usage,
            },
            isError: false,
          },
        };
      }

      if (name === "leio_ground") {
        const decision = await this.engine.systemOne(String(args.state || ""));
        return {
          jsonrpc: "2.0",
          id,
          result: {
            content: [
              {
                type: "text",
                text: JSON.stringify(
                  {
                    curie: decision.selectedConcept.curie,
                    label: decision.selectedConcept.label,
                    confidence: decision.confidence,
                    axioms_verified: decision.axiomsVerified,
                    probabilities: decision.probabilities.slice(0, 5),
                  },
                  null,
                  2
                ),
              },
            ],
            isError: false,
          },
        };
      }

      return {
        jsonrpc: "2.0",
        id,
        error: { code: -32601, message: `Unknown tool: ${name}` },
      };
    }

    return {
      jsonrpc: "2.0",
      id,
      error: { code: -32601, message: `Method not found: ${method}` },
    };
  }

  /**
   * Start high-performance HTTP server
   */
  public listen(port: number = 8080): http.Server {
    const server = http.createServer(async (req, res) => {
      // CORS headers
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, Mcp-Session-Id");
      res.setHeader("X-Protocol-Version", "2026-07-28");
      res.setHeader("X-Engine-Name", "Leio");

      if (req.method === "OPTIONS") {
        res.writeHead(204);
        res.end();
        return;
      }

      const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
      const authHeader = req.headers["authorization"] || "";
      const apiKey = authHeader.replace(/^Bearer\s+/i, "") || "public_demo_key";

      // 1. Health check
      if (req.method === "GET" && (url.pathname === "/health" || url.pathname === "/v1/health")) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            status: "healthy",
            engine: "Leio System 1",
            model: "leio-systemone-latest",
            mcp_protocol: "2026-07-28",
            pricing: "$0.042 / 1M input tokens (outputs free)",
            uptime_seconds: process.uptime(),
          })
        );
        return;
      }

      // 2. Billing / Usage ledger endpoint
      if (req.method === "GET" && (url.pathname === "/v1/billing/usage" || url.pathname === "/v1/usage")) {
        const usage = USAGE_LEDGER.get(apiKey) || {
          apiKey,
          totalRequests: 0,
          totalInputTokens: 0,
          totalOutputTokens: 0,
          totalCostUSD: 0,
          lastActive: "never",
        };
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            apiKey,
            rate: "$0.042 per 1,000,000 input tokens",
            rate_per_million_input: "$0.042",
            outputs: "FREE",
            output_pricing: "$0.00 (Free)",
            usage,
            invoice_estimate_usd: usage.totalCostUSD.toFixed(6),
          })
        );
        return;
      }

      // 3. Jev-compatible /v1/systemone
      if (req.method === "POST" && (url.pathname === "/v1/systemone" || url.pathname === "/systemone")) {
        let bodyStr = "";
        req.on("data", (chunk) => (bodyStr += chunk));
        req.on("end", async () => {
          try {
            const body = JSON.parse(bodyStr);
            const t0 = performance.now();
            const { result, usage } = await this.processSystemOne(body, apiKey);
            const dtMs = performance.now() - t0;

            res.writeHead(200, {
              "Content-Type": "application/json",
              "X-Inference-Latency-Ms": dtMs.toFixed(2),
              "X-Input-Tokens": String(usage.input_tokens),
              "X-Cost-USD": String(usage.cost_usd),
            });
            res.end(
              JSON.stringify({
                id: `leio_${crypto.randomUUID().replace(/-/g, "")}`,
                model: body.model || "leio-1-enterprise",
                questions: result,
                usage,
                _telemetry: {
                  latency_ms: Number(dtMs.toFixed(2)),
                },
              })
            );
          } catch (err: any) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: `Invalid systemone request: ${err.message}` }));
          }
        });
        return;
      }

      // 4. MCP 2026-07-28 Stateless HTTP endpoint
      if (req.method === "POST" && (url.pathname === "/v1/mcp" || url.pathname === "/mcp")) {
        let bodyStr = "";
        req.on("data", (chunk) => (bodyStr += chunk));
        req.on("end", async () => {
          try {
            const body = JSON.parse(bodyStr);
            const mcpResp = await this.processMcp(body);
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify(mcpResp));
          } catch (err: any) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(
              JSON.stringify({
                jsonrpc: "2.0",
                id: null,
                error: { code: -32700, message: `Parse error: ${err.message}` },
              })
            );
          }
        });
        return;
      }

      // 404 fallback
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          error: "Endpoint not found",
          available_endpoints: [
            "POST /v1/systemone (TypeSafe AI Jev drop-in replacement)",
            "POST /v1/mcp (Model Context Protocol 2026-07-28 stateless)",
            "GET /v1/billing/usage (Live API billing and token metering)",
            "GET /v1/health (Health check)",
          ],
        })
      );
    });

    server.listen(port);
    return server;
  }
}

export const defaultLeioServer = new LeioServer();

export async function processSystemOne(req: JevSystemOneRequest, apiKey: string = "anon_key") {
  const res = await defaultLeioServer.processSystemOne(req, apiKey);
  return {
    id: `leio_${crypto.randomUUID().replace(/-/g, "")}`,
    model: req.model || "leio-1-enterprise",
    questions: res.result,
    usage: res.usage,
  };
}
