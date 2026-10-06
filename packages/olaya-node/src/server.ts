/**
 * HTTP server for olaya.
 *
 * 1. POST /v1/systemone — Jev-style request shape (`state` + named `questions`), answered by the
 *    package's lexical primitives: `choice`, `score` (an ordered scale of `criteria` labels, via
 *    `OLaya.rate`) and `noul` (lexical proposition overlap via `OLaya.score`).
 * 2. POST /v1/mcp — MCP JSON-RPC over plain HTTP POST: one request, one JSON response. This is a
 *    subset of the 2025-03-26 Streamable HTTP transport (no SSE streaming, no sessions, no GET).
 * 3. GET /v1/health — liveness.
 *
 * There is no authentication, metering or billing: requests are not attributed to anyone.
 */

import * as http from "node:http";
import * as crypto from "node:crypto";
import { OLaya } from "./engine.js";
import { OntologyDAG } from "./ontology.js";
import {
  DEFAULT_DOMAIN,
  DEFAULT_INCIDENT_CLASSES,
  OLayaNodeMCPServer,
  SERVER_INFO,
  negotiateProtocolVersion,
} from "./mcp.js";

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
  | { type: "score"; instructions?: string; criteria: string[] }
  | { type: "noul"; instructions: string };

export interface JevChoiceResponse {
  type: "choice";
  choice: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface JevScoreResponse {
  type: "score";
  /** Expected position on the scale in [0, 1]. */
  score: number;
  /** 1-based index of the selected criterion. */
  level: number;
  confidence: number;
  /** Probability per criterion label. */
  probabilities: Record<string, number>;
}

export interface JevNoulResponse {
  type: "noul";
  noul: number;
}

export type JevQuestionResponse = JevChoiceResponse | JevScoreResponse | JevNoulResponse;

export interface SystemOneResponse {
  id: string;
  model: string;
  questions: Record<string, JevQuestionResponse>;
}

export const DEFAULT_MODEL_NAME = "olaya-lexical";

const round4 = (x: number) => Number(x.toFixed(4));
const roundMap = (m: Record<string, number>) => Object.fromEntries(Object.entries(m).map(([k, v]) => [k, round4(v)]));

export class OlayaHttpServer {
  /** Engine for the flat primitives (`choice`, `rate`, `score`); the ontology is not used by them. */
  private engine: OLaya;
  private registry = new OLayaNodeMCPServer();
  private port: number;
  private server: http.Server | null = null;

  constructor(port: number = 8080, ontology?: OntologyDAG) {
    this.port = port;
    this.engine = new OLaya(ontology || new OntologyDAG(DEFAULT_INCIDENT_CLASSES));
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
   * Answer a `/v1/systemone` payload. Throws on a malformed question so the caller can return 400.
   */
  public async processSystemOne(body: JevSystemOneRequest): Promise<Record<string, JevQuestionResponse>> {
    if (body === null || typeof body !== "object" || body.state === undefined) {
      throw new Error("`state` is required");
    }
    const rawState = typeof body.state === "string" ? body.state : JSON.stringify(body.state);
    const questions = body.questions || {};
    const result: Record<string, JevQuestionResponse> = {};

    for (const [qKey, qVal] of Object.entries(questions)) {
      if (qVal.type === "choice") {
        const options = Array.isArray(qVal.options) ? qVal.options.map(String) : [];
        if (options.length === 0) throw new Error(`question "${qKey}": choice needs a non-empty \`options\` array`);
        const choiceRes = await this.engine.choice(rawState, options, qVal.instructions);
        result[qKey] = {
          type: "choice",
          choice: choiceRes.choice,
          confidence: round4(choiceRes.confidence),
          probabilities: roundMap(choiceRes.probabilities),
        };
      } else if (qVal.type === "score") {
        const criteria = Array.isArray(qVal.criteria) ? qVal.criteria.map(String) : [];
        if (criteria.length === 0) {
          throw new Error(
            `question "${qKey}": score needs \`criteria\` (one label per level, lowest first); ` +
              "this lexical engine has no evidence to rank bare level numbers"
          );
        }
        const rating = await this.engine.rate(rawState, criteria, qVal.instructions);
        result[qKey] = {
          type: "score",
          score: round4(rating.score),
          level: rating.level,
          confidence: round4(rating.confidence),
          probabilities: roundMap(rating.probabilities),
        };
      } else if (qVal.type === "noul") {
        if (typeof qVal.instructions !== "string" || qVal.instructions.trim() === "") {
          throw new Error(`question "${qKey}": noul needs the proposition in \`instructions\``);
        }
        result[qKey] = { type: "noul", noul: round4(await this.engine.score(rawState, qVal.instructions)) };
      } else {
        throw new Error(`question "${qKey}": unknown type "${(qVal as any)?.type}"`);
      }
    }

    return result;
  }

  public mcpToolDefinitions() {
    return [
      {
        name: "olaya_systemone",
        description:
          "Answer Jev-style questions about a state with the package's lexical primitives: " +
          "choice (options), score (ordered criteria labels) and noul (a proposition in instructions).",
        inputSchema: {
          type: "object",
          properties: {
            state: { type: "string", description: "State observation or event context" },
            questions: { type: "object", description: "Named questions: {type: choice|score|noul, ...}" },
          },
          required: ["state", "questions"],
        },
      },
      {
        name: "olaya_ground",
        description:
          "Ground an observation into the best-matching class of a bundled ontology by lexical overlap, " +
          "with the subsumption/disjointness projection and a normalised-entropy confidence.",
        inputSchema: {
          type: "object",
          properties: {
            state: { type: "string", description: "Observation text" },
            domain: { type: "string", enum: this.registry.availableDomains(), default: DEFAULT_DOMAIN },
          },
          required: ["state"],
        },
      },
    ];
  }

  /**
   * Handle one MCP JSON-RPC message. Returns null for notifications (no response body).
   */
  public async processMcp(body: Record<string, any>): Promise<Record<string, any> | null> {
    const { id, method, params } = body ?? {};

    if (method === "notifications/initialized" || method === "notifications/cancelled") {
      return null;
    }

    if (method === "ping") {
      return { jsonrpc: "2.0", id, result: {} };
    }

    if (method === "initialize") {
      return {
        jsonrpc: "2.0",
        id,
        result: {
          protocolVersion: negotiateProtocolVersion(params?.protocolVersion),
          serverInfo: SERVER_INFO,
          capabilities: { tools: { listChanged: false } },
        },
      };
    }

    if (method === "tools/list") {
      return { jsonrpc: "2.0", id, result: { tools: this.mcpToolDefinitions() } };
    }

    if (method === "tools/call") {
      const name = params?.name;
      const args = params?.arguments || {};
      try {
        let payload: unknown;
        if (name === "olaya_systemone") {
          payload = await this.processSystemOne({ state: args.state, questions: args.questions });
        } else if (name === "olaya_ground") {
          payload = await this.registry.executeTool("olaya_ground", args);
        } else {
          return { jsonrpc: "2.0", id, error: { code: -32602, message: `Unknown tool: ${name}` } };
        }
        return {
          jsonrpc: "2.0",
          id,
          result: { content: [{ type: "text", text: JSON.stringify(payload, null, 2) }], isError: false },
        };
      } catch (err: any) {
        return {
          jsonrpc: "2.0",
          id,
          result: { content: [{ type: "text", text: `Error: ${err.message}` }], isError: true },
        };
      }
    }

    if (id === undefined) return null;
    return { jsonrpc: "2.0", id, error: { code: -32601, message: `Method not found: ${method}` } };
  }

  public listen(port: number = 8080): http.Server {
    const readBody = (req: http.IncomingMessage) =>
      new Promise<string>((resolve, reject) => {
        let bodyStr = "";
        req.on("data", (chunk) => (bodyStr += chunk));
        req.on("end", () => resolve(bodyStr));
        req.on("error", reject);
      });
    const json = (res: http.ServerResponse, status: number, payload: unknown, headers: Record<string, string> = {}) => {
      res.writeHead(status, { "Content-Type": "application/json", ...headers });
      res.end(JSON.stringify(payload));
    };

    const server = http.createServer(async (req, res) => {
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

      if (req.method === "OPTIONS") {
        res.writeHead(204);
        res.end();
        return;
      }

      const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);

      if (req.method === "GET" && (url.pathname === "/health" || url.pathname === "/v1/health")) {
        json(res, 200, {
          status: "ok",
          engine: SERVER_INFO.name,
          version: SERVER_INFO.version,
          model: DEFAULT_MODEL_NAME,
          uptime_seconds: process.uptime(),
        });
        return;
      }

      if (req.method === "POST" && (url.pathname === "/v1/systemone" || url.pathname === "/systemone")) {
        try {
          const body = JSON.parse(await readBody(req));
          const t0 = performance.now();
          const questions = await this.processSystemOne(body);
          const dtMs = performance.now() - t0;
          json(
            res,
            200,
            {
              id: `olaya_${crypto.randomUUID().replace(/-/g, "")}`,
              model: body.model || DEFAULT_MODEL_NAME,
              questions,
              _telemetry: { latency_ms: Number(dtMs.toFixed(2)) },
            },
            { "X-Inference-Latency-Ms": dtMs.toFixed(2) }
          );
        } catch (err: any) {
          json(res, 400, { error: `Invalid systemone request: ${err.message}` });
        }
        return;
      }

      if (req.method === "POST" && (url.pathname === "/v1/mcp" || url.pathname === "/mcp")) {
        try {
          const body = JSON.parse(await readBody(req));
          const mcpResp = await this.processMcp(body);
          if (mcpResp === null) {
            res.writeHead(202);
            res.end();
            return;
          }
          json(res, 200, mcpResp);
        } catch (err: any) {
          json(res, 400, { jsonrpc: "2.0", id: null, error: { code: -32700, message: `Parse error: ${err.message}` } });
        }
        return;
      }

      json(res, 404, {
        error: "Endpoint not found",
        available_endpoints: [
          "POST /v1/systemone (choice / score / noul questions about a state)",
          "POST /v1/mcp (MCP JSON-RPC over HTTP POST)",
          "GET /v1/health",
        ],
      });
    });

    server.listen(port);
    return server;
  }
}

/** Previous name of {@link OlayaHttpServer}, kept for callers of the 0.1.x API. */
export { OlayaHttpServer as LeioServer };

let defaultServer: OlayaHttpServer | null = null;

/** Answer a `/v1/systemone` payload in-process, without opening a socket. */
export async function processSystemOne(req: JevSystemOneRequest): Promise<SystemOneResponse> {
  defaultServer ??= new OlayaHttpServer();
  return {
    id: `olaya_${crypto.randomUUID().replace(/-/g, "")}`,
    model: req.model || DEFAULT_MODEL_NAME,
    questions: await defaultServer.processSystemOne(req),
  };
}
