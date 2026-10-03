import { describe, expect, it } from "bun:test";
import { processSystemOne, OlayaHttpServer, LeioServer, JevSystemOneRequest } from "../src/index.js";

describe("OlayaHttpServer (/v1/systemone and /v1/mcp)", () => {
  it("keeps the old class name as an alias", () => {
    expect(LeioServer).toBe(OlayaHttpServer);
  });

  it("answers choice questions from lexical evidence, without usage or cost fields", async () => {
    const req: JevSystemOneRequest = {
      state: "Critical sensor alert: Refrigeration compressor failed in cold storage unit #3. Escalate emergency repair protocol immediately.",
      questions: {
        action: {
          type: "choice",
          instructions: "Select emergency triage protocol",
          options: ["ESCALATE_P1_REPAIR", "MONITOR_HOURLY", "IGNORE"],
        },
      },
    };

    const res = await processSystemOne(req);
    expect(res.id).toStartWith("olaya_");
    expect(res.model).toBe("olaya-lexical");
    expect((res as any).usage).toBeUndefined();

    const action = res.questions.action;
    expect(action.type).toBe("choice");
    if (action.type === "choice") {
      expect(action.choice).toBe("ESCALATE_P1_REPAIR");
      expect(action.confidence).toBeGreaterThan(0.5);
      expect(action.probabilities[action.choice]).toBeGreaterThan(0);
    }
  });

  it("scores on the given criteria scale with the engine's own probabilities", async () => {
    const res = await processSystemOne({
      state: "Critical degradation detected on checkout cluster, customers cannot pay",
      questions: { severity: { type: "score", criteria: ["low", "medium", "high", "critical"] } },
    });
    const severity = res.questions.severity;
    expect(severity.type).toBe("score");
    if (severity.type === "score") {
      expect(severity.level).toBe(4);
      expect(severity.score).toBeGreaterThan(0.5);
      expect(severity.score).toBeLessThanOrEqual(1);
      expect(severity.confidence).toBeGreaterThan(0);
      const total = Object.values(severity.probabilities).reduce((a, b) => a + b, 0);
      expect(total).toBeCloseTo(1, 3);
      expect(Object.keys(severity.probabilities)).toEqual(["low", "medium", "high", "critical"]);
    }
  });

  it("reports no confidence when the state matches none of the criteria", async () => {
    const res = await processSystemOne({
      state: "A sentence about something else entirely",
      questions: { severity: { type: "score", criteria: ["low", "medium", "high"] } },
    });
    const severity = res.questions.severity;
    if (severity.type === "score") {
      expect(severity.confidence).toBeCloseTo(0, 6);
      expect(severity.probabilities.low).toBeCloseTo(1 / 3, 3);
      expect(severity.score).toBeCloseTo(0.5, 3);
    }
  });

  it("rejects score questions without criteria and noul questions without a proposition", async () => {
    await expect(
      processSystemOne({ state: "x", questions: { s: { type: "score", levels: 5 } as any } })
    ).rejects.toThrow("score needs `criteria`");
    await expect(processSystemOne({ state: "x", questions: { n: { type: "noul" } as any } })).rejects.toThrow(
      "noul needs the proposition"
    );
  });

  it("answers noul with the engine's lexical proposition score", async () => {
    const res = await processSystemOne({
      state: "Cold chain verification: temperature holds at 2.4C",
      questions: { is_cold_chain: { type: "noul", instructions: "cold chain temperature verification" } },
    });
    const q = res.questions.is_cold_chain;
    expect(q.type).toBe("noul");
    if (q.type === "noul") {
      expect(q.noul).toBeGreaterThan(0.5);
      expect(q.noul).toBeLessThanOrEqual(1);
    }
  });

  it("serves MCP JSON-RPC over HTTP POST with real protocol versions", async () => {
    const server = new OlayaHttpServer();
    const init = await server.processMcp({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { protocolVersion: "2026-07-28" },
    });
    expect(init?.result.protocolVersion).toBe("2025-03-26");
    expect(init?.result.capabilities).toEqual({ tools: { listChanged: false } });

    const tools = await server.processMcp({ jsonrpc: "2.0", id: 2, method: "tools/list" });
    expect(tools?.result.tools.map((t: any) => t.name)).toEqual(["olaya_systemone", "olaya_ground"]);
    expect(tools?.result.protocolVersion).toBeUndefined();

    const call = await server.processMcp({
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name: "olaya_ground", arguments: { state: "Fresh dairy milk beverage, one litre", domain: "gs1" } },
    });
    expect(call?.result.isError).toBe(false);
    expect(call?.result.telemetry).toBeUndefined();
    expect(JSON.parse(call?.result.content[0].text).curie).toBe("gs1:FoodBeverageTobaccoProduct");

    expect(await server.processMcp({ jsonrpc: "2.0", method: "notifications/initialized" })).toBeNull();
  });

  it("starts and responds to HTTP requests over localhost", async () => {
    const port = 9182;
    const server = new OlayaHttpServer(port);
    await server.start();

    try {
      const healthRes = await fetch(`http://127.0.0.1:${port}/v1/health`);
      expect(healthRes.status).toBe(200);
      const healthData = await healthRes.json();
      expect(healthData.status).toBe("ok");
      expect(healthData.engine).toBe("olaya-node");
      expect(healthData.pricing).toBeUndefined();

      // The billing endpoint is gone.
      const billRes = await fetch(`http://127.0.0.1:${port}/v1/billing/usage`, {
        headers: { Authorization: "Bearer key_demo_123" },
      });
      expect(billRes.status).toBe(404);

      const postRes = await fetch(`http://127.0.0.1:${port}/v1/systemone`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          state: "Frozen seafood packaging pallet",
          questions: { storage: { type: "choice", options: ["DEEP_FREEZE", "AMBIENT"] } },
        }),
      });
      expect(postRes.status).toBe(200);
      expect(postRes.headers.get("x-cost-usd")).toBeNull();
      const postData = await postRes.json();
      expect(postData.questions.storage.choice).toBe("DEEP_FREEZE");
      expect(postData.usage).toBeUndefined();

      const badRes = await fetch(`http://127.0.0.1:${port}/v1/systemone`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: "x", questions: { s: { type: "score", levels: 5 } } }),
      });
      expect(badRes.status).toBe(400);

      const notice = await fetch(`http://127.0.0.1:${port}/v1/mcp`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
      });
      expect(notice.status).toBe(202);
    } finally {
      await server.stop();
    }
  });
});
