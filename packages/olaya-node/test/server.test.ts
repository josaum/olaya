import { describe, expect, it } from "bun:test";
import {
  processSystemOne,
  computeInputTokens,
  USAGE_LEDGER,
  recordTokenUsage,
  getApiKeyUsage,
  LeioServer,
  JevSystemOneRequest,
} from "../src/index.js";

describe("Leio System 1 (Jev Compatible API & Billing)", () => {
  it("calculates input token counts correctly based on state and questions", () => {
    const req: JevSystemOneRequest = {
      state: "Package barcode scanned: 0012345678905. Fresh organic milk carton.",
      questions: {
        lane: {
          type: "choice",
          options: ["CHUTE_COLD_01", "CHUTE_DRY_02", "REJECT_BIN"],
        },
      },
    };
    const tokens = computeInputTokens(req);
    expect(tokens).toBeGreaterThan(10);
  });

  it("processes choice question conforming exactly to Jev schema", async () => {
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
    expect(res.id).toBeDefined();
    expect(res.model).toBe("leio-1-enterprise");
    expect(res.usage).toBeDefined();
    expect(res.usage.input_tokens).toBeGreaterThan(0);
    expect(res.usage.output_tokens).toBe(0); // Free output tokens like Jev
    expect(res.usage.cost_usd).toBeGreaterThan(0);

    const action = res.questions.action;
    expect(action.type).toBe("choice");
    if (action.type === "choice") {
      expect(["ESCALATE_P1_REPAIR", "MONITOR_HOURLY", "IGNORE"]).toContain(action.choice);
      expect(action.confidence).toBeGreaterThan(0.5);
      expect(action.probabilities[action.choice]).toBeGreaterThan(0);
    }
  });

  it("processes score question with levels 1..5", async () => {
    const req: JevSystemOneRequest = {
      state: "System degradation detected on checkout cluster",
      questions: {
        severity: {
          type: "score",
          levels: 5,
        },
      },
    };

    const res = await processSystemOne(req);
    const severity = res.questions.severity;
    expect(severity.type).toBe("score");
    if (severity.type === "score") {
      expect(severity.level).toBeGreaterThanOrEqual(1);
      expect(severity.level).toBeLessThanOrEqual(5);
      expect(severity.score).toBeGreaterThanOrEqual(0);
      expect(severity.score).toBeLessThanOrEqual(1);
    }
  });

  it("processes noul question with calibrated probability", async () => {
    const req: JevSystemOneRequest = {
      state: "Cold chain verification: temperature holds at 2.4C",
      questions: {
        is_compliant: {
          type: "noul",
        },
      },
    };

    const res = await processSystemOne(req);
    const isCompliant = res.questions.is_compliant;
    expect(isCompliant.type).toBe("noul");
    if (isCompliant.type === "noul") {
      expect(isCompliant.noul).toBeGreaterThanOrEqual(0);
      expect(isCompliant.noul).toBeLessThanOrEqual(1);
    }
  });

  it("meters billing usage per API key at $0.042 / 1M input tokens", () => {
    const testKey = "test_key_" + Date.now();
    recordTokenUsage(testKey, 1_000_000, 0);

    const usage = getApiKeyUsage(testKey);
    expect(usage.totalRequests).toBe(1);
    expect(usage.totalInputTokens).toBe(1_000_000);
    expect(usage.totalOutputTokens).toBe(0);
    expect(usage.totalCostUSD).toBeCloseTo(0.042, 4);
    expect(usage.invoiceEstimateUSD).toBe("$0.0420");
  });

  it("starts and responds to HTTP requests over localhost", async () => {
    const port = 9182;
    const server = new LeioServer(port);
    await server.start();

    try {
      // 1. Health check
      const healthRes = await fetch(`http://127.0.0.1:${port}/v1/health`);
      expect(healthRes.status).toBe(200);
      const healthData = await healthRes.json();
      expect(healthData.status).toBe("healthy");
      expect(healthData.engine).toBe("Leio System 1");

      // 2. Billing check
      const billRes = await fetch(`http://127.0.0.1:${port}/v1/billing/usage`, {
        headers: { Authorization: "Bearer key_demo_123" },
      });
      expect(billRes.status).toBe(200);
      const billData = await billRes.json();
      expect(billData.rate_per_million_input).toBe("$0.042");
      expect(billData.output_pricing).toBe("$0.00 (Free)");

      // 3. System 1 Decision check
      const postRes = await fetch(`http://127.0.0.1:${port}/v1/systemone`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer key_demo_123",
        },
        body: JSON.stringify({
          state: "Frozen seafood packaging pallet",
          questions: {
            storage: {
              type: "choice",
              options: ["DEEP_FREEZE", "AMBIENT"],
            },
          },
        }),
      });
      expect(postRes.status).toBe(200);
      const postData = await postRes.json();
      expect(postData.questions.storage.choice).toBe("DEEP_FREEZE");
      expect(postData.usage.cost_usd).toBeGreaterThan(0);
    } finally {
      await server.stop();
    }
  });
});
