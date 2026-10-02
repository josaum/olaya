import { describe, expect, it } from "bun:test";
import { OLayaNodeMCPServer, installNodeMCPConfig } from "../src/index.js";

describe("OLayaNodeMCPServer (MCP Protocol)", () => {
  const server = new OLayaNodeMCPServer();

  it("handles initialize request with MCP capabilities", async () => {
    const res = await server.handleMessage({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { protocolVersion: "2026-07-28" },
    });

    expect(res).not.toBeNull();
    expect(res?.result.protocolVersion).toBe("2026-07-28");
    expect(res?.result.serverInfo.name).toBe("leio-mcp-engine");
    expect(res?.result.capabilities.tools).toBeDefined();
    expect(res?.result.capabilities.stateless).toBe(true);
  });

  it("handles legacy 2024-11-05 client handshake", async () => {
    const res = await server.handleMessage({
      jsonrpc: "2.0",
      id: 11,
      method: "initialize",
      params: { protocolVersion: "2024-11-05" },
    });

    expect(res).not.toBeNull();
    expect(res?.result.protocolVersion).toBe("2024-11-05");
  });

  it("handles ping and client notifications compliant with MCP spec", async () => {
    // Ping
    const pingRes = await server.handleMessage({
      jsonrpc: "2.0",
      id: 99,
      method: "ping",
    });
    expect(pingRes).not.toBeNull();
    expect(pingRes?.id).toBe(99);
    expect(pingRes?.result).toEqual({});

    // Notifications return null (no response needed)
    const initNotice = await server.handleMessage({
      jsonrpc: "2.0",
      method: "notifications/initialized",
    });
    expect(initNotice).toBeNull();

    const cancelNotice = await server.handleMessage({
      jsonrpc: "2.0",
      method: "notifications/cancelled",
      params: { requestId: 1 },
    });
    expect(cancelNotice).toBeNull();
  });

  it("lists all decision tools via tools/list including leio ecosystem tool", async () => {
    const res = await server.handleMessage({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
    });

    expect(res).not.toBeNull();
    const tools = res?.result.tools;
    expect(Array.isArray(tools)).toBe(true);
    const names = tools.map((t: any) => t.name);
    expect(names).toContain("olaya_ground");
    expect(names).toContain("olaya_subsumes");
    expect(names).toContain("olaya_choice");
    expect(names).toContain("olaya_score");
    expect(names).toContain("leio_inspect_ecosystem");
  });

  it("executes olaya_ground on GS1 trade items", async () => {
    const res = await server.handleMessage({
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: {
        name: "olaya_ground",
        arguments: {
          state: "Certified organic whole milk gallon carton packaging",
          domain: "gs1",
        },
      },
    });

    expect(res).not.toBeNull();
    expect(res?.result.isError).toBe(false);
    const content = JSON.parse(res?.result.content[0].text);
    expect(content.curie).toBe("gs1:OrganicClaimDetails");
    expect(content.axioms_verified).toBe(true);
  });

  it("executes leio_inspect_ecosystem to ground product, model axioms, and pricing", async () => {
    // Pricing inquiry
    const priceRes = await server.handleMessage({
      jsonrpc: "2.0",
      id: 301,
      method: "tools/call",
      params: {
        name: "leio_inspect_ecosystem",
        arguments: {
          query: "Input-only token metering at $0.042 per 1M tokens with unmetered answers",
        },
      },
    });
    expect(priceRes?.result.isError).toBe(false);
    const priceContent = JSON.parse(priceRes?.result.content[0].text);
    expect(priceContent.curie).toBe("leio:InputOnlyMetering");
    expect(priceContent.domain).toBe("leio");

    // Model axiom inquiry
    const axiomRes = await server.handleMessage({
      jsonrpc: "2.0",
      id: 302,
      method: "tools/call",
      params: {
        name: "leio_inspect_ecosystem",
        arguments: {
          query: "Subsumption monotonicity guarantee enforcing P(child) <= P(parent) everywhere",
        },
      },
    });
    expect(axiomRes?.result.isError).toBe(false);
    const axiomContent = JSON.parse(axiomRes?.result.content[0].text);
    expect(axiomContent.curie).toBe("leio:SubsumptionMonotonicity");

    // Product runtime inquiry
    const runtimeRes = await server.handleMessage({
      jsonrpc: "2.0",
      id: 303,
      method: "tools/call",
      params: {
        name: "leio_inspect_ecosystem",
        arguments: {
          query: "Standalone 189 KB WebAssembly in-browser client module running in under 1 microsecond",
        },
      },
    });
    expect(runtimeRes?.result.isError).toBe(false);
    const runtimeContent = JSON.parse(runtimeRes?.result.content[0].text);
    expect(runtimeContent.curie).toBe("leio:WasmRuntime");
  });

  it("supports resources/list and resources/read including ontology://leio", async () => {
    const listRes = await server.handleMessage({
      jsonrpc: "2.0",
      id: 401,
      method: "resources/list",
    });
    const uris = listRes?.result.resources.map((r: any) => r.uri);
    expect(uris).toContain("ontology://leio");

    const readRes = await server.handleMessage({
      jsonrpc: "2.0",
      id: 402,
      method: "resources/read",
      params: { uri: "ontology://leio" },
    });
    const parsed = JSON.parse(readRes?.result.contents[0].text);
    expect(parsed.name).toBe("LeioProductModelPricing");
    expect(parsed.num_concepts).toBeGreaterThanOrEqual(30);
  });

  it("supports prompts/list and prompts/get", async () => {
    const pList = await server.handleMessage({
      jsonrpc: "2.0",
      id: 501,
      method: "prompts/list",
    });
    expect(pList?.result.prompts.length).toBeGreaterThan(0);

    const pGet = await server.handleMessage({
      jsonrpc: "2.0",
      id: 502,
      method: "prompts/get",
      params: {
        name: "explain_leio_architecture_and_pricing",
        arguments: { customer_requirement: "10M conveyor package scans per month on edge PLC" },
      },
    });
    expect(pGet?.result.messages[0].content.text).toContain("10M conveyor package scans");
  });

  it("executes olaya_subsumes down-set membership test", async () => {
    const res = await server.handleMessage({
      jsonrpc: "2.0",
      id: 4,
      method: "tools/call",
      params: {
        name: "olaya_subsumes",
        arguments: {
          state: "Heavy duty corrugated packaging box and carton container",
          target_concept: "gs1:PackagingDetails",
          domain: "gs1",
        },
      },
    });

    expect(res).not.toBeNull();
    const content = JSON.parse(res?.result.content[0].text);
    expect(content.subsumes).toBe(true);
  });

  it("supports install configuration dry-run for Claude Desktop", () => {
    const msg = installNodeMCPConfig("claude", true);
    expect(msg).toContain("[Dry-Run]");
    expect(msg).toContain("olaya");
    expect(msg).toContain("@mycelia/olaya-mcp");
  });
});
