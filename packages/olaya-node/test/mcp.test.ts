import { describe, expect, it } from "bun:test";
import {
  LATEST_PROTOCOL_VERSION,
  OLayaNodeMCPServer,
  ONTOLOGY_CATALOG,
  SUPPORTED_PROTOCOL_VERSIONS,
  installNodeMCPConfig,
  negotiateProtocolVersion,
} from "../src/index.js";

describe("OLayaNodeMCPServer (MCP Protocol)", () => {
  const server = new OLayaNodeMCPServer();

  it("negotiates only protocol revisions the code implements", async () => {
    expect(SUPPORTED_PROTOCOL_VERSIONS).toEqual(["2025-03-26", "2024-11-05"]);
    expect(negotiateProtocolVersion("2024-11-05")).toBe("2024-11-05");
    expect(negotiateProtocolVersion("2025-03-26")).toBe("2025-03-26");
    // Unknown or newer revisions get the latest one the server supports.
    expect(negotiateProtocolVersion("2026-07-28")).toBe(LATEST_PROTOCOL_VERSION);
    expect(negotiateProtocolVersion(undefined)).toBe(LATEST_PROTOCOL_VERSION);

    const res = await server.handleMessage({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "test", version: "0" } },
    });
    expect(res?.result.protocolVersion).toBe("2024-11-05");
    expect(res?.result.serverInfo.name).toBe("olaya-node");
    expect(Object.keys(res?.result.capabilities).sort()).toEqual(["resources", "tools"]);
    expect(res?.result.capabilities.tasks).toBeUndefined();
    expect(res?.result.capabilities.stateless).toBeUndefined();
  });

  it("handles ping and client notifications", async () => {
    const pingRes = await server.handleMessage({ jsonrpc: "2.0", id: 99, method: "ping" });
    expect(pingRes?.id).toBe(99);
    expect(pingRes?.result).toEqual({});

    expect(await server.handleMessage({ jsonrpc: "2.0", method: "notifications/initialized" })).toBeNull();
    expect(
      await server.handleMessage({ jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId: 1 } })
    ).toBeNull();
  });

  it("lists each decision tool exactly once, with the bundled domains", async () => {
    const res = await server.handleMessage({ jsonrpc: "2.0", id: 2, method: "tools/list" });
    const tools = res?.result.tools;
    const names = tools.map((t: any) => t.name);
    expect(names).toEqual(["olaya_ground", "olaya_subsumes", "olaya_choice", "olaya_score"]);
    expect(new Set(names).size).toBe(names.length);
    expect(names.some((n: string) => n.startsWith("leio_"))).toBe(false);
    const ground = tools.find((t: any) => t.name === "olaya_ground");
    expect(ground.inputSchema.properties.domain.enum).toEqual(Object.keys(ONTOLOGY_CATALOG));
  });

  it("executes olaya_ground on GS1 trade items", async () => {
    const res = await server.handleMessage({
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: {
        name: "olaya_ground",
        arguments: { state: "Certified organic whole milk gallon carton packaging", domain: "gs1" },
      },
    });
    expect(res?.result.isError).toBe(false);
    const content = JSON.parse(res?.result.content[0].text);
    expect(content.curie).toBe("gs1:OrganicClaimDetails");
    expect(content.axioms_verified).toBe(true);
    expect(content.confidence).toBeGreaterThan(0);
  });

  it("grounds into the bundled JSON ontologies regardless of the working directory", async () => {
    const previousCwd = process.cwd();
    process.chdir("/");
    try {
      const fresh = new OLayaNodeMCPServer();
      const res = await fresh.handleMessage({
        jsonrpc: "2.0",
        id: 31,
        method: "tools/call",
        params: {
          name: "olaya_ground",
          arguments: { state: "Patient diagnosed with juvenile myelomonocytic leukemia", domain: "doid" },
        },
      });
      expect(res?.result.isError).toBe(false);
      const content = JSON.parse(res?.result.content[0].text);
      expect(content.curie).toStartWith("DOID:");
      expect(content.label.toLowerCase()).toContain("leukemia");
      expect(content.axioms_verified).toBe(true);
      expect(fresh.getEngine("doid").ontology.size()).toBe(730);
    } finally {
      process.chdir(previousCwd);
    }
  });

  it("rejects unknown domains instead of silently answering from another ontology", async () => {
    const res = await server.handleMessage({
      jsonrpc: "2.0",
      id: 32,
      method: "tools/call",
      params: { name: "olaya_ground", arguments: { state: "anything", domain: "leio" } },
    });
    expect(res?.result.isError).toBe(true);
    expect(res?.result.content[0].text).toContain('Unknown domain "leio"');
  });

  it("describes resources with their actual class counts", async () => {
    const listRes = await server.handleMessage({ jsonrpc: "2.0", id: 401, method: "resources/list" });
    const resources = listRes?.result.resources;
    const uris = resources.map((r: any) => r.uri);
    expect(uris).toEqual(Object.keys(ONTOLOGY_CATALOG).map((k) => `ontology://${k}`));
    expect(uris).not.toContain("ontology://leio");

    const gs1 = resources.find((r: any) => r.uri === "ontology://gs1");
    expect(gs1.name).toBe("GS1 Web Vocabulary sample");
    expect(gs1.description).toContain("5 classes");
    expect(gs1.description).not.toContain("Full");

    const readRes = await server.handleMessage({
      jsonrpc: "2.0",
      id: 402,
      method: "resources/read",
      params: { uri: "ontology://incident" },
    });
    const parsed = JSON.parse(readRes?.result.contents[0].text);
    expect(parsed.num_concepts).toBe(5);
    expect(parsed.concepts.length).toBe(5);
    expect(parsed.concepts[3]).toEqual({
      curie: "INCIDENT:DB_DEADLOCK",
      label: "Database Deadlock",
      description: "",
      parents: ["INCIDENT:DATA"],
      disjoint_with: [],
    });

    const missing = await server.handleMessage({
      jsonrpc: "2.0",
      id: 403,
      method: "resources/read",
      params: { uri: "ontology://leio" },
    });
    expect(missing?.error.code).toBe(-32002);
  });

  it("does not advertise prompts", async () => {
    const res = await server.handleMessage({ jsonrpc: "2.0", id: 501, method: "prompts/list" });
    expect(res?.error.code).toBe(-32601);
  });

  it("executes olaya_subsumes as down-set membership", async () => {
    const inside = await server.handleMessage({
      jsonrpc: "2.0",
      id: 4,
      method: "tools/call",
      params: {
        name: "olaya_subsumes",
        arguments: {
          state: "Certified organic whole milk gallon",
          target_concept: "gs1:FoodBeverageTobaccoProduct",
          domain: "gs1",
        },
      },
    });
    const content = JSON.parse(inside?.result.content[0].text);
    expect(content.selected_concept).toBe("gs1:OrganicClaimDetails");
    expect(content.subsumes).toBe(true); // OrganicClaimDetails is below FoodBeverageTobaccoProduct

    const outside = await server.handleMessage({
      jsonrpc: "2.0",
      id: 5,
      method: "tools/call",
      params: {
        name: "olaya_subsumes",
        arguments: {
          state: "Heavy duty corrugated packaging box and carton container",
          target_concept: "Trade Item Product",
          domain: "gs1",
        },
      },
    });
    const outsideContent = JSON.parse(outside?.result.content[0].text);
    expect(outsideContent.selected_concept).toBe("gs1:PackagingDetails");
    expect(outsideContent.subsumes).toBe(false);
  });

  it("rates on an ordered scale with honest confidence", async () => {
    const res = await server.handleMessage({
      jsonrpc: "2.0",
      id: 6,
      method: "tools/call",
      params: {
        name: "olaya_score",
        arguments: { state: "Critical outage, customers cannot pay", criteria: ["low", "medium", "critical"] },
      },
    });
    const content = JSON.parse(res?.result.content[0].text);
    expect(content.level).toBe(3);
    expect(content.confidence).toBeGreaterThan(0);
    expect(content.scale).toEqual(["low", "medium", "critical"]);

    const noEvidence = await server.handleMessage({
      jsonrpc: "2.0",
      id: 7,
      method: "tools/call",
      params: { name: "olaya_score", arguments: { state: "nothing relevant here", criteria: ["low", "medium", "critical"] } },
    });
    const flat = JSON.parse(noEvidence?.result.content[0].text);
    expect(flat.confidence).toBeCloseTo(0, 6);
    expect(flat.probabilities.low).toBeCloseTo(1 / 3, 5);
  });

  it("supports install configuration dry-run for Claude Desktop", () => {
    const msg = installNodeMCPConfig("claude", true);
    expect(msg).toContain("[Dry-Run]");
    expect(msg).toContain("olaya");
    expect(msg).toContain("leio-decision");
  });
});
