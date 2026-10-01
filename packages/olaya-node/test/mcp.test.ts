import { describe, expect, it } from "bun:test";
import { OLayaNodeMCPServer, installNodeMCPConfig } from "../src/index.js";

describe("OLayaNodeMCPServer (MCP Protocol)", () => {
  const server = new OLayaNodeMCPServer();

  it("handles initialize request with MCP capabilities", async () => {
    const res = await server.handleMessage({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { protocolVersion: "2024-11-05" },
    });

    expect(res).not.toBeNull();
    expect(res?.result.protocolVersion).toBe("2024-11-05");
    expect(res?.result.serverInfo.name).toBe("olaya-mcp-node");
    expect(res?.result.capabilities.tools).toBeDefined();
  });

  it("lists all decision tools via tools/list", async () => {
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
