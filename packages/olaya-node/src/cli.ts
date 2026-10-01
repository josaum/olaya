#!/usr/bin/env node
/**
 * CLI Entrypoint for @mycelia/olaya-mcp
 * Zero-dependency Model Context Protocol server.
 */

import { OLayaNodeMCPServer, installNodeMCPConfig } from "./mcp.js";
import { LeioServer } from "./server.js";

const args = process.argv.slice(2);

function printHelp(): void {
  console.log(`
Leio Model Context Protocol (MCP) & System 1 Decision Server

Usage:
  npx @mycelia/olaya-mcp                     Start stdio MCP server (MCP 2026-07-28)
  npx @mycelia/olaya-mcp serve [--port 8080] Start HTTP server (Jev /v1/systemone + MCP 2026)
  npx @mycelia/olaya-mcp ground "<text>"     Ground observation from CLI
  npx @mycelia/olaya-mcp install --client claude  Configure Claude Desktop
  npx @mycelia/olaya-mcp install --client cursor  Configure Cursor IDE
`);
}

async function run(): Promise<void> {
  if (args.includes("--help") || args.includes("-h")) {
    printHelp();
    return;
  }

  const command = args[0];

  if (command === "install") {
    const clientIdx = args.indexOf("--client");
    const client = clientIdx !== -1 && args[clientIdx + 1] === "cursor" ? "cursor" : "claude";
    const dryRun = args.includes("--dry-run");
    const msg = installNodeMCPConfig(client, dryRun);
    console.log(msg);
    return;
  }

  if (command === "serve") {
    const portIdx = args.indexOf("--port");
    const port = portIdx !== -1 && args[portIdx + 1] ? parseInt(args[portIdx + 1], 10) : 8080;
    const server = new LeioServer(port);
    await server.start();
    return;
  }

  if (command === "ground") {
    const text = args[1] || "";
    const domainIdx = args.indexOf("--domain");
    const domain = domainIdx !== -1 && args[domainIdx + 1] ? args[domainIdx + 1] : "gs1";
    const server = new OLayaNodeMCPServer();
    const res = await server.executeTool("olaya_ground", { state: text, domain });
    console.log(JSON.stringify(res, null, 2));
    return;
  }

  // Default: Run stdio MCP server
  const server = new OLayaNodeMCPServer();
  server.runStdio();
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
