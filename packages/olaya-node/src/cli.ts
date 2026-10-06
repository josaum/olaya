#!/usr/bin/env node
/**
 * CLI entrypoint for olaya: a zero-dependency Model Context Protocol server.
 */

import { OLayaNodeMCPServer, installNodeMCPConfig, LATEST_PROTOCOL_VERSION, DEFAULT_DOMAIN } from "./mcp.js";
import { OlayaHttpServer } from "./server.js";

const args = process.argv.slice(2);

function printHelp(): void {
  console.log(`
O-Laya Model Context Protocol (MCP) server and lexical decision HTTP server

Usage:
  npx olaya                           Start stdio MCP server (protocol ${LATEST_PROTOCOL_VERSION} / 2024-11-05)
  npx olaya serve [--port 8080]       Start HTTP server (/v1/systemone, /v1/mcp, /v1/health)
  npx olaya ground "<text>" [--domain ${DEFAULT_DOMAIN}]
                                                   Ground an observation from the CLI
  npx olaya install --client claude   Configure Claude Desktop
  npx olaya install --client cursor   Configure Cursor IDE
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
    const server = new OlayaHttpServer(port);
    await server.start();
    return;
  }

  if (command === "ground") {
    const text = args[1] || "";
    const domainIdx = args.indexOf("--domain");
    const domain = domainIdx !== -1 && args[domainIdx + 1] ? args[domainIdx + 1] : DEFAULT_DOMAIN;
    const server = new OLayaNodeMCPServer();
    const res = await server.executeTool("olaya_ground", { state: text, domain });
    console.log(JSON.stringify(res, null, 2));
    return;
  }

  // Default: run the stdio MCP server
  const server = new OLayaNodeMCPServer();
  server.runStdio();
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
