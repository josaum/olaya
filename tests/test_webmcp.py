"""Unit and protocol tests for O-Laya WebMCP server."""

from __future__ import annotations

import json

from olaya.webmcp import OLayaMCPServer, extract_jsonld_or_microdata


def test_extract_jsonld_and_gtin():
    html_sample = """
    <!DOCTYPE html>
    <html>
    <head>
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": "Organic Cold-Pressed Orange Juice",
            "gtin13": "0012345678905",
            "description": "100% pure fresh orange juice."
        }
        </script>
    </head>
    <body>
        <h1>Organic Orange Juice</h1>
        <p>GTIN: 0012345678905</p>
    </body>
    </html>
    """
    extracted = extract_jsonld_or_microdata(html_sample)
    assert extracted.get("@type") == "Product"
    assert "Orange Juice" in extracted.get("name", "")


def test_mcp_initialize_and_tools_list():
    server = OLayaMCPServer()
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }
    resp = server.handle_message(init_req)
    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == "olaya-mcp"

    list_req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    list_resp = server.handle_message(list_req)
    tools = list_resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "olaya_ground" in tool_names
    assert "olaya_subsumes" in tool_names
    assert "olaya_choice" in tool_names
    assert "olaya_extract_and_ground" in tool_names


def test_mcp_olaya_ground_tool():
    server = OLayaMCPServer()
    call_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "olaya_ground",
            "arguments": {
                "state": "GTIN 0012345678905 barcode scan: chilled 100% orange juice bottle",
                "domain": "gs1",
            },
        },
    }
    resp = server.handle_message(call_req)
    assert resp["id"] == 3
    assert not resp["result"]["isError"]

    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["domain"] == "gs1"
    assert "curie" in content
    assert content["confidence"] > 0.0
    assert content["axioms_verified"] is True


def test_mcp_olaya_subsumes_tool():
    server = OLayaMCPServer()
    call_req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "olaya_subsumes",
            "arguments": {
                "state": "Fresh apple juice drink bottle",
                "target_concept": "Beverage",
                "domain": "gs1",
            },
        },
    }
    resp = server.handle_message(call_req)
    assert resp["id"] == 4
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["target_concept"] == "Beverage"
    assert 0.0 <= content["subsumes"] <= 1.0


def test_mcp_olaya_choice_tool():
    server = OLayaMCPServer()
    call_req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "olaya_choice",
            "arguments": {
                "state": "Payment gateway HTTP 504 timeout during checkout",
                "options": ["restart_service", "retry_gateway", "page_oncall"],
            },
        },
    }
    resp = server.handle_message(call_req)
    assert resp["id"] == 5
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["selected_choice"] in ["restart_service", "retry_gateway", "page_oncall"]


def test_mcp_resources():
    server = OLayaMCPServer()
    res_list_req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "resources/list",
        "params": {},
    }
    resp = server.handle_message(res_list_req)
    resources = resp["result"]["resources"]
    uris = [r["uri"] for r in resources]
    assert "ontology://gs1" in uris

    read_req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "resources/read",
        "params": {"uri": "ontology://gs1"},
    }
    read_resp = server.handle_message(read_req)
    contents = json.loads(read_resp["result"]["contents"][0]["text"])
    assert contents["name"] == "GS1WebVocab"
    assert contents["num_concepts"] >= 55  # may be enriched with synonyms


def test_mcp_extract_and_ground_tool():
    server = OLayaMCPServer()
    html_page = """
    <html>
      <head>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org/",
          "@type": "Product",
          "name": "Sparkling Apple Drink",
          "gtin13": "4012345678901"
        }
        </script>
      </head>
      <body>Carbonated refreshing apple beverage</body>
    </html>
    """
    call_req = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {
            "name": "olaya_extract_and_ground",
            "arguments": {"web_payload": html_page},
        },
    }
    resp = server.handle_message(call_req)
    assert resp["id"] == 8
    assert not resp["result"]["isError"]

    content = json.loads(resp["result"]["content"][0]["text"])
    assert "gs1_grounding" in content
    assert "schemaorg_grounding" in content
    assert content["gs1_grounding"]["confidence"] > 0.0


def test_mcp_compliance_ping_and_notifications():
    server = OLayaMCPServer()
    ping_req = {
        "jsonrpc": "2.0",
        "id": 101,
        "method": "ping",
    }
    resp = server.handle_message(ping_req)
    assert resp["id"] == 101
    assert resp["result"] == {}

    # Notification returns None
    init_note = server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert init_note is None

    cancel_note = server.handle_message({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}})
    assert cancel_note is None

    templates_req = {"jsonrpc": "2.0", "id": 102, "method": "resources/templates/list"}
    tpl_resp = server.handle_message(templates_req)
    assert tpl_resp["id"] == 102
    assert "resourceTemplates" in tpl_resp["result"]


def test_mcp_leio_product_ontology():
    server = OLayaMCPServer(mode="fast")
    # Test reading resource
    read_req = {
        "jsonrpc": "2.0",
        "id": 103,
        "method": "resources/read",
        "params": {"uri": "ontology://leio"},
    }
    read_resp = server.handle_message(read_req)
    contents = json.loads(read_resp["result"]["contents"][0]["text"])
    assert contents["domain"] == "leio"
    assert contents["num_concepts"] >= 30

    # Test inspect ecosystem tool
    call_req = {
        "jsonrpc": "2.0",
        "id": 104,
        "method": "tools/call",
        "params": {
            "name": "leio_inspect_ecosystem",
            "arguments": {
                "query": "Transparent token accounting at 0.042 dollars per 1 million input tokens with unmetered answers.",
            },
        },
    }
    resp = server.handle_message(call_req)
    assert resp["id"] == 104
    assert not resp["result"]["isError"]
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["domain"] == "leio"
    assert content["curie"].startswith("leio:") and ("leio:Pricing" in content["lineage"] or content["curie"] in ("leio:InputOnlyMetering", "leio:PayAsYouGoScaleTier", "leio:JevCompatibleBilling", "leio:FreeOutputTokens"))
