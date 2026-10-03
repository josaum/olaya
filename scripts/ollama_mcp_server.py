#!/usr/bin/env python3
"""
Air-gapped Local Model (Ollama) Model Context Protocol (MCP) Server.
Enables Claude Desktop and other MCP clients to execute 100% local, offline inference
against locally running Ollama instances (e.g., qwen2.5:7b, llama3.3, deepseek-r1).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

TOOLS = [
    {
        "name": "ollama_generate",
        "description": "Execute local LLM inference via Ollama. Runs 100% on device/GPU with zero external data egress. Ideal for private reasoning, code analysis, document summarization, and offline tasks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The input text or prompt to send to the local model."
                },
                "model": {
                    "type": "string",
                    "description": f"Name of local model (defaults to '{DEFAULT_MODEL}').",
                    "default": DEFAULT_MODEL
                },
                "system": {
                    "type": "string",
                    "description": "Optional system prompt instruction."
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature (default 0.7). Lower for deterministic/factual, higher for creative.",
                    "default": 0.7
                }
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "ollama_chat",
        "description": "Execute multi-turn conversational chat inference on the local model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                            "content": {"type": "string"}
                        },
                        "required": ["role", "content"]
                    },
                    "description": "Array of chat message objects: [{'role': 'user', 'content': '...'}]"
                },
                "model": {
                    "type": "string",
                    "description": f"Name of local model (defaults to '{DEFAULT_MODEL}').",
                    "default": DEFAULT_MODEL
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature (default 0.7).",
                    "default": 0.7
                }
            },
            "required": ["messages"]
        }
    },
    {
        "name": "ollama_list_models",
        "description": "List all local models currently downloaded and available in the Ollama instance.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "ollama_embeddings",
        "description": "Generate vector embeddings for input text locally using Ollama.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text to embed."
                },
                "model": {
                    "type": "string",
                    "description": f"Embedding model (defaults to '{DEFAULT_MODEL}').",
                    "default": DEFAULT_MODEL
                }
            },
            "required": ["prompt"]
        }
    }
]


def _call_ollama(endpoint: str, payload: dict[str, Any] | None = None, method: str = "POST") -> dict[str, Any]:
    url = f"{OLLAMA_HOST.rstrip('/')}{endpoint}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to local Ollama at {url}: {e}") from e


def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "ollama_generate":
        model = args.get("model") or DEFAULT_MODEL
        prompt = args.get("prompt", "")
        system = args.get("system")
        temperature = args.get("temperature", 0.7)

        req_body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        if system:
            req_body["system"] = system

        res = _call_ollama("/api/generate", req_body)
        return {
            "model": res.get("model", model),
            "response": res.get("response", ""),
            "total_duration_ms": round(res.get("total_duration", 0) / 1e6, 2),
            "eval_duration_ms": round(res.get("eval_duration", 0) / 1e6, 2),
            "prompt_tokens": res.get("prompt_eval_count", 0),
            "eval_tokens": res.get("eval_count", 0),
            "done": res.get("done", True)
        }

    elif name == "ollama_chat":
        model = args.get("model") or DEFAULT_MODEL
        messages = args.get("messages", [])
        temperature = args.get("temperature", 0.7)

        req_body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
        res = _call_ollama("/api/chat", req_body)
        msg = res.get("message", {})
        return {
            "model": res.get("model", model),
            "message": msg,
            "content": msg.get("content", ""),
            "total_duration_ms": round(res.get("total_duration", 0) / 1e6, 2),
            "eval_tokens": res.get("eval_count", 0)
        }

    elif name == "ollama_list_models":
        res = _call_ollama("/api/tags", method="GET")
        models = [
            {
                "name": m.get("name"),
                "size_gb": round(m.get("size", 0) / (1024**3), 2),
                "modified_at": m.get("modified_at"),
                "parameter_size": m.get("details", {}).get("parameter_size", "unknown"),
                "quantization": m.get("details", {}).get("quantization_level", "unknown")
            }
            for m in res.get("models", [])
        ]
        return {"models": models, "host": OLLAMA_HOST}

    elif name == "ollama_embeddings":
        model = args.get("model") or DEFAULT_MODEL
        prompt = args.get("prompt", "")
        res = _call_ollama("/api/embeddings", {"model": model, "prompt": prompt})
        return {
            "model": model,
            "embedding": res.get("embedding", []),
            "dimensions": len(res.get("embedding", []))
        }

    else:
        raise ValueError(f"Unknown tool: {name}")


def handle_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    req_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False}
                },
                "serverInfo": {
                    "name": "ollama-local",
                    "version": "1.0.0"
                }
            }
        }

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS}
        }

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            result = execute_tool(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }

    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }
    return None


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_message(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"}
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
