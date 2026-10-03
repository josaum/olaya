"""Unit tests for the air-gapped Ollama local model MCP server."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from scripts.ollama_mcp_server import execute_tool, handle_message


def test_initialize_handshake():
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    resp = handle_message(req)
    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == "ollama-local"
    assert "tools" in resp["result"]["capabilities"]


def test_tools_list():
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp = handle_message(req)
    assert resp["id"] == 2
    tool_names = [t["name"] for t in resp["result"]["tools"]]
    assert "ollama_generate" in tool_names
    assert "ollama_chat" in tool_names
    assert "ollama_list_models" in tool_names
    assert "ollama_embeddings" in tool_names


@patch("scripts.ollama_mcp_server._call_ollama")
def test_execute_generate(mock_call):
    mock_call.return_value = {
        "model": "qwen2.5:7b",
        "response": "Local test response",
        "total_duration": 500000000,
        "eval_duration": 400000000,
        "prompt_eval_count": 10,
        "eval_count": 8,
        "done": True,
    }
    result = execute_tool("ollama_generate", {"prompt": "Hello local model"})
    assert result["model"] == "qwen2.5:7b"
    assert result["response"] == "Local test response"
    assert result["eval_tokens"] == 8


@patch("scripts.ollama_mcp_server._call_ollama")
def test_execute_chat(mock_call):
    mock_call.return_value = {
        "model": "qwen2.5:7b",
        "message": {"role": "assistant", "content": "Chat reply"},
        "total_duration": 300000000,
        "eval_count": 5,
    }
    result = execute_tool("ollama_chat", {"messages": [{"role": "user", "content": "Hi"}]})
    assert result["content"] == "Chat reply"


@patch("scripts.ollama_mcp_server._call_ollama")
def test_execute_list_models(mock_call):
    mock_call.return_value = {
        "models": [
            {
                "name": "qwen2.5:7b",
                "size": 4683087332,
                "modified_at": "2026-10-02T18:43:18Z",
                "details": {"parameter_size": "7.6B", "quantization_level": "Q4_K_M"},
            }
        ]
    }
    result = execute_tool("ollama_list_models", {})
    assert len(result["models"]) == 1
    assert result["models"][0]["name"] == "qwen2.5:7b"
    assert result["models"][0]["parameter_size"] == "7.6B"


def test_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        execute_tool("nonexistent_tool", {})
