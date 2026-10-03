"""Test the extracted complete MCPB with its own interpreter, offline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import queue
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
DEFAULT_ARTIFACT = ROOT / f"dist/olaya-desktop-complete-macos-arm64-{VERSION}.mcpb"
DENIED_HOST_PATHS = [
    str(ROOT),
    str(Path(sys.base_prefix).resolve()),
    str(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))).resolve()),
]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def extract(artifact: Path, destination: Path) -> None:
    """Stream ZIP payloads and restore executable permissions on embedded Python."""
    with zipfile.ZipFile(artifact) as archive:
        seen = set()
        for entry in archive.infolist():
            relative = Path(entry.filename)
            if relative.is_absolute() or ".." in relative.parts or entry.filename in seen:
                raise ValueError("Unsafe or duplicate artifact member")
            seen.add(entry.filename)
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("Artifact contains an external symlink")
            target = destination / relative
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if mode & 0o777:
                target.chmod(mode & 0o777)


class Client:
    def __init__(self, bundle: Path, network_sandbox: bool):
        self.sequence = 0
        self.responses = queue.Queue()
        self.stderr = tempfile.TemporaryFile(mode="w+b")
        command = [str(bundle / "runtime/bin/python3.12"), "-I", "-B", str(bundle / "server/complete.py")]
        if network_sandbox:
            policy = "(version 1) (allow default) (deny network*) " + " ".join(
                f"(deny file-read* (subpath {json.dumps(path)}))" for path in DENIED_HOST_PATHS
            )
            command = ["/usr/bin/sandbox-exec", "-p", policy, *command]
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(bundle / "isolated home"),
            "TMPDIR": str(bundle / "temporary files"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        Path(env["HOME"]).mkdir(exist_ok=True)
        Path(env["TMPDIR"]).mkdir(exist_ok=True)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            cwd=bundle,
            env=env,
            text=True,
            bufsize=1,
        )

        def read():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def raw(self, text: str) -> dict:
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()
        try:
            response = self.responses.get(timeout=120)
        except queue.Empty as exc:
            raise AssertionError("Embedded MCP server timed out") from exc
        if response is None:
            self.stderr.seek(0)
            raise AssertionError("Embedded MCP exited: " + self.stderr.read().decode(errors="replace")[-4000:])
        return json.loads(response)

    def request(self, method: str, params: dict | None = None) -> dict:
        self.sequence += 1
        response = self.raw(
            json.dumps(
                {"jsonrpc": "2.0", "id": self.sequence, "method": method, "params": params or {}}, allow_nan=False
            )
        )
        assert response["id"] == self.sequence, response
        assert "error" not in response, response
        return response["result"]

    def tool(self, name: str, arguments: dict, *, error: bool = False) -> dict:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        if error:
            assert result.get("isError") is True, result
            return result
        assert not result.get("isError"), result
        value = result["structuredContent"]
        assert json.loads(result["content"][0]["text"]) == value
        return value

    def initialize(self) -> None:
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "artifact-test", "version": "1"},
            },
        )
        assert result["protocolVersion"] == "2025-11-25"
        assert result["serverInfo"]["name"] == "olaya-desktop-complete"
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.process.stdin.flush()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.reader.join(timeout=1)
        self.process.stdout.close()
        self.stderr.close()


def run_entity(trace: dict) -> dict:
    return next(node for node in trace["@graph"] if "wf:WorkflowRun" in node.get("@type", []))


def test_bundle(artifact: Path) -> dict:
    artifact_hash = digest(artifact)
    network_sandbox = Path("/usr/bin/sandbox-exec").is_file()
    with tempfile.TemporaryDirectory(prefix="olaya complete artifact test ") as directory:
        bundle = Path(directory) / "Extracted bundle with spaces"
        bundle.mkdir()
        extract(artifact, bundle)
        assert os.access(bundle / "runtime/bin/python3.12", os.X_OK), "Interpreter executable mode missing"
        provenance = json.loads((bundle / "bundle-provenance.json").read_text())
        model_hash = digest(bundle / "models/olaya_decision_model.pt")
        assert model_hash == provenance["model"]["sha256"]
        assert digest(bundle / "models/tokenizer.json") == provenance["model"]["tokenizer_sha256"]
        client = Client(bundle, network_sandbox)
        try:
            client.initialize()
            tools = client.request("tools/list")["tools"]
            assert {tool["name"] for tool in tools} == {
                "olaya_capabilities",
                "olaya_system_one",
                "olaya_workflow_compile",
                "olaya_workflow_run",
                "olaya_workflow_replay",
                "olaya_read_reference",
            }
            prompts = client.request("prompts/list")["prompts"]
            assert len(prompts) == 3
            for prompt in prompts:
                assert client.request("prompts/get", {"name": prompt["name"]})["messages"][0]["content"]["text"]
            references = client.request("resources/list")["resources"]
            assert references
            for reference in references:
                resource = client.request("resources/read", {"uri": reference["uri"]})["contents"][0]["text"]
                assert resource == client.tool("olaya_read_reference", {"path": reference["name"]})["text"]
                assert resource
            caps = client.tool("olaya_capabilities", {})
            assert caps["backend"] == "python-torch-checkpoint"
            for capability in ("offline", "workflow_compile", "workflow_execution", "workflow_replay"):
                assert caps[capability] is True
            assert caps["external_effects"] is False
            assert caps["model"]["sha256"] == model_hash
            reference_root = bundle / "skills/olaya-tbox-workflows/references"
            tbox = json.loads((reference_root / "storage-tbox.json").read_text())
            workflow = json.loads((reference_root / "storage.json").read_text())
            compiled = client.tool("olaya_workflow_compile", {"tbox": tbox, "workflow": workflow})
            assert len(compiled["workflow_hash"]) == len(compiled["tbox_hash"]) == 64
            live = client.tool(
                "olaya_workflow_run",
                {
                    "tbox": tbox,
                    "workflow": workflow,
                    "inputs": {"shipment": "Frozen seafood shipment requires cold storage."},
                },
            )
            live_run = run_entity(live["trace"])
            assert live_run["wf:executionMode"] == "live"
            assert live_run["wf:modelIdentifier"] == "olaya-checkpoint-sha256:" + model_hash
            assert live_run["wf:status"] == "completed"
            replay_args = {"tbox": tbox, "workflow": workflow, "trace": live["trace"]}
            replay = client.tool("olaya_workflow_replay", replay_args)
            assert replay["model_inference"] is False
            assert run_entity(replay["trace"])["wf:executionMode"] == "replay"
            questions = {
                "concept": {"type": "concept", "root": "INCIDENT:ROOT"},
                "subsumes": {"type": "subsumes", "concept": "INCIDENT:DATA"},
                "choice": {"type": "choice", "criteria": {"database": "Database outage", "billing": "Payment error"}},
                "score": {"type": "score", "criteria": ["routine", "urgent", "critical"]},
                "noul": {"type": "noul", "instructions": "Database service is unavailable"},
            }
            decisions = client.tool(
                "olaya_system_one",
                {"state": "Database service is unavailable; urgent incident.", "questions": questions},
            )
            assert decisions["result"]["model"] == "olaya-checkpoint-sha256:" + model_hash
            assert decisions["model"]["sha256"] == model_hash
            assert set(decisions["result"]["answers"]) == set(questions)
            for kind, answer in decisions["result"]["answers"].items():
                assert answer["type"] == kind
            assert decisions["jsonld"]["@graph"]
            duplicate = copy.deepcopy(tbox)
            duplicate["concepts"].append(copy.deepcopy(duplicate["concepts"][0]))
            cycle = copy.deepcopy(tbox)
            cycle["concepts"][0]["parents"] = [cycle["concepts"][1]["curie"]]
            cycle["concepts"][1]["parents"] = [cycle["concepts"][0]["curie"]]
            for invalid in (duplicate, cycle):
                client.tool("olaya_workflow_compile", {"tbox": invalid, "workflow": workflow}, error=True)
            client.tool("olaya_read_reference", {"path": "../../bundle-provenance.json"}, error=True)
            client.tool("olaya_read_reference", {"path": str(bundle / "bundle-provenance.json")}, error=True)
            client.tool("olaya_system_one", {"state": "x", "questions": {"bad": {"type": "unknown"}}}, error=True)
            assert client.raw("{not JSON")["error"]["code"] == -32700
            assert client.request("ping") == {}, "Malformed request broke subsequent requests"
        finally:
            client.close()
        # A fresh process cannot reuse a cached model. Remove both model inputs.
        moved = []
        for name in ("olaya_decision_model.pt", "tokenizer.json"):
            original = bundle / "models" / name
            renamed = original.with_name(name + ".unavailable")
            original.rename(renamed)
            moved.append((original, renamed))
        fresh = None
        try:
            fresh = Client(bundle, network_sandbox)
            fresh.initialize()
            absent_weights_replay = fresh.tool("olaya_workflow_replay", replay_args)
            assert absent_weights_replay["model_inference"] is False
            assert run_entity(absent_weights_replay["trace"])["wf:executionMode"] == "replay"
            # In the same fresh process a real inference request must fail rather
            # than silently constructing a random model or recovering host files.
            fresh.tool("olaya_system_one", {"state": "Database outage", "questions": questions}, error=True)
        finally:
            if fresh is not None:
                fresh.close()
            for original, renamed in moved:
                renamed.rename(original)
        assert digest(artifact) == artifact_hash, "Artifact changed during test; rerun the final build"
        return {
            "artifact": artifact.name,
            "artifact_sha256": artifact_hash,
            "model_sha256": model_hash,
            "network_denied_by_sandbox": network_sandbox,
            "host_file_reads_denied_by_sandbox": DENIED_HOST_PATHS if network_sandbox else [],
            "restricted_path": "/usr/bin:/bin",
            "tools": len(tools),
            "prompts": len(prompts),
            "references_read": len(references),
            "workflow_live_status": live_run["wf:status"],
            "all_five_primitives": True,
            "fresh_process_replay_without_model_or_tokenizer": True,
            "quality_claim": "Execution and checkpoint provenance only; no domain accuracy or calibration certification.",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    if not args.artifact.is_file():
        parser.error(f"Artifact does not exist: {args.artifact}")
    print(json.dumps(test_bundle(args.artifact.resolve()), indent=2))


if __name__ == "__main__":
    main()
