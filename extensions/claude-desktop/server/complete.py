"""Self-contained stdio MCP adapter for the actual O-Laya Python runtime."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAX_REQUEST = 2 * 1024 * 1024
VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}
MODEL = None
TOKENIZER = None
MODEL_ID = None


def manifest():
    return json.loads((ROOT / "bundle-provenance.json").read_text())


def capabilities():
    provenance = manifest()
    return {
        "backend": "python-torch-checkpoint", "offline": True,
        "platform": "macos-arm64", "workflow_compile": True,
        "workflow_execution": True, "workflow_replay": True,
        "primitives": ["concept", "choice", "score", "subsumes", "noul"],
        "external_effects": False, "model": provenance["model"],
        "runtime": provenance["runtime"],
        "limitations": ["Checkpoint quality is not certified for arbitrary TBoxes.",
                        "Terminal intents do not execute external infrastructure or application actions.",
                        "Replay checks policy consistency; it does not authenticate the trace."],
    }


def reference_list():
    return [
        {"uri": "olaya://bundle/" + p.relative_to(ROOT).as_posix(),
         "name": p.relative_to(ROOT).as_posix(),
         "mimeType": "text/markdown" if p.suffix == ".md" else "application/json"}
        for p in sorted((ROOT / "skills").rglob("*")) if p.is_file()
    ]


def reference(path):
    if not isinstance(path, str) or not any(r["name"] == path for r in reference_list()):
        raise ValueError("Unknown bundled reference")
    return (ROOT / path).read_text(encoding="utf-8")


def prompts():
    return [{"name": p.name, "description": "Load bundled " + p.name + " skill"}
            for p in sorted((ROOT / "skills").iterdir()) if (p / "SKILL.md").is_file()]


def ontology(value=None):
    from olaya.ontology import Ontology, create_incident_ontology
    if value is None:
        return create_incident_ontology()
    if not isinstance(value, dict) or not isinstance(value.get("concepts"), list):
        raise ValueError("TBox requires a concepts array")
    concepts = value["concepts"]
    if not 1 <= len(concepts) <= 4096:
        raise ValueError("TBox requires 1..4096 classes")
    ids = set()
    for row in concepts:
        if not isinstance(row, dict) or not isinstance(row.get("curie"), str) or not row["curie"]:
            raise ValueError("Invalid class identifier")
        if row["curie"] in ids:
            raise ValueError("Duplicate class identifier")
        ids.add(row["curie"])
        if not isinstance(row.get("label"), str) or not row["label"]:
            raise ValueError("Class requires a label")
        for field in ("parents", "disjoint_with"):
            if not isinstance(row.get(field, []), list) or len(row.get(field, [])) > 4096:
                raise ValueError("Invalid class relationships")
    by_id = {row["curie"]: row for row in concepts}
    for row in concepts:
        for field in ("parents", "disjoint_with"):
            if any(not isinstance(v, str) or v not in ids or v == row["curie"] for v in row.get(field, [])):
                raise ValueError("Invalid class relationship")
    # Iterative DFS avoids recursion limits on a valid long bounded chain.
    colors = {}
    for curie in ids:
        stack = [(curie, False)]
        while stack:
            current, finished = stack.pop()
            if finished:
                colors[current] = 2
                continue
            if colors.get(current) == 2:
                continue
            if colors.get(current) == 1:
                raise ValueError("Cyclic class hierarchy")
            colors[current] = 1
            stack.append((current, True))
            stack.extend((p, False) for p in by_id[current].get("parents", []))
    return Ontology.from_json(value)


def engine(tbox):
    global MODEL, TOKENIZER, MODEL_ID
    import torch
    from transformers import PreTrainedTokenizerFast
    from olaya.engine import OLaya
    if MODEL is None:
        p = ROOT / "models/olaya_decision_model.pt"
        metadata = manifest()["model"]
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual_hash != metadata["sha256"]:
            raise ValueError("Bundled checkpoint hash mismatch")
        torch.set_num_threads(2)
        TOKENIZER = PreTrainedTokenizerFast(
            tokenizer_file=str(ROOT / "models/tokenizer.json"),
            cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
            pad_token="[PAD]", unk_token="[UNK]",
        )
        # Construct architecture from checkpoint, then require every tensor to match.
        native = OLaya(ontology=tbox, tokenizer=TOKENIZER, device="cpu", weights_path=p)
        checkpoint = torch.load(p, map_location="cpu", weights_only=True)
        native.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        MODEL = native.model
        MODEL_ID = "olaya-checkpoint-sha256:" + actual_hash
    native = OLaya(ontology=tbox, tokenizer=TOKENIZER, model=MODEL, device="cpu")
    original = native.system_one
    def system_one(state, questions):
        result = original(state, questions)
        result.model = MODEL_ID
        return result
    native.system_one = system_one
    return native


def object_schema(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


OBJECT = {"type": "object"}
WORKFLOW = {"type": "object", "description": "Complete version 1 TBox-bound workflow document"}
TBOX = {"type": "object", "description": "O-Laya ontology object: name, concepts, optional properties; see bundled storage TBox"}
TOOLS = [
    {"name": "olaya_capabilities", "description": "Report runtime, bundled checkpoint provenance and exact capabilities.", "inputSchema": object_schema({})},
    {"name": "olaya_system_one", "description": "Run the bundled neural checkpoint for concept, choice, score, subsumes and noul questions; return JSON-LD provenance. Quality is not certified for arbitrary ontologies.", "inputSchema": object_schema({"tbox": TBOX, "state": {}, "questions": {"type": "object", "minProperties": 1, "maxProperties": 32}}, ["state", "questions"])},
    {"name": "olaya_workflow_compile", "description": "Compile and validate a workflow against its supplied TBox, including infrastructure and application resources.", "inputSchema": object_schema({"workflow": WORKFLOW, "tbox": TBOX}, ["workflow", "tbox"])},
    {"name": "olaya_workflow_run", "description": "Compile and run the actual Python workflow with the bundled neural checkpoint; emit JSON-LD. Terminal intents have no external side effects.", "inputSchema": object_schema({"workflow": WORKFLOW, "tbox": TBOX, "inputs": OBJECT, "run_id": {"type": "string"}}, ["workflow", "tbox", "inputs"])},
    {"name": "olaya_workflow_replay", "description": "Verify and replay an emitted live workflow trace against its workflow and TBox without model inference.", "inputSchema": object_schema({"workflow": WORKFLOW, "tbox": TBOX, "trace": OBJECT}, ["workflow", "tbox", "trace"])},
    {"name": "olaya_read_reference", "description": "Read a whitelisted packaged skill or reference.", "inputSchema": object_schema({"path": {"type": "string"}}, ["path"])},
]


def call(name, args):
    definition = next((t for t in TOOLS if t["name"] == name), None)
    if definition is None:
        raise ValueError("Unknown tool")
    schema = definition["inputSchema"]
    if not isinstance(args, dict) or set(args) - set(schema["properties"]) or set(schema["required"]) - set(args):
        raise ValueError("Missing or unknown tool arguments")
    if name == "olaya_capabilities":
        return capabilities()
    if name == "olaya_read_reference":
        return {"path": args["path"], "text": reference(args["path"])}
    tbox = ontology(args.get("tbox"))
    if name == "olaya_system_one":
        questions = args["questions"]
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 32:
            raise ValueError("Supply 1..32 typed questions")
        for q in questions.values():
            if not isinstance(q, dict) or q.get("type") not in capabilities()["primitives"]:
                raise ValueError("Invalid typed question")
            if q["type"] in {"choice", "score"}:
                criteria = q.get("criteria")
                if not isinstance(criteria, (dict, list)) or not 1 <= len(criteria) <= 128:
                    raise ValueError("Supply 1..128 choice/score criteria")
        if len(json.dumps(args["state"], allow_nan=False).encode()) > 65536:
            raise ValueError("State exceeds 64 KiB")
        result = dataclasses.asdict(engine(tbox).system_one(args["state"], questions))
        from olaya.workflow_jsonld import ontology_digest
        activity = "urn:uuid:" + str(uuid4())
        model_id = "urn:olaya:checkpoint:" + manifest()["model"]["sha256"]
        tbox_id = "urn:olaya:tbox:" + ontology_digest(tbox)
        trace = {"@context": {"o": "urn:olaya:desktop:", "prov": "http://www.w3.org/ns/prov#"}, "@graph": [
            {"@id": model_id, "@type": "prov:Entity", "o:sha256": manifest()["model"]["sha256"]},
            {"@id": tbox_id, "@type": "prov:Entity", "o:sha256": ontology_digest(tbox)},
            {"@id": activity, "@type": ["prov:Activity", "o:NeuralDecisionActivity"], "prov:used": [{"@id": model_id}, {"@id": tbox_id}]},
            {"@id": activity + "/answers", "@type": ["prov:Entity", "o:DecisionRecord"], "prov:wasGeneratedBy": {"@id": activity}, "o:answersJson": json.dumps(result["answers"], allow_nan=False)},
        ]}
        return {"result": result, "jsonld": trace, "model": manifest()["model"]}
    from olaya.workflow import compile_workflow, run_workflow
    compiled = compile_workflow(args["workflow"], tbox)
    if name == "olaya_workflow_compile":
        return {"workflow_hash": compiled.workflow_hash, "tbox_hash": compiled.tbox_hash, "document": compiled.document}
    if name == "olaya_workflow_run":
        trace = run_workflow(compiled, engine(tbox), args["inputs"], run_id=args.get("run_id"))
        return {"trace": trace, "model": manifest()["model"]}
    from olaya.workflow_replay import replay_workflow
    return {"trace": replay_workflow(compiled, args["trace"]), "model_inference": False}


def handle(request):
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
        raise ValueError("Invalid JSON-RPC request")
    if "id" not in request:
        return None
    params = request.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("Invalid parameters")
    method = request["method"]
    if method == "initialize":
        version = params.get("protocolVersion")
        result = {"protocolVersion": version if version in VERSIONS else "2025-11-25", "capabilities": {"tools": {}, "resources": {}, "prompts": {}}, "serverInfo": {"name": "olaya-desktop-complete", "version": "0.2.0"}, "instructions": "Complete offline neural O-Laya runtime. Call olaya_capabilities for model quality limits. Workflow terminal intents do not execute external effects."}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        try:
            value = call(params.get("name"), params.get("arguments", {}))
            # Check finite serialization before producing the MCP envelope.
            text = json.dumps(value, allow_nan=False)
            result = {"content": [{"type": "text", "text": text}], "structuredContent": value}
        except Exception as exc:
            # Validation errors are useful; implementation errors must not leak state or host paths.
            message = str(exc) if isinstance(exc, ValueError) else "Runtime rejected the request"
            result = {"isError": True, "content": [{"type": "text", "text": message}]}
    elif method == "resources/list":
        result = {"resources": reference_list()}
    elif method == "resources/read":
        row = next((r for r in reference_list() if r["uri"] == params.get("uri")), None)
        if row is None:
            raise ValueError("Unknown bundled resource")
        result = {"contents": [{"uri": row["uri"], "mimeType": row["mimeType"], "text": reference(row["name"])}]}
    elif method == "prompts/list":
        result = {"prompts": prompts()}
    elif method == "prompts/get":
        name = params.get("name")
        if not any(p["name"] == name for p in prompts()):
            raise ValueError("Unknown prompt")
        result = {"messages": [{"role": "user", "content": {"type": "text", "text": reference("skills/" + name + "/SKILL.md")}}]}
    else:
        return {"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32601, "message": "Method not found"}}
    return {"jsonrpc": "2.0", "id": request["id"], "result": result}


def main():
    stream = sys.stdin.buffer
    while True:
        line = stream.readline(MAX_REQUEST + 1)
        if not line:
            return
        request = None
        try:
            if len(line) > MAX_REQUEST:
                while line and not line.endswith(b"\n"):
                    line = stream.readline(MAX_REQUEST + 1)
                raise ValueError("Request exceeds 2 MiB")
            request = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            response = handle(request)
        except (ValueError, TypeError, RecursionError):
            response = {"jsonrpc": "2.0", "id": request.get("id") if isinstance(request, dict) else None, "error": {"code": -32602 if request else -32700, "message": "Invalid request"}}
        if response is not None:
            sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
