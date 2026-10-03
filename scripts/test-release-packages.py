"""Verify built Python and npm payloads from temporary directories, offline."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PYTHON_SMOKE = r"""
import importlib, json, pathlib, sys
payload = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(payload))
from olaya.ontology import Ontology, Concept
from olaya.workflow import compile_workflow, run_workflow
from olaya.workflow_replay import replay_workflow
from olaya.candidate_router import CandidateRouter
from olaya.engine import ChoiceAnswer, SystemOneResult
from olaya import webmcp
assert callable(webmcp.main)
ontology = Ontology('release-smoke')
for name in ('Workflow', 'Decision', 'Storage', 'Cold', 'Ambient', 'DispatchCold', 'DispatchAmbient', 'Review'):
    ontology.add_concept(Concept('ex:' + name, name))
ontology.add_subsumption('ex:Cold', 'ex:Storage')
ontology.add_subsumption('ex:Ambient', 'ex:Storage')
assert CandidateRouter(ontology).route_candidates('Cold')[0][0].curie == 'ex:Cold'
document = {
    'version': 1, 'id': 'urn:release:workflow', 'type': 'ex:Workflow',
    'prefixes': {'ex': 'https://example.org/release#'}, 'tbox': 'urn:release:tbox',
    'start': 'decision', 'steps': {
        'decision': {'type': 'ex:Decision', 'decide': {
            'state': 'input.shipment', 'question': {'type': 'choice', 'root': 'ex:Storage', 'options': ['ex:Cold', 'ex:Ambient']}},
            'accept': {'min_confidence': .8, 'min_margin': .15},
            'route': {'ex:Cold': 'cold', 'ex:Ambient': 'ambient'}, 'on_abstain': 'review', 'on_error': 'review'},
        'cold': {'emit': {'type': 'ex:DispatchCold'}}, 'ambient': {'emit': {'type': 'ex:DispatchAmbient'}},
        'review': {'emit': {'type': 'ex:Review'}}}}
class FakeEngine:
    def __init__(self): self.ontology = ontology
    def system_one(self, state, questions):
        assert state == 'cold shipment'
        assert set(next(iter(questions.values()))['criteria']) == {'ex:Cold', 'ex:Ambient'}
        return SystemOneResult(model='release-fake-fixture', answers={key: ChoiceAnswer(choice='ex:Cold',
            probabilities={'ex:Cold': .9, 'ex:Ambient': .1}, confidence=.9) for key in questions}, usage={}, latency_ms=0)
compiled = compile_workflow(document, ontology)
trace = run_workflow(compiled, FakeEngine(), {'shipment': 'cold shipment'})
assert any('https://example.org/release#DispatchCold' in row.get('@type', []) for row in trace['@graph'])
replay = replay_workflow(compiled, trace)
assert any(row.get('wf:executionMode') == 'replay' for row in replay['@graph'])
loaded = {}
for name, module in list(sys.modules.items()):
    if name == 'olaya' or name.startswith('olaya.'):
        location = getattr(module, '__file__', None)
        if location:
            resolved = pathlib.Path(location).resolve()
            assert resolved.is_relative_to(payload), (name, str(resolved))
            loaded[name] = str(resolved.relative_to(payload))
assert {'olaya.workflow', 'olaya.workflow_replay', 'olaya.workflow_jsonld', 'olaya.webmcp'} <= set(loaded)
print(json.dumps({'loaded_modules': loaded, 'workflow_live_and_replay': True, 'neural_inference': False}))
"""

NODE_SMOKE = r"""
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const esm = await import('@mycelia/olaya');
const cjs = createRequire(import.meta.url)('@mycelia/olaya');
for (const [format, api] of [['esm', esm], ['cjs', cjs]]) {
  assert.equal(typeof api.OntologyDAG, 'function');
  assert.equal(typeof api.OLaya, 'function');
  const ontology = new api.OntologyDAG([
    {id: 0, curie: 'T:ROOT', label: 'Storage', parents: [], disjointWith: [], synonyms: []},
    {id: 1, curie: 'T:COLD', label: 'Frozen seafood', parents: [0], disjointWith: [2], synonyms: ['cold seafood']},
    {id: 2, curie: 'T:OFFICE', label: 'Office stationery', parents: [0], disjointWith: [1], synonyms: []}
  ]);
  assert.equal(ontology.subsumes(0, 1), true);
  const engine = new api.OLaya(ontology);
  const result = await engine.systemOne('Frozen seafood');
  assert.equal(result.selectedConcept.curie, 'T:COLD', format);
  const choice = await engine.choice('Frozen seafood', ['Frozen seafood', 'Office stationery']);
  assert.equal(choice.choice, 'Frozen seafood', format);
}
console.log(JSON.stringify({formats: ['esm', 'cjs'], lexical_fixture_execution: true}));
"""


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def safe_relative(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe archive path")
    return relative


def verify_wheel(wheel: Path, destination: Path, version: str) -> dict:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "Duplicate wheel member"
        assert {"olaya/workflow.py", "olaya/workflow_replay.py", "olaya/workflow_jsonld.py", "olaya/webmcp.py"} <= set(
            names
        )
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_paths) == 1
        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode())
        assert metadata["Name"] == "olaya"
        assert metadata["Version"] == version
        entries_path = metadata_paths[0].removesuffix("METADATA") + "entry_points.txt"
        entries = configparser.ConfigParser()
        entries.read_string(archive.read(entries_path).decode())
        assert entries["console_scripts"]["olaya-mcp"] == "olaya.webmcp:main"
        for name in names:
            relative = safe_relative(name)
            if name.endswith("/"):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", PYTHON_SMOKE, str(destination)],
        cwd=destination,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "file": wheel.name,
        "sha256": sha256(wheel),
        "version": version,
        "dependency_environment": "Current interpreter's installed dependencies; every olaya module comes from the extracted wheel",
        "runtime": json.loads(result.stdout),
    }


def verify_node(tarball: Path, destination: Path, version: str) -> dict:
    with tarfile.open(tarball, "r:gz") as archive:
        members = archive.getmembers()
        names = [entry.name for entry in members]
        assert len(names) == len(set(names)), "Duplicate npm archive member"
        assert {
            "package/package.json",
            "package/dist/index.mjs",
            "package/dist/index.cjs",
            "package/dist/cli.mjs",
            "package/dist/index.d.ts",
        } <= set(names)
        for entry in members:
            relative = safe_relative(entry.name)
            assert relative.parts[0] == "package"
            assert not entry.issym() and not entry.islnk(), "npm artifact links are unsupported"
            if entry.isdir():
                continue
            assert entry.isfile(), "Unexpected npm member type"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    package = destination / "package"
    metadata = json.loads((package / "package.json").read_text())
    assert metadata["name"] == "@mycelia/olaya"
    assert metadata["version"] == version
    assert metadata["exports"]["."]["import"] == "./dist/index.mjs"
    assert metadata["exports"]["."]["require"] == "./dist/index.cjs"
    assert metadata["types"] == "./dist/index.d.ts"
    assert metadata["bin"]["olaya-mcp"] == "./dist/cli.mjs"
    node = shutil.which("node")
    assert node is not None, "Node is required to test the built package"
    script = destination / "smoke.mjs"
    script.write_text(NODE_SMOKE)
    package_link = destination / "node_modules/@mycelia/olaya"
    package_link.parent.mkdir(parents=True)
    package_link.symlink_to(package, target_is_directory=True)
    env = dict(os.environ)
    env.pop("NODE_PATH", None)
    result = subprocess.run(
        [node, str(script), str(package)],
        cwd=destination,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {"file": tarball.name, "sha256": sha256(tarball), "version": version, "runtime": json.loads(result.stdout)}


def verify_source_archive(archive_path: Path, version: str) -> dict:
    """Verify sdist identity; runtime coverage comes from its separately rebuilt wheel."""
    assert archive_path.name == f"olaya-{version}.tar.gz", "Source archive filename version mismatch"
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [entry.name for entry in members]
        assert len(names) == len(set(names)), "Duplicate source archive member"
        prefix = f"olaya-{version}"
        for entry in members:
            relative = safe_relative(entry.name)
            assert relative.parts[0] == prefix, "Unexpected source archive root"
            assert entry.isfile() or entry.isdir(), "Source archive links are unsupported"
        metadata_path = prefix + "/PKG-INFO"
        assert metadata_path in names, "Source archive package metadata missing"
        with archive.extractfile(metadata_path) as stream:
            metadata = Parser().parsestr(stream.read().decode())
        assert metadata["Name"] == "olaya"
        assert metadata["Version"] == version
    return {
        "file": archive_path.name,
        "sha256": sha256(archive_path),
        "version": version,
        "scope": "Source archive identity and integrity metadata; CI must rebuild this archive and test its resulting wheel",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-dir", type=Path, default=ROOT / "dist/python")
    parser.add_argument("--node-dir", type=Path, default=ROOT / "dist/node")
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    python_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    node_version = json.loads((ROOT / "packages/olaya-node/package.json").read_text())["version"]
    assert python_version == node_version and re.fullmatch(r"\d+\.\d+\.\d+", python_version), (
        "Python and Node release versions must match and use X.Y.Z"
    )
    wheels = sorted(args.python_dir.glob("*.whl"))
    tarballs = sorted(args.node_dir.glob("*.tgz"))
    assert len(wheels) == len(tarballs) == 1, "Expected exactly one built wheel and npm tarball"
    with tempfile.TemporaryDirectory(prefix="olaya release package test ") as temporary:
        folder = Path(temporary)
        python_output, node_output = folder / "wheel payload", folder / "node payload"
        python_output.mkdir()
        node_output.mkdir()
        report = {
            "version": python_version,
            "python": verify_wheel(wheels[0].resolve(), python_output, python_version),
            "node": verify_node(tarballs[0].resolve(), node_output, python_version),
            "limitations": "Artifact import and deterministic fixture execution; no neural accuracy, calibration, production latency, or registry publication claim.",
        }
        if args.source_archive is not None:
            report["source_archive"] = verify_source_archive(args.source_archive.resolve(), python_version)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
