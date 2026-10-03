"""Build an offline macOS arm64 MCPB from the installed, tested Python environment."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist"
VERSION = "0.2.0"
NAME = f"olaya-desktop-complete-macos-arm64-{VERSION}"
ROOT_DEPENDENCIES = ["torch", "transformers", "numpy", "pydantic", "onnxruntime", "rdflib"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency_closure():
    found = {}
    def visit(name):
        distribution = metadata.distribution(name)
        key = distribution.metadata["Name"].lower().replace("_", "-")
        if key in found:
            return
        found[key] = distribution
        for expression in distribution.requires or []:
            requirement = Requirement(expression)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                visit(requirement.name)
    for name in ROOT_DEPENDENCIES:
        visit(name)
    return found


def main():
    if sys.platform != "darwin" or platform.machine() != "arm64" or sys.version_info[:2] != (3, 12):
        raise SystemExit("Build requires macOS arm64 and the tested Python 3.12 environment")
    OUT.mkdir(exist_ok=True)
    dependencies = dependency_closure()
    base = Path(sys.base_prefix).resolve()
    with tempfile.TemporaryDirectory(prefix="olaya complete stage ") as folder:
        stage = Path(folder)
        # uv's python-build-standalone distribution relocates relative to its executable.
        # Follow symlinks so archives have actual files, not references outside the bundle.
        shutil.copytree(base, stage / "runtime", symlinks=False, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        site = stage / "runtime/lib/python3.12/site-packages"
        # CPython's preinstalled packaging tools are retained; local editable .pth files are never copied.
        for distribution in dependencies.values():
            for item in distribution.files or []:
                relative = Path(str(item))
                if relative.is_absolute() or ".." in relative.parts or relative.suffix in {".pyc", ".pth"} or relative.name == "direct_url.json":
                    continue
                source = Path(distribution.locate_file(item))
                if source.is_file():
                    target = site / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        shutil.copytree(ROOT / "src/olaya", stage / "src/olaya", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # The extension surface does not expose host-specific ABox harvesters or training imports.
        # Keep public runtime imports focused and avoid loading optional tooling at startup.
        (stage / "src/olaya/__init__.py").write_text('"""Bundled O-Laya runtime."""\n')
        (stage / "src/olaya/real_abox.py").unlink()
        shutil.copytree(ROOT / "extensions/claude-desktop/skills", stage / "skills")
        shutil.copy2(ROOT / "extensions/claude-desktop/README.md", stage / "README.md")
        (stage / "server").mkdir()
        shutil.copy2(ROOT / "extensions/claude-desktop/server/complete.py", stage / "server/complete.py")
        (stage / "models").mkdir()
        for name in ["tokenizer.json", "olaya_decision_model.pt"]:
            shutil.copy2(ROOT / "models" / name, stage / "models" / name)
        import torch
        checkpoint = torch.load(stage / "models/olaya_decision_model.pt", map_location="cpu", weights_only=True)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        provenance = {
            "model": {"file": "models/olaya_decision_model.pt", "sha256": sha(stage / "models/olaya_decision_model.pt"),
                      "tokenizer_sha256": sha(stage / "models/tokenizer.json"), "config": checkpoint["config"],
                      "historical_validation_accuracy": checkpoint.get("val_acc"),
                      "quality_status": "Historical checkpoint metrics only; no fresh domain accuracy/calibration certification."},
            "runtime": {"python": platform.python_version(), "architecture": platform.machine(),
                        "packages": {name: d.version for name, d in sorted(dependencies.items())}},
            "source_revision": revision,
            "source_files_sha256": {p.name: sha(p) for p in sorted((stage / "src/olaya").glob("*.py"))},
            "server_sha256": sha(stage / "server/complete.py"),
        }
        (stage / "bundle-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        manifest = {
            "manifest_version": "0.3", "name": "olaya-desktop-complete", "display_name": "O-Laya Complete — macOS Apple Silicon",
            "version": VERSION, "description": "Offline neural decisions, TBox workflow compilation/execution/replay, JSON-LD and three skills.",
            "long_description": "Self-contained macOS ARM64 CPython/PyTorch runtime and checkpoint. Does not support Intel Macs or Windows. Model quality is not certified for arbitrary TBoxes. Terminal intents do not execute external effects.",
            "author": {"name": "O-Laya contributors"}, "license": "MIT",
            "server": {"type": "binary", "entry_point": "runtime/bin/python3.12", "mcp_config": {"command": "${__dirname}/runtime/bin/python3.12", "args": ["-I", "-B", "${__dirname}/server/complete.py"]}},
            "compatibility": {"platforms": ["darwin"]}, "tools_generated": True, "prompts_generated": True,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        subprocess.run(["npx", "--yes", "@anthropic-ai/mcpb@2.1.2", "validate", str(stage / "manifest.json")], check=True)
        subprocess.run(["npx", "--yes", "@anthropic-ai/mcpb@2.1.2", "pack", str(stage), str(OUT / (NAME + ".mcpb"))], check=True)
        artifacts = [OUT / (NAME + ".mcpb")]
        for skill in sorted((stage / "skills").iterdir()):
            path = OUT / (skill.name + "-0.2.0.zip")
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                for source in sorted(skill.rglob("*")):
                    if source.is_file():
                        archive.write(source, source.relative_to(stage / "skills"))
            artifacts.append(path)
        # Full companion Code plugin includes the same embedded interpreter and package payload.
        plugin_path = OUT / "olaya-complete-claude-code-macos-arm64-0.2.0.zip"
        with zipfile.ZipFile(OUT / (NAME + ".mcpb")) as source, zipfile.ZipFile(plugin_path, "w", zipfile.ZIP_DEFLATED) as target:
            for item in source.infolist():
                if item.filename != "manifest.json":
                    target.writestr(item, source.read(item.filename))
            target.writestr(".claude-plugin/plugin.json", json.dumps({"name": "olaya-complete", "version": VERSION, "description": "Complete offline macOS ARM64 O-Laya runtime and skills"}))
            target.writestr(".mcp.json", json.dumps({"olaya": {"command": "${CLAUDE_PLUGIN_ROOT}/runtime/bin/python3.12", "args": ["-I", "-B", "${CLAUDE_PLUGIN_ROOT}/server/complete.py"]}}))
        artifacts.append(plugin_path)
        (OUT / "claude-desktop-complete-sha256.txt").write_text("".join(sha(p) + "  " + p.name + "\n" for p in artifacts))
        shutil.copy2(stage / "bundle-provenance.json", OUT / "claude-desktop-complete-provenance.json")
        print("Built complete artifacts:")
        for artifact in artifacts:
            print(artifact.name, artifact.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
