"""Validate release receipts, flatten assets, and generate checksums and release notes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    version = os.environ["RELEASE_VERSION"]
    commit = os.environ["GITHUB_SHA"]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.parent == root:
            continue
        target = root / path.name
        if target.exists():
            raise ValueError("Duplicate release asset basename: " + path.name)
        shutil.move(path, target)
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
    packages = json.loads((root / "package-verification.json").read_text())
    source = json.loads((root / "sdist-verification.json").read_text())
    desktop = json.loads((root / "claude-desktop-complete-verification.json").read_text())
    provenance = json.loads((root / "claude-desktop-complete-provenance.json").read_text())
    if packages["version"] != version or source["version"] != version or provenance["bundle_version"] != version:
        raise ValueError("Release version mismatch")
    if provenance["source_revision"] != commit or provenance["source_worktree_tracked_dirty"]:
        raise ValueError("Desktop provenance must match the clean release commit")
    for record in [packages["python"], packages["node"]]:
        if sha(root / record["file"]) != record["sha256"]:
            raise ValueError("Package verification hash mismatch")
    if sha(root / source["source_archive"]["file"]) != source["source_archive"]["sha256"]:
        raise ValueError("Source archive verification hash mismatch")
    if sha(root / desktop["artifact"]) != desktop["artifact_sha256"]:
        raise ValueError("Desktop verification hash mismatch")
    if not desktop["network_denied_by_sandbox"] or not desktop["fresh_process_replay_without_model_or_tokenizer"]:
        raise ValueError("Offline Desktop release checks must pass")
    if not any(root.glob("*.tar.gz")) or len(list(root.glob("*.skill"))) != 3:
        raise ValueError("Missing source archive or native skills")
    files = sorted(path for path in root.iterdir() if path.is_file())
    (root / "SHA256SUMS").write_text("".join(sha(p) + "  " + p.name + "\n" for p in files))
    accuracy = provenance["model"]["historical_validation_accuracy"]
    notes = f"""O-Laya {version}, built and verified from commit `{commit}`.

Included assets: Python wheel and source archive; `leio-decision` npm tarball;
complete macOS Apple Silicon MCPB; three native skills and skill ZIPs; companion
Claude Code plugin; package/Desktop verification reports, runtime/model provenance
and SHA-256 checksums.

The release passed Rust, Python, Node and edge CI gates, extracted wheel/npm import
checks, and complete Desktop execution with network and original-host reads denied.
All five Python decision primitives, workflow compilation/execution and inference-free
replay were exercised. Read the attached receipts for their exact scope.

The Node runtime is lexical. The Desktop runtime embeds the neural checkpoint,
CPython and native dependencies. Its historical checkpoint validation accuracy is
{accuracy:.4f}; these functional checks do not establish domain accuracy, calibration,
production latency or SOTA performance. Workflow terminal intents have no external
application/infrastructure side effects.

Hosted CI produces the unsigned MCPB. A separately named Keychain-signed variant may
be uploaded afterward with its own signature verification report. Signing keys are
not copied into hosted CI. These assets are distributed through this private GitHub
release; this workflow does not publish to npmjs.com or PyPI.
"""
    (root / "RELEASE_NOTES.md").write_text(notes)
    print(f"Validated {len(files)} assets for {version} at {commit}")


if __name__ == "__main__":
    main()
