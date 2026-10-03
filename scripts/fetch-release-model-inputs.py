"""Fetch and verify the immutable model inputs required by a clean CI checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(folder: Path, assets: list[dict]) -> None:
    for asset in assets:
        path = folder / asset["name"]
        if not path.is_file() or path.stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
            raise ValueError(f"Model input integrity check failed: {asset['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    inputs = json.loads((ROOT / ".github/release-model-inputs.json").read_text())
    assets = inputs["assets"]
    if {a["name"] for a in assets} != {"tokenizer.json", "olaya_decision_model.pt"} or len(assets) != 2:
        raise ValueError("Unexpected pinned model assets")
    if args.check_only:
        verify(args.output_dir, assets)
    else:
        with tempfile.TemporaryDirectory(prefix="olaya pinned model inputs ") as folder:
            stage = Path(folder)
            command = ["gh", "release", "download", inputs["release_tag"], "--repo", args.repo, "--dir", str(stage)]
            for asset in assets:
                command.extend(["--pattern", asset["name"]])
            subprocess.run(command, check=True)
            verify(stage, assets)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for asset in assets:
                shutil.copy2(stage / asset["name"], args.output_dir / asset["name"])
    print(f"Verified {len(assets)} model inputs from {inputs['release_tag']}")


if __name__ == "__main__":
    main()
