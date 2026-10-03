"""Validate aligned package versions and print GitHub Actions release outputs."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    node = json.loads((ROOT / "packages/olaya-node/package.json").read_text())
    if not re.fullmatch(r"\d+\.\d+\.\d+", version) or version != node["version"]:
        raise ValueError("Python and Node release versions must match and use X.Y.Z")
    print("version=" + version)
    print("tag=v" + version)


if __name__ == "__main__":
    main()
