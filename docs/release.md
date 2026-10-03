# Release packages

The release version is **0.2.0** across the Python package, Node package, complete desktop bundle, and companion Code plugin. Building artifacts, attaching them to a GitHub release, and publishing packages to registries are separate operations. A successful build does not establish model accuracy or calibration.

## Trigger

Push a commit to `main` whose commit message contains `[release]`, or dispatch [Release](../.github/workflows/release.yml) on `main` with `build_release=true`. The flag is an explicit request to build release artifacts; an ordinary development push does not request a release build. The release workflow runs the normal CI workflow before its package and native desktop jobs.

```bash
gh workflow run release.yml --ref main -f build_release=true
```

The workflow distributes built artifacts through GitHub release assets. It does not publish to npm or PyPI. A local commit, a CI artifact, or a GitHub release attachment is not evidence of registry publication. No registry credentials or publication flag are required for this GitHub release workflow.

## Inputs and runners

The complete desktop bundle requires macOS Apple Silicon and a uv-managed Python 3.12 runtime. The release build uses a `macos-15` ARM64 runner. The standalone Python wheel and Node tarball are separate package formats; the complete desktop runtime remains specific to macOS ARM64.

[release-model-inputs.json](../.github/release-model-inputs.json) records the source release tag, asset names, expected byte counts and SHA256 values for the checkpoint and tokenizer. The native CI job downloads these assets from the workflow's own repository through authenticated `gh release download`; this works with a private GitHub repository. The source release tag is `olaya-desktop-model-inputs-v1`. A missing asset, size mismatch or hash mismatch fails the build, rather than substituting a random model, unpinned checkpoint or developer cache.

The desktop bundle records checkpoint/tokenizer hashes, runtime package versions, source revision and source-file hashes. A dirty source snapshot must be disclosed in provenance. A Git revision alone does not describe uncommitted files that went into an artifact.

## Local package build and checks

From the repository root, prepare the project's development environment and build the wheel:

```bash
uv sync --extra dev
uv build --out-dir dist/python
```

Build and pack the Node package:

```bash
cd packages/olaya-node
bun install --frozen-lockfile
bun run build
mkdir -p ../../dist/node
npm pack --pack-destination ../../dist/node
```

Then return to the repository root and test the actual artifacts:

```bash
.venv/bin/python scripts/test-release-packages.py \
  --python-dir dist/python \
  --node-dir dist/node
```

The package check requires exactly one wheel and one `.tgz` in those directories. Use clean output directories for a new version. It verifies that both project manifests and both packaged versions match the release version from `pyproject.toml` and use `X.Y.Z` format. The current version produces GitHub release tag `v0.2.0`.

The wheel check extracts its payload into a temporary directory outside the checkout, starts Python with `-I -B`, and asserts that every loaded `olaya` module comes from that payload. Dependencies come from the invoking interpreter's installed environment; this check does not claim that the wheel contains PyTorch or an embedded interpreter. It verifies the workflow, replay, JSON-LD and MCP modules, the `olaya-mcp = olaya.webmcp:main` entry point, lexical routing, and a workflow run/replay using a deterministic fake backend. It does not load a neural checkpoint or measure neural quality.

The npm check verifies `dist/index.mjs`, `dist/index.cjs`, `dist/cli.mjs`, `dist/index.d.ts`, the declared exports and CLI mapping. It loads both ESM and CommonJS builds from the extracted tarball and exercises `OntologyDAG` and the lexical `OLaya` runtime on a small fixture. It does not infer performance from filenames or from source-tree tests.

The source distribution gets a separate runtime gate: rebuild a wheel from the actual source archive, then test that rebuilt wheel with the same isolated package checks. For the current version:

```bash
uv build --wheel dist/python/olaya-0.2.0.tar.gz --out-dir dist/sdist-wheel
.venv/bin/python scripts/test-release-packages.py \
  --python-dir dist/sdist-wheel \
  --node-dir dist/node \
  --source-archive dist/python/olaya-0.2.0.tar.gz \
  > dist/sdist-verification.json
```

`--source-archive` verifies the archive filename and package metadata and records its SHA256. Those checks alone do not prove the archive builds. The separate `uv build --wheel` command and runtime checks on its output supply that evidence. CI performs both steps and retains the source-archive verification report with the release assets.

## Complete desktop build and checks

Stage the exact checkpoint and tokenizer inputs in `models/`, verify them, and run the builder with the tested macOS ARM64 Python 3.12 environment:

```bash
.venv/bin/python scripts/fetch-release-model-inputs.py \
  --repo josaum/olaya --check-only
.venv/bin/python scripts/build-claude-desktop-complete.py
.venv/bin/python scripts/test-claude-desktop-complete.py \
  > dist/claude-desktop-complete-verification.json
```

The artifact test preserves executable ZIP permissions, extracts to a directory containing spaces, and executes the bundled interpreter with `-I -B` and a restricted `PATH`. On macOS it denies network access and reads from the original repository, original uv Python installation and host Hugging Face cache. It verifies six tools, three prompts, all packaged references, checkpoint-backed workflow execution and all five decision primitives. It also replays a trace in a fresh process with both model and tokenizer unavailable; a real inference request in that process must fail.

Keep the verification JSON, provenance JSON and SHA256 manifest with the release assets. These checks establish the tested artifact's execution and provenance. The historical checkpoint validation metric is not a fresh domain evaluation, and terminal workflow intents do not execute external application or infrastructure effects.

## Distribution and signing

GitHub release assets include the Python wheel and source archive, npm tarball, macOS ARM64 MCPB, companion Code plugin, skill packages, checksums, provenance and verification output produced by the release build. Build jobs retain their intermediate Actions artifacts for 14 days. The final job downloads both successful artifact sets, creates release notes and checksums, attaches assets to a draft release, and publishes that GitHub release after upload. Confirm the uploaded asset list and completed jobs before describing a release as delivered. GitHub release access follows the repository's visibility and permissions; uploading assets does not make a private repository public.

Registry publication is outside this workflow. A future npm or PyPI publication step would require explicit configuration and authorization for `@mycelia/olaya` or `olaya`, respectively. This document does not establish that either package is currently published.

Hosted CI builds are unsigned by default. A developer identity stored in a user's macOS Keychain stays on that Mac; do not export a private signing key to hosted CI. After a successful GitHub release, an authorized local signing step can download the fresh CI-built MCPB, use that Keychain identity, and upload a separate `-signed` variant without replacing the unsigned artifact. Recompute checksums and repeat artifact verification after signing or any other mutation. Signing and notarization require their own recorded verification; an unsigned functional smoke test is not a signing result.
