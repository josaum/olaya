# docs/public

Static demo pages for the lexical O-Laya engine.

- `olaya-engine.js` — the pure-JavaScript engine (a port of `packages/olaya-node/src`); it is what
  `index.html` and `playground.html` run on by default.
- `olaya-engine-wasm.js` — the same API surface with an optional Rust/WebAssembly backend.
- `ontologies/*.json` — the demo ontologies the pages can load.
- `api/` — the separately deployed edge handlers (see `spec/systemone-reliability.md`).

## The WebAssembly build is not committed

`index.html`, `playground.html` and `olaya-engine-wasm.js` try to `import('./wasm/olaya_wasm.js')`
at runtime. That directory is a build output of the `olaya-wasm` crate and is **not** checked in
(no binaries in the repository). When it is absent the dynamic import rejects, the pages log a
console warning and keep working on `olaya-engine.js`; the latency badge then stays on the JS value.

To build it, install [wasm-pack](https://rustwasm.github.io/wasm-pack/) and the target once:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

then, from the repository root:

```sh
wasm-pack build crates/olaya-wasm --release --target web --out-dir ../../docs/public/wasm
```

This writes `docs/public/wasm/olaya_wasm.js`, `olaya_wasm_bg.wasm` and the TypeScript
declarations. `wasm-pack` uses the ordinary `release` profile; for the size-optimised build use the
workspace's `wasm-release` profile and run `wasm-bindgen` yourself:

```sh
cargo build -p olaya-wasm --target wasm32-unknown-unknown --profile wasm-release
wasm-bindgen target/wasm32-unknown-unknown/wasm-release/olaya_wasm.wasm \
  --target web --out-dir docs/public/wasm
```

CI only checks that the crate compiles for `wasm32-unknown-unknown`
(`cargo build -p olaya-wasm --target wasm32-unknown-unknown --release`); its unit tests run natively
with `cargo test -p olaya-wasm` against `spec/axioms_golden.json` and `spec/decision_parity.json`.

Serve the directory with any static file server (for example `python -m http.server` from
`docs/public`); ES-module imports do not work from `file://` URLs.
