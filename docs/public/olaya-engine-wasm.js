// olaya-engine-wasm.js — Progressive WASM enhancement with JS fallback
// Tries to load the Rust WASM engine; falls back to pure JS if unavailable.

let backend = 'js';
let wasmInit = null;
let WasmEngine = null;

try {
  const wasm = await import('./wasm/olaya_wasm.js');
  await wasm.default();  // init WASM memory + bindings
  WasmEngine = wasm.OlayaWasm;
  backend = 'wasm';
} catch(e) {
  // WASM not available — fall back to pure JS engine silently
  console.debug('[olaya] WASM not available, using JS engine:', e?.message ?? e);
}

// Re-export the full JS engine API surface (always available)
export * from './olaya-engine.js';

// Additional exports: WASM backend indicator and WasmEngine class (may be null)
export { backend as BACKEND, WasmEngine };
