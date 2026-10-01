//! Shared ORT (ONNX Runtime) session helpers for Mycelia.
//!
//! Centralises the patterns previously duplicated across
//! `mycelia-gateway/src/cognitive/engines/local.rs`,
//! `mycelia-gateway/src/sisfron_cv/pipeline.rs`,
//! `mycelia-gateway/src/semantic/embeddings.rs`,
//! `mycelia-align/src/embedder.rs`, and
//! `office-parsers-rs/gliner-core/src/ort_inference.rs`.
//!
//! Policy (see workspace AGENTS.md):
//! - All Mycelia Rust crates pin `ort = "=2.0.0-rc.12"`.
//! - Execution provider preference is **CoreML (macOS) → CUDA (linux/windows) →
//!   CPU**, with a guaranteed CPU fallback when EP registration fails.
//! - `.ort` mmap format is preferred over `.onnx` when both are present.
//! - `ort::init()` is called at most once per process; helpers here never call
//!   it implicitly.

use std::path::{Path, PathBuf};

pub use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::ValueType;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OrtUtilError {
    #[error("ort session builder: {0}")]
    Builder(String),
    #[error("ort commit_from_file({path}): {message}")]
    Commit { path: String, message: String },
    #[error("model file not found: {0}")]
    NotFound(PathBuf),
}

/// Execution provider preference. Maps onto `ort` providers at build time and
/// falls back to CPU on registration failure.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EpPref {
    /// Platform default: CoreML on macOS (gated by `MYCELIA_ENABLE_COREML`
    /// when [`SessionConfig::coreml_auto_gate`] is true), CUDA on
    /// linux/windows, CPU elsewhere.
    #[default]
    Auto,
    /// Force Apple CoreML (no-op on non-macOS, falls through to CPU).
    CoreML,
    /// Force NVIDIA CUDA (no-op on macOS, falls through to CPU).
    Cuda,
    /// Force CPU.
    Cpu,
}

/// Full session configuration. Use [`SessionConfig::default()`] for
/// `(Auto EP, Level3 opt, num_cpus threads, MYCELIA_ENABLE_COREML gate on)`.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub ep: EpPref,
    pub intra_threads: usize,
    pub opt_level: GraphOptimizationLevel,
    /// CoreML model cache directory. When `None`, helpers resolve via
    /// `COREML_CACHE_DIR` env var or `{model_dir}/coreml_cache`.
    pub coreml_cache_dir: Option<PathBuf>,
    /// On macOS with [`EpPref::Auto`], gate CoreML on the `MYCELIA_ENABLE_COREML`
    /// env var. Matches `mycelia-gateway` policy: opt-in only.
    /// Has no effect for [`EpPref::CoreML`] (explicit), [`EpPref::Cuda`], or
    /// [`EpPref::Cpu`].
    pub coreml_auto_gate: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            ep: EpPref::Auto,
            intra_threads: num_cpus_or_one(),
            opt_level: GraphOptimizationLevel::Level3,
            coreml_cache_dir: None,
            coreml_auto_gate: true,
        }
    }
}

impl SessionConfig {
    /// CPU-only with `Level3` optimisation and a sensible default thread count.
    pub fn cpu() -> Self {
        Self {
            ep: EpPref::Cpu,
            ..Self::default()
        }
    }
    pub fn with_threads(mut self, n: usize) -> Self {
        self.intra_threads = n;
        self
    }
    pub fn with_opt_level(mut self, lvl: GraphOptimizationLevel) -> Self {
        self.opt_level = lvl;
        self
    }
    pub fn with_ep(mut self, ep: EpPref) -> Self {
        self.ep = ep;
        self
    }
    pub fn with_coreml_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.coreml_cache_dir = Some(dir.into());
        self
    }
    /// Disable the `MYCELIA_ENABLE_COREML` gate (always try CoreML in Auto mode
    /// on macOS). Use for crates that have their own opt-in mechanism.
    pub fn without_coreml_gate(mut self) -> Self {
        self.coreml_auto_gate = false;
        self
    }
}

/// Description of how a session was actually built. Useful for logging.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub model_path: PathBuf,
    pub size_mb: f64,
    pub provider: &'static str,
    pub used_ort_format: bool,
}

fn num_cpus_or_one() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Resolve a sibling `.ort` file when present (zero-copy mmap), else return
/// the input path unchanged.
pub fn prefer_ort_format(model_path: &str) -> (String, bool) {
    if model_path.ends_with(".onnx") {
        let candidate = model_path.replace(".onnx", ".ort");
        if Path::new(&candidate).exists() {
            return (candidate, true);
        }
    }
    (model_path.to_string(), model_path.ends_with(".ort"))
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
fn resolve_coreml_cache_dir(model_path: &str, override_dir: Option<&Path>) -> PathBuf {
    if let Some(p) = override_dir {
        return p.to_path_buf();
    }
    if let Ok(v) = std::env::var("COREML_CACHE_DIR") {
        return PathBuf::from(v);
    }
    let model_dir = Path::new(model_path).parent().unwrap_or(Path::new("."));
    model_dir.join("coreml_cache")
}

/// Is CoreML opted into for this session?
///
/// The **default stays `false`** — this crate serves the gateway's local-LLM
/// and CV paths, whose policy is documented on
/// [`SessionConfig::coreml_auto_gate`] as "opt-in only". That default is a
/// deliberate difference from the OCR path's and is not what changed here.
///
/// What changed is the **parsing**. This used to be `v.parse::<bool>()`, and
/// Rust's `FromStr for bool` accepts the exact strings `true` and `false` and
/// nothing else. So `MYCELIA_ENABLE_COREML=1` — the obvious way to turn
/// something on, and the spelling the CLI's own usage text advertised —
/// failed to parse and fell through to the off default. `1` was not the
/// inverse of `0`: both meant off, and the operator got the opposite of what
/// they asked for with no diagnostic.
///
/// It now shares [`mycelia_ocr_models::env_flag`] with the OCR path, so one
/// export means one thing everywhere in the process. See
/// `docs/adr/2026-09-21-coreml-env-gate-parsing.md`.
#[cfg(all(feature = "coreml", target_os = "macos"))]
fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" | "y" | "t" => true,
            "0" | "false" | "no" | "off" | "n" | "f" => false,
            _ => default,
        })
        .unwrap_or(default)
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
fn coreml_gate_passed(cfg: &SessionConfig) -> bool {
    if !cfg.coreml_auto_gate {
        return true;
    }
    env_flag("MYCELIA_ENABLE_COREML", false)
}

/// Build a session per [`SessionConfig`]. This is the unified entry point.
///
/// Behaviour:
/// 1. Resolve `.ort` sibling if present.
/// 2. Build a base [`SessionBuilder`](ort::session::builder::SessionBuilder)
///    with the configured optimisation level and intra-op threads.
/// 3. Register the chosen execution provider chain. If registration fails,
///    rebuild a fresh CPU-only base builder and continue.
/// 4. Commit the model and return `(Session, SessionInfo)`.
pub fn build_session(
    model_path: &str,
    cfg: &SessionConfig,
) -> Result<(Session, SessionInfo), OrtUtilError> {
    let (actual, used_ort) = prefer_ort_format(model_path);
    if !Path::new(&actual).exists() {
        return Err(OrtUtilError::NotFound(PathBuf::from(actual)));
    }

    let make_base = |threads: usize, opt: GraphOptimizationLevel| {
        Session::builder()
            .map_err(|e| OrtUtilError::Builder(e.to_string()))?
            .with_optimization_level(opt)
            .map_err(|e| OrtUtilError::Builder(e.to_string()))?
            .with_intra_threads(threads.max(1))
            .map_err(|e| OrtUtilError::Builder(e.to_string()))
    };

    #[allow(unused_mut)]
    let mut builder = make_base(cfg.intra_threads, cfg.opt_level)?;
    #[allow(unused_mut, unused_assignments)]
    let mut provider: &'static str = "cpu";

    // Helper to register a provider chain with CPU fallback on failure.
    #[allow(unused_macros)]
    macro_rules! try_register {
        ($eps:expr, $name:literal) => {{
            match builder.with_execution_providers($eps) {
                Ok(updated) => {
                    builder = updated;
                    provider = $name;
                }
                Err(e) => {
                    tracing::warn!(
                        ep = $name,
                        error = %e,
                        "EP registration failed; falling back to CPU"
                    );
                    builder = make_base(cfg.intra_threads, cfg.opt_level)?;
                }
            }
        }};
    }

    match cfg.ep {
        EpPref::Cpu => { /* nothing to register */ }

        EpPref::Auto => {
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            {
                if coreml_gate_passed(cfg) {
                    use ort::execution_providers::coreml::ModelFormat as CoreMLModelFormat;
                    use ort::execution_providers::CoreML as CoreMLEp;
                    use ort::execution_providers::ExecutionProviderDispatch;

                    let cache_dir =
                        resolve_coreml_cache_dir(&actual, cfg.coreml_cache_dir.as_deref());
                    let _ = std::fs::create_dir_all(&cache_dir);
                    let ep = CoreMLEp::default()
                        .with_model_cache_dir(cache_dir.to_string_lossy().as_ref())
                        .with_model_format(CoreMLModelFormat::MLProgram)
                        .build();
                    let providers: Vec<ExecutionProviderDispatch> = vec![ep];
                    try_register!(providers, "coreml");
                } else {
                    tracing::debug!(
                        "Auto EP: CoreML disabled by MYCELIA_ENABLE_COREML gate; using CPU"
                    );
                }
            }

            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            {
                use ort::execution_providers::{CUDAExecutionProvider, ExecutionProviderDispatch};
                let providers: Vec<ExecutionProviderDispatch> =
                    vec![CUDAExecutionProvider::default().build()];
                try_register!(providers, "cuda");
            }
        }

        EpPref::CoreML => {
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            {
                use ort::execution_providers::coreml::ModelFormat as CoreMLModelFormat;
                use ort::execution_providers::CoreML as CoreMLEp;
                use ort::execution_providers::ExecutionProviderDispatch;

                let cache_dir = resolve_coreml_cache_dir(&actual, cfg.coreml_cache_dir.as_deref());
                let _ = std::fs::create_dir_all(&cache_dir);
                let ep = CoreMLEp::default()
                    .with_model_cache_dir(cache_dir.to_string_lossy().as_ref())
                    .with_model_format(CoreMLModelFormat::MLProgram)
                    .build();
                let providers: Vec<ExecutionProviderDispatch> = vec![ep];
                try_register!(providers, "coreml");
            }
            #[cfg(not(all(feature = "coreml", target_os = "macos")))]
            {
                tracing::debug!(
                    "EpPref::CoreML requested but unavailable on this build/platform; using CPU"
                );
            }
        }

        EpPref::Cuda => {
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            {
                use ort::execution_providers::{CUDAExecutionProvider, ExecutionProviderDispatch};
                let providers: Vec<ExecutionProviderDispatch> =
                    vec![CUDAExecutionProvider::default().build()];
                try_register!(providers, "cuda");
            }
            #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
            {
                tracing::debug!(
                    "EpPref::Cuda requested but unavailable on this build/platform; using CPU"
                );
            }
        }
    }

    let size_mb = std::fs::metadata(&actual)
        .map(|m| m.len() as f64 / 1_048_576.0)
        .unwrap_or(0.0);

    let session = builder
        .commit_from_file(&actual)
        .map_err(|e| OrtUtilError::Commit {
            path: actual.clone(),
            message: e.to_string(),
        })?;

    Ok((
        session,
        SessionInfo {
            model_path: PathBuf::from(actual),
            size_mb,
            provider,
            used_ort_format: used_ort,
        },
    ))
}

/// Convenience: CPU-only session with `Level3` optimisation.
pub fn build_session_cpu(
    model_path: &str,
    intra_threads: usize,
) -> Result<(Session, SessionInfo), OrtUtilError> {
    let cfg = SessionConfig::cpu().with_threads(intra_threads);
    build_session(model_path, &cfg)
}

/// Convenience: legacy two-arg API mapping `EpPref::{Auto, Cpu}` only.
/// Prefer [`build_session`] with a full [`SessionConfig`] for new code.
pub fn build_session_auto(
    model_path: &str,
    intra_threads: usize,
    pref: EpPref,
) -> Result<(Session, SessionInfo), OrtUtilError> {
    let cfg = SessionConfig::default()
        .with_threads(intra_threads)
        .with_ep(pref);
    build_session(model_path, &cfg)
}

/// Probe the embedding dimension from a session's output's last (innermost)
/// dimension. Returns `None` if the model has no static last dim (dynamic
/// shape). Callers should treat `None` as "unknown — verify externally".
pub fn probe_output_last_dim(session: &Session, output_name: Option<&str>) -> Option<usize> {
    let outputs = session.outputs();
    let target = match output_name {
        Some(name) => outputs.iter().find(|o| o.name() == name)?,
        None => outputs.first()?,
    };
    match target.dtype() {
        ValueType::Tensor { shape, .. } => {
            // `shape` derefs to `[i64]` in ort rc.12; the last entry is the
            // innermost dim. A non-positive value means dynamic.
            let len = shape.len();
            if len == 0 {
                return None;
            }
            let last = shape.get(len - 1).copied()?;
            if last > 0 {
                Some(last as usize)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// On macOS, try common Homebrew install paths for `libonnxruntime.dylib`
/// when `ORT_DYLIB_PATH` is unset. No-op on other platforms or when the
/// variable is already set.
pub fn ensure_ort_dylib_path() {
    const ENV_KEY: &str = "ORT_DYLIB_PATH";
    if std::env::var_os(ENV_KEY).is_some() {}
    #[cfg(target_os = "macos")]
    for candidate in [
        "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib",
        "/opt/homebrew/lib/libonnxruntime.dylib",
        "/usr/local/lib/libonnxruntime.dylib",
    ] {
        if Path::new(candidate).exists() {
            std::env::set_var(ENV_KEY, candidate);
            return;
        }
    }
}
