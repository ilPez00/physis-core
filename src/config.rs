//! XDG config — one file, three shells. Precedence: flag > env > file > default.
//! File: $XDG_CONFIG_HOME/physis/config.json or $HOME/.config/physis/config.json
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

fn default_workspace() -> String { "main".into() }
fn default_budget() -> usize { 1200 }
fn default_confidence() -> f32 { 0.35 }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Workspace { pub data_dir: Option<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderCfg {
    #[serde(default)] pub model_dir: Option<String>,
    #[serde(default = "default_true")] pub allow_lexical_fallback: bool,
}
impl Default for EmbedderCfg { fn default() -> Self { Self { model_dir: None, allow_lexical_fallback: true } } }
fn default_true() -> bool { true }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OracleCfg {
    #[serde(default)] pub url: Option<String>,
    #[serde(default)] pub model: Option<String>,
    #[serde(default)] pub key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskCfg {
    #[serde(default)] pub corpus: Option<String>,
    #[serde(default)] pub budget: Option<usize>,
    #[serde(default)] pub draft: Option<bool>,
    #[serde(default)] pub confidence: Option<f32>,
}
impl Default for AskCfg { fn default() -> Self { Self { corpus: None, budget: None, draft: None, confidence: None } } }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WatchCfg {
    #[serde(default)] pub sources: Option<Vec<String>>,
    #[serde(default)] pub max: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysisConfig {
    #[serde(default = "default_workspace")] pub workspace: String,
    #[serde(default)] pub workspaces: BTreeMap<String, Workspace>,
    #[serde(default)] pub embedder: EmbedderCfg,
    #[serde(default)] pub oracle: OracleCfg,
    #[serde(default)] pub ask: AskCfg,
    #[serde(default)] pub watch: WatchCfg,
}
impl Default for PhysisConfig {
    fn default() -> Self {
        let mut m = BTreeMap::new();
        m.insert("main".into(), Workspace { data_dir: None });
        Self { workspace: "main".into(), workspaces: m, embedder: EmbedderCfg::default(), oracle: OracleCfg::default(), ask: AskCfg::default(), watch: WatchCfg::default() }
    }
}

pub fn config_path() -> PathBuf {
    if let Some(x) = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(x).join("physis/config.json");
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".config/physis/config.json")
}

fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest).to_string_lossy().to_string();
        }
    }
    p.to_string()
}

pub fn load() -> PhysisConfig { load_from(&config_path()) }

pub fn load_from(path: &Path) -> PhysisConfig {
    let s = std::fs::read_to_string(path).unwrap_or_default();
    if s.trim().is_empty() { return PhysisConfig::default(); }
    serde_json::from_str(&s).unwrap_or_default()
}

pub fn save(cfg: &PhysisConfig) -> anyhow::Result<()> { save_to(&config_path(), cfg) }

pub fn save_to(path: &Path, cfg: &PhysisConfig) -> anyhow::Result<()> {
    if let Some(p) = path.parent() { std::fs::create_dir_all(p)?; }
    std::fs::write(path, serde_json::to_string_pretty(cfg)?)?;
    Ok(())
}

/// Resolve data_dir: env PHYSIS_CORE_DIR wins > active workspace data_dir > default
pub fn data_dir(cfg: &PhysisConfig) -> PathBuf {
    if let Some(dir) = std::env::var_os(crate::store::DATA_DIR_ENV).filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(ws) = cfg.workspaces.get(&cfg.workspace) {
        if let Some(d) = ws.data_dir.as_deref().filter(|s| !s.is_empty()) {
            return PathBuf::from(expand_tilde(d));
        }
    }
    crate::store::data_dir()
}

pub fn oracle_env_override(cfg: &PhysisConfig) -> Option<(crate::oracle::OracleConfig, String)> {
    // env wins
    if let Some(v) = crate::oracle::from_env() { return Some(v); }
    let url = cfg.oracle.url.clone().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| "https://openrouter.ai/api/v1".into());
    let model = cfg.oracle.model.clone().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| "openai/gpt-4o".into());
    let key = cfg.oracle.key.clone().filter(|s| !s.trim().is_empty())?;
    Some((crate::oracle::OracleConfig { url, model, has_key: true }, key))
}

pub fn ask_defaults(cfg: &PhysisConfig) -> (Option<String>, Option<usize>, Option<bool>, Option<f32>) {
    (cfg.ask.corpus.clone(), cfg.ask.budget, cfg.ask.draft, cfg.ask.confidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tmp() -> PathBuf { std::env::temp_dir().join(format!("physis-cfg-test-{}", std::process::id())) }
    #[test] fn missing_is_default() { assert_eq!(load_from(&PathBuf::from("/tmp/not-here-xyz123.json")).workspace, "main"); }
    #[test] fn roundtrip() {
        let p = tmp().join("roundtrip.json");
        let _ = std::fs::remove_file(&p);
        let mut cfg = PhysisConfig::default();
        cfg.oracle.url = Some("http://localhost:11434/v1".into());
        save_to(&p, &cfg).unwrap();
        let loaded = load_from(&p);
        assert_eq!(loaded.oracle.url, cfg.oracle.url);
        let _ = std::fs::remove_file(&p);
    }
    #[test] fn tilde_expand() {
        std::env::set_var("HOME", "/home/testuser");
        assert_eq!(expand_tilde("~/notes"), "/home/testuser/notes");
        assert_eq!(expand_tilde("/abs"), "/abs");
    }
    #[test] fn data_dir_env_wins() {
        std::env::set_var("PHYSIS_CORE_DIR", "/tmp/envwin");
        let cfg = PhysisConfig::default();
        assert_eq!(data_dir(&cfg), PathBuf::from("/tmp/envwin"));
        std::env::remove_var("PHYSIS_CORE_DIR");
    }
}
