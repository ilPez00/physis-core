//! Human and JSON presentations of the same workspace service.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand};
use serde_json::{json, Value};

use crate::system::{PackStrategy, Workspace, API_VERSION};

#[derive(Debug, Args)]
pub struct SystemArgs {
    /// Filesystem workspace to inspect; also the working directory for run.
    #[arg(long, default_value = ".", global = true)]
    root: PathBuf,
    /// Core observation store. Default: PHYSIS_CORE_DIR or ROOT/.physis/system.
    #[arg(long, global = true)]
    state: Option<PathBuf>,
    /// Emit one versioned JSON response, including structured errors.
    #[arg(long, global = true)]
    json: bool,
    /// Stop excluding this build/vendor directory name (repeatable). A
    /// published package is mostly `dist/`, so the defaults hide the thing
    /// being inspected: `--include-dir dist`.
    #[arg(long = "include-dir", global = true, value_name = "NAME")]
    include_dir: Vec<String>,
    #[command(subcommand)]
    command: SystemCommand,
}

#[derive(Debug, Subcommand)]
enum SystemCommand {
    /// Discover operations, state location, representations, and limits.
    Capabilities,
    /// Inspect workspace files and recent recorded work.
    Inspect,
    /// List addressable files (hidden/build/vendor directories excluded).
    List {
        #[arg(long, default_value_t = 100, value_parser = clap::value_parser!(u32).range(1..=5000))]
        limit: u32,
    },
    /// Find file content and remembered work through Physis BM25.
    Find {
        query: String,
        #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u32).range(1..=1000))]
        limit: u32,
    },
    /// Pack a token-budgeted context bundle: the lines themselves, not pointers.
    Pack {
        query: String,
        #[arg(long, default_value_t = 2000, value_parser = clap::value_parser!(u32).range(50..=200_000))]
        budget: u32,
        /// Windows one file may contribute, so a single long file cannot fill the budget.
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=100))]
        per_file: u32,
        /// Ranking stage. `files-then-windows` shortlists files first; `windows`
        /// ranks every window directly. Kept selectable so the two stay comparable.
        #[arg(long, default_value = "files-then-windows", value_parser = ["files-then-windows", "windows"])]
        strategy: String,
        /// Skip the neighbour window that closes a one-window gap inside a file.
        #[arg(long, default_value_t = false)]
        no_bridge: bool,
    },
    /// What this workspace's log says about how this kind of action has gone.
    Predict {
        /// The argv you would run, after `--`.
        #[arg(last = true, required = true)]
        argv: Vec<String>,
    },
    /// Read a relative path, file:<id>, or obs:<sequence>. Append :start-end for one region.
    Read {
        target: String,
        #[arg(long, default_value_t = 32768, value_parser = clap::value_parser!(u32).range(1..=1048576))]
        max_bytes: u32,
    },
    /// Recall this workspace's notes and executions, newest first.
    History {
        query: Option<String>,
        #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..=1000))]
        limit: u32,
    },
    /// Record a development note; does not alter classification quality.
    Remember {
        text: String,
        #[arg(long, default_value = "unverified", value_parser = ["unverified", "success", "failure", "inconclusive"])]
        outcome: String,
        #[arg(long, default_value = "operator")]
        actor: String,
    },
    /// Run explicit argv in the workspace, recording intent, output, and exit.
    /// This is ordinary process execution, not a sandbox or machine-safety gate.
    Run {
        #[arg(long)]
        intent: String,
        #[arg(long, default_value = "operator")]
        actor: String,
        #[arg(last = true, required = true, num_args = 1..)]
        argv: Vec<String>,
    },
    /// Export references/history to a NEW ordinary directory (a snapshot).
    Export { destination: PathBuf },
}

impl SystemCommand {
    fn name(&self) -> &'static str {
        match self {
            Self::Capabilities => "capabilities",
            Self::Inspect => "inspect",
            Self::List { .. } => "list",
            Self::Find { .. } => "find",
            Self::Pack { .. } => "pack",
            Self::Predict { .. } => "predict",
            Self::Read { .. } => "read",
            Self::History { .. } => "history",
            Self::Remember { .. } => "remember",
            Self::Run { .. } => "run",
            Self::Export { .. } => "export",
        }
    }
}

impl SystemArgs {
    pub fn run(&self) -> Result<()> {
        let operation = self.command.name();
        let result = (|| -> Result<(Workspace, Value)> {
            let root = self.root.canonicalize()?;
            let state = self
                .state
                .clone()
                .or_else(|| {
                    std::env::var_os("PHYSIS_CORE_DIR")
                        .filter(|v| !v.is_empty())
                        .map(PathBuf::from)
                })
                .unwrap_or_else(|| root.join(".physis/system"));
            let workspace = Workspace::open(&root, &state)?.including(&self.include_dir);
            let data = match &self.command {
                SystemCommand::Capabilities => workspace.capabilities(),
                SystemCommand::Inspect => workspace.inspect()?,
                SystemCommand::List { limit } => {
                    let mut inventory = workspace.inventory()?;
                    let total = inventory.objects.len();
                    inventory.truncated |= total > *limit as usize;
                    inventory.objects.truncate(*limit as usize);
                    json!({"inventory":inventory, "files_enumerated":total})
                }
                SystemCommand::Find { query, limit } => workspace.find(query, *limit as usize)?,
                SystemCommand::Pack {
                    query,
                    budget,
                    per_file,
                    strategy,
                    no_bridge,
                } => workspace.pack_with(
                    query,
                    *budget as usize,
                    *per_file as usize,
                    match strategy.as_str() {
                        "windows" => PackStrategy::Windows,
                        _ => PackStrategy::FilesThenWindows,
                    },
                    !no_bridge,
                )?,
                SystemCommand::Predict { argv } => workspace.predict(argv)?,
                SystemCommand::Read { target, max_bytes } => {
                    workspace.read(target, *max_bytes as usize)?
                }
                SystemCommand::History { query, limit } => {
                    workspace.history(query.as_deref(), *limit as usize)?
                }
                SystemCommand::Remember {
                    text,
                    outcome,
                    actor,
                } => workspace.remember(text, outcome, actor)?,
                SystemCommand::Run {
                    argv,
                    intent,
                    actor,
                } => workspace.run(argv, intent, actor)?,
                SystemCommand::Export { destination } => workspace.export(destination)?,
            };
            Ok((workspace, data))
        })();
        match result {
            Ok((workspace, data)) => {
                let failed = operation == "run" && data["result"]["process_success"] == false;
                if self.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({"api_version":API_VERSION,
                        "operation":operation, "workspace":workspace.root, "state":workspace.state,
                        "ok":!failed, "data":data, "error":Value::Null}))?
                    );
                } else {
                    render(operation, &data);
                }
                if failed {
                    // Preserve a failed child's exit status for scripts/agents.
                    let code = data["result"]["exit_code"].as_i64().unwrap_or(1);
                    std::process::exit(if (1..=255).contains(&code) {
                        code as i32
                    } else {
                        1
                    });
                }
                Ok(())
            }
            Err(error) => {
                if self.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({"api_version":API_VERSION,
                        "operation":operation, "workspace":self.root, "ok":false,
                        "data":Value::Null, "error":{"message":format!("{error:#}")}}))?
                    );
                }
                Err(error)
            }
        }
    }
}

fn render(operation: &str, data: &Value) {
    match operation {
        "capabilities" => {
            println!(
                "Physis workspace interface ({})",
                data["api_version"].as_str().unwrap_or("")
            );
            println!(
                "Workspace: {}\nState: {}",
                data["workspace"].as_str().unwrap_or(""),
                data["state"].as_str().unwrap_or("")
            );
            for op in data["operations"].as_array().into_iter().flatten() {
                println!(
                    "  {:<13} {:<5} {}",
                    op["name"].as_str().unwrap_or(""),
                    if op["writes"] == true {
                        "write"
                    } else {
                        "read"
                    },
                    op["purpose"].as_str().unwrap_or("")
                );
            }
            println!("\nAdd --json for the versioned contract and all limits.");
        }
        "inspect" => {
            println!(
                "Workspace: {}",
                data["capabilities"]["workspace"].as_str().unwrap_or("")
            );
            println!(
                "{} files · {} workspace observations in the last {} log records",
                data["files"], data["workspace_records_in_window"], data["history_window"]
            );
            println!(
                "Inventory truncated: {} · skipped symlinks: {} · scan errors: {}",
                data["inventory_truncated"],
                data["skipped_links"],
                data["scan_errors"].as_array().map_or(0, Vec::len)
            );
            println!(
                "Runs without a recorded finish: {}",
                data["runs_without_recorded_finish_in_window"]
            );
            println!("State: {}\nUse list, find, read, or history; --json includes the full inventory summary.",
                data["capabilities"]["state"].as_str().unwrap_or(""));
        }
        "export" => {
            println!(
                "Created reference snapshot: {}",
                data["directory"].as_str().unwrap_or("")
            );
            println!(
                "{} file references · {} observations · live: false",
                data["manifest"]["files"], data["manifest"]["observations"]
            );
        }
        "read" => {
            if let Some(text) = data["text"].as_str() {
                let range = match (data["lines"]["start_line"].as_u64(), data["lines"]["end_line"].as_u64()) {
                    (Some(a), Some(b)) => format!(":{a}-{b}"),
                    _ => String::new(),
                };
                println!(
                    "{}  {}{range}",
                    short_id(data["object"]["id"].as_str().unwrap_or("")),
                    data["object"]["path"].as_str().unwrap_or("")
                );
                println!("{text}");
                if data["truncated"] == true {
                    println!("[partial read; raise --max-bytes for more]");
                }
            } else {
                println!("{}", serde_json::to_string_pretty(data).unwrap());
            }
        }
        "pack" => {
            println!(
                "packed {} of {} budget tokens · {} strategy · {} windows ranked, {} skipped as too large · {} files",
                data["used_tokens"],
                data["budget_tokens"],
                data["strategy"].as_str().unwrap_or(""),
                data["windows_ranked"],
                data["windows_over_budget"],
                data["files_scanned"]
            );
            for chunk in data["chunks"].as_array().into_iter().flatten() {
                println!(
                    "--- {}:{}-{}",
                    chunk["path"].as_str().unwrap_or(""),
                    chunk["start_line"],
                    chunk["end_line"]
                );
                println!("{}", chunk["text"].as_str().unwrap_or(""));
            }
            println!(
                "budget counted by {}; re-read a region with `system read path:start-end`",
                data["token_counter"].as_str().unwrap_or("")
            );
        }
        "predict" => {
            if data["status"] == "NOT MEASURED" {
                println!(
                    "NOT MEASURED · {} · {}",
                    data["kind"].as_str().unwrap_or(""),
                    data["reason"].as_str().unwrap_or("")
                );
            } else {
                println!(
                    "{}  failure probability {:.2}  ({} of {} run(s) of this kind failed; \
workspace rate {:.2} over {} run(s))",
                    data["kind"].as_str().unwrap_or(""),
                    data["failure_probability"].as_f64().unwrap_or(0.0),
                    data["failures_of_this_kind"],
                    data["runs_of_this_kind"],
                    data["workspace_failure_rate"].as_f64().unwrap_or(0.0),
                    data["runs_total"]
                );
                for entry in data["recent"].as_array().into_iter().flatten() {
                    println!(
                        "  obs:{}  {}  {}",
                        entry["seq"],
                        if entry["failed"] == true { "failed " } else { "ok     " },
                        entry["argv"]
                            .as_array()
                            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(" "))
                            .unwrap_or_default()
                    );
                }
                println!("{}", data["not_claimed"].as_str().unwrap_or(""));
            }
        }
        "find" => {
            println!(
                "BM25 lexical search · {} documents · {} content bytes",
                data["documents_ranked"], data["content_bytes"]
            );
            for hit in data["hits"].as_array().into_iter().flatten() {
                let label = hit["label"].as_str().unwrap_or("");
                let excerpt = hit["excerpt"].as_str().unwrap_or("");
                println!("{}  {label}", short_id(hit["id"].as_str().unwrap_or("")));
                // The first matching line is often the path itself, and printing
                // the same string twice is pure cost.
                if !excerpt.trim().is_empty() && excerpt.trim() != label.trim() {
                    println!("  {excerpt}");
                }
            }
            println!(
                "{} files without complete indexed text; inventory truncated: {}",
                data["files_without_complete_text"], data["inventory_truncated"]
            );
        }
        "list" => {
            for object in data["inventory"]["objects"]
                .as_array()
                .into_iter()
                .flatten()
            {
                println!(
                    "{}  {}",
                    short_id(object["id"].as_str().unwrap_or("")),
                    object["path"].as_str().unwrap_or("")
                );
            }
            println!(
                "{} files enumerated; output truncated: {}",
                data["files_enumerated"], data["inventory"]["truncated"]
            );
        }
        "history" => {
            for record in data["records"].as_array().into_iter().flatten() {
                let subject = record["subject"].as_str().unwrap_or("");
                println!(
                    "obs:{}  {}  {subject}",
                    record["seq"],
                    record["source"].as_str().unwrap_or(""),
                );
                // The payload of a note repeats its subject verbatim, so
                // printing both doubled the cost of every recall. Print the
                // fields the subject does not carry, and only those.
                let body: Value =
                    serde_json::from_str(record["body"].as_str().unwrap_or("")).unwrap_or(Value::Null);
                match &body {
                    Value::Object(map) => {
                        let extras: Vec<String> = map
                            .iter()
                            .filter(|(key, value)| {
                                key.as_str() != "workspace"
                                    && value.as_str() != Some(subject)
                                    && !matches!(value, Value::Null)
                            })
                            .map(|(key, value)| match value.as_str() {
                                Some(text) if text.len() > 160 => {
                                    format!("{key}={}…", &text[..160])
                                }
                                Some(text) => format!("{key}={text}"),
                                None => format!("{key}={value}"),
                            })
                            .collect();
                        if !extras.is_empty() {
                            println!("  {}", extras.join(" · "));
                        }
                    }
                    // A legacy `fs` observation stores a content hash, not JSON.
                    _ => {
                        let raw = record["body"].as_str().unwrap_or("");
                        if !raw.is_empty() && raw != subject {
                            println!("  {raw}");
                        }
                    }
                }
            }
            println!(
                "{} matches in the last {} global records",
                data["matched_in_window"], data["history_window"]
            );
        }
        "run" => {
            println!(
                "run {} · exit {} · {} ms\nintent {} → outcome {}",
                data["id"].as_str().unwrap_or(""),
                data["result"]["exit_code"],
                data["result"]["duration_ms"],
                data["start_id"].as_str().unwrap_or(""),
                data["finish_id"].as_str().unwrap_or("")
            );
            print!(
                "{}{}",
                data["stdout_tail"].as_str().unwrap_or(""),
                data["stderr_tail"].as_str().unwrap_or("")
            );
            println!(
                "\nlogs: {}\n{}",
                data["result"]["stdout"].as_str().unwrap_or(""),
                data["result"]["stderr"].as_str().unwrap_or("")
            );
        }
        "remember" => println!(
            "Remembered {}: {}",
            data["id"].as_str().unwrap_or(""),
            data["record"]["subject"].as_str().unwrap_or("")
        ),
        _ => println!("{}", serde_json::to_string_pretty(data).unwrap()),
    }
}

/// Displays print an abbreviated file ID: the full SHA-256 is 43 tokens for a
/// caller that has to read it back, and `read` resolves any unambiguous prefix.
/// `--json` still carries the full ID.
fn short_id(id: &str) -> String {
    match id.strip_prefix("file:") {
        Some(hash) if hash.len() > 12 => format!("file:{}", &hash[..12]),
        _ => id.to_string(),
    }
}
