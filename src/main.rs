use anyhow::{Context, Result};
use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use walkdir::WalkDir;

#[derive(Debug, Deserialize, Serialize)]
struct FileEdit {
    filepath: String,
    search: String,
    replace: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct EditResponse {
    edits: Vec<FileEdit>,
}

/// Make edits to a local folder using LM Studio backed API
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The API URL to use for requests
    #[arg(long, default_value = "http://localhost:1234/v1/chat/completions")]
    api_url: String,

    /// The model to use
    #[arg(long, default_value = "local-model")]
    model: String,
}

fn build_system_prompt(current_dir: &std::path::Path) -> (String, usize) {
    let mut context = String::from(
        "You are an expert developer. You provide code edits by responding with a structured JSON object according to the schema provided.\nEnsure that the code in the search block matches the file exactly. Do not truncate or omit parts of the block.\n\nCurrent codebase context:\n\n",
    );

    let mut file_count = 0;
    for entry in WalkDir::new(current_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        let path_str = path.to_string_lossy();

        // Skip common large/binary/build folders
        if path_str.contains(".git")
            || path_str.contains("target")
            || path_str.contains("node_modules")
        {
            continue;
        }

        // read_to_string natively fails on non-UTF8 files (skipping binaries/images)
        if let Ok(content) = fs::read_to_string(path) {
            let rel_path = path.strip_prefix(current_dir).unwrap_or(path);
            context.push_str(&format!(
                "--- FILE: {} ---\n{}\n\n",
                rel_path.display(),
                content
            ));
            file_count += 1;
        }
    }

    (context, file_count)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let current_dir = env::current_dir().context("Failed to get current directory")?;

    let (context, file_count) = build_system_prompt(&current_dir);

    println!(
        "Loaded {} files into context. Context size: {} bytes.",
        file_count,
        context.len()
    );

    let client = Client::new();

    // Initialize conversational history
    let mut history = vec![json!({"role": "system", "content": context})];

    // 2. REPL Loop
    loop {
        print!("\nlmcli> ");
        io::stdout().flush().context("Failed to flush stdout")?;

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .context("Failed to read from stdin")?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            break;
        }
        if input.is_empty() {
            continue;
        }

        history.push(json!({"role": "user", "content": input}));

        let payload = json!({
            "model": args.model,
            "messages": history,
            "temperature": 0.1, // Low temperature forces deterministic, factual code edits
            "stream": false,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "edit_response",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "edits": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "filepath": {"type": "string"},
                                        "search": {"type": "string"},
                                        "replace": {"type": "string"}
                                    },
                                    "required": ["filepath", "search", "replace"],
                                    "additionalProperties": false
                                }
                            }
                        },
                        "required": ["edits"],
                        "additionalProperties": false
                    }
                }
            }
        });

        let res = client.post(&args.api_url).json(&payload).send().await;

        match res {
            Ok(response) => {
                let status = response.status();
                let body: serde_json::Value = response.json().await.unwrap_or(json!({}));

                if status.is_success() {
                    if let Some(reply) = body["choices"][0]["message"]["content"].as_str() {
                        println!("\n{}", reply);
                        history.push(json!({"role": "assistant", "content": reply}));

                        // Apply diffs if found
                        match apply_diffs(reply, &current_dir) {
                            Ok(true) => {
                                // Reload context
                                let (new_context, file_count) = build_system_prompt(&current_dir);
                                history[0] = json!({"role": "system", "content": new_context});
                                println!(
                                    "✅ Reloaded {} files into context after edits.",
                                    file_count
                                );
                            }
                            Ok(false) => {
                                // No diffs applied
                            }
                            Err(e) => {
                                eprintln!("Error applying diffs: {}", e);
                            }
                        }
                    } else {
                        eprintln!("Error: Malformed response: {}", body);
                        // Revert the last user message to avoid poisoning the history vector with unanswered prompts
                        history.pop();
                    }
                } else {
                    eprintln!("API Error: Status {}. Response: {}", status, body);
                    history.pop();
                }
            }
            Err(e) => {
                eprintln!("Network Error: {}", e);
                history.pop();
            }
        }
    }

    Ok(())
}

// Helper function to normalize line endings and trim trailing whitespace
fn normalize_content(content: &str) -> String {
    let mut normalized = content
        .lines()
        .map(|line| line.trim_end())
        .collect::<Vec<_>>()
        .join("\n");

    if content.ends_with('\n') {
        normalized.push('\n');
    }

    normalized
}

// Applies diffs found in the structured LLM response
fn apply_diffs(response: &str, current_dir: &Path) -> Result<bool> {
    let mut files_changed = false;

    // The response string should be a JSON object conforming to `EditResponse`
    let edit_response: EditResponse = match serde_json::from_str(response) {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("Error parsing JSON response: {}", e);
            return Ok(false);
        }
    };

    for edit in edit_response.edits {
        let filepath = &edit.filepath;
        let search_block = &edit.search;
        let replace_block = &edit.replace;

        let full_path = current_dir.join(filepath);
        if !full_path.exists() {
            eprintln!("Warning: File not found: {}", filepath);
            continue;
        }

        let original_content = match fs::read_to_string(&full_path) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Error reading file {}: {}", filepath, e);
                continue;
            }
        };

        let new_content = if original_content.contains(search_block) {
            // Exact match
            original_content.replace(search_block, replace_block)
        } else {
            // Fallback: fuzzy match (normalize line endings and trailing whitespace)
            let norm_original = normalize_content(&original_content);
            let norm_search = normalize_content(search_block);

            if norm_original.contains(&norm_search) {
                // If fuzzy match works, we replace the fuzzy part.
                // Note: this may alter surrounding whitespace in the original file slightly.
                let norm_replace = normalize_content(replace_block);
                norm_original.replace(&norm_search, &norm_replace)
            } else {
                eprintln!(
                    "Warning: Search block not found in {} (even with fuzzy matching)",
                    filepath
                );
                continue;
            }
        };

        if original_content != new_content {
            match fs::write(&full_path, new_content) {
                Ok(_) => {
                    println!("✅ Applied edit to {}", filepath);
                    files_changed = true;
                }
                Err(e) => {
                    eprintln!("Error writing to file {}: {}", filepath, e);
                }
            }
        } else {
            println!("ℹ️ No changes needed for {}", filepath);
        }
    }

    Ok(files_changed)
}
