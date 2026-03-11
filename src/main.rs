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
                            Ok((true, summary)) => {
                                // Update history with concise summary instead of full JSON
                                let last_idx = history.len() - 1;
                                history[last_idx] =
                                    json!({"role": "assistant", "content": summary});

                                // Reload context
                                let (new_context, file_count) = build_system_prompt(&current_dir);
                                history[0] = json!({"role": "system", "content": new_context});
                                println!(
                                    "✅ Reloaded {} files into context after edits.",
                                    file_count
                                );
                            }
                            Ok((false, summary)) => {
                                // Update history with concise summary even if no files changed
                                let last_idx = history.len() - 1;
                                history[last_idx] =
                                    json!({"role": "assistant", "content": summary});
                            }
                            Err(e) => {
                                eprintln!("Error applying diffs: {}", e);
                                let error_summary = format!("Error applying edits: {}", e);
                                let last_idx = history.len() - 1;
                                history[last_idx] =
                                    json!({"role": "assistant", "content": error_summary});
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

// Finds all fuzzy matches of a search string in the original content.
// Returns a list of (start_byte, end_byte) tuples indicating the byte range of the match.
// It ignores differences in line endings and trailing whitespace.
fn find_all_fuzzy_matches(original: &str, search: &str) -> Vec<(usize, usize)> {
    if search.is_empty() {
        return vec![];
    }

    let search_lines: Vec<&str> = search.split_inclusive('\n').map(|l| l.trim_end()).collect();
    if search_lines.is_empty() {
        return vec![];
    }

    let mut orig_lines = Vec::new();
    let mut current_start = 0;

    for line_raw in original.split_inclusive('\n') {
        let end = current_start + line_raw.len();
        let trimmed = line_raw.trim_end();
        orig_lines.push((trimmed, current_start, end));
        current_start = end;
    }

    let mut matches = Vec::new();
    let mut i = 0;
    while i + search_lines.len() <= orig_lines.len() {
        let mut is_match = true;
        for j in 0..search_lines.len() {
            if orig_lines[i + j].0 != search_lines[j] {
                is_match = false;
                break;
            }
        }

        if is_match {
            let start_byte = orig_lines[i].1;
            let end_byte = orig_lines[i + search_lines.len() - 1].2;
            matches.push((start_byte, end_byte));
            i += search_lines.len(); // Skip the matched lines to avoid overlapping matches
        } else {
            i += 1;
        }
    }

    matches
}

// Applies diffs found in the structured LLM response
// Returns a tuple of (bool, String) where the boolean indicates if files were changed,
// and the string contains a concise summary of the edits applied or failed.
fn apply_diffs(response: &str, current_dir: &Path) -> Result<(bool, String)> {
    let mut files_changed = false;
    let mut summary = String::new();

    // The response string should be a JSON object conforming to `EditResponse`
    let edit_response: EditResponse = match serde_json::from_str(response) {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("Error parsing JSON response: {}", e);
            return Ok((false, format!("Error parsing JSON response: {}", e)));
        }
    };

    let canonical_current_dir = match current_dir.canonicalize() {
        Ok(path) => path,
        Err(e) => anyhow::bail!("Failed to canonicalize current directory: {}", e),
    };

    for edit in edit_response.edits {
        let filepath = &edit.filepath;
        let search_block = &edit.search;
        let replace_block = &edit.replace;

        let full_path = current_dir.join(filepath);
        if !full_path.exists() {
            eprintln!("Warning: File not found: {}", filepath);
            summary.push_str(&format!("Failed to edit {}: File not found.\n", filepath));
            continue;
        }

        let canonical_full_path = match full_path.canonicalize() {
            Ok(path) => path,
            Err(e) => anyhow::bail!("Failed to canonicalize path {}: {}", filepath, e),
        };

        if !canonical_full_path.starts_with(&canonical_current_dir) {
            anyhow::bail!(
                "Security Error: Attempted path traversal detected for file: {}",
                filepath
            );
        }

        let original_content = match fs::read_to_string(&canonical_full_path) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Error reading file {}: {}", filepath, e);
                summary.push_str(&format!("Failed to read {}: {}\n", filepath, e));
                continue;
            }
        };

        let new_content = if original_content.contains(search_block) {
            // Exact match
            original_content.replace(search_block, replace_block)
        } else {
            // Fallback: fuzzy match (ignore line endings and trailing whitespace)
            let matches = find_all_fuzzy_matches(&original_content, search_block);
            if matches.is_empty() {
                eprintln!(
                    "Warning: Search block not found in {} (even with fuzzy matching)",
                    filepath
                );
                summary.push_str(&format!(
                    "Failed to edit {}: Search block not found.\n",
                    filepath
                ));
                continue;
            }

            // Replace all fuzzy matches from right to left to avoid invalidating indices
            let mut modified_content = original_content.clone();
            for (start, end) in matches.into_iter().rev() {
                modified_content.replace_range(start..end, replace_block);
            }
            modified_content
        };

        if original_content != new_content {
            match fs::write(&canonical_full_path, new_content) {
                Ok(_) => {
                    println!("✅ Applied edit to {}", filepath);
                    summary.push_str(&format!("Successfully edited {}.\n", filepath));
                    files_changed = true;
                }
                Err(e) => {
                    eprintln!("Error writing to file {}: {}", filepath, e);
                    summary.push_str(&format!("Failed to write {}: {}\n", filepath, e));
                }
            }
        } else {
            println!("ℹ️ No changes needed for {}", filepath);
            summary.push_str(&format!("No changes needed for {}.\n", filepath));
        }
    }

    if summary.is_empty() {
        summary.push_str("No edits requested.\n");
    }

    Ok((files_changed, summary.trim_end().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_all_fuzzy_matches() {
        let original = "line1\r\n  line2  \nline3\nline4\r\n";
        let search = "  line2\nline3\r\n";
        let matches = find_all_fuzzy_matches(original, search);
        assert_eq!(matches, vec![(7, 23)]);

        let mut modified = original.to_string();
        for (start, end) in matches.into_iter().rev() {
            modified.replace_range(start..end, "replaced\n");
        }
        assert_eq!(modified, "line1\r\nreplaced\nline4\r\n");
    }
}
