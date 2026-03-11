use anyhow::{Context, Result};
use clap::Parser;
use ignore::WalkBuilder;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

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

// Assembles the initial system prompt by concatenating all valid UTF-8 files to provide codebase context, isolating file ingestion from the main REPL loop.
fn build_system_prompt(current_dir: &Path) -> (String, usize) {
    let initial_context = String::from(
        "You are an expert developer. You provide code edits by responding with a structured JSON object according to the schema provided.\nEnsure that the code in the search block matches the file exactly. Do not truncate or omit parts of the block.\n\nCurrent codebase context:\n\n",
    );

    WalkBuilder::new(current_dir)
        .build()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_some_and(|ft| ft.is_file()))
        .filter_map(|entry| {
            let path = entry.path();
            // read_to_string natively fails on non-UTF8 files (skipping binaries/images)
            fs::read_to_string(path).ok().map(|content| {
                let rel_path = path.strip_prefix(current_dir).unwrap_or(path);
                format!("--- FILE: {} ---\n{}\n\n", rel_path.display(), content)
            })
        })
        .fold((initial_context, 0), |(mut context, count), file_str| {
            context.push_str(&file_str);
            (context, count + 1)
        })
}

// Processes the LLM response, applies edits, and manages the history state accordingly.
async fn process_llm_interaction(
    client: &Client,
    args: &Args,
    current_dir: &Path,
    history: &mut Vec<serde_json::Value>,
) {
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

    let response = match client.post(&args.api_url).json(&payload).send().await {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Network Error: {}", e);
            history.pop();
            return;
        }
    };

    let status = response.status();
    let body: serde_json::Value = response.json().await.unwrap_or_default();

    if !status.is_success() {
        eprintln!("API Error: Status {}. Response: {}", status, body);
        history.pop();
        return;
    }

    let Some(reply) = body["choices"][0]["message"]["content"].as_str() else {
        eprintln!("Error: Malformed response: {}", body);
        // Revert the last user message to avoid poisoning the history vector with unanswered prompts
        history.pop();
        return;
    };

    println!("\n{}", reply);
    history.push(json!({"role": "assistant", "content": reply}));

    // Apply diffs if found
    match apply_diffs(reply, current_dir) {
        Ok((true, summary)) => {
            // Update history with concise summary instead of full JSON
            if let Some(last) = history.last_mut() {
                *last = json!({"role": "assistant", "content": summary});
            }

            // Reload context
            let (new_context, file_count) = build_system_prompt(current_dir);
            history[0] = json!({"role": "system", "content": new_context});
            println!("✅ Reloaded {} files into context after edits.", file_count);
        }
        Ok((false, summary)) => {
            // Update history with concise summary even if no files changed
            if let Some(last) = history.last_mut() {
                *last = json!({"role": "assistant", "content": summary});
            }
        }
        Err(e) => {
            eprintln!("Error applying diffs: {}", e);
            let error_summary = format!("Error applying edits: {}", e);
            if let Some(last) = history.last_mut() {
                *last = json!({"role": "assistant", "content": error_summary});
            }
        }
    }
}

// Entrypoint for the application, initializing the CLI, managing conversational history state, and orchestrating the REPL loop with the LM API.
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
        process_llm_interaction(&client, &args, &current_dir, &mut history).await;
    }

    Ok(())
}

// Finds byte ranges of search strings in file content, ignoring whitespace and line-ending differences, providing a fallback for imperfect LLM outputs.
fn find_all_fuzzy_matches(original: &str, search: &str) -> Vec<(usize, usize)> {
    if search.is_empty() {
        return vec![];
    }

    let search_lines: Vec<&str> = search.lines().map(str::trim_end).collect();
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
    let mut skip_until = 0;
    let window_size = search_lines.len();

    for (i, window) in orig_lines.windows(window_size).enumerate() {
        if i < skip_until {
            continue;
        }

        let is_match = window
            .iter()
            .zip(&search_lines)
            .all(|((orig, _, _), search)| orig == search);

        if is_match {
            let start_byte = window[0].1;
            let end_byte = window[window_size - 1].2;
            matches.push((start_byte, end_byte));
            skip_until = i + window_size; // Skip the matched lines to avoid overlapping matches
        }
    }

    matches
}

// Pre-processes LLM responses to strip markdown formatting, cleanly isolating raw JSON payloads prior to structured deserialization.
#[allow(clippy::collapsible_if)]
fn strip_markdown_code_blocks(response: &str) -> &str {
    let start = response.find(['{', '[']);
    let end = response.rfind(['}', ']']).map(|i| i + 1);

    if let (Some(s), Some(e)) = (start, end) {
        if s < e {
            return &response[s..e];
        }
    }

    // Fallback for empty blocks or unparseable JSON
    let mut stripped = response.trim();
    if let Some(s) = stripped.strip_prefix("```") {
        stripped = s.trim_start();
        if let Some(first_line_end) = stripped.find('\n') {
            // Strip any language identifiers (e.g., `json`) on the first line
            // only if the line doesn't already contain JSON structures like { or [
            if !stripped[..first_line_end].contains(['{', '[']) {
                stripped = stripped[first_line_end + 1..].trim_start();
            }
        } else {
            // Block is entirely empty or a single line without newlines after ````
            stripped = "";
        }
    }

    if let Some(s) = stripped.strip_suffix("```") {
        stripped = s.trim_end();
    }

    stripped
}

// Applies JSON-structured search-and-replace edits to local files securely, returning a compact LLM-friendly success or failure summary.
fn apply_diffs(response: &str, current_dir: &Path) -> Result<(bool, String)> {
    let mut files_changed = false;
    let mut summary = String::new();

    let cleaned_response = strip_markdown_code_blocks(response);

    // The response string should be a JSON object conforming to `EditResponse`
    let edit_response: EditResponse = match serde_json::from_str(cleaned_response) {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("Error parsing JSON response: {}", e);
            return Ok((false, format!("Error parsing JSON response: {}", e)));
        }
    };

    let canonical_current_dir = current_dir
        .canonicalize()
        .context("Failed to canonicalize current directory")?;

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

        let canonical_full_path = full_path
            .canonicalize()
            .with_context(|| format!("Failed to canonicalize path {}", filepath))?;

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
    fn test_strip_markdown_code_blocks() {
        // Plain JSON
        let plain = "{\"foo\": \"bar\"}";
        assert_eq!(strip_markdown_code_blocks(plain), "{\"foo\": \"bar\"}");

        // Wrapped with ```json
        let wrapped = "```json\n{\"foo\": \"bar\"}\n```";
        assert_eq!(strip_markdown_code_blocks(wrapped), "{\"foo\": \"bar\"}");

        // Wrapped with upper case JSON
        let wrapped_upper = "```JSON\n{\"foo\": \"bar\"}\n```";
        assert_eq!(
            strip_markdown_code_blocks(wrapped_upper),
            "{\"foo\": \"bar\"}"
        );

        // Wrapped with no language tag
        let wrapped_no_tag = "```\n{\"foo\": \"bar\"}\n```";
        assert_eq!(
            strip_markdown_code_blocks(wrapped_no_tag),
            "{\"foo\": \"bar\"}"
        );

        // Inline wrapped
        let inline = "```json{\"foo\": \"bar\"}```";
        assert_eq!(strip_markdown_code_blocks(inline), "{\"foo\": \"bar\"}");

        // Inline wrapped bracket
        let inline_bracket = "```json[{\"foo\": \"bar\"}]```";
        assert_eq!(
            strip_markdown_code_blocks(inline_bracket),
            "[{\"foo\": \"bar\"}]"
        );

        // Leading/trailing whitespace
        let whitespace = "   \n\n```json\n  {\"foo\": \"bar\"}  \n```   \n\n";
        assert_eq!(strip_markdown_code_blocks(whitespace), "{\"foo\": \"bar\"}");

        // Empty block
        let empty = "```json\n```";
        assert_eq!(strip_markdown_code_blocks(empty), "");
    }

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
