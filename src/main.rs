use anyhow::{Context, Result};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
};
use clap::Parser as ClapParser;
use ignore::WalkBuilder;
use pulldown_cmark::{Event, Parser as MarkdownParser, Tag, TagEnd};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::env;
use std::fs;
use std::path::Path;

/// Make edits to a local folder using LM Studio backed API
#[derive(ClapParser, Debug)]
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
        "You are an expert developer. You provide code edits by responding with a standard Unified Diff.\nEnsure that the diff can be applied directly using standard patching tools. Use the exact file paths provided in the context.\n\nCurrent codebase context:\n\n",
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
    client: &Client<OpenAIConfig>,
    args: &Args,
    current_dir: &Path,
    history: &mut Vec<ChatCompletionRequestMessage>,
) {
    let request = match CreateChatCompletionRequestArgs::default()
        .model(&args.model)
        .messages(history.clone())
        .temperature(0.1)
        .build()
    {
        Ok(req) => req,
        Err(e) => {
            eprintln!("Error building request: {}", e);
            history.pop();
            return;
        }
    };

    let response = match client.chat().create(request).await {
        Ok(res) => res,
        Err(e) => {
            eprintln!("API Error: {}", e);
            history.pop();
            return;
        }
    };

    let reply = match response
        .choices
        .first()
        .and_then(|c| c.message.content.as_ref())
    {
        Some(content) => content.clone(),
        None => {
            eprintln!("Error: Response did not contain any content.");
            history.pop();
            return;
        }
    };

    println!("\n{}", reply);

    // Let's create an assistant message for the history using the builder pattern
    let assistant_msg = ChatCompletionRequestAssistantMessageArgs::default()
        .content(reply.clone())
        .build()
        .expect("Failed to build assistant message");

    history.push(assistant_msg.into());

    // Apply diffs if found
    match apply_diffs(&reply, current_dir) {
        Ok((true, summary)) => {
            // Update history with concise summary instead of full text
            if let Some(last) = history.last_mut() {
                let summary_msg = ChatCompletionRequestAssistantMessageArgs::default()
                    .content(summary)
                    .build()
                    .expect("Failed to build assistant message");
                *last = summary_msg.into();
            }

            // Reload context
            let (new_context, file_count) = build_system_prompt(current_dir);
            if let Some(first) = history.first_mut() {
                let system_msg = ChatCompletionRequestSystemMessageArgs::default()
                    .content(new_context)
                    .build()
                    .expect("Failed to build system message");
                *first = system_msg.into();
            }
            println!("✅ Reloaded {} files into context after edits.", file_count);
        }
        Ok((false, summary)) => {
            // Update history with concise summary even if no files changed
            if let Some(last) = history.last_mut() {
                let summary_msg = ChatCompletionRequestAssistantMessageArgs::default()
                    .content(summary)
                    .build()
                    .expect("Failed to build assistant message");
                *last = summary_msg.into();
            }
        }
        Err(e) => {
            eprintln!("Error applying diffs: {}", e);
            let error_summary = format!("Error applying edits: {}", e);
            if let Some(last) = history.last_mut() {
                let error_msg = ChatCompletionRequestAssistantMessageArgs::default()
                    .content(error_summary)
                    .build()
                    .expect("Failed to build assistant message");
                *last = error_msg.into();
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

    let config = OpenAIConfig::new()
        .with_api_base(&args.api_url)
        .with_api_key("not-needed-for-local");
    let client = Client::with_config(config);

    // Initialize conversational history
    let system_msg = ChatCompletionRequestSystemMessageArgs::default()
        .content(context)
        .build()
        .expect("Failed to build system message");
    let mut history = vec![system_msg.into()];

    // Initialize rustyline editor
    let mut rl = DefaultEditor::new().context("Failed to initialize rustyline editor")?;

    // 2. REPL Loop
    loop {
        let readline = rl.readline("\nlmcli> ");
        match readline {
            Ok(line) => {
                let input = line.trim();

                if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
                    break;
                }
                if input.is_empty() {
                    continue;
                }

                // Add to rustyline history
                let _ = rl.add_history_entry(input);

                let user_msg = ChatCompletionRequestUserMessageArgs::default()
                    .content(input.to_string())
                    .build()
                    .expect("Failed to build user message");
                history.push(user_msg.into());

                process_llm_interaction(&client, &args, &current_dir, &mut history).await;
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    Ok(())
}

// Pre-processes LLM responses to extract the first markdown code block encountered.
fn strip_markdown_code_blocks(response: &str) -> String {
    let parser = MarkdownParser::new(response);
    let mut in_code_block = false;
    let mut code_content = String::new();

    for event in parser {
        match event {
            Event::Start(Tag::CodeBlock(_)) => {
                in_code_block = true;
            }
            Event::End(TagEnd::CodeBlock) => {
                if in_code_block {
                    // We only care about the *first* code block (assumed to be the unified diff)
                    return code_content;
                }
            }
            Event::Text(text) => {
                if in_code_block {
                    code_content.push_str(&text);
                }
            }
            _ => {}
        }
    }

    // Fallback: If no code blocks were found, return the original response
    // in case the LLM returned the raw diff directly.
    response.trim().to_string()
}

// Applies unified diffs to local files securely, returning a compact LLM-friendly success or failure summary.
fn apply_diffs(response: &str, current_dir: &Path) -> Result<(bool, String)> {
    let mut files_changed = false;
    let mut summary = String::new();

    let cleaned_response = strip_markdown_code_blocks(response);

    let patch = match diffy::Patch::from_str(&cleaned_response) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error parsing unified diff: {}", e);
            return Ok((false, format!("Error parsing unified diff: {}", e)));
        }
    };

    let canonical_current_dir = current_dir
        .canonicalize()
        .context("Failed to canonicalize current directory")?;

    for _hunk in patch.hunks() {
        let filepath = patch
            .original()
            .unwrap_or_else(|| patch.modified().unwrap_or(""));
        // Remove `a/` or `b/` prefix often found in diffs
        let filepath = filepath
            .strip_prefix("a/")
            .or_else(|| filepath.strip_prefix("b/"))
            .unwrap_or(filepath);

        if filepath.is_empty() {
            continue; // Could not determine file
        }

        let full_path = current_dir.join(filepath);
        if !full_path.exists() {
            eprintln!("Warning: File not found: {}", filepath);
            summary.push_str(&format!("Failed to edit {}: File not found.\n", filepath));
            continue; // Skip file if not found
        }

        let canonical_full_path = match full_path.canonicalize() {
            Ok(p) => p,
            Err(_) => continue,
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

        let new_content = match diffy::apply(&original_content, &patch) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Error applying diff to {}: {}", filepath, e);
                summary.push_str(&format!("Failed to apply diff to {}: {}\n", filepath, e));
                continue;
            }
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

        // We applied the whole patch to the file in one go using `diffy::apply` and `patch`
        // So we can break here since we handled the file.
        break;
    }

    if summary.is_empty() {
        summary.push_str("No edits requested or recognized.\n");
    }

    Ok((files_changed, summary.trim_end().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_markdown_code_blocks() {
        // Plain diff without backticks
        let plain = "--- a/file\n+++ b/file\n@@ -1 +1 @@\n-foo\n+bar";
        assert_eq!(strip_markdown_code_blocks(plain), plain);

        // Wrapped with ```diff
        let wrapped = "Here is the diff:\n```diff\n--- a/file\n+++ b/file\n@@ -1 +1 @@\n-foo\n+bar\n```\nEnjoy!";
        assert_eq!(
            strip_markdown_code_blocks(wrapped),
            "--- a/file\n+++ b/file\n@@ -1 +1 @@\n-foo\n+bar\n"
        );

        // Multiple code blocks (should extract first)
        let multiple = "```diff\nfirst\n```\nSome text\n```rust\nsecond\n```";
        assert_eq!(strip_markdown_code_blocks(multiple), "first\n");

        // Empty block
        let empty = "```diff\n```";
        assert_eq!(strip_markdown_code_blocks(empty), "");
    }
}
