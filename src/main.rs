use anyhow::{Context, Result};
use clap::Parser;
use reqwest::Client;
use serde_json::json;
use std::env;
use std::fs;
use std::io::{self, Write};
use walkdir::WalkDir;

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

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let current_dir = env::current_dir().context("Failed to get current directory")?;

    let mut context = String::from(
        "You are an expert developer. Provide clean, precise diffs or code edits based on the following codebase:\n\n",
    );

    // 1. Ingest codebase context
    let mut file_count = 0;
    for entry in WalkDir::new(&current_dir)
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
            let rel_path = path.strip_prefix(&current_dir).unwrap_or(path);
            context.push_str(&format!(
                "--- FILE: {} ---\n{}\n\n",
                rel_path.display(),
                content
            ));
            file_count += 1;
        }
    }

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
            "stream": false
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
