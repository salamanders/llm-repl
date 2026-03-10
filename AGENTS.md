# lmcli Architecture & Reference

Building the absolute minimum viable REPL for LM Studio requires keeping the payload small, utilizing native error-handling for implicit filtering, and relying on standard HTTP clients. Rust is an optimal choice due to its low footprint, minimal runtime, and efficiency when handling text streams.

## Table of Contents

- [Architecture](#architecture)
- [Technical Choices](#technical-choices)
- [Implemented Files](#implemented-files)
- [Limitations to Address](#limitations-to-address)

## Architecture

* **File Ingestion:** Recursively reads files in the current directory. By utilizing `fs::read_to_string`, the application natively fails on non-UTF8 files, acting as a free and lightweight binary filter.
* **API Protocol:** Uses LM Studio's OpenAI-compatible endpoint (`http://localhost:1234/v1/chat/completions` by default) because it seamlessly handles the `messages` array for conversation history and system contexts, keeping the state entirely in your control.
* **Context Management:** Pre-loads the ingested files into the `system` prompt, then appends user interactions and assistant responses to an in-memory history vector to maintain the loop.

## Technical Choices

* **[anyhow](https://crates.io/crates/anyhow):** Used for robust and idiomatic error handling across the application. It simplifies propagating errors with contextual information.
* **[clap](https://crates.io/crates/clap):** Used to parse command-line arguments, allowing users to override the default API URL and model name easily and cleanly.
* **[reqwest](https://crates.io/crates/reqwest):** An ergonomic HTTP client for Rust, used here asynchronously (with `tokio`) to handle API requests to the local LM Studio instance.
* **[walkdir](https://crates.io/crates/walkdir):** Provides efficient recursive directory traversal, avoiding symlink loops and manually implementing directory recursion.
* **[serde_json](https://crates.io/crates/serde_json):** Used to construct and parse the JSON payloads required by the OpenAI-compatible API.

## Implemented Files

* [`Cargo.toml`](./Cargo.toml) - Defines the project metadata and its dependencies.
* [`src/main.rs`](./src/main.rs) - Contains the main REPL implementation, CLI argument parsing, file ingestion, and the interaction loop with the LM Studio API.

## Limitations to Address

* **Context Window Limits:** This naive approach blindly concatenates text. If your directory's token count exceeds the loaded model's maximum context limits (e.g., 8k or 32k), LM Studio will truncate the prompt or error out entirely.
* **Blocking Generation:** This baseline uses `stream: false` and waits for the entire generation to finish before returning output. For extensive diffs, the REPL will appear to freeze.
