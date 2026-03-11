# llm-repl
Make edits to a local folder using LM Studio backed API

## Prerequisites

- [Rust and Cargo](https://rustup.rs/) (latest stable version recommended)
- [LM Studio](https://lmstudio.ai/) running locally with an API server active.

## How to Build

To build the project in release mode:

```bash
cargo build --release
```

The compiled binary will be located at `target/release/lmcli`.

## How to Run

To run the application directly:

```bash
cargo run
```

You can also pass arguments (if any are supported by the CLI):

```bash
cargo run -- [args]
```
