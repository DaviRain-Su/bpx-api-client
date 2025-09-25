# Repository Guidelines

## Project Structure & Module Organization
- `client/` holds the main Rust crate; `src/lib.rs` exposes `BpxClient` with submodules under `routes/` and optional WebSocket support in `ws/`.
- `types/` defines shared Backpack API data structures consumed by the client.
- `examples/` is a companion crate; `src/` offers runnable samples that demonstrate real API flows.
- Shared tooling lives at the workspace root (`Cargo.toml`, `justfile`, `rustfmt.toml`); build artifacts land in `target/`.

## Build, Test, and Development Commands
- `cargo build --all-targets` or `just build` compiles every workspace crate in debug mode.
- `cargo test` or `just test` runs the available unit and doc tests across crates.
- `just check` chains `cargo check`, `cargo +nightly fmt --check`, and `cargo clippy` to enforce the full lint gate.
- `just fix` formats with nightly `rustfmt` and applies auto-fixable Clippy suggestions.

## Coding Style & Naming Conventions
- Follow Rust 2021 defaults: four-space indentation, snake_case modules/functions, UpperCamelCase types, SCREAMING_SNAKE_CASE constants.
- Respect `rustfmt.toml` (120-column width, reordered imports). Always format with `cargo +nightly fmt --all` before sending a PR.
- Resolve lint warnings reported by `cargo clippy --all-targets --all-features`; do not suppress without justification.

## Testing Guidelines
- Co-locate unit tests inside their modules using `#[cfg(test)]`. Add integration tests under `client/tests/` when exercising public APIs end-to-end.
- Prefer async-compatible test harnesses (e.g., `tokio::test`) when touching HTTP or WebSocket flows.
- Ensure new WebSocket features include regression coverage behind the `ws` feature flag.

## Commit & Pull Request Guidelines
- Mirror the existing commit style: short, imperative subject lines ("Add order client", "Fix withdrawal route"), ~50 characters when possible.
- Keep commits focused; update generated files (e.g., JSON fixtures) in separate commits when large.
- PRs should describe the change, list test commands run, and link to any Backpack Exchange tickets or issues. Include screenshots or sample responses if the API surface changes.

## Security & Configuration Tips
- Never hard-code live API secrets; provide them via environment variables before running examples.
- Use the provided `BACKPACK_API_BASE_URL` and `BACKPACK_WS_URL` constants to avoid mixing staging/production endpoints.
