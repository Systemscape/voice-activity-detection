# List available recipes
default:
    @just --list

# Build the release binaries
alias b := build
build:
    cargo build --release

# Build and run the speech detector app (extra args are passed through)
alias r := run
run *ARGS:
    cargo run --release -- {{ARGS}}
