#!/bin/sh

# c++ will be required on install wasm-opt
apt update
apt install g++ -y
# commands for rust setup
cargo install ic-file-uploader
export PATH="$PATH:~/.cargo/bin"
# Reload the shell configuration
export PATH="$PATH:/usr/bin"
# continue with rust setup
# Add support for the WebAssembly System Interface (WASI) target to your Rust toolchain:
rustup target add wasm32-wasi 
# Add wasi2ic tool, which is needed to convert the WASI-compiled Wasm to IC-compatible Wasm
cargo install wasi2ic