# ICP-Bootcamp-Advanced-AI-GPT2

Launch a Chat GPT 2 fully on chain and interact with it from a frontend that is also on chain.

## Bootcamp Installation Instructions

This project is meant to run with a Codespace, and it will already setup everything related with Rust, Wasm tooling on top of dfx tooling.

After you start the codespace, please wait for it to finish it's setup (tip: look into a temporary file called "nohup.out").

Afterwards, do each of these steps:

1. Install "wasm-opt" and while you wai (average time ~12 minutes)

```bash
cargo install wasm-opt
```

2. Download the model and add it to the right folder

The project uses the GPT-2 Open Instruct v1 model. You'll need to download the following file from Hugging Face: model.safetensors (`https://huggingface.co/vicgalle/gpt2-open-instruct-v1/tree/main`)

Please add it to folder `lib/model`.

3. After wasm-opt has finished, you are ready to deploy

Run `dfx start` and in another tab `dfx deploy`

4. Upload the model files:

```bash
ic-file-uploader model append_safetensors_bytes lib/model/model.safetensors
dfx canister call model store_safetensors_bytes_to_stable
dfx canister call model load_safetensors_bytes_from_stable

ic-file-uploader model append_config_bytes lib/model/config.json
dfx canister call model setup_model

ic-file-uploader model append_tokenizer_bytes lib/model/tokenizer.json
dfx canister call model store_tokenizer_bytes_to_stable
dfx canister call model load_tokenizer_bytes_from_stable
dfx canister call model setup_tokenizer
```

## Interacting with the Model

You can interact with the model in two ways:

1. Direct token inference:

```bash
dfx canister call model inference '(vec {1; 2}, 1:nat8, 0.2:float64)'
```

Parameters:

- Input token sequence
- Generation length (nat8)
- Sampling temperature (float64)

2. Text generation:

```bash
dfx canister call model generate '("what is the capital of France?", 10:nat8, 0.2:float64)'
```

Parameters:

- Input text
- Generation length (nat8)
- Sampling temperature (float64)

Note: The maximum length of response (number of tokens generated / generation length) depends on the input length. There is a finite amount of compute and generating the next work for a longer sequence of tokens requires more computation.

# README FROM GPT2 Project

# GPT-2 on the Internet Computer

This project implements GPT-2 inference on the Internet Computer blockchain, providing a straightforward way to deploy and interact with the model.

## Analysis

### Raw Instruction Counts

The following table shows the number of instructions executed for different input lengths and generation lengths:

| Input Length | gen_1   | gen_2   | gen_4   | gen_8   |
| ------------ | ------- | ------- | ------- | ------- |
| 1            | 1.12B   | 2.25B   | 4.49B   | 9.00B   |
| 2            | 1.15B   | 2.27B   | 4.52B   | 9.02B   |
| 4            | 1.72B   | 2.84B   | 5.09B   | 9.60B   |
| 8            | 3.04B   | 4.17B   | 6.42B   | 10.93B  |
| 16           | 5.70B   | 6.83B   | 9.09B   | 13.62B  |
| 32           | 11.06B  | 12.20B  | 14.47B  | 19.01B  |
| 64           | 22.15B  | 23.30B  | 25.59B  | 30.18B  |
| 128          | 44.53B  | 45.72B  | 48.08B  | 52.82B  |
| 256          | 91.77B  | 93.05B  | 95.58B  | 100.66B |
| 512          | 196.22B | 197.77B | 200.80B | 206.88B |
| 1024         | 445.26B | 445.26B | 445.26B | 445.26B |

Observations:

1. The 1024 input length shows identical values across all generation lengths (445.26B), which might indicate a limit or issue at that input size
2. Input length has an exponential impact on instruction count

### Incremental Generation Cost Analysis

When analyzing the incremental cost of generating additional tokens, we observe some interesting patterns:

1. **Stable Cost Region (Input Length 1-64):**

   - Generating 1 additional token: ~1.13B instructions
   - Generating 3 additional tokens: ~3.38B instructions (1.13B per token)
   - Generating 7 additional tokens: ~7.90B instructions (1.13B per token)

   This shows a remarkably stable cost for generating additional tokens when working with shorter inputs.

2. **Increasing Cost Region (Input Length > 64):**
   The cost of generating additional tokens starts to increase more significantly:

| Input Length | Cost for +1 token | Cost for +3 tokens | Cost for +7 tokens |
| ------------ | ----------------- | ------------------ | ------------------ |
| 128          | 1.19B             | 3.55B              | 8.29B              |
| 256          | 1.28B             | 3.81B              | 8.89B              |
| 512          | 1.55B             | 4.58B              | 10.66B             |

### Processing Input is the key driver of computational costs

There is the initial processing of the model and prep that contributes to overhead.
We see a near linear rate of scaling with input.

## Prerequisites

- Install the Internet Computer SDK (`dfx`): https://internetcomputer.org/docs/current/developer-docs/getting-started/install
- Install Rust and Cargo: https://rustup.rs/
- NodeJS and npm: https://nodejs.org/

## Installation

1. Install the IC file uploader utility:

```bash
cargo install ic-file-uploader
```

2. Add support for the WebAssembly System Interface (WASI) target to your Rust toolchain:

```bash
rustup target add wasm32-wasi
```

3. Add wasi2ic tool, which is needed to convert the WASI-compiled Wasm to IC-compatible Wasm

```bash
cargo install wasi2ic
```

Install "wasm-opt" and wait (average time ~12 minutes)

```bash
cargo install wasm-opt
```

## Model Setup

The project uses the GPT-2 Open Instruct v1 model. You'll need to download the following files from Hugging Face:

- model.safetensors
- config.json
- tokenizer.json

These files can be found at: `https://huggingface.co/vicgalle/gpt2-open-instruct-v1/tree/main`

## Deployment and Usage

1. Start the local Internet Computer network:

```bash
dfx start --background
```

2. Upload the model files:

```bash
ic-file-uploader model append_safetensors_bytes /path/to/model.safetensors
dfx canister call model store_safetensors_bytes_to_stable
dfx canister call model load_safetensors_bytes_from_stable

ic-file-uploader model append_config_bytes /path/to/config.json
dfx canister call model setup_model

ic-file-uploader model append_tokenizer_bytes /path/to/tokenizer.json
dfx canister call model store_tokenizer_bytes_to_stable
dfx canister call model load_tokenizer_bytes_from_stable
dfx canister call model setup_tokenizer
```

## Interacting with the Model

You can interact with the model in two ways:

1. Direct token inference:

```bash
dfx canister call model inference '(vec {1; 2}, 1:nat8, 0.2:float64)'
```

Parameters:

- Input token sequence
- Generation length (nat8)
- Sampling temperature (float64)

2. Text generation:

```bash
dfx canister call model generate '("what is the capital of France?", 10:nat8, 0.2:float64)'
```

Parameters:

- Input text
- Generation length (nat8)
- Sampling temperature (float64)

Note: The maximum length of response (number of tokens generated / generation length) depends on the input length. There is a finite amount of compute and generating the next work for a longer sequence of tokens requires more computation.

## Constants

The model uses the following special tokens:

```javascript
EOT = 50257;
PREFIX_TOKENS = [50258, 198];
SUFFIX_TOKENS = [628, 50259, 198];
```

## Development Notes

- Model responses may vary based on the sampling temperature
- Longer input sequences will reduce the maximum possible generation length
- Ensure all model files are properly uploaded before making inference calls
