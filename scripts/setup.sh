#!/bin/sh

npm i
dfx identity new codespace_dev --storage-mode=plaintext
dfx identity use codespace_dev
dfx start --background --clean
npm i -g ic-mops
mops install
