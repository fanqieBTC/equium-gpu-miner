#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BIN="${BIN:-$ROOT_DIR/target/release/equium-miner}"
KEYPAIR_PATH="${KEYPAIR_PATH:-$HOME/.config/solana/id.json}"
ENGINE="${ENGINE:-cuda}"
THREADS="${THREADS:-8}"
MAX_NONCES_PER_ROUND="${MAX_NONCES_PER_ROUND:-256}"
ROUND_STALL_SECS="${ROUND_STALL_SECS:-300}"
ADVANCE_COOLDOWN_SECS="${ADVANCE_COOLDOWN_SECS:-120}"
export ROUND_STALL_SECS ADVANCE_COOLDOWN_SECS

if [[ -z "${RPC_URL:-}" ]]; then
  echo "RPC_URL is required, for example:"
  echo "  RPC_URL='https://mainnet.helius-rpc.com/?api-key=YOUR_KEY' KEYPAIR_PATH='$HOME/.config/solana/id.json' $0"
  exit 2
fi

if [[ ! -f "$KEYPAIR_PATH" ]]; then
  echo "Keypair file not found: $KEYPAIR_PATH"
  echo "Keep the private key outside the repo, then set KEYPAIR_PATH=/path/to/keypair."
  exit 2
fi

if [[ ! -x "$BIN" ]]; then
  echo "Miner binary not found or not executable: $BIN"
  echo "Build it first with: cargo build -p equium-cli-miner --release"
  exit 2
fi

args=(
  --engine "$ENGINE"
  --rpc-url "$RPC_URL"
  --keypair "$KEYPAIR_PATH"
  --threads "$THREADS"
  --max-nonces-per-round "$MAX_NONCES_PER_ROUND"
)

if [[ -n "${MAX_BLOCKS:-}" ]]; then
  args+=(--max-blocks "$MAX_BLOCKS")
fi

exec "$BIN" "${args[@]}"
