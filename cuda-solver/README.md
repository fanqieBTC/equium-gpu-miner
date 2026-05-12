# Equium CUDA Solver

Equium CLI miner 的 CUDA 求解器后端。

该求解器面向 Equihash `(96,5)`，通过 JSON 行协议与 Rust miner 通信。推荐使用
常驻 daemon 模式，避免反复初始化 CUDA context：

```bash
./clients/cuda-solver/equium-cuda-solver --daemon
```

为 RTX 4090 级别显卡构建：

```bash
cd clients/cuda-solver
CUDA_ARCH=sm_89 ./build-cuda.sh
```

使用 CUDA 后端启动 CLI miner：

```bash
RPC_URL='https://mainnet.helius-rpc.com/?api-key=YOUR_KEY' \
KEYPAIR_PATH="$HOME/.config/solana/id.json" \
ENGINE=cuda \
THREADS=8 \
MAX_NONCES_PER_ROUND=256 \
./scripts/run-equium-miner.sh
```

当传入 target 时，CUDA solver 只返回满足目标难度的候选解；Rust CLI 在提交链上
交易前仍会重新校验 Equihash 解和 target，作为最后一道保护。
