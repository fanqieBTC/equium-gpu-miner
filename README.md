# Equium CUDA GPU Miner

这是一个面向 Equium 的 CUDA GPU 挖矿加速项目，核心目标是把原本偏 CPU
的 Equihash `(96,5)` 求解流程迁移到 GPU 侧执行，并保留 Rust CLI 的链上提交
与结果校验逻辑。

项目当前围绕 RTX 4090 级别显卡做了多轮优化，主要包括：

- **CUDA 常驻守护进程**：避免每次求解都重新启动进程和初始化 CUDA context；
- **GPU Equihash/Wagner 求解路径**：减少 CPU 端重复计算，将主要搜索流程放到 CUDA；
- **目标难度过滤**：solver 侧只返回满足当前 target 的候选解，减少无效提交；
- **CPU 二次验证**：提交链上交易前仍由 Rust CLI 复验，保证结果可靠；
- **多实例调优**：针对显存访问和调度延迟，测试不同并发实例数以提升实际 H/s。

在实测过程中，早期原型只有低个位数 H/s；经过 daemon 化、target filtering、
数据结构与并发调优后，单张 RTX 4090 可以达到数百 H/s。具体速度会受到显卡、
驱动、CUDA 版本、RPC 状态和实例数量影响。

## 项目结构

- `cuda-solver/`：独立 CUDA Equihash `(96,5)` 求解器，支持 JSON 行协议的常驻
  daemon 模式；
- `scripts/run-equium-miner.sh`：通用启动脚本，通过环境变量读取 RPC 与钱包路径；
- `patches/equium-cli-gpu.patch`：给上游 Equium Rust CLI 使用的补丁，用于接入
  CUDA solver 后端和 CPU 校验路径。

## 接入 Equium

推荐基于已验证的 Equium 版本安装：

```bash
git clone https://github.com/HannaPrints/equium.git
cd equium
git checkout 3455eb69ef63201f5bc127366d55728dbfcda339
```

应用 CUDA miner 补丁并复制求解器文件：

```bash
git apply /path/to/equium-gpu-miner/patches/equium-cli-gpu.patch
cp -R /path/to/equium-gpu-miner/cuda-solver clients/cuda-solver
cp /path/to/equium-gpu-miner/scripts/run-equium-miner.sh scripts/run-equium-miner.sh
```

上面的提交是本项目验证过的兼容版本。若要接入其他 Equium 版本，可能需要根据
CLI 文件变化手动调整补丁。

构建 CUDA 求解器和 CLI：

```bash
cd clients/cuda-solver
CUDA_ARCH=sm_89 ./build-cuda.sh
cd ../..
cargo build -p equium-cli-miner --release
```

使用自己的 RPC endpoint 和 Solana keypair 路径启动：

```bash
RPC_URL='https://mainnet.helius-rpc.com/?api-key=YOUR_KEY' \
KEYPAIR_PATH="$HOME/.config/solana/id.json" \
ENGINE=cuda \
THREADS=8 \
MAX_NONCES_PER_ROUND=256 \
./scripts/run-equium-miner.sh
```

## 优化思路

Equihash `(96,5)` 更偏 memory-bound，单看 CUDA occupancy 不能完全代表真实算力。
这次优化里收益比较明显的方向是：

- 让 CUDA solver 常驻，避免频繁初始化；
- 控制 host/device 之间的数据拷贝量；
- 在 solver 内部完成 target 判断；
- 保持 Rust 端 verifier 作为提交前保护；
- 根据显卡和驱动情况调整并发实例数。不同机器的最佳实例数可能不同，建议从
  1、2、4 个实例分别测试，选择实际总 H/s 最高且稳定的配置。
