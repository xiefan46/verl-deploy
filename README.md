# verl-deploy

RunPod 一键部署 verl 训练环境。

## RunPod 基础镜像

使用 `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`

创建 Pod 时在 Docker Image 处填入此镜像名（已包含 PyTorch 2.8.0 + CUDA 12.8.1 + cuDNN）。

## 策略

环境缓存的 source of truth 是 **HF Hub dataset** (`xiefan46/verl-env-cache`，private)。

| 场景 | 脚本 | 行为 | 耗时 |
|------|------|------|------|
| 重建 cache | `rebuild_env.sh` | 完整安装 → 打包 → 推送 HF Hub | ~25-35 min |
| 日常启动 | `setup_env.sh` | 从 HF Hub 下载 cache → 解压 → 激活 | ~3-5 min |
| 重入容器 | `setup_env.sh` | 检测到本地 env 已存在 → 直接激活 | ~10 s |
| 装 MagiAttention | `setup_magi.sh` | 从 HF Hub 下载预编译 wheel → pip install | ~1-2 min |
| 重建 Magi cache | `rebuild_magi.sh` | 源码编译 (Hopper) → 构建 wheel → 推送 HF Hub | ~10-20 min |

> 旧版用 Network Volume 缓存的方式已废弃。HF Hub 的好处是任何 pod 不需 mount network volume 即可拉取，pod-to-pod 速率快。
>
> MagiAttention cache 单独用 dataset `xiefan46/magi-env-cache`（避免污染 verl env cache）。

## 快速开始

### Fully Async OPD（8×H100，Multi-Teacher 蒸馏）

```bash
git clone -b async-opd https://github.com/xiefan46/verl.git /root/verl
git clone https://github.com/xiefan46/verl-deploy.git /root/verl-deploy
bash /root/verl-deploy/setup_env.sh /root/verl
source ~/.bashrc && conda activate verl
bash /root/verl-deploy/patch_mbridge.sh    # 修 qwen3_vl text-only 崩溃 bug
bash /root/verl-deploy/download_models.sh
tmux new -s verl
bash /root/verl/tests/special_e2e/run_fully_async_policy_opd.sh
```

### Weight Sync Benchmark（6×H100，Qwen3-30B MoE）

```bash
git clone -b weight-sync-benchmark https://github.com/xiefan46/verl.git /root/verl
git clone https://github.com/xiefan46/verl-deploy.git /root/verl-deploy
bash /root/verl-deploy/setup_env.sh /root/verl
source ~/.bashrc && conda activate verl
tmux new -s verl
bash /root/verl/tests/special_e2e/run_weight_sync_benchmark.sh
```

### 最小示例（单卡 H100，Qwen2-0.5B）

```bash
git clone https://github.com/xiefan46/verl.git /root/verl
git clone https://github.com/xiefan46/verl-deploy.git /root/verl-deploy
bash /root/verl-deploy/setup_env.sh /root/verl
source ~/.bashrc && conda activate verl
tmux new -s verl
bash /root/verl/examples/tuning/0.5b/qwen2-0.5b_grpo-lora_1_h100_fsdp_vllm.sh
```

### NCCL Suspend/Resume E2E（2×H100，Colocated GRPO）

```bash
git clone -b feat/nccl-comm-suspend-resume https://github.com/xiefan46/verl.git /root/verl
git clone https://github.com/xiefan46/verl-deploy.git /root/verl-deploy
bash /root/verl-deploy/setup_env.sh /root/verl
source ~/.bashrc && conda activate verl
bash /root/verl-deploy/upgrade_nccl.sh
bash /root/verl-deploy/download_models.sh Qwen/Qwen2.5-0.5B-Instruct
tmux new -s verl
NCCL_NVLS_ENABLE=0 bash /root/verl/tests/special_e2e/run_nccl_suspend_e2e.sh
```

### NCCL Suspend/Resume Profile（8×H100，单节点）

```bash
git clone -b feat/nccl-comm-suspend-resume https://github.com/xiefan46/verl.git /root/verl
git clone https://github.com/xiefan46/verl-deploy.git /root/verl-deploy
bash /root/verl-deploy/setup_env.sh /root/verl
source ~/.bashrc && conda activate verl
bash /root/verl-deploy/upgrade_nccl.sh
cd /root/verl && torchrun --nproc_per_node=8 tests/utils/test_nccl_suspend.py
torchrun --nproc_per_node=8 tests/utils/profile_nccl_memory.py
```

### NCCL Profile 多节点（2×8 H100 Instant Cluster）

创建 Instant Cluster 后，分别 SSH 到两台机器。

**两台机器上分别执行（setup 阶段）：**

```bash
git clone -b feat/nccl-comm-suspend-resume https://github.com/xiefan46/verl.git /root/verl
git clone https://github.com/xiefan46/verl-deploy.git /root/verl-deploy
bash /root/verl-deploy/setup_env.sh
source ~/.bashrc && conda activate verl
bash /root/verl-deploy/upgrade_nccl.sh
```

**确认内部 IP（两台都执行）：**

```bash
ip addr show ens1 | grep 'inet '
# 记下 Node 0 的 IP，假设是 10.0.0.1
```

**两台同时执行 torchrun（两个 SSH 窗口）：**

```bash
# Node 0:
cd /root/verl && \
NCCL_SOCKET_IFNAME=ens1 \
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=10.0.0.1 --master_port=29500 \
  tests/utils/profile_nccl_memory.py

# Node 1:
cd /root/verl && \
NCCL_SOCKET_IFNAME=ens1 \
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=10.0.0.1 --master_port=29500 \
  tests/utils/profile_nccl_memory.py
```

注意事项：
- 两边 torchrun 要几乎同时执行（先启动的会等另一边，默认超时 10 分钟）
- 第一次跑可加 `NCCL_DEBUG=INFO` 确认走了 InfiniBand/RoCE
- 如果 `ens1` 不对，用 `ip link show` 找高速网卡名
- 跨节点的 NCCL network buffer 属于 `ncclMemPersist`（suspend 不会释放），这正是多节点测量的价值

## 缓存重建（HF cache 损坏或要更新依赖时）

```bash
# 在任一 pod 上执行（pod 出口带宽快，HF 上传 3-5 min）
bash /root/verl-deploy/rebuild_env.sh /root/verl
```

会完整跑：conda create → pip 装全部依赖 → 编译 TE/Apex/flash-attn → 打包 → 推送到
`xiefan46/verl-env-cache`。完成后所有新 pod 跑 `setup_env.sh` 即可拉新版本。

```bash
# 只重建本地，不推送 HF
SKIP_HF_UPLOAD=1 bash rebuild_env.sh
```

首次运行会要求 `hf auth login` — 准备好 [Write token](https://huggingface.co/settings/tokens) 粘贴。

## 选项

```bash
# 自定义 verl 路径（默认 /root/verl）
bash setup_env.sh /path/to/verl

# 强制重新解压（删本地 env，重新从 HF 拉/解压）
FORCE_RESTORE=1 bash setup_env.sh

# 自定义缓存仓库
HF_CACHE_REPO=user/repo bash setup_env.sh
```

## 监控

### Ray Dashboard

在本地终端建立 SSH 端口转发，然后浏览器打开 `http://localhost:8265`：

```bash
ssh -L 8265:localhost:8265 root@<host> -p <port> -i ~/.ssh/id_ed25519
```

可以在 Actors 页面查看各角色的 GPU 分配和状态。

### Wandb

```bash
# 远程机器上登录（一次性）
wandb login  # 按提示粘贴 API key

# 跑训练时加上 wandb logger
bash tests/special_e2e/run_fully_async_policy_opd.sh \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='async-opd-test'
```

训练开始后终端会打印 wandb run 链接，在本地浏览器打开即可看到实时曲线。

### GPU 使用

```bash
# 远程机器上查看各卡占用
nvidia-smi
# 持续监控
watch -n 1 nvidia-smi
```

## 环境说明

- **Python**: 3.11 (conda via Miniforge)
- **环境位置**: `/opt/conda/envs/verl` (Container Disk，高速 IO)
- **缓存 source of truth**: HF Hub dataset `xiefan46/verl-env-cache` (private)
- **本地缓存路径**: `/root/verl_cache/verl_env.tar.zst` (~4.7 GB)
- **verl 安装方式**: editable install (`pip install --no-deps -e .`)，修改代码无需重装
- **Megatron 依赖**: mbridge + megatron-core + Transformer Engine + Apex（包含在 cache 中）
- **依赖版本参考**: [verl 官方 Dockerfile](https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.vllm)，修改依赖前务必对照确保一致
