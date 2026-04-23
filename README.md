# verl-deploy

RunPod 一键部署 verl 训练环境。

## 策略

| 场景 | 行为 | 耗时 |
|------|------|------|
| 首次安装 | 完整安装到 Container Disk → 压缩到 Network Volume | 20-30 min |
| 后续启动 | 从 Network Volume 解压 → 激活 | ~1 min |

## 快速开始

### Fully Async OPD（6×H100，Multi-Teacher 蒸馏）

```bash
git clone -b async-opd https://github.com/xiefan46/verl.git /root/verl
git clone https://github.com/xiefan46/verl-deploy.git /root/verl-deploy
bash /root/verl-deploy/setup_env.sh /root/verl
source ~/.bashrc && conda activate verl
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

## 选项

```bash
# 自定义 verl 路径（默认 /root/verl）
bash setup_env.sh /path/to/verl

# 强制重新安装（忽略缓存）
FORCE_INSTALL=1 bash setup_env.sh /root/verl
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

- **Python**: 3.12 (conda via Miniforge)
- **环境位置**: `/opt/conda/envs/verl` (Container Disk，高速 IO)
- **缓存位置**: `/workspace/verl_cache/verl_env.tar.zst` (Network Volume，跨重启持久化)
- **verl 安装方式**: editable install (`pip install --no-deps -e .`)，修改代码无需重装
- **Megatron 依赖**: mbridge + megatron-core（自动安装）
