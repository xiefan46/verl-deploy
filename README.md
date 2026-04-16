# verl-deploy

RunPod 一键部署 verl 训练环境。

## 策略

| 场景 | 行为 | 耗时 |
|------|------|------|
| 首次安装 | 完整安装到 Container Disk → 压缩到 Network Volume | 20-30 min |
| 后续启动 | 从 Network Volume 解压 → 激活 | ~1 min |

## 用法

```bash
# 1. Clone 仓库
git clone https://github.com/xiefan46/verl.git /root/verl
git clone https://github.com/xiefan46/verl-deploy.git /root/verl-deploy

# 2. 一键部署（首次会完整安装，后续自动从缓存恢复）
bash /root/verl-deploy/setup_env.sh /root/verl

# 3. 激活环境（首次需要 source bashrc，新终端则直接 conda activate 即可）
source ~/.bashrc && conda activate verl

# 4. 开始训练
bash /root/verl/examples/tuning/0.5b/qwen2-0.5b_grpo-lora_1_h100_fsdp_vllm.sh
```

## 选项

```bash
# 自定义 verl 路径（默认 /root/verl）
bash setup_env.sh /path/to/verl

# 强制重新安装（忽略缓存）
FORCE_INSTALL=1 bash setup_env.sh /root/verl
```

## 环境说明

- **Python**: 3.12 (conda via Miniforge)
- **环境位置**: `/opt/conda/envs/verl` (Container Disk，高速 IO)
- **缓存位置**: `/workspace/verl_cache/verl_env.tar.zst` (Network Volume，跨重启持久化)
- **verl 安装方式**: editable install (`pip install --no-deps -e .`)，修改代码无需重装
