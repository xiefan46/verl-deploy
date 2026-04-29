# verl-deploy

RunPod 一键部署 verl 训练环境的脚本集合。

## 脚本

| 脚本 | 用途 |
|------|------|
| `setup_env.sh` | 一键搭建 verl conda 环境（含 PyTorch、vLLM、Megatron、TE、Apex） |
| `upgrade_nccl.sh` | 升级 NCCL 到 2.29.7+ 以支持 ncclCommSuspend/Resume |

## 环境配置

- **基础镜像**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **conda 环境**: `/opt/conda/envs/verl` (Python 3.11)
- **缓存**: `/workspace/verl_cache/verl_env.tar.zst` (Network Volume) 或 `/root/verl_env.tar.zst` (SCP 上传)
- **verl 仓库**: 默认 `/root/verl`，editable install

## setup_env.sh 三条路径

1. **本地已存在**（重入同一容器）：跳过安装，重新链接 verl，补装缺失依赖
2. **从缓存恢复**（新容器，有缓存）：解压 → 补装依赖 → 链接 verl → ~1-2 min
3. **完整安装**（无缓存）：conda create → pip install 全部依赖 → 编译 TE/Apex → ~20-30 min

关键设计：verl editable install 放在所有依赖（包括 TE/Apex 编译）之后，避免被覆盖。

## upgrade_nccl.sh

替换 PyTorch 自带的 libnccl.so 为系统 apt 安装的最新版。验证用 ctypes `ncclGetVersion()`（不依赖 `torch.cuda.nccl.version()`，那是编译时版本）。

## 多节点 (Instant Cluster)

两台机器分别 setup 后，用 torchrun 的 `--nnodes`/`--node_rank`/`--master_addr` 参数协调。NCCL 需要指定高速网卡：`NCCL_SOCKET_IFNAME=ens1`。
