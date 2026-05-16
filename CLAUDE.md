# verl-deploy

RunPod 一键部署 verl 训练环境的脚本集合。

## 脚本

| 脚本 | 用途 |
|------|------|
| `setup_env.sh` | 从 HF Hub 缓存快速恢复 verl 环境（~3-5 min） |
| `rebuild_env.sh` | 从零完整重建 verl 环境并推送到 HF Hub（~25-35 min，cache 丢失或更新时用） |
| `setup_magi.sh` | 从 HF Hub 缓存装好 MagiAttention（~1-2 min, 走预编译 wheel） |
| `rebuild_magi.sh` | 从源码编译 MagiAttention 并推送到 HF Hub（~10-20 min on Hopper, cache 失效时用） |
| `upgrade_nccl.sh` | 升级 NCCL 到 2.29.7+ 以支持 ncclCommSuspend/Resume |
| `download_models.sh` | 下载 HF 模型到本地 |
| `patch_mbridge.sh` | 卸载 cache 自带的官方 mbridge，editable install fork 仓库（修 qwen3_vl 文本-only 崩溃 bug，详见 ISEEKYAN/mbridge#47） |

## 环境配置

- **基础镜像**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **conda 环境**: `/opt/conda/envs/verl` (Python 3.11)
- **环境缓存** (source of truth): HF Hub dataset `xiefan46/verl-env-cache` (private)
- **本地解压目录**: `/root/verl_cache/verl_env.tar.zst`
- **verl 仓库**: 默认 `/root/verl`，editable install

## setup_env.sh

简化的两条路径（从 HF Hub 拉缓存）：

1. **本地 env 已存在**（重入同一容器）：重新链接 verl 即可
2. **本地 env 不存在**：本地有 archive 就用本地，否则从 HF 下载 → 解压 → 链接 verl

如果 HF cache 损坏或不存在，会提示运行 `rebuild_env.sh`。

## rebuild_env.sh

完整重建 + 推送 HF 的独立脚本，四步：

1. 完整安装（conda create + pip install + 编译 TE/Apex/flash-attn）~20-30 min
2. 链接 verl + 准备数据 + 验证
3. 打包压缩成 `tar.zst`
4. 推送到 HF Hub（pod 出口带宽快，~3-5 min）

`SKIP_HF_UPLOAD=1 bash rebuild_env.sh` 只构建本地不推送。

关键设计：verl editable install 放在所有依赖（包括 TE/Apex 编译）之后，避免被覆盖。

## setup_magi.sh / rebuild_magi.sh

MagiAttention 不在 PyPI, 只能从源码编译 (Hopper 上 10-20 min)。镜像 verl env cache 模式:

- `setup_magi.sh`: 从 HF Hub `xiefan46/magi-env-cache` 拉预编译 wheel → `pip install` (~1-2 min)。如果 HF cache 不存在自动 fallback 到 `rebuild_magi.sh` 编译。
- `rebuild_magi.sh`: clone + submodule init + `pip install --no-build-isolation .` → 构建 wheel → 推到 HF (~10-20 min)

依赖 wheel 名: `magi_attention.whl` (固定名, 不带版本号, 方便 setup 脚本下载)。

**setup_env.sh 自动 chain**: setup_env.sh 跑完后会**检测 `${VERL_ROOT}/verl/experimental/tree_training/_magi_backend.py` 是否存在**:
- 存在 (用户在 magi 集成分支如 `local-bench-prep`) → 自动调用 setup_magi.sh, 一条命令搞定全部
- 不存在 (main 分支等) → 跳过 Magi 安装
- 强制跳过: `SKIP_MAGI=1 bash setup_env.sh`

HF token 只输一次: setup_env.sh 里 `hf auth login` 之后 token 写到 `~/.cache/huggingface/token`, setup_magi.sh 里的 `ensure_hf_login` 直接命中 cached token, 不会再提示。

环境变量:
- `FORCE_REINSTALL=1 bash setup_magi.sh` — 卸载现有 + 重新下载装 (升级 / cache 损坏)
- `SKIP_HF_UPLOAD=1 bash rebuild_magi.sh` — 只构建本地
- `MAGI_BRANCH=main bash rebuild_magi.sh` — 指定分支 (默认 main)
- `SKIP_MAGI=1 bash setup_env.sh` — 强制跳过 Magi 子步骤

**仅支持 Hopper (H100 / H200)**。Ampere / Blackwell 需要额外 env vars (`MAGI_ATTENTION_PREBUILD_FFA=0` 等), 当前脚本不覆盖。

## upgrade_nccl.sh

替换 PyTorch 自带的 libnccl.so 为系统 apt 安装的最新版。验证用 ctypes `ncclGetVersion()`（不依赖 `torch.cuda.nccl.version()`，那是编译时版本）。

## download_models.sh

下载 HF 模型到 `${HOME}/models/`。

两条路径：
1. **本地已存在**（`${HOME}/models/` 有 safetensors）：跳过
2. **从 HuggingFace 下载**：保存到本地

```bash
# 下载默认模型（Async OPD 用的 3 个模型）
bash download_models.sh

# 下载指定模型
bash download_models.sh Qwen/Qwen2.5-0.5B-Instruct
bash download_models.sh Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-0.6B
```

首次运行会提示输入 HF token（从 https://huggingface.co/settings/tokens 获取）。

## patch_mbridge.sh

`setup_env.sh` 用 HF cache 装的是官方 `ISEEKYAN/mbridge@641a5a0`，这个 commit 有 bug：mixed text+image 训练时，全文本 microbatch 会在 `qwen3_vl/utils.py:split_deepstack_embs` 崩 (`'NoneType' object is not iterable`)。详见 [ISEEKYAN/mbridge#47](https://github.com/ISEEKYAN/mbridge/issues/47)。

修复在 fork [`xiefan46/mbridge` 的 `fix/qwen3vl-text-only-deepstack` 分支](https://github.com/xiefan46/mbridge/tree/fix/qwen3vl-text-only-deepstack)（基于 641a5a0 + 一行 guard + 4 个回归测试）。

```bash
# setup_env.sh 之后跑这个
bash patch_mbridge.sh

# 自定义 ref（commit 或别的分支）
MBRIDGE_REF=ef5f92e bash patch_mbridge.sh
MBRIDGE_REPO=https://github.com/foo/bar.git MBRIDGE_REF=main bash patch_mbridge.sh
```

脚本流程：
1. 卸载现有 mbridge
2. clone fork 到 `${MBRIDGE_LOCAL_DIR}` (默认 `/root/mbridge`)
3. `pip install -e .` (editable，改代码不用重装)
4. 校验 `split_deepstack_embs` 源码包含 `deepstack_visual_embeds is None` guard，没有就报错

## 常见问题

### NCCL NVLS 报错 (8 卡 H100)

```
Failed to bind NVLink SHARP (NVLS) Multicast memory: CUDA error 1 'invalid argument'
```

原因：NCCL 2.29+ 在 H100 NVSwitch 上自动尝试 NVLS (硬件 multicast)，但 RunPod 的 Fabric Manager 可能未正确配置。加环境变量跳过：

```bash
NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=8 ...
```

不影响功能，只是退回普通 NVLink P2P 通信。

### NCCL 版本与 CUDA driver 不兼容

```
CUDA driver version is insufficient for CUDA runtime version
```

原因：`upgrade_nccl.sh` 从 apt 装的 NCCL 可能是 cuda13.x 版本，而机器 driver 只支持 CUDA 12.x。脚本已自动检测 CUDA driver 版本选择兼容包，但如果仍出问题，手动指定：

```bash
apt-get install -y --allow-downgrades --allow-change-held-packages libnccl2=2.29.7-1+cuda12.9 libnccl-dev=2.29.7-1+cuda12.9
```

### HF 模型权重下载失败 (mbridge Weights not found)

```
ValueError: Weights ['model.language_model.layers...'] not found in safetensors files
```

原因：HF 模型只下载了 config/tokenizer，safetensors 权重文件未下载（无认证或 Ray worker 子进程没读到 token）。手动下载：

```bash
export HF_HOME=/workspace/.cache/huggingface
export HF_TOKEN=$(cat /workspace/.cache/huggingface/token)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-2B-Instruct')
"
```

确认下载完整：`ls /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*/*.safetensors`

## 多节点 (Instant Cluster)

两台机器分别 setup 后，用 torchrun 的 `--nnodes`/`--node_rank`/`--master_addr` 参数协调。NCCL 需要指定高速网卡：`NCCL_SOCKET_IFNAME=ens1`。
