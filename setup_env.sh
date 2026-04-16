#!/bin/bash
# setup_env.sh — 一键搭建 verl 训练环境（conda + 压缩缓存加速重启）
#
# 策略：
#   首次安装：在 Container Disk (/opt/conda/envs/verl) 安装 conda 环境 → 压缩到 Network Volume (/workspace/)
#   后续启动：从 Network Volume 解压 (~1min) → 激活，跳过 20-30min 的重新安装
#
# 用法: bash setup_env.sh [verl_repo_path]
# 强制重装: FORCE_INSTALL=1 bash setup_env.sh [verl_repo_path]
# 默认 verl 路径: /root/verl
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

# ─── 辅助函数 ───
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${RESET} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${RESET} $*"; exit 1; }

# ─── 配置 ───
CONDA_DIR="/opt/conda"                          # conda 安装位置 (Container Disk)
ENV_NAME="verl"                                  # conda 环境名
ENV_DIR="${CONDA_DIR}/envs/${ENV_NAME}"          # 环境路径
CACHE_DIR="/workspace/verl_cache"                # Network Volume 缓存目录
ENV_ARCHIVE="${CACHE_DIR}/verl_env.tar.zst"      # 压缩包路径
FA_WHEEL_CACHE="${CACHE_DIR}/wheels"              # flash-attn wheel 缓存
VERL_ROOT="${1:-/root/verl}"                      # verl 仓库路径（可通过参数传入）
PYTHON_VER="3.12"

# ─── 检查 verl 仓库 ───
[ -d "$VERL_ROOT" ] || err "verl 仓库不存在: $VERL_ROOT (用法: bash setup_env.sh /path/to/verl)"

installed() { "${ENV_DIR}/bin/python" -c "import $1" 2>/dev/null; }
has_pkg()   { "${ENV_DIR}/bin/python" -c "import importlib.metadata; importlib.metadata.version('$1')" 2>/dev/null; }

# ─── 系统工具 ───
log "安装系统工具..."
command -v tmux &>/dev/null || { apt-get update -qq && apt-get install -y -qq tmux zstd > /dev/null; }
command -v zstd &>/dev/null || { apt-get update -qq && apt-get install -y -qq zstd > /dev/null; }

# ─── 安装 conda (如果不存在) ───
if ! command -v conda &>/dev/null && [ ! -f "${CONDA_DIR}/bin/conda" ]; then
    log "安装 Miniforge..."
    INSTALLER="/tmp/miniforge.sh"
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O "$INSTALLER"
    bash "$INSTALLER" -b -p "$CONDA_DIR"
    rm -f "$INSTALLER"
fi
export PATH="${CONDA_DIR}/bin:$PATH"
eval "$(conda shell.bash hook)"

# ─── 快速路径：环境已在本地（重入同一容器） ───
if [ -d "$ENV_DIR" ] && installed torch && [ "${FORCE_INSTALL:-}" != "1" ]; then
    log "环境已存在于本地，直接激活"
    conda activate "$ENV_NAME"

    # 重新链接 verl（代码可能有更新）
    pip install --no-deps -e "$VERL_ROOT" --quiet

    log "验证环境..."
    python -c "
import torch; print(f'PyTorch: {torch.__version__}')
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
import vllm; print(f'vLLM: {vllm.__version__}')
import verl; print('verl: OK')
print('\n=== All checks passed! ===')
"
    echo -e "\n${BOLD}${GREEN}  环境已就绪（本地已存在）${RESET}\n"
    exit 0
fi

# ─── 快速路径：从缓存恢复（新容器，有缓存） ───
if [ -f "$ENV_ARCHIVE" ] && [ ! -d "$ENV_DIR" ] && [ "${FORCE_INSTALL:-}" != "1" ]; then
    log "发现缓存，从 Network Volume 解压环境..."
    SECONDS=0
    mkdir -p "$ENV_DIR"
    zstd -d "$ENV_ARCHIVE" --stdout | tar xf - -C "$ENV_DIR"
    log "环境解压完成 (${SECONDS}s)"

    conda activate "$ENV_NAME"

    # 重新以 editable 模式链接 verl（代码可能有更新）
    log "重新链接 verl..."
    pip install --no-deps -e "$VERL_ROOT" --quiet

    # 准备数据
    if [ ! -f ~/data/gsm8k/train.parquet ]; then
        log "准备 GSM8K 数据..."
        mkdir -p ~/data/gsm8k
        python "$VERL_ROOT/examples/data_preprocess/gsm8k.py" --local_save_dir ~/data/gsm8k
    fi

    log "验证环境..."
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
import vllm; print(f'vLLM: {vllm.__version__}')
import verl; print('verl: OK')
print('\n=== All checks passed! ===')
"
    echo -e "\n${BOLD}${GREEN}========================================${RESET}"
    echo -e "${BOLD}${GREEN}  环境已从缓存恢复! (${SECONDS}s)${RESET}"
    echo -e "${BOLD}${GREEN}========================================${RESET}"
    echo -e "\n${YELLOW}激活环境:${RESET}"
    echo -e "${GREEN}   conda activate ${ENV_NAME}${RESET}\n"
    exit 0
fi

# ─── 完整安装路径 ───
log "开始完整安装..."
SECONDS=0

# [1/6] 创建 conda 环境
if [ ! -d "$ENV_DIR" ] || [ "${FORCE_INSTALL:-}" = "1" ]; then
    if [ -d "$ENV_DIR" ]; then
        log "[1/6] 删除旧环境..."
        conda env remove -y -n "$ENV_NAME" --quiet 2>/dev/null || rm -rf "$ENV_DIR"
    fi
    log "[1/6] 创建 conda 环境 (Python ${PYTHON_VER})..."
    conda create -y -n "$ENV_NAME" python="${PYTHON_VER}" --quiet
else
    log "[1/6] conda 环境已存在，跳过"
fi
conda activate "$ENV_NAME"

# [2/6] PyTorch + vLLM + 基础依赖
log "[2/6] 安装 PyTorch + vLLM + 基础依赖..."
installed torch  && log "  torch 已安装: $(python -c 'import torch;print(torch.__version__)'), 跳过" || pip install torch torchvision torchaudio
installed vllm   && log "  vllm 已安装，跳过"          || pip install vllm
installed ray    && log "  ray 已安装，跳过"            || pip install "ray[default]"
installed tensordict && log "  tensordict 已安装，跳过"  || pip install "tensordict>=0.8.0,<=0.10.0,!=0.9.0"
pip install -r "$VERL_ROOT/requirements.txt" --quiet

# [3/6] flash-attn
log "[3/6] 安装 flash-attn..."
if has_pkg flash_attn; then
    log "  flash-attn 已安装，跳过"
else
    LOCAL_WHL=$(ls ${FA_WHEEL_CACHE}/flash_attn-*.whl 2>/dev/null | head -1 || true)
    if [ -n "$LOCAL_WHL" ]; then
        log "  从缓存安装: $LOCAL_WHL"
        pip install --no-cache-dir "$LOCAL_WHL"
    else
        PY_VER=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
        TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0].rsplit('.',1)[0])")
        CXX11_ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
        WHEEL="flash_attn-2.7.3+cu12torch${TORCH_VER}cxx11abi${CXX11_ABI}-${PY_VER}-${PY_VER}-linux_x86_64.whl"
        WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/${WHEEL}"
        log "  尝试下载预编译 wheel: ${WHEEL}"
        wget -nv "${WHEEL_URL}" && pip install --no-cache-dir "${WHEEL}" && rm -f "${WHEEL}" \
            || { log "  预编译 wheel 不可用，源码编译..."; MAX_JOBS=8 pip install flash-attn --no-build-isolation; }
        mkdir -p "$FA_WHEEL_CACHE"
        pip wheel flash-attn --no-build-isolation --no-deps -w "$FA_WHEEL_CACHE" 2>/dev/null || true
    fi
fi

# [4/6] 安装 verl
log "[4/6] 安装 verl..."
installed verl && log "  verl 已安装，跳过" || pip install --no-deps -e "$VERL_ROOT"

# [5/6] 准备数据
log "[5/6] 准备 GSM8K 数据..."
if [ ! -f ~/data/gsm8k/train.parquet ]; then
    mkdir -p ~/data/gsm8k
    python "$VERL_ROOT/examples/data_preprocess/gsm8k.py" --local_save_dir ~/data/gsm8k
else
    log "  GSM8K 数据已存在，跳过"
fi

# [6/6] 压缩环境到 Network Volume
if [ -d "/workspace" ]; then
    log "[6/6] 压缩环境到 Network Volume..."
    mkdir -p "$CACHE_DIR"
    PACK_START=$SECONDS
    tar cf - -C "$ENV_DIR" . | zstd -T0 -3 -o "$ENV_ARCHIVE"
    PACK_TIME=$((SECONDS - PACK_START))
    ARCHIVE_SIZE=$(du -sh "$ENV_ARCHIVE" | cut -f1)
    log "  压缩完成: ${ARCHIVE_SIZE} (${PACK_TIME}s)"
else
    warn "[6/6] /workspace 不存在（非 RunPod 环境），跳过缓存"
fi

# 验证
log "验证环境..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')

import vllm
print(f'vLLM: {vllm.__version__}')

import verl
print('verl: OK')

import importlib.metadata
try:
    fa_ver = importlib.metadata.version('flash_attn')
    print(f'flash-attn: {fa_ver}')
except importlib.metadata.PackageNotFoundError:
    print('flash-attn: NOT INSTALLED')

import os
assert os.path.exists(os.path.expanduser('~/data/gsm8k/train.parquet')), 'GSM8K data missing!'
print('GSM8K data: OK')

print('\n=== All checks passed! ===')
"

TOTAL_TIME=$SECONDS
echo -e "
${BOLD}${GREEN}========================================${RESET}
${BOLD}${GREEN}  Setup complete! (${TOTAL_TIME}s)${RESET}
${BOLD}${GREEN}========================================${RESET}

${YELLOW}# 激活环境:${RESET}
${GREEN}   conda activate ${ENV_NAME}${RESET}

${YELLOW}# 启动 tmux:${RESET}
${GREEN}   tmux new -s verl${RESET}

${YELLOW}# 开始训练:${RESET}
${GREEN}   bash ${VERL_ROOT}/examples/tuning/0.5b/qwen2-0.5b_grpo-lora_1_h100_fsdp_vllm.sh${RESET}

${YELLOW}# 后续启动只需:${RESET}
${GREEN}   bash setup_env.sh  # 自动从缓存恢复 (~1min)${RESET}
"
