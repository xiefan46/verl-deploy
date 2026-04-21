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

# 全路径，避免 conda activate 在非交互 shell 中不生效
PY="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

# ─── 检查 verl 仓库 ───
[ -d "$VERL_ROOT" ] || err "verl 仓库不存在: $VERL_ROOT (用法: bash setup_env.sh /path/to/verl)"

installed() { "$PY" -c "import $1" 2>/dev/null; }
has_pkg()   { "$PY" -c "import importlib.metadata; importlib.metadata.version('$1')" 2>/dev/null; }

# flash-attn 完整性校验：metadata 存在 + C 扩展可加载
fa_ok() {
    "$PY" -c "
from flash_attn.flash_attn_interface import flash_attn_func
print('flash-attn C extension OK')
" 2>/dev/null
}

# ─── 系统工具 ───
export DEBIAN_FRONTEND=noninteractive
NEED_PKGS=""
command -v tmux &>/dev/null || NEED_PKGS="tmux"
command -v zstd &>/dev/null || NEED_PKGS="${NEED_PKGS} zstd"
if [ -n "$NEED_PKGS" ]; then
    log "安装系统工具:${NEED_PKGS}..."
    apt-get install -y ${NEED_PKGS} 2>/dev/null || { apt-get update && apt-get install -y ${NEED_PKGS}; }
else
    log "系统工具已就绪"
fi

# ─── 安装 conda (如果不存在) ───
if ! command -v conda &>/dev/null && [ ! -f "${CONDA_DIR}/bin/conda" ]; then
    log "安装 Miniforge..."
    INSTALLER="/tmp/miniforge.sh"
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O "$INSTALLER"
    bash "$INSTALLER" -b -p "$CONDA_DIR"
    rm -f "$INSTALLER"
fi
export PATH="${CONDA_DIR}/bin:$PATH"

# ─── 初始化 conda（让 conda activate 在用户 shell 中可用） ───
if ! grep -q "conda initialize" ~/.bashrc 2>/dev/null; then
    log "初始化 conda (写入 ~/.bashrc)..."
    conda init bash --quiet >/dev/null 2>&1
fi

# ─── 快速路径：环境已在本地（重入同一容器） ───
if [ -d "$ENV_DIR" ] && installed torch && [ "${FORCE_INSTALL:-}" != "1" ]; then
    log "环境已存在于本地，直接激活"

    # 重新链接 verl（代码可能有更新）
    $PIP install --no-deps -e "$VERL_ROOT" --quiet

    # 确保开发/测试工具已安装（旧缓存可能缺少）
    installed pytest    || $PIP install pytest pytest-asyncio --quiet
    installed wandb     || $PIP install wandb --quiet
    installed qwen_vl_utils || $PIP install qwen-vl-utils --quiet
    installed mathruler     || $PIP install mathruler --quiet

    # 准备数据（可能上次没跑完）
    if [ ! -f ~/data/gsm8k/train.parquet ]; then
        log "准备 GSM8K 数据..."
        mkdir -p ~/data/gsm8k
        $PY "$VERL_ROOT/examples/data_preprocess/gsm8k.py" --local_save_dir ~/data/gsm8k
    fi

    # 压缩缓存（可能上次没跑到这一步）
    if [ -d "/workspace" ] && [ ! -f "$ENV_ARCHIVE" ]; then
        log "压缩环境到 Network Volume..."
        mkdir -p "$CACHE_DIR"
        tar cf - -C "$ENV_DIR" . | zstd -T0 -3 -o "$ENV_ARCHIVE"
        log "  压缩完成: $(du -sh "$ENV_ARCHIVE" | cut -f1)"
    fi

    log "验证环境..."
    $PY -c "
import torch; print(f'PyTorch: {torch.__version__}')
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
import vllm; print(f'vLLM: {vllm.__version__}')
import verl; print('verl: OK')
print('\n=== All checks passed! ===')
"
    echo -e "\n${BOLD}${GREEN}  环境已就绪（本地已存在）${RESET}"
    echo -e "${YELLOW}激活环境: source ~/.bashrc && conda activate ${ENV_NAME}${RESET}\n"
    exit 0
fi

# ─── 快速路径：从缓存恢复（新容器，有缓存） ───
if [ -f "$ENV_ARCHIVE" ] && [ ! -d "$ENV_DIR" ] && [ "${FORCE_INSTALL:-}" != "1" ]; then
    log "发现缓存，从 Network Volume 解压环境..."
    SECONDS=0
    mkdir -p "$ENV_DIR"
    zstd -d "$ENV_ARCHIVE" --stdout | tar xf - -C "$ENV_DIR"
    log "环境解压完成 (${SECONDS}s)"

    # 重新以 editable 模式链接 verl（代码可能有更新）
    log "重新链接 verl..."
    $PIP install --no-deps -e "$VERL_ROOT" --quiet

    # 确保开发/测试工具已安装（旧缓存可能缺少）
    installed pytest    || $PIP install pytest pytest-asyncio --quiet
    installed wandb     || $PIP install wandb --quiet
    installed qwen_vl_utils || $PIP install qwen-vl-utils --quiet
    installed mathruler     || $PIP install mathruler --quiet

    # 准备数据
    if [ ! -f ~/data/gsm8k/train.parquet ]; then
        log "准备 GSM8K 数据..."
        mkdir -p ~/data/gsm8k
        $PY "$VERL_ROOT/examples/data_preprocess/gsm8k.py" --local_save_dir ~/data/gsm8k
    fi

    log "验证环境..."
    $PY -c "
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
    echo -e "\n${YELLOW}激活环境: source ~/.bashrc && conda activate ${ENV_NAME}${RESET}\n"
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

# [2/6] PyTorch + vLLM + 基础依赖
log "[2/6] 安装 PyTorch + vLLM + 基础依赖..."
installed torch  && log "  torch 已安装: $($PY -c 'import torch;print(torch.__version__)'), 跳过" || $PIP install torch torchvision torchaudio
installed vllm   && log "  vllm 已安装，跳过"          || $PIP install vllm
installed ray    && log "  ray 已安装，跳过"            || $PIP install "ray[default]"
installed tensordict && log "  tensordict 已安装，跳过"  || $PIP install "tensordict>=0.8.0,<=0.10.0,!=0.9.0"
installed pytest  && log "  pytest 已安装，跳过"          || $PIP install pytest pytest-asyncio
installed cupy    && log "  cupy 已安装，跳过"            || $PIP install cupy-cuda12x
installed wandb   && log "  wandb 已安装，跳过"           || $PIP install wandb
installed qwen_vl_utils && log "  qwen-vl-utils 已安装，跳过" || $PIP install qwen-vl-utils
$PIP install -r "$VERL_ROOT/requirements.txt" --quiet

# 修复 libstdc++ CXXABI 版本不足（conda env 的版本比系统新，需要优先加载）
ENV_LIBCXX="${ENV_DIR}/lib/libstdc++.so.6"
if [ -f "$ENV_LIBCXX" ]; then
    export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
    log "  已设置 LD_LIBRARY_PATH 优先使用 conda env 的 libstdc++"
fi

# [3/6] flash-attn
log "[3/6] 安装 flash-attn..."
if fa_ok; then
    log "  flash-attn 已安装且 C 扩展完好，跳过"
else
    # 清理可能的残留（中断编译留下的不完整安装）
    if has_pkg flash_attn; then
        warn "flash-attn metadata 存在但 C 扩展加载失败，清理残留重装..."
        $PIP uninstall -y flash-attn 2>/dev/null || true
    fi

    LOCAL_WHL=$(ls ${FA_WHEEL_CACHE}/flash_attn-*.whl 2>/dev/null | head -1 || true)
    if [ -n "$LOCAL_WHL" ]; then
        log "  从缓存安装: $LOCAL_WHL"
        $PIP install --no-cache-dir "$LOCAL_WHL"
    else
        PY_VER=$($PY -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
        TORCH_VER=$($PY -c "import torch; print(torch.__version__.split('+')[0].rsplit('.',1)[0])")
        CXX11_ABI=$($PY -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
        WHEEL="flash_attn-2.7.3+cu12torch${TORCH_VER}cxx11abi${CXX11_ABI}-${PY_VER}-${PY_VER}-linux_x86_64.whl"
        WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/${WHEEL}"
        log "  尝试下载预编译 wheel: ${WHEEL}"
        wget -nv "${WHEEL_URL}" && $PIP install --no-cache-dir "${WHEEL}" && rm -f "${WHEEL}" \
            || {
                # 自动检测 GPU 架构，只编译需要的 sm 版本（避免编译 sm_80+sm_90 全量组合）
                GPU_ARCH=$($PY -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f'{cap[0]}.{cap[1]}')
else:
    print('')
" 2>/dev/null)
                if [ -n "$GPU_ARCH" ]; then
                    export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
                    log "  检测到 GPU 架构: sm_${GPU_ARCH//./}，仅编译该架构"
                fi

                # 安装 ninja 以启用并行 .cu 编译（distutils 串行，ninja 并行）
                $PIP install ninja --quiet 2>/dev/null || true

                # 根据 CPU 核数设置并行编译任务数
                NCPU=$(nproc 2>/dev/null || echo 8)
                NJOBS=$((NCPU > 16 ? 16 : NCPU))

                log "  预编译 wheel 不可用，源码编译（MAX_JOBS=${NJOBS}，ARCH=${GPU_ARCH:-all}）..."
                FA_START=$SECONDS
                MAX_JOBS=$NJOBS $PIP install flash-attn --no-build-isolation -v 2>&1 \
                    | while IFS= read -r line; do
                        # 只打印编译关键行：进入新文件 / 链接 / 错误 / 警告
                        case "$line" in
                            *"building"*|*"Building"*|*"compiling"*|*".cu"*|*".cpp"*|*"linking"*|*"Linking"*|*"error"*|*"Error"*|*"warning:"*)
                                echo -e "${GREEN}[fa-build $(( SECONDS - FA_START ))s]${RESET} $line"
                                ;;
                        esac
                    done
                FA_TIME=$((SECONDS - FA_START))
                log "  源码编译完成 (${FA_TIME}s)"
            }
        mkdir -p "$FA_WHEEL_CACHE"
        $PIP wheel flash-attn --no-build-isolation --no-deps -w "$FA_WHEEL_CACHE" 2>/dev/null || true
    fi

    # 最终校验
    fa_ok || err "flash-attn 安装失败：C 扩展无法加载，请检查 CUDA 版本和编译日志"
fi

# [4/6] 安装 verl
log "[4/6] 安装 verl..."
installed verl && log "  verl 已安装，跳过" || $PIP install --no-deps -e "$VERL_ROOT"

# [5/6] 准备数据
log "[5/6] 准备 GSM8K 数据..."
if [ ! -f ~/data/gsm8k/train.parquet ]; then
    mkdir -p ~/data/gsm8k
    $PY "$VERL_ROOT/examples/data_preprocess/gsm8k.py" --local_save_dir ~/data/gsm8k
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
$PY -c "
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
${GREEN}   source ~/.bashrc && conda activate ${ENV_NAME}${RESET}

${YELLOW}# 启动 tmux:${RESET}
${GREEN}   tmux new -s verl${RESET}

${YELLOW}# 开始训练:${RESET}
${GREEN}   bash ${VERL_ROOT}/examples/tuning/0.5b/qwen2-0.5b_grpo-lora_1_h100_fsdp_vllm.sh${RESET}

${YELLOW}# 后续启动只需:${RESET}
${GREEN}   bash setup_env.sh  # 自动从缓存恢复 (~1min)${RESET}
"
