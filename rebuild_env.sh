#!/bin/bash
# rebuild_env.sh — 从零重建 verl 环境并推送到 HF Hub
#
# 当 HF cache 丢失、损坏，或依赖版本要更新时使用。
# 流程：
#   [1/4] 完整安装 (conda create + pip + 编译 TE/Apex/flash-attn)  ~20-30 min
#   [2/4] 链接 verl + 准备数据 + 验证
#   [3/4] 打包压缩 (tar + zstd)                                     ~3-5 min
#   [4/4] 推送到 HF Hub (xiefan46/verl-env-cache)                   ~3-5 min
#
# 依赖版本参考: https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.vllm
# 修改依赖前务必对照官方 Dockerfile。
#
# RunPod 基础镜像: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
#
# 用法:
#   bash rebuild_env.sh [verl_repo_path]              # 默认 /root/verl
#   SKIP_HF_UPLOAD=1 bash rebuild_env.sh              # 只构建本地，不推送
#   HF_CACHE_REPO=user/repo bash rebuild_env.sh       # 自定义仓库

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${RESET} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${RESET} $*"; exit 1; }

# ─── 配置 ───
CONDA_DIR="/opt/conda"
ENV_NAME="verl"
ENV_DIR="${CONDA_DIR}/envs/${ENV_NAME}"
HF_CACHE_REPO="${HF_CACHE_REPO:-xiefan46/verl-env-cache}"
CACHE_DIR="/root/verl_cache"
ENV_ARCHIVE="${CACHE_DIR}/verl_env.tar.zst"
FA_WHEEL_CACHE="${CACHE_DIR}/wheels"
VERL_ROOT="${1:-/root/verl}"
PYTHON_VER="3.11"
PY="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

[ -d "$VERL_ROOT" ] || err "verl 仓库不存在: $VERL_ROOT (用法: bash rebuild_env.sh /path/to/verl)"

installed() { "$PY" -c "import $1" 2>/dev/null; }
has_pkg()   { "$PY" -c "import importlib.metadata; importlib.metadata.version('$1')" 2>/dev/null; }

fa_ok()   { "$PY" -c "from flash_attn.flash_attn_interface import flash_attn_func" 2>/dev/null; }
te_ok()   { "$PY" -c "import transformer_engine; import transformer_engine.pytorch" 2>/dev/null; }
apex_ok() { "$PY" -c "import apex; from apex.normalization.fused_layer_norm import FusedLayerNorm" 2>/dev/null; }

# ─── HF helpers ───
ensure_hf_cli() {
    command -v hf >/dev/null 2>&1 && return
    log "安装 HF CLI..."
    pip3 install -qU "huggingface_hub[cli]" hf_transfer
    command -v hf >/dev/null 2>&1 || err "hf CLI 安装失败"
}

ensure_hf_login() {
    if hf auth whoami >/dev/null 2>&1; then
        log "HF 已登录: $(hf auth whoami 2>/dev/null | head -1)"
        return
    fi
    if [ -n "${HF_TOKEN:-}" ]; then
        log "使用 HF_TOKEN 环境变量"
        return
    fi
    log "HF 未登录，请粘贴 Write token (https://huggingface.co/settings/tokens):"
    hf auth login
}

install_verl() {
    log "安装 verl (editable)..."
    $PIP install --no-deps -e "$VERL_ROOT"
    installed verl || $PIP install --no-deps --force-reinstall -e "$VERL_ROOT"
    installed verl || err "verl 安装失败，请检查 $VERL_ROOT"
}

install_megatron_deps() {
    local MBRIDGE_REF="641a5a0"
    local MEGATRON_REF="core_v0.16.0"
    local TE_REF="release_v2.12"

    installed mbridge       || { log "  安装 mbridge (@${MBRIDGE_REF})..."; $PIP install --force-reinstall git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_REF} --quiet; }
    installed megatron.core || { log "  安装 megatron-core (${MEGATRON_REF})..."; $PIP install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@${MEGATRON_REF} --quiet; }

    if ! apex_ok; then
        log "  安装 Apex（源码编译，约 5-10 分钟）..."
        $PIP install nvidia-mathdx ninja --quiet 2>/dev/null || true
        APEX_START=$SECONDS
        NCPU=$(nproc 2>/dev/null || echo 8)
        MAX_JOBS=$NCPU $PIP install -v --no-cache-dir --no-build-isolation \
            --config-settings "--build-option=--cpp_ext" \
            --config-settings "--build-option=--cuda_ext" \
            git+https://github.com/NVIDIA/apex.git 2>&1 \
            | while IFS= read -r line; do
                case "$line" in
                    *"building"*|*"Building"*|*"compiling"*|*".cu"*|*".cpp"*|*"linking"*|*"Linking"*|*"error"*|*"Error"*)
                        echo -e "${GREEN}[apex-build $(( SECONDS - APEX_START ))s]${RESET} $line"
                        ;;
                esac
            done
        log "    Apex 编译完成 ($((SECONDS - APEX_START))s)"
    else
        log "  Apex 已安装，跳过"
    fi

    if ! te_ok; then
        log "  安装 Transformer Engine (${TE_REF})（源码编译，约 10-20 分钟）..."
        $PIP install ninja --quiet 2>/dev/null || true
        export NVTE_FRAMEWORK=pytorch
        TE_START=$SECONDS
        NCPU=$(nproc 2>/dev/null || echo 8)
        MAX_JOBS=$NCPU NVTE_BUILD_THREADS_PER_JOB=4 \
        $PIP install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@${TE_REF} 2>&1 \
            | while IFS= read -r line; do
                case "$line" in
                    *"building"*|*"Building"*|*"compiling"*|*".cu"*|*".cpp"*|*"linking"*|*"Linking"*|*"error"*|*"Error"*)
                        echo -e "${GREEN}[te-build $(( SECONDS - TE_START ))s]${RESET} $line"
                        ;;
                esac
            done
        log "    TE 编译完成 ($((SECONDS - TE_START))s)"
    else
        log "  TE 已安装，跳过"
    fi
}

# ─── 系统工具 ───
export DEBIAN_FRONTEND=noninteractive
NEED_PKGS=""
command -v tmux  &>/dev/null || NEED_PKGS="tmux"
command -v zstd  &>/dev/null || NEED_PKGS="${NEED_PKGS} zstd"
command -v cmake &>/dev/null || NEED_PKGS="${NEED_PKGS} cmake"
command -v ninja &>/dev/null || NEED_PKGS="${NEED_PKGS} ninja-build"
if [ -n "$NEED_PKGS" ]; then
    log "安装系统工具:${NEED_PKGS}..."
    apt-get install -y ${NEED_PKGS} 2>/dev/null || { apt-get update && apt-get install -y ${NEED_PKGS}; }
fi

# ─── 安装 conda ───
if ! command -v conda &>/dev/null && [ ! -f "${CONDA_DIR}/bin/conda" ]; then
    log "安装 Miniforge..."
    INSTALLER="/tmp/miniforge.sh"
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O "$INSTALLER"
    bash "$INSTALLER" -b -p "$CONDA_DIR"
    rm -f "$INSTALLER"
fi
export PATH="${CONDA_DIR}/bin:$PATH"
if ! grep -q "conda initialize" ~/.bashrc 2>/dev/null; then
    conda init bash --quiet >/dev/null 2>&1
fi

# ─── HF 登录（提前，避免后面卡 token 输入） ───
if [ "${SKIP_HF_UPLOAD:-}" != "1" ]; then
    ensure_hf_cli
    ensure_hf_login
fi

TOTAL_START=$SECONDS

# ──────────────────────────────────────────────────────────
# [1/4] 完整安装
# ──────────────────────────────────────────────────────────
log "[1/4] 完整安装..."

# 创建/重建 conda 环境
if [ -d "$ENV_DIR" ]; then
    log "  删除旧环境..."
    conda env remove -y -n "$ENV_NAME" --quiet 2>/dev/null || rm -rf "$ENV_DIR"
fi
log "  创建 conda 环境 (Python ${PYTHON_VER})..."
conda create -y -n "$ENV_NAME" python="${PYTHON_VER}" pip --quiet

# PyTorch + vLLM + 基础依赖（版本对齐官方 Dockerfile.stable.vllm）
log "  安装 PyTorch + vLLM + 基础依赖..."
$PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
$PIP install vllm==0.18.0
$PIP install "ray[default]"
$PIP install "tensordict>=0.8.0,<=0.10.0,!=0.9.0"
$PIP install pytest pytest-asyncio
$PIP install cupy-cuda12x
$PIP install wandb
$PIP install qwen-vl-utils
$PIP install pybind11 wheel --quiet
$PIP install codetiming mathruler pylatexenc cachetools nvtx matplotlib liger_kernel --quiet
$PIP install --no-deps trl==0.27.0 --quiet
$PIP install transformers==5.3.0 --quiet
$PIP install -r "$VERL_ROOT/requirements.txt" --quiet
$PIP install "numpy>=2.0" --quiet  # cupy 要求 numpy>=2.0

# libstdc++ CXXABI 修复
ENV_LIBCXX="${ENV_DIR}/lib/libstdc++.so.6"
if [ -f "$ENV_LIBCXX" ]; then
    export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

# flash-attn 源码编译
log "  安装 flash-attn 2.8.3（源码编译 MAX_JOBS=4）..."
$PIP install ninja --quiet 2>/dev/null || true
FA_START=$SECONDS
MAX_JOBS=4 $PIP install --no-build-isolation flash_attn==2.8.3 -v 2>&1 \
    | while IFS= read -r line; do
        case "$line" in
            *"building"*|*"Building"*|*"compiling"*|*".cu"*|*".cpp"*|*"linking"*|*"Linking"*|*"error"*|*"Error"*)
                echo -e "${GREEN}[fa-build $(( SECONDS - FA_START ))s]${RESET} $line"
                ;;
        esac
    done
log "    flash-attn 编译完成 ($((SECONDS - FA_START))s)"
mkdir -p "$FA_WHEEL_CACHE"
$PIP wheel flash-attn==2.8.3 --no-build-isolation --no-deps -w "$FA_WHEEL_CACHE" 2>/dev/null || true
fa_ok || err "flash-attn 安装失败"

# Megatron 依赖
log "  安装 Megatron 依赖 (mbridge + megatron-core + Apex + TE)..."
install_megatron_deps

# ──────────────────────────────────────────────────────────
# [2/4] verl + 数据 + 验证
# ──────────────────────────────────────────────────────────
log "[2/4] 链接 verl + 准备数据 + 验证..."
install_verl

if [ ! -f ~/data/gsm8k/train.parquet ]; then
    log "  准备 GSM8K 数据..."
    mkdir -p ~/data/gsm8k
    $PY "$VERL_ROOT/examples/data_preprocess/gsm8k.py" --local_save_dir ~/data/gsm8k
fi

log "  验证..."
$PY -c "
import torch; print(f'PyTorch: {torch.__version__}')
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
import vllm; print(f'vLLM: {vllm.__version__}')
import verl; print('verl: OK')
import flash_attn; print(f'flash-attn: {flash_attn.__version__}')
import transformer_engine; print(f'TE: {transformer_engine.__version__}')
import apex; print('Apex: OK')
import megatron.core; print('Megatron-Core: OK')
import mbridge; print('mbridge: OK')
print('\n=== All checks passed! ===')
"

# ──────────────────────────────────────────────────────────
# [3/4] 打包
# ──────────────────────────────────────────────────────────
log "[3/4] 打包环境..."
mkdir -p "$CACHE_DIR"
PACK_START=$SECONDS
tar cf - -C "$ENV_DIR" . | zstd -T0 -3 -f -o "$ENV_ARCHIVE"
ARCHIVE_SIZE=$(du -sh "$ENV_ARCHIVE" | cut -f1)
log "  打包完成: ${ARCHIVE_SIZE} ($((SECONDS - PACK_START))s)"

# ──────────────────────────────────────────────────────────
# [4/4] 推送 HF
# ──────────────────────────────────────────────────────────
if [ "${SKIP_HF_UPLOAD:-}" = "1" ]; then
    log "[4/4] SKIP_HF_UPLOAD=1，跳过 HF 上传"
    log "  手动上传命令:"
    log "    hf upload --repo-type=dataset --private ${HF_CACHE_REPO} ${ENV_ARCHIVE} verl_env.tar.zst"
else
    log "[4/4] 推送到 HF Hub: ${HF_CACHE_REPO}..."
    export HF_HUB_ENABLE_HF_TRANSFER=1
    UPLOAD_START=$SECONDS
    if hf upload --repo-type=dataset --private "$HF_CACHE_REPO" \
            "$ENV_ARCHIVE" verl_env.tar.zst; then
        log "  上传完成 ($((SECONDS - UPLOAD_START))s)"
        if [ -d "$FA_WHEEL_CACHE" ] && [ "$(ls -A "$FA_WHEEL_CACHE" 2>/dev/null)" ]; then
            log "  推送 wheels..."
            hf upload --repo-type=dataset --private "$HF_CACHE_REPO" \
                "$FA_WHEEL_CACHE" wheels >/dev/null 2>&1 || warn "  wheels 上传失败（不影响主 cache）"
        fi
    else
        warn "  HF 上传失败。本地 cache 仍可用: $ENV_ARCHIVE"
        warn "  手动重试: hf upload --repo-type=dataset --private $HF_CACHE_REPO $ENV_ARCHIVE verl_env.tar.zst"
    fi
fi

TOTAL_TIME=$((SECONDS - TOTAL_START))
echo -e "\n${BOLD}${GREEN}========================================${RESET}"
echo -e "${BOLD}${GREEN}  环境重建完成! (${TOTAL_TIME}s)${RESET}"
echo -e "${BOLD}${GREEN}========================================${RESET}"
echo -e "\n${YELLOW}激活: source ~/.bashrc && conda activate ${ENV_NAME}${RESET}"
echo -e "${YELLOW}下次新 pod: bash setup_env.sh (~3-5min 从 HF 恢复)${RESET}\n"
