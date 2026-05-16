#!/bin/bash
# setup_env.sh — 从 HF Hub 缓存快速恢复 verl 环境
#
# 流程：
#   1. 本地 conda env 已存在 → 重新链接 verl 即可
#   2. 本地不存在 → 从 HF Hub 下载缓存 → 解压 → 链接 verl
#
# HF cache 仓库: xiefan46/verl-env-cache (private dataset)
# 如果 cache 丢失或要更新，运行: bash rebuild_env.sh
#
# 用法:
#   bash setup_env.sh [verl_repo_path]                # 默认 /root/verl
#   FORCE_RESTORE=1 bash setup_env.sh                 # 强制重新解压
#   HF_CACHE_REPO=user/repo bash setup_env.sh         # 自定义仓库

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
VERL_ROOT="${1:-/root/verl}"
PY="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

[ -d "$VERL_ROOT" ] || err "verl 仓库不存在: $VERL_ROOT (用法: bash setup_env.sh /path/to/verl)"

installed() { "$PY" -c "import $1" 2>/dev/null; }

# ─── 系统工具 ───
export DEBIAN_FRONTEND=noninteractive
NEED_PKGS=""
command -v zstd &>/dev/null || NEED_PKGS="${NEED_PKGS} zstd"
command -v tmux &>/dev/null || NEED_PKGS="${NEED_PKGS} tmux"
if [ -n "$NEED_PKGS" ]; then
    log "安装系统工具:${NEED_PKGS}..."
    apt-get install -y ${NEED_PKGS} 2>/dev/null || { apt-get update && apt-get install -y ${NEED_PKGS}; }
fi

# ─── 安装 conda（base image 通常已有，预防情况） ───
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
    log "HF 未登录"
    log "  Token: ${BOLD}https://huggingface.co/settings/tokens${RESET}"
    hf auth login
}

install_verl() {
    log "链接 verl (editable)..."
    $PIP install --no-deps -e "$VERL_ROOT" --quiet
    installed verl || $PIP install --no-deps --force-reinstall -e "$VERL_ROOT" --quiet
    installed verl || err "verl 安装失败，请检查 $VERL_ROOT"
}

# ─── 主流程 ───

# 1. 本地 env 已存在 → 跳过解压
if [ -d "$ENV_DIR" ] && installed torch && [ "${FORCE_RESTORE:-}" != "1" ]; then
    log "本地 env 已存在，跳过解压"
else
    if [ "${FORCE_RESTORE:-}" = "1" ] && [ -d "$ENV_DIR" ]; then
        log "FORCE_RESTORE=1: 删除本地 env"
        rm -rf "$ENV_DIR"
    fi

    # 2. 没有本地 archive → 从 HF 拉
    if [ ! -f "$ENV_ARCHIVE" ]; then
        ensure_hf_cli
        ensure_hf_login
        log "从 HF 下载 cache: ${HF_CACHE_REPO}"
        mkdir -p "$CACHE_DIR"
        export HF_HUB_ENABLE_HF_TRANSFER=1
        SECONDS=0
        if ! hf download --repo-type=dataset "$HF_CACHE_REPO" verl_env.tar.zst --local-dir "$CACHE_DIR"; then
            err "HF 下载失败。如果 cache 不存在或损坏，运行 'bash rebuild_env.sh' 完整重建"
        fi
        log "下载完成 (${SECONDS}s): $(du -sh "$ENV_ARCHIVE" | cut -f1)"
    else
        log "使用本地缓存: $ENV_ARCHIVE ($(du -sh "$ENV_ARCHIVE" | cut -f1))"
    fi

    # 3. 解压
    log "解压到 ${ENV_DIR}..."
    SECONDS=0
    mkdir -p "$ENV_DIR"
    zstd -d "$ENV_ARCHIVE" --stdout | tar xf - -C "$ENV_DIR"
    log "解压完成 (${SECONDS}s)"
fi

# 4. 补装可能缺失的轻量依赖（cache 可能是旧版本）
# 已装的包通过 `installed X ||` 短路跳过，几乎零成本
log "检查关键依赖..."
installed pytest        || $PIP install pytest pytest-asyncio --quiet
installed wandb         || $PIP install wandb --quiet
installed qwen_vl_utils || $PIP install qwen-vl-utils --quiet
installed mathruler     || $PIP install mathruler --quiet
installed mbridge       || { log "  补装 mbridge..."; $PIP install --force-reinstall git+https://github.com/ISEEKYAN/mbridge.git@641a5a0 --quiet; }
installed megatron.core || { log "  补装 megatron-core..."; $PIP install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.16.0 --quiet; }
$PIP install "numpy>=2.0" --quiet  # cupy 要求 numpy>=2.0

# 5. 检查重型编译依赖（TE/Apex/flash-attn）— 缺失则提示重建，不在此处编译
heavy_missing=""
"$PY" -c "from flash_attn.flash_attn_interface import flash_attn_func" 2>/dev/null || heavy_missing="${heavy_missing} flash-attn"
"$PY" -c "import transformer_engine, transformer_engine.pytorch" 2>/dev/null || heavy_missing="${heavy_missing} transformer-engine"
"$PY" -c "import apex; from apex.normalization.fused_layer_norm import FusedLayerNorm" 2>/dev/null || heavy_missing="${heavy_missing} apex"
if [ -n "$heavy_missing" ]; then
    err "重型编译依赖缺失:${heavy_missing} (cache 可能过期)。请运行: bash rebuild_env.sh"
fi

# 6. verl editable install (每次都跑：确保 .pth 链接到当前 verl_repo 路径)
install_verl

# 5. GSM8K 数据
if [ ! -f ~/data/gsm8k/train.parquet ]; then
    log "准备 GSM8K 数据..."
    mkdir -p ~/data/gsm8k
    $PY "$VERL_ROOT/examples/data_preprocess/gsm8k.py" --local_save_dir ~/data/gsm8k
fi

# 6. 验证
log "验证环境..."
$PY -c "
import torch; print(f'PyTorch: {torch.__version__}')
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
import vllm; print(f'vLLM: {vllm.__version__}')
import verl; print('verl: OK')
import transformer_engine; print(f'TE: {transformer_engine.__version__}')
import apex; print('Apex: OK')
print('\n=== All checks passed! ===')
"

echo -e "\n${BOLD}${GREEN}========================================${RESET}"
echo -e "${BOLD}${GREEN}  verl 环境就绪${RESET}"
echo -e "${BOLD}${GREEN}========================================${RESET}"

# ─── 7. MagiAttention (按需自动安装) ───
# 检测 verl 仓库是否含 Magi backend 文件 — 是就自动装 Magi。
# 没有 (e.g. main 分支 / 其他 feature 分支) 跳过。
# 强制跳过: SKIP_MAGI=1 bash setup_env.sh
MAGI_BACKEND_FILE="${VERL_ROOT}/verl/experimental/tree_training/_magi_backend.py"
if [ "${SKIP_MAGI:-}" = "1" ]; then
    log "SKIP_MAGI=1, 跳过 MagiAttention 安装"
elif [ -f "$MAGI_BACKEND_FILE" ]; then
    log "检测到 Magi backend (${MAGI_BACKEND_FILE##*tree_training/}), 自动安装 MagiAttention..."
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
    SETUP_MAGI="${SCRIPT_DIR}/setup_magi.sh"
    if [ -f "$SETUP_MAGI" ]; then
        # HF token 已经在上面 ensure_hf_login 里 cache 过 (~/.cache/huggingface/token),
        # setup_magi.sh 里的 ensure_hf_login 直接命中, 不会再提示。
        bash "$SETUP_MAGI"
    else
        warn "$SETUP_MAGI 不存在, 跳过 Magi 安装 (手动: bash $SCRIPT_DIR/setup_magi.sh)"
    fi
else
    log "verl 仓库不含 Magi backend 文件, 跳过 MagiAttention 安装"
    log "  (如果切到含 Magi 的分支后想装: bash $(dirname "$0")/setup_magi.sh)"
fi

echo -e "\n${BOLD}${GREEN}========================================${RESET}"
echo -e "${BOLD}${GREEN}  全部就绪${RESET}"
echo -e "${BOLD}${GREEN}========================================${RESET}"
echo -e "\n${YELLOW}激活: source ~/.bashrc && conda activate ${ENV_NAME}${RESET}"
echo -e "${YELLOW}重建 verl cache (HF 丢失时): bash rebuild_env.sh${RESET}"
echo -e "${YELLOW}重建 magi cache (HF 丢失时): bash rebuild_magi.sh${RESET}\n"
