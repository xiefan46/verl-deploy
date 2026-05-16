#!/bin/bash
# setup_magi.sh — 从 HF Hub 缓存快速装好 MagiAttention
#
# 流程（跟 setup_env.sh 同模式）：
#   1. magi_attention 已 installed → 跳过
#   2. 本地不存在 → 从 HF Hub 下载预编译 wheel → pip install
#   3. HF 上没有 wheel → 提示运行 rebuild_magi.sh
#
# HF cache 仓库: xiefan46/magi-env-cache (private dataset)
# 缓存内容: magi_attention-X.X.X-cp311-cp311-linux_x86_64.whl (Hopper 编译产物)
#
# 用法:
#   bash setup_magi.sh                                # 默认 verl conda env
#   FORCE_REINSTALL=1 bash setup_magi.sh              # 强制卸载重装（升级 / cache 坏）
#   HF_CACHE_REPO=user/repo bash setup_magi.sh        # 自定义仓库
#   MAGI_WHEEL_NAME=magi.whl bash setup_magi.sh       # 自定义 HF 文件名

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
HF_CACHE_REPO="${HF_CACHE_REPO:-xiefan46/magi-env-cache}"
MAGI_WHEEL_NAME="${MAGI_WHEEL_NAME:-magi_attention.whl}"
CACHE_DIR="/root/magi_cache"
WHEEL_PATH="${CACHE_DIR}/${MAGI_WHEEL_NAME}"
PY="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

[ -d "$ENV_DIR" ] || err "verl conda env 不存在 ($ENV_DIR)。先跑: bash setup_env.sh"

installed_magi() {
    "$PY" -c "from magi_attention.api import flex_flash_attn_func, magi_attn_flex_key, calc_attn" 2>/dev/null
}

# ─── HF helpers (跟 setup_env.sh 一致) ───
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

# ─── 主流程 ───

# 1. FORCE_REINSTALL=1 → 卸载现有
if [ "${FORCE_REINSTALL:-}" = "1" ]; then
    if installed_magi; then
        log "FORCE_REINSTALL=1: 卸载现有 magi_attention..."
        $PIP uninstall -y magi_attention
    fi
    # 同时丢弃本地 wheel 缓存
    [ -f "$WHEEL_PATH" ] && { log "  丢弃本地 wheel 缓存"; rm -f "$WHEEL_PATH"; }
fi

# 2. 已安装 → 跳过
if installed_magi; then
    VERSION=$("$PY" -c "import magi_attention; print(magi_attention.__version__)" 2>/dev/null || echo "unknown")
    log "magi_attention 已安装 (${VERSION})，跳过"
    log "  如需强制重装: FORCE_REINSTALL=1 bash setup_magi.sh"
    exit 0
fi

# 3. 准备 wheel — 本地有就用本地，没有就从 HF 拉
if [ ! -f "$WHEEL_PATH" ]; then
    ensure_hf_cli
    ensure_hf_login
    log "从 HF Hub 下载预编译 wheel: ${HF_CACHE_REPO}/${MAGI_WHEEL_NAME}"
    mkdir -p "$CACHE_DIR"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    SECONDS=0
    if ! hf download --repo-type=dataset "$HF_CACHE_REPO" "$MAGI_WHEEL_NAME" --local-dir "$CACHE_DIR"; then
        err "HF 下载失败。如果 cache 不存在或损坏，运行 'bash rebuild_magi.sh' 从源码编译并推送"
    fi
    log "下载完成 (${SECONDS}s): $(du -sh "$WHEEL_PATH" | cut -f1)"
else
    log "使用本地 wheel 缓存: $WHEEL_PATH ($(du -sh "$WHEEL_PATH" | cut -f1))"
fi

# 4. 装依赖 (轻量, 大头是 nvshmem ~300MB)
log "安装 MagiAttention 依赖 (nvshmem-cu12 + cutlass-dsl + ...)..."
$PIP install --quiet "nvidia-nvshmem-cu12==3.4.5" "nvidia-cutlass-dsl==4.3.5" einops

# 5. 装 wheel 本体
log "安装 magi_attention wheel..."
$PIP install --no-deps "$WHEEL_PATH" --force-reinstall

# 6. 验证
log "验证 import..."
$PY - <<EOF
from magi_attention.api import (
    flex_flash_attn_func, magi_attn_flex_key, calc_attn,
    AttnRanges, AttnMaskType, DistAttnConfig,
    compute_pad_size, dispatch, undispatch, get_most_recent_key,
)
import magi_attention
print(f"magi_attention {magi_attention.__version__} — all key symbols importable")
EOF

echo -e "\n${BOLD}${GREEN}========================================${RESET}"
echo -e "${BOLD}${GREEN}  MagiAttention 就绪${RESET}"
echo -e "${BOLD}${GREEN}========================================${RESET}"
echo -e "\n${YELLOW}如需重新编译并更新 HF cache: bash rebuild_magi.sh${RESET}\n"
