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
    # 同时丢弃本地 wheel 缓存 (原始 + 重命名后版本)
    [ -f "$WHEEL_PATH" ] && { log "  丢弃本地 wheel 缓存"; rm -f "$WHEEL_PATH"; }
    rm -f "${CACHE_DIR}"/magi_attention-*-cp*.whl 2>/dev/null || true
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
    if ! hf download --repo-type=dataset "$HF_CACHE_REPO" "$MAGI_WHEEL_NAME" --local-dir "$CACHE_DIR" 2>/dev/null; then
        warn "HF cache 不存在 (xiefan46/magi-env-cache 第一次用 / cache 失效)"
        warn "自动 fallback 到源码编译 (Hopper 上 10-20 min, 完成后会自动推到 HF cache)"
        SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
        REBUILD="${SCRIPT_DIR}/rebuild_magi.sh"
        [ -f "$REBUILD" ] || err "rebuild_magi.sh 不存在: $REBUILD"
        log "调用 rebuild_magi.sh ..."
        exec bash "$REBUILD"   # exec 替换本 shell, rebuild_magi 跑完即完整安装好
    fi
    log "下载完成 (${SECONDS}s): $(du -sh "$WHEEL_PATH" | cut -f1)"
else
    log "使用本地 wheel 缓存: $WHEEL_PATH ($(du -sh "$WHEEL_PATH" | cut -f1))"
fi

# 4. 装依赖 (轻量, 大头是 nvshmem ~300MB)
# debugpy: Magi utils/__init__.py 隐式 import 但没在 install_requires 声明
log "安装 MagiAttention 依赖 (nvshmem-cu12 + cutlass-dsl + einops + debugpy)..."
$PIP install --quiet "nvidia-nvshmem-cu12==3.4.5" "nvidia-cutlass-dsl==4.3.5" einops debugpy

# 5. 装 wheel 本体
#
# rebuild_magi.sh 上传时把 wheel 简化成 `magi_attention.whl` (无版本/平台 tag),
# 但 pip 解析 wheel 文件名要求标准 5 段格式 (name-version-pythontag-abitag-platform.whl)。
# 单文件 `magi_attention.whl` 会触发 `ERROR: Invalid wheel filename`.
# 这里从 wheel 内 dist-info 抽真版本号 + 当前 Python/平台 tag, 重命名后再 install.
log "重命名 wheel 为 pip 可解析的标准名..."
WHEEL_VER=$(unzip -p "$WHEEL_PATH" '*.dist-info/METADATA' 2>/dev/null \
    | grep -m1 '^Version:' | awk '{print $2}')
[ -n "$WHEEL_VER" ] || err "读不出 wheel 版本; 解压检查 dist-info/METADATA"

PY_TAG=$("$PY" -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
PLAT_TAG=$("$PY" -c "from sysconfig import get_platform; print(get_platform().replace('-','_').replace('.','_'))")
PROPER_WHEEL="${CACHE_DIR}/magi_attention-${WHEEL_VER}-${PY_TAG}-${PY_TAG}-${PLAT_TAG}.whl"

# 用 cp 不用 mv: 保留 magi_attention.whl 作为 setup 脚本下次启动的标志 (上面 step 3
# 的本地 cache 命中靠 [[ -f "$WHEEL_PATH" ]] 判断)。重命名版本旁存。
cp "$WHEEL_PATH" "$PROPER_WHEEL"
log "  $(basename "$WHEEL_PATH") -> $(basename "$PROPER_WHEEL")"

log "安装 magi_attention wheel..."
$PIP install --no-deps "$PROPER_WHEEL" --force-reinstall

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
