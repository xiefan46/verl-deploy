#!/bin/bash
# download_models.sh — 登录 HuggingFace 并下载模型到本地
#
# 策略（类似 setup_env.sh）：
#   1. 本地已存在（${HOME}/models/）：跳过
#   2. Network Volume 有缓存（/workspace/models_cache/）：复制到本地
#   3. 都没有：从 HuggingFace 下载 → 保存到本地 → 缓存到 Network Volume
#
# 用法:
#   bash download_models.sh                          # 下载默认模型
#   bash download_models.sh model1 model2 model3     # 下载指定模型
#
# HF Token: https://huggingface.co/settings/tokens

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${RESET} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${RESET} $*"; exit 1; }

HF_MODELS_DIR="${HF_MODELS_DIR:-${HOME}/models}"
CACHE_DIR="${MODELS_CACHE_DIR:-/workspace/models_cache}"

# ─── HuggingFace 登录 ───
if [ -z "${HF_TOKEN:-}" ]; then
    if [ ! -f "${HOME}/.cache/huggingface/token" ] && [ ! -f "/workspace/.cache/huggingface/token" ]; then
        log "HuggingFace token 未设置"
        log "获取 token: ${BOLD}https://huggingface.co/settings/tokens${RESET}"
        read -rp "请输入 HF token（回车跳过）: " HF_TOKEN_INPUT
        if [ -n "${HF_TOKEN_INPUT}" ]; then
            export HF_TOKEN="${HF_TOKEN_INPUT}"
            mkdir -p "${HOME}/.cache/huggingface"
            echo "${HF_TOKEN}" > "${HOME}/.cache/huggingface/token"
            log "HF token 已保存"
        else
            warn "跳过 HF 登录，部分模型可能无法下载"
        fi
    else
        log "HuggingFace token 已存在，跳过登录"
    fi
else
    log "HuggingFace token 已通过环境变量设置"
fi

# ─── 默认模型列表（Async OPD） ───
DEFAULT_MODELS=(
    "Qwen/Qwen3-VL-2B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-VL-4B-Instruct"
)

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# ─── 检查模型是否有 safetensors ───
has_safetensors() {
    local dir="$1"
    [ -d "${dir}" ] && find "${dir}" -name "*.safetensors" -maxdepth 2 | head -1 | grep -q .
}

# ─── 下载/恢复模型 ───
log "模型目录: ${HF_MODELS_DIR}"
log "缓存目录: ${CACHE_DIR}"
mkdir -p "${HF_MODELS_DIR}"

for model_id in "${MODELS[@]}"; do
    local_dir="${HF_MODELS_DIR}/${model_id}"
    cache_dir="${CACHE_DIR}/${model_id}"

    # 路径 1: 本地已存在
    if has_safetensors "${local_dir}"; then
        log "本地已存在: ${model_id}"
        continue
    fi

    # 路径 2: 从 Network Volume 缓存恢复
    if has_safetensors "${cache_dir}"; then
        log "从缓存恢复: ${model_id} (${cache_dir} → ${local_dir})"
        mkdir -p "${local_dir}"
        cp -a "${cache_dir}/." "${local_dir}/"
        log "恢复完成: ${model_id}"
        continue
    fi

    # 路径 3: 从 HuggingFace 下载
    log "下载模型: ${BOLD}${model_id}${RESET} → ${local_dir}"
    if python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${model_id}',
    local_dir='${local_dir}',
)
print('Done: ${model_id}')
"; then
        # 下载成功，缓存到 Network Volume
        if [ -d "/workspace" ]; then
            log "缓存到 Network Volume: ${cache_dir}"
            mkdir -p "${cache_dir}"
            cp -a "${local_dir}/." "${cache_dir}/"
            log "缓存完成: ${model_id}"
        fi
    else
        warn "下载失败: ${model_id}，请检查网络或 HF token"
    fi
done

log "所有模型准备完成"
log ""
log "运行 OPD 测试:"
log "  bash /root/verl/tests/special_e2e/run_fully_async_policy_opd.sh"
