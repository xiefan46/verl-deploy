#!/bin/bash
# download_models.sh — 登录 HuggingFace 并下载模型到本地
#
# 用法:
#   bash download_models.sh                          # 下载默认模型
#   bash download_models.sh model1 model2 model3     # 下载指定模型
#
# 模型下载到 ${HF_MODELS_DIR}/<model_id>/（默认 ${HOME}/models/）
# 下载完成后可直接用于 verl 训练，test script 会自动从该路径加载。
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

# ─── 下载模型 ───
log "模型保存目录: ${HF_MODELS_DIR}"
mkdir -p "${HF_MODELS_DIR}"

for model_id in "${MODELS[@]}"; do
    local_dir="${HF_MODELS_DIR}/${model_id}"
    if [ -d "${local_dir}" ] && ls "${local_dir}"/*.safetensors &>/dev/null; then
        log "模型已存在: ${model_id} → ${local_dir}"
        continue
    fi

    log "下载模型: ${BOLD}${model_id}${RESET} → ${local_dir}"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${model_id}',
    local_dir='${local_dir}',
)
print('Done: ${model_id}')
" || warn "下载失败: ${model_id}，请检查网络或 HF token"
done

log "所有模型下载完成"
log ""
log "运行 OPD 测试:"
log "  bash /root/verl/tests/special_e2e/run_fully_async_policy_opd.sh"
