#!/bin/bash
# patch_mbridge.sh — 用 fork 仓库替换 setup_env.sh 装的官方 mbridge
#
# 背景: 官方 mbridge@641a5a0 (CI base) 的 qwen3_vl/utils.py split_deepstack_embs
# 在 microbatch 全文本时会崩 (deepstack_visual_embeds=None 触发 TypeError)。
# 详见 ISEEKYAN/mbridge#47 和 fork 的 fix/qwen3vl-text-only-deepstack 分支。
#
# 流程:
#   1. 卸载原 mbridge
#   2. clone fork 到 ${MBRIDGE_LOCAL_DIR} (默认 /root/mbridge)
#   3. pip install -e . (editable，方便后续改代码不用重装)
#   4. 验证补丁函数已生效
#
# 用法:
#   bash patch_mbridge.sh                                    # 用默认 fork + 分支
#   MBRIDGE_REF=ef5f92e bash patch_mbridge.sh                # 指定 commit
#   MBRIDGE_REPO=https://github.com/foo/bar.git \
#     MBRIDGE_REF=main bash patch_mbridge.sh                 # 完全自定义

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${RESET} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${RESET} $*"; exit 1; }

ENV_DIR="/opt/conda/envs/verl"
PY="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

MBRIDGE_REPO="${MBRIDGE_REPO:-https://github.com/xiefan46/mbridge.git}"
MBRIDGE_REF="${MBRIDGE_REF:-fix/qwen3vl-text-only-deepstack}"
MBRIDGE_LOCAL_DIR="${MBRIDGE_LOCAL_DIR:-/root/mbridge}"

[ -x "$PY" ] || err "verl env 不存在: $ENV_DIR (先跑 setup_env.sh)"

# ─── 1. 卸载原 mbridge ───
if "$PY" -c "import mbridge" 2>/dev/null; then
    log "卸载现有 mbridge..."
    $PIP uninstall -y mbridge --quiet
else
    log "未检测到已装的 mbridge，跳过卸载"
fi

# ─── 2. clone fork ───
if [ -d "$MBRIDGE_LOCAL_DIR/.git" ]; then
    log "已存在 ${MBRIDGE_LOCAL_DIR}，fetch 更新并 checkout ${MBRIDGE_REF}..."
    cd "$MBRIDGE_LOCAL_DIR"
    git fetch origin --quiet
    git checkout "${MBRIDGE_REF}" --quiet
    git pull --ff-only --quiet 2>/dev/null || true
else
    log "clone ${MBRIDGE_REPO}@${MBRIDGE_REF} → ${MBRIDGE_LOCAL_DIR}..."
    git clone --branch "${MBRIDGE_REF}" "${MBRIDGE_REPO}" "$MBRIDGE_LOCAL_DIR" --quiet \
        || err "clone 失败: ${MBRIDGE_REPO}@${MBRIDGE_REF}"
fi

# ─── 3. editable install ───
log "pip install -e ${MBRIDGE_LOCAL_DIR} (editable)..."
$PIP install -e "$MBRIDGE_LOCAL_DIR" --no-deps --quiet

# ─── 4. 验证补丁生效 ───
log "验证补丁..."
"$PY" -c "
import inspect, mbridge
from mbridge.models.qwen3_vl.utils import split_deepstack_embs

# editable install (PEP 660) 下 mbridge.__file__ 可能是 None，用 __path__ 拿目录
pkg_path = list(mbridge.__path__)[0] if hasattr(mbridge, '__path__') else mbridge.__file__
print(f'mbridge install path: {pkg_path}')
src = inspect.getsource(split_deepstack_embs)
guard_block = src.split('return')[0]
if 'deepstack_visual_embeds is None' not in guard_block:
    raise SystemExit('FAIL: split_deepstack_embs 没有 deepstack=None guard，补丁未生效')
print('OK: split_deepstack_embs 包含 deepstack_visual_embeds is None 守卫')
"

cd "$MBRIDGE_LOCAL_DIR" && CURRENT_SHA=$(git rev-parse --short HEAD)

echo -e "\n${BOLD}${GREEN}========================================${RESET}"
echo -e "${BOLD}${GREEN}  mbridge 补丁应用完成${RESET}"
echo -e "${BOLD}${GREEN}========================================${RESET}"
echo -e "${YELLOW}本地仓库:    ${MBRIDGE_LOCAL_DIR} (${CURRENT_SHA})${RESET}"
echo -e "${YELLOW}分支/ref:    ${MBRIDGE_REF}${RESET}"
echo -e "${YELLOW}改 mbridge 代码无需重装 (editable)${RESET}\n"
