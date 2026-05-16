#!/bin/bash
# rebuild_magi.sh — 从源码编译 MagiAttention 并推到 HF Hub
#
# 当 HF cache 丢失/损坏，或要升级 Magi 版本时用。
# 流程：
#   [1/4] Clone + submodule init
#   [2/4] pip install 依赖 + 源码编译 (--no-build-isolation)             ~10-20 min on H100/H200
#   [3/4] 构建可分发 wheel (pip wheel ... -w 缓存目录)                   ~1-2 min
#   [4/4] 推到 HF Hub (xiefan46/magi-env-cache)                          ~1-2 min
#
# 官方 install doc (Hopper path):
#   https://sandai-org.github.io/MagiAttention/docs/main/user_guide/install.html
#
# 仅支持 Hopper (H100 / H200)。Ampere / Blackwell 需要额外 env vars,
# 见上面链接 — 此脚本不覆盖。
#
# 用法:
#   bash rebuild_magi.sh                                # 默认 verl conda env
#   SKIP_HF_UPLOAD=1 bash rebuild_magi.sh               # 只构建本地, 不推送
#   MAGI_BRANCH=main bash rebuild_magi.sh               # 自定义分支 (默认 main)
#   HF_CACHE_REPO=user/repo bash rebuild_magi.sh        # 自定义 HF 仓库

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
MAGI_REPO="${MAGI_REPO:-https://github.com/SandAI-org/MagiAttention.git}"
MAGI_BRANCH="${MAGI_BRANCH:-main}"
MAGI_DIR="${MAGI_DIR:-/root/MagiAttention}"
CACHE_DIR="/root/magi_cache"
MAGI_WHEEL_NAME="${MAGI_WHEEL_NAME:-magi_attention.whl}"
PY="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

[ -d "$ENV_DIR" ] || err "verl conda env 不存在 ($ENV_DIR)。先跑: bash setup_env.sh"

installed_magi() {
    "$PY" -c "from magi_attention.api import flex_flash_attn_func, magi_attn_flex_key, calc_attn" 2>/dev/null
}

# ─── HF helpers (跟 rebuild_env.sh 一致) ───
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

# ─── 检查 GPU / 编译器 ───
if ! command -v nvcc &>/dev/null; then
    err "nvcc 未找到，MagiAttention 编译需要 CUDA toolkit。RunPod devel image 应自带。"
fi
log "nvcc: $(nvcc --version | grep release)"

$PY -c "import torch; assert torch.cuda.is_available()" || err "torch.cuda 不可用"
GPU_NAME=$("$PY" -c "import torch; print(torch.cuda.get_device_name(0))")
log "GPU: ${GPU_NAME}"
if ! echo "$GPU_NAME" | grep -qE "H100|H200"; then
    warn "Magi 在非 Hopper 上需要额外 env vars (MAGI_ATTENTION_PREBUILD_FFA=0 等)"
    warn "本脚本只走 Hopper path. 当前 GPU 可能编译失败"
fi

# ─── HF 登录（提前） ───
if [ "${SKIP_HF_UPLOAD:-}" != "1" ]; then
    ensure_hf_cli
    ensure_hf_login
fi

TOTAL_START=$SECONDS

# ──────────────────────────────────────────────────────────
# [1/4] Clone + submodule
# ──────────────────────────────────────────────────────────
log "[1/4] Clone MagiAttention..."
if [ -d "$MAGI_DIR" ]; then
    log "  目录已存在, 拉最新"
    cd "$MAGI_DIR"
    git fetch origin "$MAGI_BRANCH"
    git checkout "$MAGI_BRANCH"
    git pull origin "$MAGI_BRANCH"
else
    git clone --branch "$MAGI_BRANCH" "$MAGI_REPO" "$MAGI_DIR"
    cd "$MAGI_DIR"
fi
log "  初始化 submodule (CUTLASS + Flash-Attention fork)..."
git submodule update --init --recursive

# ──────────────────────────────────────────────────────────
# [2/4] 安装依赖 + 编译
# ──────────────────────────────────────────────────────────
log "[2/4] 安装依赖 + 编译 magi_attention (10-20 min on Hopper)..."
$PIP install -r requirements.txt --quiet

# Magi utils/__init__.py 间接 import debugpy (dev tool, 没在 requirements 里).
# Setup.py 的 prebuild FFA JIT 阶段会 import magi_attention 给 JIT 预热,
# 这时缺 debugpy 会 fail. 装上避免 prebuild step 挂.
$PIP install debugpy --quiet

# RunPod 标准 image 是 PyTorch 2.8.0 + CUDA 12.8.1, 但 Magi setup.py 在 CUDA 13
# 之下会 raise RuntimeError 阻止 build。Hopper 上影响是 WGMMA 同步, 性能下降
# ~10-20%, 不影响功能 / 数值正确性 / tree-vs-dense 比例 (我们 demo 关注的指标)。
# 长期 (V2) 建议切 NGC 25.10 镜像 (CUDA 13.0), 这里先 bypass。
export MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12="${MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12:-1}"

# Magi 还有一个 magi_attn_comm 扩展, 是 internode IBGDA/RDMA 的 GroupCast/
# GroupReduce 内核 (跨节点 CP 用)。需要 infiniband header (libmlx5-dev),
# RunPod 标准 image 没装。我们 V1 单 GPU + V3 单节点 multi-DP 都不需要这个
# 内核, 跳过即可。V3+ 真要跨节点 CP 时再装 libmlx5-dev 并去掉这个 flag。
export MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD="${MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD:-1}"

# AOT 预编译 FFA kernel — 默认开。
# 不开的话, runtime 每个新 (head_dim, num_heads, dtype) 组合都会 JIT 编译,
# 第一次跑 ~1-3 min/shape, 用户体验差。开了以后 setup.py 把预设 shape
# (常见 head_dim 64/128, num_heads, bf16/fp16) 一次性 AOT 进 .so, 走
# pip install 时一次性付完, wheel push 到 HF 后所有 pod 受益。
# 代价: build 时间 ~30 min (vs 17s), wheel 体积 ~200-500MB (vs 32MB)。
# 不常见 shape (e.g. H.T5 用的 D=32) 仍走 JIT, 但 e2e 训练用的 model shape
# (Qwen2.5: D=64/128) 在预设集里。
export MAGI_ATTENTION_PREBUILD_FFA="${MAGI_ATTENTION_PREBUILD_FFA:-1}"

log "  MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12=${MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12} (绕过 CUDA 13 check)"
log "  MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD=${MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD} (跳过 internode comm 内核)"
log "  MAGI_ATTENTION_PREBUILD_FFA=${MAGI_ATTENTION_PREBUILD_FFA} (AOT 预编译 FFA, build 慢但 runtime 不 JIT)"

BUILD_START=$SECONDS
LAST_NINJA=0
# Filter: 显示 ninja [N/M] 进度 + Magi 关键阶段 + 错误/警告。
# Ninja 编译时每个 .cu 文件一行 [n/total], 看到数字递增就知道在跑。
$PIP install --no-build-isolation . 2>&1 \
    | while IFS= read -r line; do
        PRINT=0
        case "$line" in
            # ninja progress like "[15/238] /usr/local/cuda/bin/nvcc ..."
            \[*/*\]*) PRINT=1 ;;
            # pip / setuptools milestones
            *"Building wheel"*|*"Successfully built"*|*"Successfully installed"*) PRINT=1 ;;
            # magi-specific milestones (prebuild, comm build phase headers, etc.)
            *"Building magi_attn"*|*"Prebuilding"*|*"Generating "*|*"Compiling "*|*"Linking "*) PRINT=1 ;;
            # errors and warnings (always show)
            *"error:"*|*"Error:"*|*"ERROR"*|*"fatal"*|*"Failed"*|*"failed"*) PRINT=1 ;;
            *"warning:"*|*"Warning:"*) PRINT=1 ;;
        esac
        if [ "$PRINT" = "1" ]; then
            echo -e "${GREEN}[magi-build $(( SECONDS - BUILD_START ))s]${RESET} $line"
        fi
    done
log "  编译完成 ($((SECONDS - BUILD_START))s)"

# 验证
installed_magi || err "magi_attention 编译完成但 import 失败。检查 ${MAGI_DIR}"
VERSION=$("$PY" -c "import magi_attention; print(magi_attention.__version__)")
log "  magi_attention ${VERSION} 编译验证通过"

# ──────────────────────────────────────────────────────────
# [3/4] 构建 wheel (用于上传)
# ──────────────────────────────────────────────────────────
log "[3/4] 构建可分发 wheel..."
mkdir -p "$CACHE_DIR"
rm -f "$CACHE_DIR"/magi_attention-*.whl  # 清旧 wheel

WHEEL_START=$SECONDS
$PIP wheel . --no-deps --no-build-isolation -w "$CACHE_DIR" --quiet
BUILT_WHEEL=$(ls "$CACHE_DIR"/magi_attention-*.whl | head -1)
[ -f "$BUILT_WHEEL" ] || err "wheel 构建失败"

# 拷成固定名字方便上传 / 下载
WHEEL_FIXED="${CACHE_DIR}/${MAGI_WHEEL_NAME}"
cp "$BUILT_WHEEL" "$WHEEL_FIXED"
log "  wheel: $(basename "$BUILT_WHEEL"), 大小 $(du -sh "$WHEEL_FIXED" | cut -f1), 用时 $((SECONDS - WHEEL_START))s"

# ──────────────────────────────────────────────────────────
# [4/4] 推送 HF
# ──────────────────────────────────────────────────────────
if [ "${SKIP_HF_UPLOAD:-}" = "1" ]; then
    log "[4/4] SKIP_HF_UPLOAD=1，跳过 HF 上传"
    log "  手动上传:"
    log "    hf upload --repo-type=dataset --private ${HF_CACHE_REPO} ${WHEEL_FIXED} ${MAGI_WHEEL_NAME}"
else
    log "[4/4] 推送到 HF Hub: ${HF_CACHE_REPO}/${MAGI_WHEEL_NAME}..."
    export HF_HUB_ENABLE_HF_TRANSFER=1
    UPLOAD_START=$SECONDS
    if hf upload --repo-type=dataset --private "$HF_CACHE_REPO" \
            "$WHEEL_FIXED" "$MAGI_WHEEL_NAME"; then
        log "  上传完成 ($((SECONDS - UPLOAD_START))s)"
    else
        warn "  HF 上传失败。本地 wheel 仍可用: $WHEEL_FIXED"
        warn "  手动重试: hf upload --repo-type=dataset --private $HF_CACHE_REPO $WHEEL_FIXED $MAGI_WHEEL_NAME"
    fi
fi

TOTAL_TIME=$((SECONDS - TOTAL_START))
echo -e "\n${BOLD}${GREEN}========================================${RESET}"
echo -e "${BOLD}${GREEN}  MagiAttention 重建完成! (${TOTAL_TIME}s)${RESET}"
echo -e "${BOLD}${GREEN}========================================${RESET}"
echo -e "\n${YELLOW}下次新 pod: bash setup_magi.sh (~1-2 min 从 HF 装好)${RESET}\n"
