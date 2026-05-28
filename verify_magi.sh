#!/bin/bash
# verify_magi.sh — 验证 MagiAttention 安装状态 + verl 代码兼容性
#
# 用途：装完 setup_magi.sh 之后跑一次, 确认:
#   1. magi_attention 能 import (版本号)
#   2. magi_attn_comm C++ 扩展 .so 文件确实落盘
#   3. verl 代码用到的所有 API symbol 当前 wheel 都暴露
#      (Magi upstream 2026 中改过名字: magi_attn_flex_dispatch/undispatch
#       → dispatch/undispatch, 老 wheel 跟新 wheel 不兼容)
#
# 退出码:
#   0 = PASS (全部 symbol 可用)
#   1 = FAIL (有缺失, 输出列出缺哪些 + 建议修复)
#
# 用法:
#   bash verify_magi.sh                    # 默认 verl conda env
#   ENV_NAME=myenv bash verify_magi.sh     # 自定义 env

set -uo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${RESET} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${RESET} $*"; }

CONDA_DIR="${CONDA_DIR:-/opt/conda}"
ENV_NAME="${ENV_NAME:-verl}"
ENV_DIR="${CONDA_DIR}/envs/${ENV_NAME}"
PY="${ENV_DIR}/bin/python"

[ -x "$PY" ] || { err "Python not found at $PY (env=$ENV_NAME)"; exit 1; }

# ─── verl 代码当前依赖的 Magi API symbol 清单 ───
# 来源 (grep "from magi_attention" verl/ tests/):
#   verl/utils/prefix_tree_magi.py        : DistAttnConfig, magi_attn_flex_key
#                                           magi_attention.common.AttnRanges
#                                           magi_attention.common.enum.AttnMaskType
#                                           magi_attention.meta.solver.dispatch_solver.DispatchConfig
#   verl/models/mcore/prefix_tree_merge.py: calc_attn, dispatch, undispatch       (NEW API)
#   verl/models/transformers/monkey_patch.py: calc_attn, get_most_recent_key
#   verl/workers/engine/fsdp/transformer_impl.py:
#                                           magi_attn_flex_dispatch                (OLD API)
#                                           magi_attn_flex_undispatch              (OLD API)
#
# REQUIRED_API_SYMBOLS = verl 代码导入的全部 magi_attention.api symbol
REQUIRED_API_SYMBOLS=(
    DistAttnConfig
    calc_attn
    dispatch
    get_most_recent_key
    magi_attn_flex_dispatch
    magi_attn_flex_key
    magi_attn_flex_undispatch
    undispatch
)

# 同 module 内的非 api 路径 (验证子模块结构完整)
REQUIRED_SUBMODULES=(
    magi_attention.common:AttnRanges
    magi_attention.common.enum:AttnMaskType
    magi_attention.meta.solver.dispatch_solver:DispatchConfig
)

# ─── 1. 基础 import + version ───
echo
echo -e "${BOLD}========== [1/4] magi_attention import & version ==========${RESET}"
"$PY" - <<'EOF' || { err "magi_attention import failed"; exit 1; }
import sys
try:
    import magi_attention
except Exception as e:
    print(f"FAIL: cannot import magi_attention: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
ver = getattr(magi_attention, "__version__", "unknown")
print(f"magi_attention version: {ver}")
print(f"location: {magi_attention.__file__}")
EOF
IMPORT_RC=$?
[ $IMPORT_RC -eq 0 ] || { err "Step 1 failed (exit $IMPORT_RC)"; exit 1; }

# ─── 2. C++ extension (.so) 落盘检查 ───
echo
echo -e "${BOLD}========== [2/4] magi_attn_comm C++ extension files ==========${RESET}"
"$PY" - <<'EOF'
import glob, os, magi_attention
mdir = os.path.dirname(magi_attention.__file__)
print(f"package dir: {mdir}")
all_sos = sorted(glob.glob(mdir + "/**/*.so", recursive=True))
if not all_sos:
    print("  (no .so files found in package)")
else:
    print(f"  {len(all_sos)} .so file(s):")
    for f in all_sos:
        rel = os.path.relpath(f, mdir)
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"    {rel}  ({size:.1f} MB)")

comm_files = sorted(glob.glob(mdir + "/magi_attn_comm*"))
print(f"magi_attn_comm* entries (top-level): {len(comm_files)}")
for f in comm_files:
    print(f"    {os.path.basename(f)}")

# Try to import the C++ extension directly
try:
    import magi_attention.magi_attn_comm as comm
    print(f"  magi_attn_comm import OK: {comm.__file__ if hasattr(comm, '__file__') else 'builtin'}")
except Exception as e:
    print(f"  magi_attn_comm import FAILED: {type(e).__name__}: {e}")
EOF

# ─── 3. magi_attention.api symbol 覆盖检查 (KEY GATE) ───
echo
echo -e "${BOLD}========== [3/4] magi_attention.api symbol coverage ==========${RESET}"
REQUIRED_PY="["
for s in "${REQUIRED_API_SYMBOLS[@]}"; do
    REQUIRED_PY+="'$s',"
done
REQUIRED_PY+="]"

API_OUT=$("$PY" - <<EOF
import magi_attention.api as api
required = ${REQUIRED_PY}
all_syms = sorted([s for s in dir(api) if not s.startswith("_")])
missing = [s for s in required if not hasattr(api, s)]
print("---ALL_API_SYMBOLS_BEGIN---")
for s in all_syms:
    print(s)
print("---ALL_API_SYMBOLS_END---")
print("---MISSING_BEGIN---")
for s in missing:
    print(s)
print("---MISSING_END---")
EOF
)

# Pretty-print all api symbols
echo "magi_attention.api exposes:"
echo "$API_OUT" | sed -n '/---ALL_API_SYMBOLS_BEGIN---/,/---ALL_API_SYMBOLS_END---/p' \
    | grep -v "^---" | awk '{printf "  %s\n", $0}'

# Pretty-print required-vs-present table
echo
echo "verl-required symbols (PASS = present, FAIL = missing):"
MISSING=$(echo "$API_OUT" | sed -n '/---MISSING_BEGIN---/,/---MISSING_END---/p' \
    | grep -v "^---" | grep -v "^$")
for s in "${REQUIRED_API_SYMBOLS[@]}"; do
    if echo "$MISSING" | grep -qx "$s"; then
        printf "  ${RED}FAIL${RESET}  %s\n" "$s"
    else
        printf "  ${GREEN}PASS${RESET}  %s\n" "$s"
    fi
done

# ─── 4. 子模块 (非 api) 路径检查 ───
echo
echo -e "${BOLD}========== [4/4] non-api submodule paths ==========${RESET}"
SUBMODULE_FAIL=0
for entry in "${REQUIRED_SUBMODULES[@]}"; do
    modpath="${entry%%:*}"
    symname="${entry##*:}"
    if "$PY" -c "from ${modpath} import ${symname}" 2>/dev/null; then
        printf "  ${GREEN}PASS${RESET}  from %s import %s\n" "$modpath" "$symname"
    else
        printf "  ${RED}FAIL${RESET}  from %s import %s\n" "$modpath" "$symname"
        SUBMODULE_FAIL=1
    fi
done

# ─── Final verdict ───
echo
echo -e "${BOLD}====================== VERDICT ======================${RESET}"

if [ -n "$MISSING" ]; then
    echo -e "${RED}${BOLD}FAIL${RESET}: magi_attention.api 缺少 verl 代码需要的 symbol:"
    echo "$MISSING" | awk '{printf "    - %s\n", $0}'
    echo
    echo -e "${YELLOW}原因:${RESET} Magi upstream 2026 中重命名了 dispatch API."
    echo "  老名字: magi_attn_flex_dispatch / magi_attn_flex_undispatch"
    echo "  新名字: dispatch / undispatch"
    echo
    echo -e "${YELLOW}修复方向 (二选一):${RESET}"
    echo "  A) 改 verl 代码到新 API (推荐, 最干净):"
    echo "     verl/workers/engine/fsdp/transformer_impl.py 1392/1409 行"
    echo "     magi_attn_flex_dispatch  → dispatch"
    echo "     magi_attn_flex_undispatch → undispatch"
    echo
    echo "  B) Rebuild Magi 老版本 (不推荐, 老版本以后会越来越孤立):"
    echo "     MAGI_BRANCH=<commit-before-rename> bash rebuild_magi.sh"
    [ $SUBMODULE_FAIL -ne 0 ] && {
        echo
        echo -e "${RED}另外子模块也有缺失, 见 [4/4] 输出${RESET}"
    }
    exit 1
fi

if [ $SUBMODULE_FAIL -ne 0 ]; then
    echo -e "${RED}${BOLD}FAIL${RESET}: 子模块路径有缺失, 见 [4/4] 输出"
    exit 1
fi

echo -e "${GREEN}${BOLD}PASS${RESET}: magi_attention 全部 verl-required symbol 可用."
exit 0
