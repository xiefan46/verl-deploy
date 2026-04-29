#!/bin/bash
# upgrade_nccl.sh — 升级 NCCL 到 2.29.7+ 以支持 ncclCommSuspend/ncclCommResume
#
# 用法: bash upgrade_nccl.sh
#
# 策略:
#   1. 检测当前 NCCL 版本
#   2. 从 NVIDIA apt 源安装最新 NCCL
#   3. 找到 PyTorch 实际加载的 libnccl.so 并替换
#   4. 验证 ncclCommSuspend 符号存在
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${RESET} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${RESET} $*"; exit 1; }

PY="${PYTHON:-python3}"

# ─── Step 1: 检测当前版本 ───
log "Step 1: 检测当前 NCCL 版本..."

CURRENT_VER=$($PY -c "
import torch
major, minor, patch = torch.cuda.nccl.version()
print(f'{major}.{minor}.{patch}')
" 2>/dev/null || echo "unknown")

log "  当前 PyTorch 报告的 NCCL: $CURRENT_VER"

CURRENT_OK=$($PY -c "
import torch
major, minor, patch = torch.cuda.nccl.version()
ver = major * 10000 + minor * 100 + patch
print('yes' if ver >= 22907 else 'no')
" 2>/dev/null || echo "unknown")

if [ "$CURRENT_OK" = "yes" ]; then
    log "✅ NCCL 已经 >= 2.29.7，无需升级"
    # 直接跳到验证
    exec bash "$0" --verify-only 2>/dev/null || true
    exit 0
fi

# ─── Step 2: 找到 PyTorch 实际加载的 libnccl ───
log ""
log "Step 2: 定位 PyTorch 当前使用的 libnccl.so..."

# 方法: 启动 Python 加载 NCCL，从 /proc 读取 maps
NCCL_PATH=$($PY -c "
import torch, os, subprocess, sys, time

# Force NCCL to load
torch.cuda.nccl.version()

# Read own /proc/self/maps
with open('/proc/self/maps') as f:
    for line in f:
        if 'nccl' in line.lower():
            # Extract path (last field)
            parts = line.strip().split()
            if len(parts) >= 6:
                path = parts[-1]
                if os.path.exists(path):
                    print(path)
                    sys.exit(0)

# Fallback: search common locations
import ctypes.util
lib = ctypes.util.find_library('nccl')
if lib:
    print(lib)
    sys.exit(0)

# Fallback: search in torch lib dir
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
for f in os.listdir(torch_lib):
    if 'nccl' in f:
        print(os.path.join(torch_lib, f))
        sys.exit(0)

print('not_found')
" 2>/dev/null)

if [ "$NCCL_PATH" = "not_found" ] || [ -z "$NCCL_PATH" ]; then
    warn "  无法定位 PyTorch 使用的 libnccl.so"
    warn "  尝试搜索系统中所有 libnccl..."
    find / -name "libnccl.so*" -type f 2>/dev/null | head -10
    echo ""
else
    log "  PyTorch 加载的 libnccl: $NCCL_PATH"
    ls -la "$NCCL_PATH" 2>/dev/null || true
    # 如果是软链接，显示目标
    if [ -L "$NCCL_PATH" ]; then
        log "  -> 实际指向: $(readlink -f "$NCCL_PATH")"
    fi
fi

# ─── Step 3: 安装兼容当前 CUDA driver 的 NCCL ───
log ""
log "Step 3: 安装 NCCL (需要 >= 2.29.7 且兼容当前 CUDA driver)..."

# 检测 CUDA driver 版本，选择兼容的 NCCL 包
CUDA_DRIVER_VER=$($PY -c "
import subprocess, re
out = subprocess.check_output(['nvidia-smi'], text=True)
m = re.search(r'CUDA Version:\s+(\d+)\.(\d+)', out)
if m: print(f'{m.group(1)}.{m.group(2)}')
else: print('unknown')
" 2>/dev/null || echo "unknown")
log "  CUDA driver 版本: $CUDA_DRIVER_VER"

apt-get update -qq

# 找到 >= 2.29.7 且匹配 CUDA major 版本的最新 NCCL
# CUDA 12.x driver → 优先选 cuda12.x 的包，避免 cuda13 的包
CUDA_MAJOR=$(echo "$CUDA_DRIVER_VER" | cut -d. -f1)
if [ "$CUDA_MAJOR" = "12" ]; then
    # 找 cuda12.x 版本中 >= 2.29.7 的最新版
    NCCL_PKG=$(apt-cache madison libnccl2 2>/dev/null \
        | grep "cuda12\." \
        | awk -F'|' '{print $2}' \
        | tr -d ' ' \
        | sort -V \
        | while read ver; do
            # 提取 NCCL 版本号 (e.g., 2.29.7 from 2.29.7-1+cuda12.9)
            nccl_ver=$(echo "$ver" | grep -oP '^\d+\.\d+\.\d+')
            major=$(echo "$nccl_ver" | cut -d. -f1)
            minor=$(echo "$nccl_ver" | cut -d. -f2)
            patch=$(echo "$nccl_ver" | cut -d. -f3)
            num=$((major * 10000 + minor * 100 + patch))
            if [ "$num" -ge 22907 ]; then
                echo "$ver"
            fi
        done | tail -1)

    if [ -n "$NCCL_PKG" ]; then
        log "  选择 NCCL 版本: $NCCL_PKG (匹配 CUDA $CUDA_MAJOR)"
        apt-get install -y --allow-downgrades --allow-change-held-packages \
            "libnccl2=${NCCL_PKG}" "libnccl-dev=${NCCL_PKG}" 2>&1 | tail -3
    else
        warn "  未找到 cuda${CUDA_MAJOR} 的 NCCL >= 2.29.7，尝试安装最新版..."
        apt-get install -y --allow-change-held-packages libnccl2 libnccl-dev 2>&1 | tail -3
    fi
else
    log "  CUDA $CUDA_MAJOR，直接安装最新 NCCL..."
    apt-get install -y --allow-change-held-packages libnccl2 libnccl-dev 2>&1 | tail -3
fi

NEW_SYSTEM_VER=$(dpkg -l libnccl2 2>/dev/null | grep ^ii | awk '{print $3}')
log "  系统 NCCL 版本: $NEW_SYSTEM_VER"

# 找到新装的 libnccl.so
NEW_NCCL=$(find /usr/lib -name "libnccl.so.2" -type f -o -name "libnccl.so.2" -type l 2>/dev/null | head -1)
if [ -z "$NEW_NCCL" ]; then
    NEW_NCCL=$(find /usr -name "libnccl.so*" 2>/dev/null | head -1)
fi
log "  新 NCCL 库路径: $NEW_NCCL"

# 检查新版本是否有 ncclCommSuspend
if [ -n "$NEW_NCCL" ] && [ -f "$(readlink -f "$NEW_NCCL")" ]; then
    if nm -D "$(readlink -f "$NEW_NCCL")" 2>/dev/null | grep -q ncclCommSuspend; then
        log "  ✅ 新 NCCL 包含 ncclCommSuspend 符号"
    else
        err "  ❌ 新 NCCL 不包含 ncclCommSuspend，版本仍然不够"
    fi
fi

# ─── Step 4: 替换 PyTorch 使用的 libnccl ───
log ""
log "Step 4: 替换 PyTorch 加载的 libnccl..."

if [ "$NCCL_PATH" != "not_found" ] && [ -n "$NCCL_PATH" ] && [ -f "$NCCL_PATH" ]; then
    # 备份原文件
    BACKUP="${NCCL_PATH}.backup_$(date +%Y%m%d%H%M%S)"
    cp "$NCCL_PATH" "$BACKUP"
    log "  备份: $BACKUP"

    # 替换
    cp "$(readlink -f "$NEW_NCCL")" "$NCCL_PATH"
    log "  已替换: $NCCL_PATH"
elif [ "$NCCL_PATH" = "not_found" ]; then
    # PyTorch 可能通过 LD_LIBRARY_PATH 或 rpath 找 nccl
    # 尝试设置 LD_LIBRARY_PATH
    log "  未找到 PyTorch 使用的 libnccl，尝试通过 LD_LIBRARY_PATH 覆盖..."

    NCCL_DIR=$(dirname "$(readlink -f "$NEW_NCCL")")
    export LD_LIBRARY_PATH="${NCCL_DIR}:${LD_LIBRARY_PATH:-}"
    log "  设置 LD_LIBRARY_PATH=$NCCL_DIR:\$LD_LIBRARY_PATH"

    # 写入 bashrc 使其持久化
    if ! grep -q "# NCCL upgrade" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# NCCL upgrade for ncclCommSuspend support" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=${NCCL_DIR}:\${LD_LIBRARY_PATH:-}" >> ~/.bashrc
        log "  已写入 ~/.bashrc"
    fi

    # 也尝试直接放到 torch/lib 目录
    TORCH_LIB=$($PY -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
    if [ -d "$TORCH_LIB" ]; then
        cp "$(readlink -f "$NEW_NCCL")" "${TORCH_LIB}/libnccl.so.2"
        log "  也复制到 ${TORCH_LIB}/libnccl.so.2"
    fi
fi

# ─── Step 5: 验证 ───
log ""
log "Step 5: 验证..."

# NOTE: torch.cuda.nccl.version() 返回的是 PyTorch 编译时链接的版本号，
# 替换 .so 不会改变此值。真正的验证应检查运行时加载的 .so 是否包含 ncclCommSuspend。

COMPILE_VER=$($PY -c "
import torch
major, minor, patch = torch.cuda.nccl.version()
print(f'{major}.{minor}.{patch}')
" 2>/dev/null || echo "unknown")

log "  PyTorch 编译时 NCCL 版本: $COMPILE_VER (不随 .so 替换改变，可忽略)"

# 用 ctypes 验证运行时加载的 libnccl 是否有 ncclCommSuspend
CTYPES_RESULT=$($PY -c "
import ctypes
try:
    lib = ctypes.CDLL('libnccl.so.2')
    has_suspend = hasattr(lib, 'ncclCommSuspend')
    has_resume = hasattr(lib, 'ncclCommResume')
    # 尝试读取运行时版本
    try:
        ver_fn = lib.ncclGetVersion
        ver_fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
        ver_fn.restype = ctypes.c_int
        ver = ctypes.c_int(0)
        ver_fn(ctypes.byref(ver))
        v = ver.value
        runtime_ver = f'{v // 10000}.{(v % 10000) // 100}.{v % 100}'
    except Exception:
        runtime_ver = 'unknown'
    print(f'{has_suspend and has_resume}|{runtime_ver}')
except Exception as e:
    print(f'False|load_failed: {e}')
" 2>/dev/null || echo "False|unknown")

CTYPES_OK=$(echo "$CTYPES_RESULT" | cut -d'|' -f1)
RUNTIME_VER=$(echo "$CTYPES_RESULT" | cut -d'|' -f2)

log "  运行时 NCCL 版本 (ncclGetVersion): $RUNTIME_VER"
log "  ncclCommSuspend + ncclCommResume: $CTYPES_OK"

echo ""
if [ "$CTYPES_OK" = "True" ]; then
    echo -e "${GREEN}========================================${RESET}"
    echo -e "${GREEN}  ✅ NCCL 升级成功!${RESET}"
    echo -e "${GREEN}  运行时版本: $RUNTIME_VER${RESET}"
    echo -e "${GREEN}  ncclCommSuspend/Resume: 可用${RESET}"
    echo -e "${GREEN}========================================${RESET}"
else
    echo -e "${RED}========================================${RESET}"
    echo -e "${RED}  ❌ NCCL 升级未完全生效${RESET}"
    echo -e "${RED}  运行时版本: $RUNTIME_VER${RESET}"
    echo -e "${RED}  ncclCommSuspend/Resume: $CTYPES_OK${RESET}"
    echo -e "${RED}========================================${RESET}"
    echo ""
    echo "调试信息:"
    echo "  PyTorch 实际加载的 libnccl:"
    $PY -c "
import torch, os
torch.cuda.nccl.version()
with open('/proc/self/maps') as f:
    for line in f:
        if 'nccl' in line.lower():
            print('   ', line.strip())
" 2>/dev/null || echo "  (无法读取)"
    echo ""
    echo "  系统中所有 libnccl:"
    find / -name "libnccl.so*" 2>/dev/null | head -10 | sed 's/^/    /'
fi
