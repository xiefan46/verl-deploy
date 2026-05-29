#!/bin/bash
# verify_magi_cache_bug.sh
# Run minimal_magi_cache_repro.py twice (size=1 fix vs size=1000 default)
# and show verdicts side by side.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPRO="${SCRIPT_DIR}/minimal_magi_cache_repro.py"
[ -f "$REPRO" ] || { echo "ERR: $REPRO not found"; exit 1; }

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
RESET='\033[0m'

run_one() {
    local size="$1"
    local label="$2"
    local logfile="/tmp/magi_cache_size${size}.log"
    echo -e "${BOLD}=== Run: MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE=$size  ($label) ===${RESET}"
    MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE="$size" \
        torchrun --standalone --nproc_per_node=1 "$REPRO" 2>&1 | tee "$logfile"
    echo
}

run_one 1    "our fix — should NOT drift"
run_one 1000 "Magi default — should DRIFT if bug isolated"

echo -e "${BOLD}=== Side-by-side verdicts ===${RESET}"
echo
echo -e "${BOLD}cache_size=1${RESET}:"
grep -E "Pass [AC]|VERDICT|A - C|A - C\|/" /tmp/magi_cache_size1.log | sed 's/^/  /'
echo
echo -e "${BOLD}cache_size=1000${RESET}:"
grep -E "Pass [AC]|VERDICT|A - C|A - C\|/" /tmp/magi_cache_size1000.log | sed 's/^/  /'
echo

# Interpretation guide
echo -e "${BOLD}=== How to read this ===${RESET}"
echo "  cache_size=1 PASS + cache_size=1000 DRIFT → bug 100% isolated to Magi."
echo "                                              File an issue with SandAI."
echo "  cache_size=1 PASS + cache_size=1000 PASS  → bug requires external state"
echo "                                              (FSDP / vLLM / CUDA allocator);"
echo "                                              Magi alone is fine."
echo "  cache_size=1 DRIFT (any)                  → setup error (check Magi install)."
