#!/bin/bash
# verify_magi_repros.sh
# Run all three Magi cache-corruption repros at size=1 (our fix) and
# size=1000 (Magi default) and show side-by-side verdicts.
#
# Repro 1: Magi only — random tensors
# Repro 2: Magi + vLLM init/sleep
# Repro 3: Magi + FSDP2-wrapped Qwen2 (sourcing Q/K/V from a real model)
#
# We've already run repro 1; it PASSED at both sizes. So if repro 2 or 3
# DRIFTS at size=1000 but PASSES at size=1, that pinpoints which external
# dependency (vLLM / FSDP2) is the real precondition.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
RESET='\033[0m'

run_one() {
    local script="$1"
    local size="$2"
    local tag="$3"
    local logfile="/tmp/magi_${tag}_size${size}.log"
    echo -e "${BOLD}── ${tag} @ size=${size} ──${RESET}"
    MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE="$size" \
        torchrun --standalone --nproc_per_node=1 "${SCRIPT_DIR}/${script}" 2>&1 | tee "$logfile"
    echo
}

# ── Repro 1 (sanity, already known-PASS at both sizes) ──
echo -e "${BOLD}=== Repro 1: Magi only ===${RESET}"
run_one minimal_magi_cache_repro.py 1    repro1
run_one minimal_magi_cache_repro.py 1000 repro1

# ── Repro 2: Magi + vLLM ──
echo -e "${BOLD}=== Repro 2: Magi + vLLM ===${RESET}"
run_one repro2_magi_with_vllm.py 1    repro2
run_one repro2_magi_with_vllm.py 1000 repro2

# ── Repro 3: Magi + FSDP2 ──
echo -e "${BOLD}=== Repro 3: Magi + FSDP2 ===${RESET}"
run_one repro3_magi_with_fsdp.py 1    repro3
run_one repro3_magi_with_fsdp.py 1000 repro3

# ── Summary ──
echo -e "${BOLD}═══ Summary: |A-C|/|A| at each cache size ═══${RESET}"
echo

for tag in repro1 repro2 repro3; do
    s1=$(grep -oE "\|A - C\|/\|A\| = [0-9.]+%" "/tmp/magi_${tag}_size1.log" 2>/dev/null | tail -1)
    s1k=$(grep -oE "\|A - C\|/\|A\| = [0-9.]+%" "/tmp/magi_${tag}_size1000.log" 2>/dev/null | tail -1)
    v1=$(grep -oE "VERDICT: [A-Za-z≈ ]+" "/tmp/magi_${tag}_size1.log" 2>/dev/null | head -1)
    v1k=$(grep -oE "VERDICT: [A-Za-z≈ ]+" "/tmp/magi_${tag}_size1000.log" 2>/dev/null | head -1)
    printf "  ${BOLD}%s${RESET}\n" "$tag"
    printf "    size=1:    %s   (%s)\n" "${s1:-N/A}" "${v1:-N/A}"
    printf "    size=1000: %s   (%s)\n" "${s1k:-N/A}" "${v1k:-N/A}"
    echo
done

echo -e "${BOLD}═══ Interpretation ═══${RESET}"
echo "  Decision table (only size=1000 results matter — size=1 always PASSES):"
echo "    repro1=PASS, repro2=DRIFT, repro3=PASS  → vLLM is the precondition"
echo "    repro1=PASS, repro2=PASS,  repro3=DRIFT → FSDP2 is the precondition"
echo "    repro1=PASS, repro2=DRIFT, repro3=DRIFT → either independently triggers"
echo "    repro1=PASS, repro2=PASS,  repro3=PASS  → still need vLLM + FSDP2 together"
echo "                                              (would need a 4th repro)"
