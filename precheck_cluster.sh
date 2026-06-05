#!/bin/bash
# precheck_cluster.sh — verify a RunPod (or similar) cluster has a usable
# cross-node network BEFORE wasting time on env install / model download.
#
# Run on each node. Exits 0 only if all checks pass; nonzero with a
# clear diagnosis otherwise.
#
# Usage:
#   bash precheck_cluster.sh                     # local checks only
#   bash precheck_cluster.sh <peer_internal_ip>  # also ping peer node
#
# Override the iface auto-detect with:
#   IFACE_PATTERN='^(ens|ib)[0-9]+$'   # iface name regex
#   IFACE_NET='10.'                    # private IP prefix
#   MIN_IFACES=1                       # min cluster ifaces required (default 1)
#   MIN_HCAS=1                         # min HCAs ACTIVE required (default 1)

set -u

IFACE_PATTERN="${IFACE_PATTERN:-^(ens|ib)[0-9]+$}"
IFACE_NET="${IFACE_NET:-10.}"
MIN_IFACES="${MIN_IFACES:-1}"
MIN_HCAS="${MIN_HCAS:-1}"
PEER_IP="${1:-}"

PASS=$'\033[32m✓\033[0m'
FAIL=$'\033[31m✗\033[0m'
WARN=$'\033[33m!\033[0m'

errors=0
warnings=0

check_pass()  { printf "  %s %s\n" "$PASS" "$1"; }
check_fail()  { printf "  %s %s\n" "$FAIL" "$1"; errors=$((errors+1)); }
check_warn()  { printf "  %s %s\n" "$WARN" "$1"; warnings=$((warnings+1)); }

section() { printf "\n── %s ──\n" "$1"; }

section "host"
echo "  hostname=$(hostname)  eth0=$(ip -4 -o addr show eth0 2>/dev/null | awk '{print $4}')"

# ─── 1. cluster network ifaces ───
section "1. cluster network ifaces (need >=$MIN_IFACES iface matching $IFACE_PATTERN with ${IFACE_NET}* IP)"

mapfile -t ifaces < <(
    ip -4 -o addr show | awk '{sub(/@.*/, "", $2); print $2, $4}' |
        awk -v pat="$IFACE_PATTERN" -v net="$IFACE_NET" '
            $1 ~ pat && index($2, net) == 1 { print $1 " " $2 }
        ' | sort -u
)

if [[ ${#ifaces[@]} -lt $MIN_IFACES ]]; then
    check_fail "found ${#ifaces[@]} cluster ifaces (need $MIN_IFACES+). Only seeing:"
    ip -4 -o addr show | awk '{sub(/@.*/, "", $2); print "      "$2" "$4}'
    echo
    echo "  Likely cause: this pod was NOT provisioned as an Instant Cluster"
    echo "  (or the cluster network namespace isn't set up). Re-provision on"
    echo "  RunPod and pick the explicit Instant Cluster product. Without"
    echo "  cluster ifaces, cross-node NCCL has no transport."
else
    check_pass "found ${#ifaces[@]} cluster ifaces:"
    for line in "${ifaces[@]}"; do echo "      $line"; done
fi

# Primary IP for cross-node ping
if [[ ${#ifaces[@]} -gt 0 ]]; then
    OWN_IP=$(echo "${ifaces[0]}" | awk '{print $2}' | cut -d/ -f1)
else
    OWN_IP=""
fi

# ─── 2. HCAs ───
section "2. RDMA HCAs (need >=$MIN_HCAS ACTIVE)"

if [[ ! -d /sys/class/infiniband ]]; then
    check_fail "no /sys/class/infiniband — HCAs not exposed to container"
else
    active=0; total=0
    for h in /sys/class/infiniband/*/; do
        [[ -d $h ]] || continue
        name=$(basename "$h")
        total=$((total+1))
        st=$(cat "$h/ports/1/state" 2>/dev/null || echo "?")
        ll=$(cat "$h/ports/1/link_layer" 2>/dev/null || echo "?")
        if [[ "$st" == *ACTIVE* ]]; then
            active=$((active+1))
            check_pass "$name: $ll, $st"
        else
            check_warn "$name: $ll, $st (not ACTIVE — won't be used)"
        fi
    done
    if [[ $active -lt $MIN_HCAS ]]; then
        check_fail "only $active/$total HCAs ACTIVE (need $MIN_HCAS+)"
    else
        check_pass "$active/$total HCAs ACTIVE"
    fi
fi

# ─── 3. verbs devices ───
section "3. kernel verbs devices (/dev/infiniband)"

if [[ ! -d /dev/infiniband ]]; then
    check_fail "/dev/infiniband missing — NCCL can't use IB transport"
else
    uverbs=$(ls /dev/infiniband/uverbs* 2>/dev/null | wc -l)
    if [[ $uverbs -lt 1 ]]; then
        check_fail "no uverbs* devices in /dev/infiniband — verbs userspace blocked"
    else
        check_pass "$uverbs uverbs devices: $(ls /dev/infiniband/uverbs* | xargs -n1 basename | tr '\n' ' ')"
    fi
fi

# ─── 4. NCCL version (must be 2.29.7+ for suspend/resume) ───
section "4. NCCL version on disk (need 2.29.7+ for ncclCommSuspend)"

# Don't trust torch.cuda.nccl.version() — it reports the version torch was
# built against, not what's loaded at runtime. Check the .so directly.
nccl_so=$(find /opt/conda -name 'libnccl.so.2.*' 2>/dev/null | head -1)
[[ -z "$nccl_so" ]] && nccl_so=$(find /usr -name 'libnccl.so.2.*' 2>/dev/null | head -1)

if [[ -z "$nccl_so" ]]; then
    check_warn "libnccl.so.2.* not found on disk — torch may be using bundled NCCL"
else
    ver=$(basename "$nccl_so" | sed 's/libnccl\.so\.//')
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    patch=$(echo "$ver" | cut -d. -f3)
    if [[ "$major" -gt 2 ]] || \
       { [[ "$major" -eq 2 ]] && [[ "$minor" -gt 29 ]]; } || \
       { [[ "$major" -eq 2 ]] && [[ "$minor" -eq 29 ]] && [[ "$patch" -ge 7 ]]; }; then
        check_pass "NCCL $ver at $nccl_so"
    else
        check_warn "NCCL $ver < 2.29.7. Run upgrade_nccl.sh before training (suspend feature needs 2.29.7+)"
    fi
fi

# ─── 5. cross-node connectivity (only if peer IP given) ───
section "5. cross-node ping (peer=${PEER_IP:-<not provided>})"

if [[ -z "$PEER_IP" ]]; then
    echo "  (skipped — no peer IP. Re-run as: bash precheck_cluster.sh <peer_internal_ip>)"
elif [[ -z "$OWN_IP" ]]; then
    check_fail "can't ping — no cluster iface to send from"
else
    if ping -c 2 -W 3 -I "${ifaces[0]%% *}" "$PEER_IP" >/dev/null 2>&1; then
        check_pass "ping $PEER_IP via $(echo "${ifaces[0]}" | awk '{print $1}') OK (own IP $OWN_IP)"
    else
        check_fail "ping $PEER_IP via $(echo "${ifaces[0]}" | awk '{print $1}') FAILED. Cluster network isn't routing between nodes."
    fi
fi

# ─── verdict ───
section "verdict"
if [[ $errors -gt 0 ]]; then
    printf "  %s FAIL — %d error(s), %d warning(s). Do NOT proceed to Step 1.\n" "$FAIL" "$errors" "$warnings"
    exit 1
elif [[ $warnings -gt 0 ]]; then
    printf "  %s PASS with %d warning(s). Safe to proceed but check the warnings above.\n" "$WARN" "$warnings"
    exit 0
else
    printf "  %s PASS — cluster looks healthy. Safe to run Step 1 (clone / setup_env / download_models).\n" "$PASS"
    exit 0
fi
