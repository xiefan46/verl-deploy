#!/bin/bash
# diagnose_cluster_net.sh — dump everything needed to figure out why
# setup_cluster.sh can't find a usable cluster network.
#
# Run on each node; paste the output back. Covers:
#   - All ifaces (with state, IPs, MAC), including ones without IPs
#   - All RDMA HCAs + their link_layer (IB vs RoCE) and port state
#   - /dev/infiniband presence (kernel verbs available?)
#   - Default route + hostname (to compare two nodes' network topology)
#   - NCCL version (must be >= 2.29.7 for suspend/resume)

set -u

hr() { printf '\n──── %s ────\n' "$1"; }

hr "hostname + default route"
hostname
ip route get 1.1.1.1 2>/dev/null || echo "(no default route)"

hr "all ifaces (incl. DOWN / no IP)"
ip -o link show | awk -F': ' '{print $2, $3}' | sed 's/<[^>]*>//' | column -t

hr "all IPv4 addresses"
ip -4 -o addr show | awk '{sub(/@.*/, "", $2); print $2, $4}'

hr "RDMA HCAs (/sys/class/infiniband)"
if [[ -d /sys/class/infiniband ]]; then
    for h in /sys/class/infiniband/*/; do
        name=$(basename "$h")
        ll=$(cat "$h/ports/1/link_layer" 2>/dev/null || echo "?")
        st=$(cat "$h/ports/1/state" 2>/dev/null || echo "?")
        rate=$(cat "$h/ports/1/rate" 2>/dev/null || echo "?")
        echo "$name: link_layer=$ll  state=$st  rate=$rate"
    done
else
    echo "(no /sys/class/infiniband — HCAs not exposed to container)"
fi

hr "/dev/infiniband (verbs devices)"
ls -1 /dev/infiniband/ 2>/dev/null || echo "(missing — kernel verbs not exposed)"

hr "NCCL version"
python -c "import torch; print('torch', torch.__version__); print('nccl', torch.cuda.nccl.version())" 2>/dev/null \
    || echo "(python/torch not available)"

hr "ibstat (if installed)"
ibstat 2>/dev/null | head -40 || echo "(ibstat not installed — apt-get install -y infiniband-diags if needed)"

hr "Done"
echo "Paste the entire output above. Most important:"
echo "  - link_layer (InfiniBand vs Ethernet) — determines RoCE vs native IB"
echo "  - port state (must be ACTIVE for HCA to be usable)"
echo "  - which node-internal IP is reachable from the OTHER node"
