#!/usr/bin/env bash
# Division 6: remove the backplane created by net_up.sh: the N namespaces (their veth ends go with them) and the
# bridge. Safe to run when nothing exists.
#
#   bash emane/scripts/net_down.sh <N>
set -u
source "$(dirname "$0")/lib.sh"
N=${1:-}
[[ "$N" =~ ^[0-9]+$ && "$N" -ge 1 ]] || die "usage: net_down.sh <N>"
W=$(id_width "$N")
for (( i = 1; i <= N; i++ )); do
    ip netns del "$(node_name "$i" "$W")" 2>/dev/null     # deleting the namespace destroys backchan0 and so its veth peer
    ip link del "$(veth_name "$i" "$W")" 2>/dev/null       # in case only the root end was left behind
done
ip link del "$BR" 2>/dev/null
echo "backplane down: $BR and $N namespaces removed"
