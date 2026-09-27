#!/usr/bin/env bash
# Division 6: create the backplane: bridge tdmabr0 (10.99.0.254/24) plus one network namespace per radio,
# each plugged into the bridge with a veth pair. Idempotent: running it twice leaves the same wiring.
#
#   bash emane/scripts/net_up.sh <N>          (N radios, 1..253)
set -u
source "$(dirname "$0")/lib.sh"
N=${1:-}
[[ "$N" =~ ^[0-9]+$ && "$N" -ge 1 && "$N" -le 253 ]] || die "usage: net_up.sh <N> (1..253 radios)"
W=$(id_width "$N")

# CONCEPT: veth pairs & Linux bridges - the bridge is a virtual Ethernet switch living in the root namespace;
# a veth pair is a virtual cable: one end (vethXX) is plugged into the switch, the other (backchan0) into the radio's
# namespace. Everything EMANE sends between radios (OTA frames, events, emanesh) travels over this backplane.
if ! ip link show "$BR" >/dev/null 2>&1; then
    ip link add name "$BR" type bridge
fi
# CONCEPT: Multicast groups & IGMP snooping - with snooping ON a bridge forwards a multicast group only to ports that
# sent an IGMP join; OFF means "flood to every port", which is exactly what the OTA and event groups need here.
ip link set "$BR" type bridge stp_state 0 forward_delay 0 mcast_snooping 0
ip addr replace "$BR_IP" dev "$BR"
ip link set "$BR" up
ip route replace "$MCAST" dev "$BR"          # publishers in the root namespace (pathloss.py, emaneevent-*) use the bridge

for (( i = 1; i <= N; i++ )); do
    n=$(node_name "$i" "$W"); v=$(veth_name "$i" "$W")
    # CONCEPT: Linux network namespaces - each radio gets a private network stack (own interfaces, addresses, routes,
    # ARP table); "ip netns exec nodeXX ..." runs a command inside it. The emane daemon of NEM i lives in nodeXX.
    [[ -e "/var/run/netns/$n" ]] || ip netns add "$n"
    if ! ip link show "$v" >/dev/null 2>&1; then
        ip link add "$v" type veth peer name backchan0 netns "$n"
    fi
    ip link set "$v" master "$BR" up
    ip -n "$n" addr replace "$BACKNET.$i/24" dev backchan0
    ip -n "$n" link set lo up
    ip -n "$n" link set backchan0 up
    ip -n "$n" route replace "$MCAST" dev backchan0
done

# Firewall trap: with br_netfilter, bridged frames pass the iptables FORWARD chain; a DROP policy silently eats the
# OTA and event multicast. (Docker Desktop can set FORWARD to DROP on the host; inside the container it is ACCEPT.)
pol=$(iptables -S FORWARD 2>/dev/null | head -1)
nf=$(cat /proc/sys/net/bridge/bridge-nf-call-iptables 2>/dev/null || echo absent)
if [[ "$pol" == *DROP* && "$nf" == "1" ]]; then
    echo "WARNING: iptables FORWARD policy is DROP and bridge-nf-call-iptables=1: bridged multicast will be dropped." >&2
    echo "         fix: iptables -P FORWARD ACCEPT   (or: sysctl -w net.bridge.bridge-nf-call-iptables=0)" >&2
fi
echo "backplane up: $BR $BR_IP, $N namespaces $(node_name 1 "$W")..$(node_name "$N" "$W") (backchan0 = $BACKNET.1..$N), FORWARD '${pol:-n/a}'"
