#!/usr/bin/env bash
# Division 6: settings and helpers shared by the emane/scripts/*.sh orchestration scripts. Sourced, never run.
# The naming rule: Node_XX (natural-order index i) -> NEM i -> netns nodeXX -> backchan0 10.99.0.i -> emane0 10.100.0.i

BR=${BR:-tdmabr0}                 # the backplane bridge in the root namespace
BR_IP=10.99.0.254/24
MCAST=224.1.2.8                   # OTA (port 45702) and event (port 45703) multicast group
BACKNET=10.99.0                   # backchan0 = 10.99.0.<nem>
RADIONET=10.100.0                 # emane0    = 10.100.0.<nem>
EMANE_RT=${EMANE_RT:-1}           # 1 = start emane with real-time scheduling (-r); 0 = without
EMANE_LOGLEVEL=${EMANE_LOGLEVEL:-3}

die() { echo "ERROR: $*" >&2; exit 1; }

scenario_dir() {   # $1 = scenario directory argument -> absolute path, or die with usage
    [[ -n "${1:-}" && -d "$1" ]] || die "usage: $(basename "$0") <scenario_dir>   (e.g. emane/scenario/grid16_300m)"
    ( cd "$1" && pwd )
}

scenario_n() {     # $1 = scenario dir -> number of radios (one platformNN.xml each)
    ls "$1"/platform*.xml 2>/dev/null | wc -l | tr -d ' '
}

id_width() {       # $1 = N -> digits in nodeXX/vethXX/platformXX (2, or more for 100+ radios; generate_scenario.py agrees)
    local w=${#1}
    (( w < 2 )) && w=2
    echo "$w"
}

node_name() { printf "node%0${2:-2}d" "$1"; }   # node_name <i> <width>
veth_name() { printf "veth%0${2:-2}d" "$1"; }   # veth_name <i> <width>

pid_alive() {      # $1 = pidfile
    [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null
}

control_ok() {     # $1 = backplane IP of a radio -> 0 if its control port (TCP 47000) answers
    timeout 3 emanesh "$1" get config nems mac >/dev/null 2>&1
}

wait_ready() {     # $1 = backplane IP, $2 = timeout seconds (default 60) -> 0 once the control port answers
    local ip=$1 deadline=$(( $(date +%s) + ${2:-60} ))
    until control_ok "$ip"; do
        (( $(date +%s) >= deadline )) && return 1
        sleep 0.25
    done
}

mac_stat() {       # $1 = backplane IP, $2 = MAC statistic name -> its value (empty if unreachable)
    timeout 3 emanesh "$1" get stat nems mac "$2" 2>/dev/null | awk -F'= ' 'END { print $2 }'
}

node_names() {     # $1 = scenario dir -> lines "<nem> <node name>" in NEM order, from nemmap.json
    python3 - "$1/nemmap.json" <<'PY'
import json, sys
nemmap = json.load(open(sys.argv[1]))
for name, entry in sorted(nemmap.items(), key=lambda kv: kv[1]["nem"]):
    print(entry["nem"], name)
PY
}
