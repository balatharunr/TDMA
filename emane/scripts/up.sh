#!/usr/bin/env bash
# Division 6: bring up every radio of a generated scenario: backplane, DTD validation of the XML, one emane daemon
# per radio inside its namespace, wait for all control ports, then publish the pathloss events that mirror our
# 500 m graph and read them back. NO TDMA schedule is loaded here (that is Division 7): the radios stay silent.
#
#   bash emane/scripts/up.sh emane/scenario/grid16_300m
#   EMANE_RT=0 bash emane/scripts/up.sh ...      start the daemons without real-time scheduling
#
# Idempotent: it first runs down.sh, so a second "up" is a clean restart.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/lib.sh"
ROOT=$(cd "$HERE/../.." && pwd)
SCEN=$(scenario_dir "${1:-}")
N=$(scenario_n "$SCEN")
(( N > 0 )) || die "no platform*.xml in $SCEN (run: python3 emane/gen/generate_scenario.py --schedule ... --out $SCEN)"
W=$(id_width "$N")
t0=$(date +%s)

echo "=== up: $SCEN ($N radios)  $(date -u +'%Y-%m-%dT%H:%M:%SZ')  emane $(emane --version 2>/dev/null) ==="
bash "$HERE/down.sh" "$SCEN" --quiet
mkdir -p "$SCEN/logs" "$SCEN/run"

echo "[1/5] backplane"
bash "$HERE/net_up.sh" "$N" | sed 's/^/  /'

echo "[2/5] validating the XML against the installed DTDs (/usr/share/emane/dtd)"
if command -v xmllint >/dev/null 2>&1; then
    for f in "$SCEN"/platform*.xml "$SCEN"/tdmanem.xml "$SCEN"/tdmaradiomodel.xml "$SCEN"/transvirtual.xml; do
        xmllint --noout --valid "$f" 2>&1 | sed 's/^/  /' || true
        [[ ${PIPESTATUS[0]} -eq 0 ]] || die "invalid XML: $f"
    done
    echo "  $(( N + 3 )) files valid"
else
    echo "  xmllint not installed, skipped"
fi

echo "[3/5] starting $N emane daemons (one NEM each, inside their namespaces, real-time scheduling: $EMANE_RT)"
# CONCEPT: Bash scripting for orchestration (pidfiles) - every daemon writes its pid to run/nodeXX.pid, which is
# how down.sh and status.sh find it again; logs go to logs/nodeXX.log.
cd "$SCEN" || exit 1                                # the relative definition="..." paths in the XML resolve from here
start_daemon() {                                    # $1 = i, $2 = "-r" or ""
    local n; n=$(node_name "$1" "$W")
    # shellcheck disable=SC2086
    ip netns exec "$n" emane $2 -d -l "$EMANE_LOGLEVEL" -f "$SCEN/logs/$n.log" \
        --pidfile "$SCEN/run/$n.pid" --uuidfile "$SCEN/run/$n.uuid" "platform$(printf "%0${W}d" "$1").xml"
}
rt=""; (( EMANE_RT )) && rt="-r"
for (( i = 1; i <= N; i++ )); do start_daemon "$i" "$rt"; done
sleep 1
for (( i = 1; i <= N; i++ )); do
    n=$(node_name "$i" "$W")
    if ! pid_alive "$SCEN/run/$n.pid"; then
        echo "  $n: exited right after start$( [[ -n $rt ]] && echo ' (real-time scheduling refused?)'); retrying without -r"
        rm -f "$SCEN/run/$n.pid"
        start_daemon "$i" ""
    fi
done
cd "$ROOT" || exit 1

echo "[4/5] waiting for the $N control ports"
failed=0
for (( i = 1; i <= N; i++ )); do
    n=$(node_name "$i" "$W")
    if ! wait_ready "$BACKNET.$i" 60; then
        echo "  $n: control port $BACKNET.$i:47000 not answering; log tail:"
        tail -3 "$SCEN/logs/$n.log" | sed 's/^/    /'
        failed=$(( failed + 1 ))
    fi
done
(( failed == 0 )) || die "$failed radio(s) did not come up"
echo "  $N/$N control ports answering, emane0 present on $(for (( i = 1; i <= N; i++ )); do ip -n "$(node_name "$i" "$W")" -br addr show emane0 2>/dev/null | grep -c "$RADIONET"; done | awk '{s+=$1} END {print s}')/$N radios ($(( $(date +%s) - t0 )) s since start)"

echo "[5/5] pathloss events: EMANE's connectivity := our 500 m graph (links.csv), then read back from every radio"
python3 "$ROOT/emane/bridge/pathloss.py" --scenario "$SCEN" --publish --verify \
    --csv "$ROOT/results/emane/pathloss_matrix.csv" --readback-out "$ROOT/results/emane/d6_pathloss_readback.txt" | sed 's/^/  /'
rc=${PIPESTATUS[0]}
(( rc == 0 )) || die "pathloss publish/verify failed (exit $rc)"
echo "=== up: $N radios running, no TDMA schedule loaded yet ($(( $(date +%s) - t0 )) s) ==="
