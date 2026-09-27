#!/usr/bin/env bash
# Division 6: stop every emane daemon of a scenario (by pidfile) and remove the backplane. Safe to run twice.
#
#   bash emane/scripts/down.sh emane/scenario/grid16_300m
#   bash emane/scripts/down.sh <scenario_dir> --quiet     (used by up.sh for a clean restart)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/lib.sh"
SCEN=$(scenario_dir "${1:-}")
QUIET=${2:-}
N=$(scenario_n "$SCEN")
(( N > 0 )) || N=1
W=$(id_width "$N")

stopped=0
for (( i = 1; i <= N; i++ )); do
    pf="$SCEN/run/$(node_name "$i" "$W").pid"
    if pid_alive "$pf"; then kill "$(cat "$pf")" 2>/dev/null && stopped=$(( stopped + 1 )); fi
done
for _ in $(seq 1 20); do                              # give them up to 5 s to exit cleanly
    alive=0
    for (( i = 1; i <= N; i++ )); do pid_alive "$SCEN/run/$(node_name "$i" "$W").pid" && alive=1; done
    (( alive == 0 )) && break
    sleep 0.25
done
for (( i = 1; i <= N; i++ )); do
    pf="$SCEN/run/$(node_name "$i" "$W").pid"
    pid_alive "$pf" && kill -9 "$(cat "$pf")" 2>/dev/null
done
rm -f "$SCEN"/run/*.pid "$SCEN"/run/*.uuid
pkill -f "emane .*platform[0-9]*.xml" 2>/dev/null     # strays started by hand (the smoke test uses the same bridge)
bash "$HERE/net_down.sh" "$N" >/dev/null 2>&1
[[ "$QUIET" == "--quiet" ]] || echo "down: $stopped daemon(s) stopped, backplane $BR and $N namespaces removed"
