#!/usr/bin/env bash
# Division 6: one line per radio: process state, control port, emane0 address, schedule state; then a health count.
# Exit code 0 only when every radio is healthy.
#
#   bash emane/scripts/status.sh emane/scenario/grid16_300m
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/lib.sh"
SCEN=$(scenario_dir "${1:-}")
N=$(scenario_n "$SCEN")
(( N > 0 )) || die "no platform*.xml in $SCEN"
W=$(id_width "$N")

healthy=0
while read -r nem name; do
    n=$(node_name "$nem" "$W"); ip="$BACKNET.$nem"
    if pid_alive "$SCEN/run/$n.pid"; then proc=running; else proc=STOPPED; fi
    # CONCEPT: EMANE control port & emanesh - each radio's control port (TCP 47000 on its backchan0 address) answers
    # configuration, statistics and table queries; that is how we look inside a running radio without touching it.
    if control_ok "$ip"; then ctl=OK; else ctl=FAIL; fi
    addr=$(ip -n "$n" -br addr show emane0 2>/dev/null | awk '{print $3}'); addr=${addr%/*}; [[ -n "$addr" ]] || addr=none
    acc=$(mac_stat "$ip" scheduler.scheduleAcceptFull)
    case "$acc" in
        0) sched=none ;;
        ''|*[!0-9]*) sched=unknown ;;
        *) sched="loaded (full schedules accepted: $acc)" ;;
    esac
    printf "%-8s NEM %2d  emane: %-8s control: %-4s emane0: %-12s schedule: %s\n" "$name" "$nem" "$proc" "$ctl" "$addr" "$sched"
    [[ "$proc" == running && "$ctl" == OK && "$addr" == "$RADIONET.$nem" ]] && healthy=$(( healthy + 1 ))
done < <(node_names "$SCEN")
echo "$healthy/$N radios healthy"
(( healthy == N ))
