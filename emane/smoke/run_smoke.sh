#!/usr/bin/env bash
# Division 5 smoke test: 2 EMANE TDMA radios, each in its own network namespace, inside this one container.
# It proves, before we scale to 16 radios in Division 6, that:
#   1. the backplane wiring works (multicast over a Linux bridge between namespaces),
#   2. a TDMA radio is SILENT without a schedule (ping: 100 % loss),
#   3. after publishing a 2-slot schedule the ping succeeds (~0 % loss),
#   4. the schedule can be read back from each radio with emanesh,
#   5. how well this host keeps 1 ms slots (TX slots fired late, receptions processed after the slot ended).
#
# Usage (inside the container, any directory):
#   bash emane/smoke/run_smoke.sh                 # 1 ms slots (the brief's value)  -> results/emane/d5_smoke.txt
#   SLOT_US=10000 bash emane/smoke/run_smoke.sh   # 10 ms slots                    -> results/emane/d5_smoke_10000us.txt
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
SMOKE="$ROOT/emane/smoke"
BR=tdmabr0
N=2
SLOT_US="${SLOT_US:-1000}"
if [[ "$SLOT_US" == "1000" ]]; then
    OUT="$ROOT/results/emane/d5_smoke.txt"
    SCHEDULE="$SMOKE/schedule.xml"
else
    OUT="$ROOT/results/emane/d5_smoke_${SLOT_US}us.txt"
    SCHEDULE="$SMOKE/run/schedule_${SLOT_US}us.xml"     # same hand-written schedule, other slot length
fi
mkdir -p "$ROOT/results/emane" "$SMOKE/logs" "$SMOKE/run"
[[ "$SLOT_US" == "1000" ]] || sed "s/slotduration=\"1000\"/slotduration=\"$SLOT_US\"/" "$SMOKE/schedule.xml" > "$SCHEDULE"

teardown() {
    echo
    echo "[8/8] teardown"
    for i in $(seq 1 $N); do
        n=$(printf "node%02d" "$i")
        pf="$SMOKE/run/$n.pid"
        if [[ -f "$pf" ]]; then kill "$(cat "$pf")" 2>/dev/null; rm -f "$pf"; fi
        ip netns del "$n" 2>/dev/null
    done
    pkill -f "emane platform[12].xml" 2>/dev/null
    ip link del "$BR" 2>/dev/null
    echo "[8/8] done"
}

quiet_cleanup() {   # leftovers from an aborted run (idempotent)
    for i in $(seq 1 $N); do ip netns del "$(printf "node%02d" "$i")" 2>/dev/null; done
    pkill -f "emane platform[12].xml" 2>/dev/null
    ip link del "$BR" 2>/dev/null
    rm -f "$SMOKE"/run/*.pid "$SMOKE"/run/*.uuid
    return 0
}

wait_ready() {   # $1 = backplane IP of a radio; returns 0 once its control port answers
    for _ in $(seq 1 40); do
        if emanesh "$1" get config nems mac >/dev/null 2>&1; then return 0; fi
        sleep 0.25
    done
    return 1
}

start_radio() {   # $1 = i ; tries real-time priority first (-r), then without it
    local i="$1" n
    n=$(printf "node%02d" "$i")
    # CONCEPT: EMANE architecture - one emane daemon per radio, started INSIDE that radio's namespace so the
    # TUN device emane0 and the backplane interface backchan0 it uses are the namespace's own.
    if ip netns exec "$n" emane "platform$i.xml" -r -d -l 3 -f "$SMOKE/logs/$n.log" \
            --pidfile "$SMOKE/run/$n.pid" --uuidfile "$SMOKE/run/$n.uuid" >/dev/null 2>&1 \
       && sleep 0.5 && [[ -f "$SMOKE/run/$n.pid" ]] && kill -0 "$(cat "$SMOKE/run/$n.pid")" 2>/dev/null; then
        echo "  $n: emane NEM $i started (real-time priority)"
    else
        rm -f "$SMOKE/run/$n.pid"
        ip netns exec "$n" emane "platform$i.xml" -d -l 3 -f "$SMOKE/logs/$n.log" \
            --pidfile "$SMOKE/run/$n.pid" --uuidfile "$SMOKE/run/$n.uuid"
        echo "  $n: emane NEM $i started WITHOUT real-time priority (-r refused, see logs/$n.log)"
    fi
}

tx_slots() {   # $1 = backplane IP; prints the TX slot indexes from the radio's ScheduleInfoTable
    emanesh "$1" get table nems mac scheduler.ScheduleInfoTable 2>/dev/null \
        | awk -F'|' '$5 ~ /TX/ { gsub(/ /, "", $4); s = s (s ? "," : "") $4 } END { print (s ? s : "none") }'
}

stat() {   # $1 = backplane IP, $2 = statistic name; prints its value
    emanesh "$1" get stat nems mac "$2" 2>/dev/null | awk -F'= ' '{print $2}' | tail -1
}

main() {
    echo "=== D5 smoke test  $(date -u +'%Y-%m-%dT%H:%M:%SZ')  emane $(emane --version 2>/dev/null | head -1)  kernel $(uname -r)  slot = $SLOT_US us ==="
    quiet_cleanup
    trap teardown EXIT

    echo
    echo "[1/8] wiring: bridge $BR (10.99.0.254/24) + $N namespaces"
    # CONCEPT: veth + Linux bridge - the bridge is a virtual switch in the root namespace; each radio's namespace
    # is plugged into it with a veth pair: "vethNN" on the bridge side, "backchan0" inside the namespace.
    ip link add name "$BR" type bridge
    # CONCEPT: multicast - snooping OFF makes the bridge flood the OTA and event multicast groups to every port
    ip link set "$BR" type bridge stp_state 0 forward_delay 0 mcast_snooping 0
    ip addr add 10.99.0.254/24 dev "$BR"
    ip link set "$BR" up
    ip route replace 224.1.2.8 dev "$BR"        # publishers in the root namespace send multicast via the bridge
    for i in $(seq 1 $N); do
        n=$(printf "node%02d" "$i"); v=$(printf "veth%02d" "$i")
        # CONCEPT: Network namespace (first contact) - a private network stack per radio: own interfaces, IPs, routes
        ip netns add "$n"
        ip link add "$v" type veth peer name backchan0 netns "$n"
        ip link set "$v" master "$BR" up
        ip -n "$n" addr add "10.99.0.$i/24" dev backchan0
        ip -n "$n" link set lo up
        ip -n "$n" link set backchan0 up
        ip -n "$n" route replace 224.1.2.8 dev backchan0
        echo "  $n: backchan0 10.99.0.$i <-> $v on $BR"
    done

    echo
    echo "[2/8] starting $N emane daemons (one NEM each)"
    cd "$SMOKE" || exit 1          # relative definition= paths in the XML resolve from here
    for i in $(seq 1 $N); do start_radio "$i"; done
    cd "$ROOT" || exit 1
    for i in $(seq 1 $N); do
        n=$(printf "node%02d" "$i")
        if wait_ready "10.99.0.$i"; then
            echo "  $n: control port OK, emane0 = $(ip -n "$n" -br addr show emane0 2>/dev/null | awk '{print $3}')"
        else
            echo "  $n: control port NOT answering; last log lines:"; tail -5 "$SMOKE/logs/$n.log"; exit 1
        fi
    done

    echo
    echo "[3/8] pathloss 1<->2 = 80 dB (precomputed propagation: this event IS the topology)"
    emaneevent-pathloss 1:2 80 -i "$BR" && echo "  published: $(emanesh 10.99.0.1 get table nems phy PathlossEventInfoTable 2>/dev/null | grep -E '^\| [0-9]' | tr -s ' ')"
    sleep 0.5

    echo
    echo "[4/8] ping BEFORE any schedule (expect 100% loss; the '+N errors' are 'Destination Host Unreachable': even ARP cannot get through)"
    # CONCEPT: No schedule -> silent radio - until a TDMA schedule event arrives every slot is IDLE:
    # the radio never transmits, and the outgoing packets just queue up.
    ip netns exec node01 ping -c 5 -W 1 10.100.0.2 2>&1 | tail -2 | head -1 | sed 's/^/  [before schedule] /'
    echo "  scheduleAcceptFull = $(stat 10.99.0.1 scheduler.scheduleAcceptFull) (no schedule yet)"

    echo
    echo "[5/8] publish the 2-slot schedule: emaneevent-tdmaschedule -i $BR ${SCHEDULE#"$ROOT/"}"
    emaneevent-tdmaschedule -i "$BR" "$SCHEDULE"
    echo "  exit code: $?  (1 would mean the XML failed EMANE's XSD validation)"
    sleep 1

    echo
    echo "[6/8] ping AFTER the schedule (expect ~0% loss)"
    # ARP first: the very first frames may be processed late at 1 ms slots (see [7/8]) and the kernel only retries ARP once a second
    ip netns exec node01 ping -c 2 -W 3 10.100.0.2 >/dev/null 2>&1
    ip netns exec node01 ping -c 5 -i 0.2 -W 2 10.100.0.2 2>&1 | tail -2 | sed 's/^/  [after schedule] /'

    echo
    echo "[7/8] read back the schedule and the slot statistics from each radio (emanesh)"
    for i in $(seq 1 $N); do
        echo "  --- NEM $i (10.99.0.$i) ---"
        emanesh "10.99.0.$i" get table nems mac scheduler.StructureInfoTable scheduler.ScheduleInfoTable 2>&1 | sed 's/^/  /'
        emanesh "10.99.0.$i" get stat nems mac scheduler.scheduleAcceptFull scheduler.scheduleAcceptUpdate \
            numTxSlotValid numTxSlotErrorMissed numRxSlotValid numRxSlotErrorMissed 2>&1 | sed 's/^/  /'
        # TxSlotStatusTable: how late the TX slot timer fired, in fractions of a slot; >= 1.0 = after the slot ended = Missed.
        # RxSlotStatusTable: same for receptions; a reception processed after its slot ended is dropped ("Slot Error").
        emanesh "10.99.0.$i" get table nems mac TxSlotStatusTable RxSlotStatusTable 2>&1 | sed 's/^/  /'
    done
    echo "  [readback] NEM 1: TX slot $(tx_slots 10.99.0.1) | NEM 2: TX slot $(tx_slots 10.99.0.2)"
    echo "  [timing @ $SLOT_US us] TX slots missed: NEM1 $(stat 10.99.0.1 numTxSlotErrorMissed)/$(stat 10.99.0.1 numTxSlotValid), NEM2 $(stat 10.99.0.2 numTxSlotErrorMissed)/$(stat 10.99.0.2 numTxSlotValid); receptions processed too late (dropped): NEM1 $(stat 10.99.0.1 numRxSlotErrorMissed) of $(( $(stat 10.99.0.1 numRxSlotValid) + $(stat 10.99.0.1 numRxSlotErrorMissed) )), NEM2 $(stat 10.99.0.2 numRxSlotErrorMissed) of $(( $(stat 10.99.0.2 numRxSlotValid) + $(stat 10.99.0.2 numRxSlotErrorMissed) ))"
}

main 2>&1 | tee "$OUT"
exit "${PIPESTATUS[0]}"
