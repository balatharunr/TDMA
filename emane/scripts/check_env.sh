#!/usr/bin/env bash
# Division 5: environment check for the Part 2 lab (inside the tdma-emane container, or WSL2 Ubuntu for Path B).
# Prints one PASS/FAIL line per check with a fix hint, saves the PCR-curve rows that the D8 collision
# experiment relies on, and exits 1 if anything failed.
#
# Usage (repo root inside the container):  bash emane/scripts/check_env.sh
set -u

pass=0
fail=0
PASS() { echo "PASS  $1"; pass=$((pass + 1)); }
FAIL() { echo "FAIL  $1"; echo "      fix: $2"; fail=$((fail + 1)); }

MODEL_DIR=/usr/share/emane/xml/models/mac/tdmaeventscheduler
OUT_DIR=results/emane
mkdir -p "$OUT_DIR"

echo "== EMANE installation =="
ver="$(emane --version 2>/dev/null | head -1)"
if [[ "$ver" == *1.5.3* ]]; then PASS "emane version $ver"; else FAIL "emane --version (got '${ver:-nothing}')" "rebuild the image: docker build -t tdma-emane emane/docker"; fi

if python3 -c "import emane.events" 2>/dev/null; then PASS "python3 'import emane.events'"; else FAIL "python3 'import emane.events'" "install python3-emane_*.deb and python3-pkg-resources"; fi

if emaneevent-tdmaschedule --help >/dev/null 2>&1; then PASS "emaneevent-tdmaschedule --help"; else FAIL "emaneevent-tdmaschedule --help" "the emane package did not install its python tools"; fi

for f in tdmanem.xml tdmaradiomodel.xml tdmabasemodelpcr.xml; do
    if [[ -f "$MODEL_DIR/$f" ]]; then PASS "$MODEL_DIR/$f"; else FAIL "$MODEL_DIR/$f missing" "install emane-model-tdmaeventscheduler (part of the emane*.deb set)"; fi
done

if [[ -f /usr/share/emane/dtd/platform.dtd ]]; then PASS "/usr/share/emane/dtd/platform.dtd"; else FAIL "/usr/share/emane/dtd/platform.dtd missing" "install the emane-dtds package"; fi

pkgdir="$(python3 -c "import emane.events, os; print(os.path.dirname(emane.events.__file__))" 2>/dev/null)"
xsd="${pkgdir:-/nonexistent}/schema/tdmaschedule.xsd"
if [[ -f "$xsd" ]]; then PASS "schedule XSD $xsd"; else FAIL "tdmaschedule.xsd not found under $pkgdir" "install python3-emane_*.deb"; fi

echo "== Kernel features (need --privileged) =="
# CONCEPT: TUN device emane0 - EMANE's virtual transport opens /dev/net/tun to create each radio's emane0 interface
if [[ -c /dev/net/tun ]]; then PASS "/dev/net/tun"; else FAIL "/dev/net/tun missing" "run the container with --privileged (emane/docker/run.ps1 does)"; fi

ns="chk$$"; br="chkbr$$"; v="chkv$$"
# CONCEPT: Network namespace (first contact) - a throw-away namespace + bridge + veth pair, exactly what the lab wiring does
if ip netns add "$ns" 2>/dev/null && ip link add "$br" type bridge 2>/dev/null \
   && ip link add "$v" type veth peer name backchan0 netns "$ns" 2>/dev/null \
   && ip link set "$v" master "$br" up 2>/dev/null; then
    PASS "netns + bridge + veth create/delete"
else
    FAIL "netns/bridge/veth creation" "needs CAP_NET_ADMIN: run the container with --privileged"
fi
ip link del "$v" 2>/dev/null; ip link del "$br" 2>/dev/null; ip netns del "$ns" 2>/dev/null

echo "== Firewall trap =="
pol="$(iptables -S FORWARD 2>/dev/null | head -1)"
nf="$(cat /proc/sys/net/bridge/bridge-nf-call-iptables 2>/dev/null || echo absent)"
if [[ "$pol" == *DROP* && "$nf" == "1" ]]; then
    FAIL "FORWARD policy is DROP and bridge-nf-call-iptables=1: bridged multicast (OTA + events) would be silently dropped" \
         "iptables -P FORWARD ACCEPT    (or: sysctl -w net.bridge.bridge-nf-call-iptables=0)"
else
    PASS "firewall: FORWARD policy '${pol:-n/a}', bridge-nf-call-iptables=$nf"
fi

echo "== Tools =="
for t in ping tcpdump iptables xmllint iperf3; do
    if command -v "$t" >/dev/null 2>&1; then PASS "$t"; else FAIL "$t missing" "apt-get install -y $t (see the Dockerfile)"; fi
done
if python3 -c "import networkx" 2>/dev/null; then PASS "python3 networkx"; else FAIL "python3 networkx" "apt-get install -y python3-networkx"; fi

echo "== PCR curve (delivery probability vs SINR, per data rate) =="
# The D8 hidden-terminal experiment relies on 11M giving 0 % delivery at SINR <= 1 dB (an equal-power collision
# is ~0 dB), while 1M would still deliver at 0 dB (capture effect). Record the installed rows as evidence.
if python3 - "$MODEL_DIR/tdmabasemodelpcr.xml" "$OUT_DIR/d5_pcr_curve.txt" <<'PY'
import sys, xml.etree.ElementTree as ET
src, dst = sys.argv[1], sys.argv[2]
root = ET.parse(src).getroot()
lines = [f"# {src}  packetsize={root.get('packetsize')}"]
first_nonzero = {}
for dr in root.iter("datarate"):
    bps = dr.get("bps")
    rows = [(float(e.get("sinr")), float(e.get("por"))) for e in dr.iter("entry")]
    lines.append(f"datarate {bps}: " + ", ".join(f"{s:g} dB -> {p:g} %" for s, p in rows))
    nz = [s for s, p in rows if p > 0]
    first_nonzero[bps] = min(nz) if nz else None
open(dst, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
ok = first_nonzero.get("11M") is not None and first_nonzero["11M"] > 1.0 \
     and first_nonzero.get("1M") is not None and first_nonzero["1M"] <= -2.0
sys.exit(0 if ok else 1)
PY
then PASS "PCR curve: 11M is 0 % at SINR <= 1 dB and 1M delivers at 0 dB (saved to $OUT_DIR/d5_pcr_curve.txt)"
else FAIL "PCR curve differs from the guide's A.7 (see $OUT_DIR/d5_pcr_curve.txt)" "keep 11M only if its first non-zero row is above 1 dB; otherwise pick the lowest rate that is"; fi

echo
echo "$pass passed, $fail failed"
[[ $fail -eq 0 ]]
