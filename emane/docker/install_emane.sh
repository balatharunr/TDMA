#!/usr/bin/env bash
# Path B (fallback): install EMANE 1.5.3 directly inside the WSL2 Ubuntu 24.04 distro.
# Same steps as the Dockerfile, run with sudo. Division 5.
#
#   wsl -d Ubuntu
#   bash /mnt/y/TDMA/emane/docker/install_emane.sh
set -euo pipefail

EMANE_VERSION="${EMANE_VERSION:-1.5.3}"
EMANE_DISTRO="${EMANE_DISTRO:-ubuntu-24_04}"
BUNDLE="emane-${EMANE_VERSION}-release-1.${EMANE_DISTRO}.amd64.tar.gz"
URL="https://adjacentlink.com/downloads/emane/${BUNDLE}"

echo "== apt packages =="
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    ca-certificates wget iproute2 iputils-ping tcpdump iperf3 iptables procps psmisc less nano \
    python3 python3-networkx python3-matplotlib python3-pytest \
    python3-lxml python3-protobuf python3-pkg-resources libxml2-utils

echo "== EMANE ${EMANE_VERSION} (${EMANE_DISTRO}) =="
tmp="$(mktemp -d)"
cd "$tmp"
wget -q "$URL"
tar xzf "$BUNDLE"
cd "emane-${EMANE_VERSION}-release-1/debs/${EMANE_DISTRO}/amd64"
rm -f emane-spectrum-tools*.deb emane-model-lte*.deb *dev*.deb
sudo apt-get install -y ./emane*.deb ./python3-emane_*.deb
cd /
rm -rf "$tmp"

echo "== installed =="
emane --version
python3 -c "import emane.events; print('python3-emane OK')"
