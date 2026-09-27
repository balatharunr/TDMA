# One consolidated container for Part 2: Ubuntu 24.04 + EMANE 1.5.3 + tools (Division 5).
# The 16 emulated radios of Division 6 all live inside this ONE container, each in its own
# Linux network namespace, so the brief's "one consolidated Docker container" is met literally.
#
# Build (repo root):   docker build -t tdma-emane emane/docker      (or emane/docker/build.ps1)
# Run   (repo root):   emane/docker/run.ps1                          (privileged + repo mounted at /work)
FROM ubuntu:24.04
ARG EMANE_VERSION=1.5.3
ARG EMANE_DISTRO=ubuntu-24_04
ENV DEBIAN_FRONTEND=noninteractive

# Tools: iproute2 (ip netns / bridge / veth), iputils-ping, tcpdump (E2 timing capture), iperf3,
# iptables (the FORWARD-policy check), libxml2-utils (xmllint to validate our XML against EMANE's DTDs),
# python3-pkg-resources because EMANE 1.5.3's emane.events imports pkg_resources.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget iproute2 iputils-ping tcpdump iperf3 iptables procps psmisc less nano \
        python3 python3-networkx python3-matplotlib python3-pytest \
        python3-lxml python3-protobuf python3-pkg-resources libxml2-utils \
    && rm -rf /var/lib/apt/lists/*

# EMANE: the prebuilt bundle from Adjacent Link (the "lean install" CORE's own Dockerfiles use).
# We skip the spectrum tools, the LTE model and the -dev packages: not needed for a TDMA scenario.
RUN set -eux; cd /tmp; \
    wget -q https://adjacentlink.com/downloads/emane/emane-${EMANE_VERSION}-release-1.${EMANE_DISTRO}.amd64.tar.gz; \
    tar xzf emane-${EMANE_VERSION}-release-1.${EMANE_DISTRO}.amd64.tar.gz; \
    cd emane-${EMANE_VERSION}-release-1/debs/${EMANE_DISTRO}/amd64; \
    rm -f emane-spectrum-tools*.deb emane-model-lte*.deb *dev*.deb; \
    apt-get update; apt-get install -y ./emane*.deb ./python3-emane_*.deb; \
    rm -rf /tmp/emane-* /var/lib/apt/lists/*

WORKDIR /work
CMD ["bash"]
