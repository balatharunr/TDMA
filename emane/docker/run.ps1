# Start the Part 2 container and open a shell in it. Division 5.
#
#   .\emane\docker\run.ps1            create/start the container "tdma-emane" and open bash in it
#   .\emane\docker\run.ps1 -NoShell   only make sure it is running (scripts then use: docker exec tdma-emane ...)
#
# The container is privileged (it creates network namespaces, a bridge, veth pairs and TUN devices)
# and the repo root is bind-mounted at /work, so everything written under results/ lands on Y: directly.
# It keeps running in the background ("sleep infinity"); open more shells with: docker exec -it tdma-emane bash
# Remove it with: docker rm -f tdma-emane
param([switch]$NoShell)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # emane\docker -> repo root
$name = 'tdma-emane'

$state = docker inspect -f '{{.State.Status}}' $name 2>$null
if (-not $state) {
    # CONCEPT: Docker privileged container & bind mount - --privileged gives the container the kernel
    # capabilities for netns/bridge/TUN; -v mounts the Windows repo folder at /work inside Linux.
    docker run -d --privileged --name $name -v "${root}:/work" -w /work tdma-emane sleep infinity | Out-Null
    Write-Host "created and started container $name (repo mounted at /work)"
} elseif ($state -ne 'running') {
    docker start $name | Out-Null
    Write-Host "started container $name"
} else {
    Write-Host "container $name is already running"
}

if (-not $NoShell) { docker exec -it -w /work $name bash }
