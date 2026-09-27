# Build the one consolidated Part 2 image (Ubuntu 24.04 + EMANE 1.5.3). Division 5.
# Usage (any directory):  .\emane\docker\build.ps1
# Needs Docker Desktop running. The first build downloads about 250 MB and takes a few minutes.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # emane\docker -> repo root
docker build -t tdma-emane (Join-Path $root 'emane\docker')
