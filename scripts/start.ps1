#Requires -Version 5.1
<#
.SYNOPSIS
  Detect the GPU backend and start the matching OneJev compose stack.
.EXAMPLE
  .\scripts\start.ps1
  .\scripts\start.ps1 -Backend vulkan -Build
  .\scripts\start.ps1 -Action down
#>
[CmdletBinding()]
param(
  [ValidateSet('auto', 'torch', 'rocm', 'vulkan', 'gguf', 'gguf-rocm', 'cpu')]
  [string]$Backend = 'auto',

  [ValidateSet('up', 'down', 'logs', 'ps')]
  [string]$Action = 'up',

  [switch]$Build,
  [switch]$Foreground
)

$ErrorActionPreference = 'Stop'

$AllProfiles  = @('torch', 'rocm', 'gguf', 'gguf-rocm', 'vulkan', 'cpu')

function Get-Backend {
  if ($env:QEV_BACKEND) { return $env:QEV_BACKEND }
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    if ((& nvidia-smi -L 2>$null | Out-String).Trim()) { return 'torch' }
  }
  return 'cpu'
}

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'docker not found' }
& docker compose version *> $null
if ($LASTEXITCODE -ne 0) { throw 'docker compose v2 not found' }

$profileArgs = @()
foreach ($p in $AllProfiles) { $profileArgs += @('--profile', $p) }
switch ($Action) {
  'down' { & docker compose @profileArgs down --remove-orphans; exit $LASTEXITCODE }
  'logs' { & docker compose @profileArgs logs -f --tail=200; exit $LASTEXITCODE }
  'ps'   { & docker compose @profileArgs ps; exit $LASTEXITCODE }
}

if ($Backend -eq 'auto') { $Backend = Get-Backend }
if ($Backend -notin $AllProfiles) {
  Write-Warning "unknown backend '$Backend'; falling back to 'cpu'"
  $Backend = 'cpu'
}

# Docker Desktop on Windows only passes through NVIDIA (CUDA via WSL2); the
# ROCm and Vulkan passthrough stacks need a Linux host.
if ($Backend -in @('rocm', 'gguf-rocm', 'vulkan')) {
  Write-Warning "backend '$Backend' needs Linux GPU passthrough; on Windows use 'torch' (NVIDIA) or 'cpu'."
}

if (-not (Test-Path .env) -and (Test-Path .env.example)) {
  Copy-Item .env.example .env
  Write-Host 'created .env from .env.example'
}

switch ($Action) {
  'up' {
    # The onejev/qev images are only built locally from the repo
    # Dockerfile and never published to a registry. Build them first
    # (cached, unless -Build) so compose does not try to pull
    # onejev/qev and die with "pull access denied".
    $buildArgs = @('--profile', $Backend, 'build')
    if ($Build) { $buildArgs += '--no-cache' }
    Write-Host "backend '$Backend' -> docker compose $($buildArgs -join ' ')"
    & docker compose @buildArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $composeArgs = @('--profile', $Backend, 'up')
    if (-not $Foreground) { $composeArgs += '-d' }
    Write-Host "backend '$Backend' -> docker compose $($composeArgs -join ' ')"
    & docker compose @composeArgs
    $svc = & docker compose @profileArgs ps --services | Select-Object -First 1
    $addr = if ($svc) { & docker compose @profileArgs port $svc 8000 | Select-Object -First 1 }
    $port = if ($addr) { ($addr -split ':')[-1] } else { '8000' }
    Write-Host "API and playground: http://localhost:$port"
    Write-Host "Stop: .\scripts\start.ps1 -Action down"
  }
}
