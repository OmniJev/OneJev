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
$GgufBackends = @('gguf', 'gguf-rocm', 'vulkan', 'cpu')
$Upstream = @{
  torch = 'qev:8000'; rocm = 'qev-rocm:8000'
  gguf = 'qev-gguf:8000'; 'gguf-rocm' = 'qev-gguf:8000'; vulkan = 'qev-gguf:8000'; cpu = 'qev-gguf:8000'
}
$ApiPort = @{
  torch = '8000'; rocm = '8000'
  gguf = '8001'; 'gguf-rocm' = '8001'; vulkan = '8001'; cpu = '8001'
}

function Get-Backend {
  if ($env:QEV_BACKEND) { return $env:QEV_BACKEND }
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    if ((& nvidia-smi -L 2>$null | Out-String).Trim()) { return 'torch' }
  }
  if (Get-Command rocm-smi -ErrorAction SilentlyContinue) { return 'rocm' }
  $gpus = ((Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue).Name -join ' ')
  if ($gpus -match 'Radeon|AMD') { return 'rocm' }
  if ($gpus -match 'Intel') { return 'vulkan' }
  return 'cpu'
}

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'docker not found' }
& docker compose version *> $null
if ($LASTEXITCODE -ne 0) { throw 'docker compose v2 not found' }

if ($Action -eq 'down') {
  $profileArgs = @()
  foreach ($p in $AllProfiles) { $profileArgs += @('--profile', $p) }
  & docker compose @profileArgs down --remove-orphans
  exit $LASTEXITCODE
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

# Match the playground to the started backend; overrides QEV_UPSTREAM in .env.
if (-not $env:QEV_UPSTREAM) { $env:QEV_UPSTREAM = $Upstream[$Backend] }

switch ($Action) {
  'logs' { & docker compose --profile $Backend logs -f --tail=200 }
  'ps'   { & docker compose --profile $Backend ps }
  'up' {
    if ($GgufBackends -contains $Backend) {
      $dir = if ($env:LLAMA_MODELS_DIR) { $env:LLAMA_MODELS_DIR } else { './models' }
      if (-not (Get-ChildItem -Path $dir -Filter *.gguf -ErrorAction SilentlyContinue)) {
        Write-Warning "no *.gguf in $dir; add the model and its mmproj file (LLAMA_MODEL / LLAMA_MMPROJ)."
      }
    }
    $composeArgs = @('--profile', $Backend, 'up')
    if ($Build) { $composeArgs += '--build' }
    if (-not $Foreground) { $composeArgs += '-d' }
    Write-Host "backend '$Backend' -> docker compose $($composeArgs -join ' ')"
    & docker compose @composeArgs
    $playground = if ($env:PLAYGROUND_PORT) { $env:PLAYGROUND_PORT } else { '8080' }
    Write-Host "API:        http://localhost:$($ApiPort[$Backend])   (playground upstream: $env:QEV_UPSTREAM)"
    Write-Host "Playground: http://localhost:$playground"
    Write-Host "Stop:       .\scripts\start.ps1 -Action down"
  }
}
