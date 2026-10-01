#!/usr/bin/env bash
# Detect the GPU backend and start the matching OneJev compose stack.
#
#   ./scripts/start.sh                 # detect hardware, start detached
#   ./scripts/start.sh vulkan --build  # force a backend and rebuild
#   ./scripts/start.sh down            # stop every profile
#   ./scripts/start.sh logs            # follow logs of the detected backend
#
# backend: auto (default) | torch (NVIDIA CUDA) | rocm (AMD) | vulkan (AMD/Intel)
#          | gguf (NVIDIA + GGUF) | gguf-rocm (AMD + GGUF) | cpu (no GPU)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ALL_PROFILES=(torch rocm gguf gguf-rocm vulkan cpu)

log()  { printf '\033[36m%s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: scripts/start.sh [backend] [options]
       scripts/start.sh down | logs | ps

backend   auto (default), torch (NVIDIA/CUDA), rocm (AMD/ROCm),
          vulkan (AMD/Intel), gguf (NVIDIA + GGUF), gguf-rocm (AMD + GGUF),
          cpu (fallback)

options   --build       rebuild images before starting
          --foreground  run in the foreground (default: detached)
          -h, --help    show this help

env       QEV_BACKEND              same values as `backend`, overrides detection
          QEV_NO_AUTODETECT_GID=1  do not read video/render GIDs from the host
EOF
}

detect_backend() {
  if [ -n "${QEV_BACKEND:-}" ]; then printf '%s' "$QEV_BACKEND"; return; fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    printf 'torch'; return
  fi
  if command -v rocm-smi >/dev/null 2>&1 || [ -e /dev/kfd ]; then
    printf 'rocm'; return
  fi
  if compgen -G '/dev/dri/renderD*' >/dev/null 2>&1 || command -v vulkaninfo >/dev/null 2>&1; then
    printf 'vulkan'; return
  fi
  printf 'cpu'
}

detect_gids() {
  [ "${QEV_NO_AUTODETECT_GID:-0}" = "1" ] && return 0
  command -v getent >/dev/null 2>&1 || return 0
  local gid
  if [ -z "${AMD_VIDEO_GID:-}" ]; then
    gid="$(getent group video  | cut -d: -f3 || true)"; [ -n "$gid" ] && export AMD_VIDEO_GID="$gid"
  fi
  gid="$(getent group render | cut -d: -f3 || true)"
  if [ -n "$gid" ]; then
    [ -z "${AMD_RENDER_GID:-}" ] && export AMD_RENDER_GID="$gid"
    [ -z "${DRI_RENDER_GID:-}" ] && export DRI_RENDER_GID="$gid"
  fi
}

upstream_for() {
  case "$1" in
    torch)  printf 'qev:8000' ;;
    rocm)   printf 'qev-rocm:8000' ;;
    gguf|gguf-rocm|vulkan|cpu) printf 'qev-gguf:8000' ;;
  esac
}

is_gguf_backend() { case "$1" in gguf|gguf-rocm|vulkan|cpu) return 0 ;; *) return 1 ;; esac; }

check_models() {
  is_gguf_backend "$1" || return 0
  local dir="${LLAMA_MODELS_DIR:-./models}"
  compgen -G "$dir/*.gguf" >/dev/null 2>&1 && return 0
  warn "no *.gguf in $dir; add the model and its mmproj file (see .env.example: LLAMA_MODEL / LLAMA_MMPROJ)."
}

backend=""
action="up"
build=0
foreground=0

while [ $# -gt 0 ]; do
  case "$1" in
    auto|torch|rocm|vulkan|gguf|gguf-rocm|cpu) backend="$1" ;;
    down|logs|ps) action="$1" ;;
    --build) build=1 ;;
    --foreground|-f) foreground=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
  shift
done

command -v docker >/dev/null 2>&1 || die "docker not found"
docker compose version >/dev/null 2>&1 || die "docker compose v2 not found"

if [ "$action" = "down" ]; then
  profile_args=()
  for p in "${ALL_PROFILES[@]}"; do profile_args+=(--profile "$p"); done
  docker compose "${profile_args[@]}" down --remove-orphans
  exit 0
fi

[ -z "$backend" ] && backend="$(detect_backend)"
case "$backend" in torch|rocm|vulkan|gguf|gguf-rocm|cpu) ;; *) die "unknown backend: $backend" ;; esac

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  log "created .env from .env.example"
fi
detect_gids
# The playground upstream must match the started backend, so this exported value
# intentionally takes precedence over QEV_UPSTREAM in .env.
export QEV_UPSTREAM="${QEV_UPSTREAM:-$(upstream_for "$backend")}"

case "$action" in
  logs) docker compose --profile "$backend" logs -f --tail=200 ;;
  ps)   docker compose --profile "$backend" ps ;;
  up)
    check_models "$backend"
    up_args=(--profile "$backend" up)
    [ "$build" = 1 ] && up_args+=(--build)
    [ "$foreground" = 0 ] && up_args+=(-d)
    log "backend '$backend' -> docker compose ${up_args[*]}"
    docker compose "${up_args[@]}"
    case "$backend" in torch|rocm) api_port=8000 ;; *) api_port=8001 ;; esac
    log "API:        http://localhost:${api_port}   (playground upstream: ${QEV_UPSTREAM})"
    log "Playground: http://localhost:${PLAYGROUND_PORT:-8080}"
    log "Stop:       ./scripts/start.sh down"
    ;;
esac
