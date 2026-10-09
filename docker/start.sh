#!/usr/bin/env bash
# Detect the GPU backend and start the matching OneJev compose stack.
#
#   ./docker/start.sh                 # detect hardware, start detached
#   ./docker/start.sh vulkan --build  # force a backend and rebuild
#   ./docker/start.sh down            # stop every profile
#   ./docker/start.sh logs            # follow logs of the running backend
#
# backend: auto (default) | torch (NVIDIA CUDA) | rocm (AMD) | vulkan (AMD/Intel)
#          | gguf (NVIDIA + GGUF) | gguf-rocm (AMD + GGUF) | cpu (no GPU)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

ALL_PROFILES=(torch rocm gguf gguf-rocm vulkan cpu)

log()  { printf '\033[36m%s\033[0m\n' "$*"; }
die()  { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: docker/start.sh [backend] [options]
       docker/start.sh down | logs | ps

backend   auto (default), torch (NVIDIA/CUDA), rocm (AMD/ROCm),
          vulkan (AMD/Intel), gguf (NVIDIA + GGUF), gguf-rocm (AMD + GGUF),
          cpu (fallback)

options   --build       rebuild images from scratch (cached build by default)
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

profile_args=()
for p in "${ALL_PROFILES[@]}"; do profile_args+=(--profile "$p"); done
case "$action" in
  down) exec docker compose "${profile_args[@]}" down --remove-orphans ;;
  logs) exec docker compose "${profile_args[@]}" logs -f --tail=200 ;;
  ps)   exec docker compose "${profile_args[@]}" ps ;;
esac

[ -z "$backend" ] && backend="$(detect_backend)"
case "$backend" in torch|rocm|vulkan|gguf|gguf-rocm|cpu) ;; *) die "unknown backend: $backend" ;; esac

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  log "created .env from .env.example"
fi
detect_gids

case "$action" in
  up)
    # The onejev/qev images are only built locally from the repo
    # Dockerfile and never published to a registry. Build them first
    # (cached, unless --build) so compose does not try to pull
    # onejev/qev and die with "pull access denied".
    build_args=(--profile "$backend" build)
    [ "$build" = 1 ] && build_args+=(--no-cache)
    log "backend '$backend' -> docker compose ${build_args[*]}"
    docker compose "${build_args[@]}"
    up_args=(--profile "$backend" up)
    [ "$foreground" = 0 ] && up_args+=(-d)
    log "backend '$backend' -> docker compose ${up_args[*]}"
    docker compose "${up_args[@]}"
    svc="$(docker compose "${profile_args[@]}" ps --services | head -n1)"
    port="$(docker compose "${profile_args[@]}" port "$svc" 8000 2>/dev/null | head -n1 || true)"
    log "API and playground: http://localhost:${port##*:}"
    log "Stop: ./docker/start.sh down"
    ;;
esac
