#!/usr/bin/env bash
set -euo pipefail
URL="${QEV_BASE_URL:-http://localhost:8000}"
curl -s "$URL/v1/systemone" -H "Content-Type: application/json" -d @"$(dirname "$0")/request.json" | python3 -m json.tool
