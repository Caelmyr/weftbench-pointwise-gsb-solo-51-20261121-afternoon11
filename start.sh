#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
docker compose up -d --build --wait
printf 'WeftBench 浏览器入口 http://127.0.0.1:%s\n' "${WEFTBENCH_PORT:-19954}"
