#!/usr/bin/env bash

set -euo pipefail

container_name="rocm723_visdrone"

if ! docker container inspect "${container_name}" >/dev/null 2>&1; then
    echo "错误：找不到容器 ${container_name}。请先按 WINDOWS_WSL_ROCM_CN.md 创建容器。" >&2
    exit 1
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "${container_name}")" == "true" ]]; then
    echo "${container_name} 已在运行，正在连接主终端……"
    exec docker attach "${container_name}"
fi

echo "正在启动 ${container_name} 并打开主终端……"
exec docker start -ai "${container_name}"
