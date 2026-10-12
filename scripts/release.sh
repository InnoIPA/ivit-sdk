#!/bin/bash
# release.sh — 一鍵觸發指定平台的 tarball 與 deb 建置
#
# Usage: ./scripts/release.sh <platform> [--skip-deb] [--skip-tarball]
#   platform: x86_64-nvidia | x86_64-intel | aarch64-jetson | all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PLATFORM="${1:-}"
shift || true

SKIP_DEB=false
SKIP_TARBALL=false

for arg in "$@"; do
    case "$arg" in
        --skip-deb)     SKIP_DEB=true ;;
        --skip-tarball) SKIP_TARBALL=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

if [[ -z "$PLATFORM" ]]; then
    echo "Usage: $0 <platform> [--skip-deb] [--skip-tarball]"
    echo "  Platforms: x86_64-nvidia | x86_64-intel | aarch64-jetson | all"
    exit 1
fi

build_platform() {
    local plat="$1"
    echo ""
    echo "============================================================"
    echo "  Building all packages for: $plat"
    echo "============================================================"

    if [[ "$SKIP_TARBALL" != true ]]; then
        echo "--- Tarball ---"
        "$SCRIPT_DIR/build_tarball.sh" "$plat"
    fi

    if [[ "$SKIP_DEB" != true ]]; then
        echo "--- Deb ---"
        "$SCRIPT_DIR/build_deb.sh" "$plat"
    fi
}

if [[ "$PLATFORM" == "all" ]]; then
    for p in x86_64-nvidia x86_64-intel aarch64-jetson; do
        build_platform "$p"
    done
else
    build_platform "$PLATFORM"
fi

echo ""
echo "=== Release build complete ==="
