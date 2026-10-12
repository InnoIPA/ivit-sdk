#!/bin/bash
# verify_package.sh — 驗證各方案產出是否可用
#
# Usage: ./scripts/verify_package.sh <type> <path>
#   type: tarball | deb
#   path: tarball 路徑 / .deb 路徑

set -euo pipefail

TYPE="${1:-}"
TARGET="${2:-}"

if [[ -z "$TYPE" || -z "$TARGET" ]]; then
    echo "Usage: $0 <type> <path>"
    echo "  Types: tarball | deb"
    echo ""
    echo "Examples:"
    echo "  $0 tarball build-tarball-x86_64-nvidia/ivit-sdk-1.1.0-x86_64-nvidia.tar.xz"
    echo "  $0 deb build/ivit-sdk_1.1.0_amd64-nvidia.deb"
    exit 1
fi

PASS=0
FAIL=0

check() {
    local desc="$1"
    shift
    echo -n "  [$desc] ... "
    if "$@" >/dev/null 2>&1; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
    fi
}

# The backend library libivit.so must link for the package's platform (from its file name).
expected_backend_lib() {
    case "$1" in
        *nvidia*|*jetson*) echo "libnvinfer" ;;
        *intel*) echo "libopenvino" ;;
        *) echo "" ;;
    esac
}

links_backend() {  # <libivit.so> <library prefix>
    readelf -d "$1" 2>/dev/null | grep NEEDED | grep "$2" >/dev/null
}

verify_tarball() {
    local tarball="$1"
    local tmpdir
    tmpdir=$(mktemp -d)
    trap "rm -rf $tmpdir" EXIT

    echo "=== Verifying tarball: $tarball ==="

    check "Extract tarball" tar -xf "$tarball" -C "$tmpdir"

    local sdk_dir
    sdk_dir=$(ls -d "$tmpdir"/ivit-sdk-* 2>/dev/null | head -1)
    if [[ -z "$sdk_dir" ]]; then
        echo "  FAIL: No SDK directory found after extraction"
        FAIL=$((FAIL + 1))
        return
    fi

    check "libivit.so exists" test -f "$sdk_dir/lib/libivit.so"
    local backend_lib
    backend_lib=$(expected_backend_lib "$(basename "$tarball")")
    if [[ -n "$backend_lib" ]]; then
        check "libivit.so links ${backend_lib}" links_backend "$sdk_dir/lib/libivit.so" "$backend_lib"
    fi
    check "Headers exist" test -d "$sdk_dir/include/ivit"
    check "CMake config exists" test -d "$sdk_dir/lib/cmake/ivit"
    check "setup_env.sh exists" test -f "$sdk_dir/bin/setup_env.sh"
    check "INSTALL.md exists" test -f "$sdk_dir/INSTALL.md"

    # 測試 Python import (若有 wheel)
    if ls "$sdk_dir"/python/*.whl >/dev/null 2>&1; then
        check "Python wheel exists" test -f "$sdk_dir"/python/*.whl
    fi
}

# Each check runs a single command, so the grep has to live inside it: piping `check ... | grep`
# would grep check's own "PASS" line (which contains the search text) and count nothing.
# grep reads all input (no -q): with pipefail, -q exiting early kills dpkg-deb with SIGPIPE and
# fails the check even though the file is there.
deb_lists() { dpkg-deb -c "$1" | grep "$2" >/dev/null; }
deb_declares_depends() { dpkg-deb --info "$1" | grep "Depends" >/dev/null; }

verify_deb() {
    local deb="$1"
    echo "=== Verifying .deb: $deb ==="

    check ".deb file valid" dpkg-deb --info "$deb"
    check "Contains libivit.so" deb_lists "$deb" "libivit.so"
    check "Contains headers" deb_lists "$deb" "include/ivit"
    check "Has dependencies" deb_declares_depends "$deb"

    local backend_lib
    backend_lib=$(expected_backend_lib "$(basename "$deb")")
    if [[ -n "$backend_lib" ]]; then
        local tmpdir lib
        tmpdir=$(mktemp -d)
        dpkg-deb -x "$deb" "$tmpdir"
        lib=$(find "$tmpdir" -name "libivit.so*" -type f | head -1)
        check "libivit.so links ${backend_lib}" links_backend "$lib" "$backend_lib"
        rm -rf "$tmpdir"
    fi
}

case "$TYPE" in
    tarball) verify_tarball "$TARGET" ;;
    deb)     verify_deb "$TARGET" ;;
    *)       echo "Unknown type: $TYPE"; exit 1 ;;
esac

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[[ $FAIL -eq 0 ]] || exit 1
