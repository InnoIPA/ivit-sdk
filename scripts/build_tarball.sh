#!/bin/bash
# build_tarball.sh — 依平台參數編譯 + 打包 tarball
#
# Usage: ./scripts/build_tarball.sh <platform>
#   platform: x86_64-nvidia | x86_64-intel | aarch64-jetson
#
# Output: build-tarball-<platform>/ivit-sdk-<version>-<platform>.tar.xz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION=$(grep 'project(ivit VERSION' "$PROJECT_ROOT/CMakeLists.txt" | sed 's/.*VERSION \([^ )]*\).*/\1/')

PLATFORM="${1:-}"
if [[ -z "$PLATFORM" ]]; then
    echo "Usage: $0 <platform>"
    echo "  Platforms: x86_64-nvidia | x86_64-intel | aarch64-jetson"
    exit 1
fi

# 設定 CMake 選項
case "$PLATFORM" in
    x86_64-nvidia)
        CMAKE_OPTS="-DIVIT_USE_OPENVINO=OFF -DIVIT_USE_TENSORRT=ON"
        ARCH="x86_64"
        ;;
    x86_64-intel)
        CMAKE_OPTS="-DIVIT_USE_OPENVINO=ON -DIVIT_USE_TENSORRT=OFF"
        ARCH="x86_64"
        ;;
    aarch64-jetson)
        CMAKE_OPTS="-DIVIT_USE_OPENVINO=OFF -DIVIT_USE_TENSORRT=ON"
        ARCH="aarch64"
        ;;
    *)
        echo "Error: Unknown platform '$PLATFORM'"
        exit 1
        ;;
esac

# CMake only warns when it cannot find a requested backend and then builds without it. Refuse to
# package in that case, rather than shipping e.g. an "nvidia" SDK that has no TensorRT.
require_backend() {
    local log="$1" backend
    case "$PLATFORM" in
        x86_64-intel) backend="OpenVINO" ;;
        *) backend="TensorRT" ;;
    esac
    if ! grep -qE "^-- +${backend}: +ON" "$log"; then
        echo "Error: ${backend} was requested for $PLATFORM but CMake disabled it (see $log)." >&2
        echo "       Install the ${backend} development files, or point CMake at them." >&2
        exit 1
    fi
}

BUILD_DIR="$PROJECT_ROOT/build-tarball-$PLATFORM"
INSTALL_DIR="$BUILD_DIR/install"
SDK_NAME="ivit-sdk-${VERSION}-${PLATFORM}"

echo "=== Building tarball for $PLATFORM ==="
echo "  Version:  $VERSION"
echo "  Build:    $BUILD_DIR"
echo "  Install:  $INSTALL_DIR"

# 編譯
mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DIVIT_BUNDLE_DEPS=ON \
    -DIVIT_BUILD_TESTS=OFF \
    -DIVIT_BUILD_EXAMPLES=ON \
    -DIVIT_BUILD_PYTHON=ON \
    $CMAKE_OPTS \
    2>&1 | tee "$BUILD_DIR/configure.log"
require_backend "$BUILD_DIR/configure.log"

cmake --build "$BUILD_DIR" -j"$(nproc)"
cmake --install "$BUILD_DIR"

# 組裝 tarball 目錄結構
PACKAGE_DIR="$BUILD_DIR/$SDK_NAME"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

# 複製安裝產物
cp -r "$INSTALL_DIR"/* "$PACKAGE_DIR/"

# 複製範例原始碼
mkdir -p "$PACKAGE_DIR/examples"
cp -r "$PROJECT_ROOT/examples/cpp" "$PACKAGE_DIR/examples/" 2>/dev/null || true
cp -r "$PROJECT_ROOT/examples/python" "$PACKAGE_DIR/examples/" 2>/dev/null || true

# 複製安裝說明
cp "$PROJECT_ROOT/packaging/INSTALL.md" "$PACKAGE_DIR/" 2>/dev/null || true

# 建置 Python wheel
if command -v pip3 &>/dev/null; then
    echo "=== Building Python wheel ==="
    mkdir -p "$PACKAGE_DIR/python"
    cd "$PROJECT_ROOT"
    pip3 wheel . --no-deps --wheel-dir "$PACKAGE_DIR/python" 2>/dev/null || \
        echo "Warning: Python wheel build failed, skipping"
    cd -
fi

# 建立 tarball。xz rather than gzip: with TensorRT 11 the NVIDIA package is 3.5 GB unpacked and
# 2.35 GB as .tar.gz, over GitHub's 2 GiB per-asset limit; .tar.xz is 1.41 GB.
cd "$BUILD_DIR"
XZ_OPT="${XZ_OPT:--T0 -6}" tar -cJf "${SDK_NAME}.tar.xz" "$SDK_NAME"

echo ""
echo "=== Tarball created ==="
echo "  $BUILD_DIR/${SDK_NAME}.tar.xz"
