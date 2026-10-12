#!/bin/bash
# ============================================================================
# download_deps.sh
# 下載並準備 iVIT-SDK 依賴庫
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(dirname "$SCRIPT_DIR")"
DEPS_DIR="${IVIT_DEPS_DIR:-${SDK_ROOT}/deps}"   # same default as the CMake IVIT_DEPS_DIR
INSTALL_DIR="${DEPS_DIR}/install"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 檢測系統架構
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    ARCH_SUFFIX="x64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH_SUFFIX="aarch64"
else
    log_error "Unsupported architecture: $ARCH"
    exit 1
fi

log_info "Detected architecture: $ARCH ($ARCH_SUFFIX)"

# 建立目錄
mkdir -p "$DEPS_DIR"
mkdir -p "$INSTALL_DIR"

# ============================================================================
# 下載 OpenVINO
# ============================================================================
download_openvino() {
    # 先檢查系統是否已安裝 OpenVINO (pip 或系統套件)
    if python3 -c "import openvino" 2>/dev/null; then
        local SYS_OV_VER=$(python3 -c "import openvino; print(openvino.__version__)" 2>/dev/null)
        log_info "System OpenVINO detected: ${SYS_OV_VER}"
        log_info "Skipping download - CMake will use system OpenVINO via find_package()"
        log_info "If you prefer a bundled version, uninstall system OpenVINO first."
        return
    fi

    if [ -d "${INSTALL_DIR}/runtime" ]; then
        log_info "OpenVINO already exists in deps/, skipping..."
        return
    fi

    # Archive names on storage.openvinotoolkit.org carry the full build id and live under the
    # release directory, e.g. packages/2024.6/linux/l_openvino_toolkit_ubuntu24_2024.6.0.17404.4c0f47d2335_x86_64.tgz
    local OV_RELEASE="2024.6"
    local OV_BUILD="2024.6.0.17404.4c0f47d2335"
    local VERSION="2024.6.0"

    # Read the OS version in a subshell: /etc/os-release also defines VERSION, which would
    # otherwise overwrite the OpenVINO version above.
    local OS_ID=""
    [ -f /etc/os-release ] && OS_ID="$(. /etc/os-release && echo "${VERSION_ID}")"
    local OS_SUFFIX
    case "$OS_ID" in
        20.04) OS_SUFFIX="ubuntu20" ;;
        22.04) OS_SUFFIX="ubuntu22" ;;
        24.04) OS_SUFFIX="ubuntu24" ;;
        *) OS_SUFFIX="ubuntu22" ;;
    esac
    local OV_ARCH="x86_64"
    [ "$ARCH" = "aarch64" ] && OV_ARCH="arm64"
    if [ "$OV_ARCH" = "arm64" ] && [ "$OS_SUFFIX" != "ubuntu20" ]; then
        log_error "OpenVINO ${VERSION} has no ${OS_SUFFIX} arm64 archive; install it with pip instead:"
        log_info "  pip install openvino>=${VERSION}"
        return 1
    fi

    log_info "Downloading OpenVINO ${VERSION}..."
    log_info "Alternative: install via pip instead:"
    log_info "  pip install openvino>=${VERSION}"

    local URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/${OV_RELEASE}/linux/l_openvino_toolkit_${OS_SUFFIX}_${OV_BUILD}_${OV_ARCH}.tgz"
    local ARCHIVE="${DEPS_DIR}/openvino-${VERSION}.tgz"

    wget -q --show-progress -O "$ARCHIVE" "$URL" || {
        log_error "Failed to download OpenVINO from: ${URL}"
        log_info "Please install OpenVINO manually:"
        log_info "  pip install openvino>=${VERSION}"
        log_info "  # or follow: https://docs.openvino.ai/2024/get-started/install-openvino.html"
        rm -f "$ARCHIVE"
        return 1
    }

    # 驗證下載的檔案是否為有效的 gzip
    if ! file "$ARCHIVE" | grep -q "gzip"; then
        log_error "Downloaded file is not a valid archive (URL may have changed)"
        log_info "Please install OpenVINO manually:"
        log_info "  pip install openvino>=${VERSION}"
        rm -f "$ARCHIVE"
        return 1
    fi

    log_info "Extracting OpenVINO..."
    tar -xzf "$ARCHIVE" -C "${INSTALL_DIR}" --strip-components=1

    rm -f "$ARCHIVE"
    log_info "OpenVINO installed to ${INSTALL_DIR}"
}

# ============================================================================
# 設定 TensorRT (需要手動下載)
# ============================================================================
setup_tensorrt() {
    log_warn "TensorRT requires manual download from NVIDIA Developer website"
    log_info "Please download TensorRT from: https://developer.nvidia.com/tensorrt"
    log_info "Then extract to: ${INSTALL_DIR}/tensorrt"

    cat << 'EOF'

TensorRT 安裝步驟:
1. 前往 https://developer.nvidia.com/tensorrt
2. 登入 NVIDIA Developer 帳號
3. 下載對應 CUDA 版本的 TensorRT (推薦 tar 格式)
4. 解壓縮到 deps/install/tensorrt:
   tar -xzf TensorRT-*.tgz
   mv TensorRT-* deps/install/tensorrt

EOF

    # 建立目錄結構提示
    mkdir -p "${INSTALL_DIR}/tensorrt"
    cat > "${INSTALL_DIR}/tensorrt/README.txt" << 'EOF'
Please download and extract TensorRT here.

Required files:
  include/NvInfer.h
  lib/libnvinfer.so*
  lib/libnvonnxparser.so*

Download from: https://developer.nvidia.com/tensorrt
EOF
}

# ============================================================================
# 驗證安裝
# ============================================================================
verify_installation() {
    log_info "Verifying installation..."

    local has_error=0

    # 檢查 OpenVINO
    if [ -f "${INSTALL_DIR}/runtime/lib/intel64/libopenvino.so" ] || \
       [ -f "${INSTALL_DIR}/runtime/lib/libopenvino.so" ]; then
        log_info "✓ OpenVINO: OK"
    else
        log_warn "✗ OpenVINO: Not found"
        has_error=1
    fi

    # 檢查 TensorRT
    if [ -f "${INSTALL_DIR}/tensorrt/lib/libnvinfer.so" ]; then
        log_info "✓ TensorRT: OK"
    else
        log_warn "✗ TensorRT: Not found (optional, requires manual download)"
    fi

    return $has_error
}

# ============================================================================
# 建立 CMake 設定檔
# ============================================================================
create_cmake_config() {
    log_info "Creating CMake configuration..."

    cat > "${DEPS_DIR}/deps_config.cmake" << EOF
# Auto-generated dependency configuration
# Generated by download_deps.sh

set(IVIT_DEPS_INSTALL_DIR "${INSTALL_DIR}" CACHE PATH "Dependencies install directory" FORCE)

# OpenVINO
if(EXISTS "${INSTALL_DIR}/runtime/cmake/OpenVINOConfig.cmake")
    set(OpenVINO_DIR "${INSTALL_DIR}/runtime/cmake" CACHE PATH "OpenVINO CMake dir" FORCE)
endif()

# TensorRT
if(EXISTS "${INSTALL_DIR}/tensorrt/lib/libnvinfer.so")
    set(TENSORRT_ROOT "${INSTALL_DIR}/tensorrt" CACHE PATH "TensorRT root" FORCE)
endif()

message(STATUS "Loaded bundled dependencies from: ${INSTALL_DIR}")
EOF

    log_info "CMake configuration saved to: ${DEPS_DIR}/deps_config.cmake"
}

# ============================================================================
# 主程式
# ============================================================================
main() {
    log_info "========================================="
    log_info "iVIT-SDK Dependencies Downloader"
    log_info "========================================="

    local download_ov=1
    local download_trt=0

    # 解析參數
    while [[ $# -gt 0 ]]; do
        case $1 in
            --openvino-only)
                download_trt=0
                shift
                ;;
            --all)
                download_ov=1
                download_trt=1
                shift
                ;;
            --tensorrt)
                download_trt=1
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --openvino-only     Only download OpenVINO"
                echo "  --tensorrt          Show TensorRT setup instructions"
                echo "  --all               Download all dependencies"
                echo "  --help              Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # 下載依賴
    if [ $download_ov -eq 1 ]; then
        download_openvino
    fi

    if [ $download_trt -eq 1 ]; then
        setup_tensorrt
    fi

    # 建立設定檔
    create_cmake_config

    # 驗證
    verify_installation

    log_info "========================================="
    log_info "Dependencies download complete!"
    log_info ""
    log_info "To build with bundled dependencies, run:"
    log_info "  mkdir build && cd build"
    log_info "  cmake .. -DIVIT_BUNDLE_DEPS=ON -C ../deps/deps_config.cmake"
    log_info "  make -j\$(nproc)"
    log_info "========================================="
}

main "$@"
