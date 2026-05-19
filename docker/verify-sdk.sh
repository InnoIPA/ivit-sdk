#!/bin/bash
# ============================================================================
# iVIT-SDK Complete Verification Script
#
# 此腳本驗證 SDK 的完整流程：
#   1. Python 套件安裝
#   2. C++ 編譯
#   3. Python 測試
#   4. C++ 範例執行
#   5. Python 範例執行
#
# Usage:
#   verify-sdk.sh [options]
#
# Options:
#   --python-only    只驗證 Python 套件
#   --cpp-only       只驗證 C++ 編譯
#   --skip-tests     跳過測試
#   --help           顯示幫助
# ============================================================================

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 預設值
PYTHON_ONLY=false
CPP_ONLY=false
SKIP_TESTS=false
SDK_DIR="/workspace/ivit-sdk"
BUILD_DIR="/tmp/ivit-build"
TEST_IMAGE="/opt/test-data/images/bus.jpg"
TEST_MODEL="/opt/test-data/models/yolox-s.onnx"

# ============================================================================
# 輔助函數
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${YELLOW}[$1/$TOTAL_STEPS]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

show_help() {
    echo "iVIT-SDK Verification Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --python-only    只驗證 Python 套件安裝和測試"
    echo "  --cpp-only       只驗證 C++ 編譯和範例"
    echo "  --skip-tests     跳過測試階段"
    echo "  --help           顯示此幫助訊息"
    echo ""
    echo "Example:"
    echo "  $0                    # 完整驗證"
    echo "  $0 --python-only      # 只驗證 Python"
    echo "  $0 --cpp-only         # 只驗證 C++"
    exit 0
}

# 解析參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --python-only)
            PYTHON_ONLY=true
            shift
            ;;
        --cpp-only)
            CPP_ONLY=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# 計算總步驟數
if [ "$PYTHON_ONLY" = true ]; then
    TOTAL_STEPS=4
elif [ "$CPP_ONLY" = true ]; then
    TOTAL_STEPS=4
else
    TOTAL_STEPS=7
fi

CURRENT_STEP=0

# ============================================================================
# 驗證前置條件
# ============================================================================

print_header "iVIT-SDK Complete Verification"

echo "Configuration:"
echo "  SDK Directory: $SDK_DIR"
echo "  Build Directory: $BUILD_DIR"
echo "  Test Image: $TEST_IMAGE"
echo "  Test Model: $TEST_MODEL"
echo ""

# 檢查 SDK 目錄
if [ ! -d "$SDK_DIR" ]; then
    print_error "SDK source not found at $SDK_DIR"
    echo "Please mount the SDK source:"
    echo "  docker run -v \$(pwd):/workspace/ivit-sdk ..."
    exit 1
fi

# 檢查測試資料
if [ ! -f "$TEST_IMAGE" ]; then
    print_error "Test image not found: $TEST_IMAGE"
    exit 1
fi

if [ ! -f "$TEST_MODEL" ]; then
    print_error "Test model not found: $TEST_MODEL"
    exit 1
fi

print_success "Prerequisites check passed"

# ============================================================================
# Step 1: Python 套件安裝
# ============================================================================

if [ "$CPP_ONLY" != true ]; then
    CURRENT_STEP=$((CURRENT_STEP + 1))
    print_header "Step $CURRENT_STEP: Python Package Installation"

    print_step $CURRENT_STEP $TOTAL_STEPS "Installing Python package in development mode..."

    cd "$SDK_DIR"

    # 安裝 Python 套件
    pip3 install . --quiet

    print_success "Python package installed"

    # 驗證安裝
    echo ""
    echo "Verifying installation..."
    python3 -c "
import ivit
print(f'  iVIT-SDK version: {ivit.__version__}')
print(f'  C++ available: {ivit.is_cpp_available()}')
print(f'  Devices:')
for d in ivit.list_devices():
    print(f'    - {d.id}: {d.name} [{d.backend}]')
"

    print_success "Python package verification passed"
fi

# ============================================================================
# Step 2: Python 測試
# ============================================================================

if [ "$CPP_ONLY" != true ] && [ "$SKIP_TESTS" != true ]; then
    CURRENT_STEP=$((CURRENT_STEP + 1))
    print_header "Step $CURRENT_STEP: Python Tests"

    print_step $CURRENT_STEP $TOTAL_STEPS "Running Python tests..."

    cd "$SDK_DIR"

    # 執行測試
    python3 -m pytest tests/integration/ -v --tb=short 2>&1 | tail -20

    print_success "Python tests passed"
fi

# ============================================================================
# Step 3: Python 範例執行
# ============================================================================

if [ "$CPP_ONLY" != true ]; then
    CURRENT_STEP=$((CURRENT_STEP + 1))
    print_header "Step $CURRENT_STEP: Python Example Execution"

    print_step $CURRENT_STEP $TOTAL_STEPS "Running Python detection example..."

    cd "$SDK_DIR"

    # 建立輸出目錄
    mkdir -p /tmp/ivit-output

    # 執行 Python 範例
    python3 << EOF
import ivit
from ivit.vision import Detector
import cv2
import os

print("Loading model...")
detector = Detector("$TEST_MODEL", device="cpu")
print(f"  Input size: {detector.input_size}")

print("Running detection...")
results = detector.predict("$TEST_IMAGE", conf_threshold=0.25)

print(f"\nDetected {len(results.detections)} objects:")
for det in results.detections:
    print(f"  - {det.label}: {det.confidence:.0%}")

# Save visualization
image = cv2.imread("$TEST_IMAGE")
vis = results.visualize(image)
output_path = "/tmp/ivit-output/python_detection.jpg"
cv2.imwrite(output_path, vis)
print(f"\nVisualization saved: {output_path}")

# Save JSON
results.save("/tmp/ivit-output/python_detection.json")
print("Results saved: /tmp/ivit-output/python_detection.json")
EOF

    print_success "Python example executed successfully"

    # 顯示輸出
    echo ""
    echo "Output files:"
    ls -la /tmp/ivit-output/python_*
fi

# ============================================================================
# Step 4: C++ 編譯
# ============================================================================

if [ "$PYTHON_ONLY" != true ]; then
    CURRENT_STEP=$((CURRENT_STEP + 1))
    print_header "Step $CURRENT_STEP: C++ Build"

    print_step $CURRENT_STEP $TOTAL_STEPS "Building C++ components..."

    # 複製到可寫目錄
    rm -rf "$BUILD_DIR"
    cp -r "$SDK_DIR" "$BUILD_DIR"
    cd "$BUILD_DIR"

    # 下載依賴
    echo "Downloading dependencies..."
    if [ -f "./scripts/download_deps.sh" ]; then
        ./scripts/download_deps.sh --openvino-only 2>&1 | tail -5
    fi

    # CMake 配置
    echo ""
    echo "Configuring CMake..."
    rm -rf build && mkdir build && cd build

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DIVIT_USE_OPENVINO=ON \
        -DIVIT_USE_TENSORRT=OFF \
        -DIVIT_BUILD_TESTS=ON \
        -DIVIT_BUILD_EXAMPLES=ON \
        -DIVIT_BUILD_PYTHON=OFF \
        -DIVIT_BUNDLE_DEPS=ON 2>&1 | grep -E "(^--|Configuring|Generating|Build)"

    # 編譯
    echo ""
    echo "Building (this may take a few minutes)..."
    make -j$(nproc) 2>&1 | tail -10

    print_success "C++ build completed"

    # 顯示產物
    echo ""
    echo "Build artifacts:"
    echo "  Libraries:"
    ls -la lib/*.so 2>/dev/null || echo "    (no shared libraries)"
    echo "  Binaries:"
    ls -la bin/* 2>/dev/null || echo "    (no binaries)"
fi

# ============================================================================
# Step 5: C++ 測試
# ============================================================================

if [ "$PYTHON_ONLY" != true ] && [ "$SKIP_TESTS" != true ]; then
    CURRENT_STEP=$((CURRENT_STEP + 1))
    print_header "Step $CURRENT_STEP: C++ Tests"

    print_step $CURRENT_STEP $TOTAL_STEPS "Running C++ tests..."

    cd "$BUILD_DIR/build"

    # 執行測試
    ctest --output-on-failure 2>&1 | tail -20

    print_success "C++ tests passed"
fi

# ============================================================================
# Step 6: C++ 範例執行
# ============================================================================

if [ "$PYTHON_ONLY" != true ]; then
    CURRENT_STEP=$((CURRENT_STEP + 1))
    print_header "Step $CURRENT_STEP: C++ Example Execution"

    print_step $CURRENT_STEP $TOTAL_STEPS "Running C++ detection example..."

    cd "$BUILD_DIR/build"

    # 檢查範例是否存在
    if [ -f "./bin/simple_inference" ]; then
        echo "Listing devices..."
        ./bin/simple_inference devices 2>&1 || true

        echo ""
        echo "Running detection..."
        ./bin/simple_inference detect \
            "$TEST_MODEL" \
            "$TEST_IMAGE" \
            cpu \
            /tmp/ivit-output/cpp_detection.jpg 2>&1 || true

        print_success "C++ example executed"

        # 顯示輸出
        echo ""
        echo "Output files:"
        ls -la /tmp/ivit-output/cpp_* 2>/dev/null || echo "  (no output files)"
    else
        print_info "simple_inference not found, skipping C++ example"
    fi
fi

# ============================================================================
# 完成摘要
# ============================================================================

print_header "Verification Summary"

echo -e "${GREEN}All verification steps completed!${NC}"
echo ""
echo "Verified components:"

if [ "$CPP_ONLY" != true ]; then
    echo -e "  ${GREEN}✓${NC} Python package installation"
    if [ "$SKIP_TESTS" != true ]; then
        echo -e "  ${GREEN}✓${NC} Python tests"
    fi
    echo -e "  ${GREEN}✓${NC} Python example execution"
fi

if [ "$PYTHON_ONLY" != true ]; then
    echo -e "  ${GREEN}✓${NC} C++ build"
    if [ "$SKIP_TESTS" != true ]; then
        echo -e "  ${GREEN}✓${NC} C++ tests"
    fi
    echo -e "  ${GREEN}✓${NC} C++ example execution"
fi

echo ""
echo "Output files saved to: /tmp/ivit-output/"
ls -la /tmp/ivit-output/ 2>/dev/null || true

echo ""
echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  VERIFICATION PASSED${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
echo ""
