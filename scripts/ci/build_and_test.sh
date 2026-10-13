#!/usr/bin/env bash
# Build the SDK from a clean environment and run the C++ and Python suites.
#
# One script for CI and for local use, so a green local run means a green CI run.
#
#   scripts/ci/build_and_test.sh                 # OpenVINO build
#   scripts/ci/build_and_test.sh --tensorrt      # OpenVINO + TensorRT build
#
# --tensorrt needs no CUDA toolkit: TensorRT/CUDA runtime libraries come from pip and the
# TensorRT headers from NVIDIA's open-source repository (same release as the pip wheels).
# Without a GPU the TensorRT code is still compiled and linked, and the GPU-only tests skip.
#
# Needs: cmake >= 3.18, g++ (C++17), libopencv-dev, libgtest-dev, python3-dev, python3-venv.
set -euo pipefail

TENSORRT=0
TORCH=1
WORK=""
JOBS="$(nproc)"
TRT_VERSION="11.4"        # pip: tensorrt-cu12==11.4.*  /  headers: NVIDIA/TensorRT tag v11.4
TRT_TAG="v11.4"
ONNX_PARSER_BRANCH="release/11.4-GA"
TRT_MAJOR="${TRT_VERSION%%.*}"

while [ $# -gt 0 ]; do
    case "$1" in
        --tensorrt) TENSORRT=1 ;;
        --no-torch) TORCH=0 ;;          # skips the three trainer tests
        --work) WORK="$2"; shift ;;
        --jobs) JOBS="$2"; shift ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-$ROOT/build-ci}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
VENV="$WORK/venv"
BUILD="$WORK/build"
STAGE="$WORK/stage"

cleanup() { # the build drops the extension module into the source tree; never leave it there
    rm -f "$ROOT"/python/ivit/_ivit_core*.so
    find "$ROOT" -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
}
trap cleanup EXIT

step() { printf '\n==> %s\n' "$*"; }

step "Python environment ($VENV)"
python3 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip
PACKAGES=(numpy opencv-python-headless onnx pytest requests tqdm "pybind11>=2.11" "openvino>=2026.0")
"$PY" -m pip install --quiet "${PACKAGES[@]}"
if [ "$TORCH" = 1 ]; then
    "$PY" -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
    "$PY" -m pip install --quiet onnxscript   # torch's ONNX exporter needs it
fi
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

CMAKE_ARGS=(
    -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
    -DIVIT_USE_OPENVINO=ON -DIVIT_BUILD_PYTHON=ON -DIVIT_BUILD_TESTS=ON -DIVIT_BUILD_EXAMPLES=ON
    -DOpenVINO_DIR="$SITE/openvino/cmake"
    -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)"
    # Build the extension for the venv's interpreter, not whichever python CMake finds first on
    # PATH (a second Python there produced a module the venv could not import).
    -DPython_EXECUTABLE="$PY"
)
LIBPATH="$SITE/openvino/libs:$BUILD/lib"

if [ "$TENSORRT" = 1 ]; then
    step "TensorRT $TRT_VERSION headers and libraries"
    "$PY" -m pip install --quiet "tensorrt-cu12==$TRT_VERSION.*" nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12
    TRT_ROOT="$WORK/trt_root"
    CUDA_ROOT="$WORK/cuda_root"
    mkdir -p "$TRT_ROOT/include" "$TRT_ROOT/lib" "$CUDA_ROOT/bin" "$CUDA_ROOT/lib64"
    RAW="https://raw.githubusercontent.com/NVIDIA/TensorRT/$TRT_TAG/include"
    for header in NvInfer NvInferImpl NvInferLegacyDims NvInferPluginBase NvInferPlugin \
                  NvInferPluginUtils NvInferRuntimeBase NvInferRuntimeCommon NvInferRuntime \
                  NvInferRuntimePlugin NvInferVersion NvOnnxConfig; do
        curl -fsSL --retry 3 -o "$TRT_ROOT/include/$header.h" "$RAW/$header.h"
    done
    curl -fsSL --retry 3 -o "$TRT_ROOT/include/NvOnnxParser.h" \
        "https://raw.githubusercontent.com/onnx/onnx-tensorrt/$ONNX_PARSER_BRANCH/NvOnnxParser.h"
    ln -sf "$SITE/tensorrt_libs/libnvinfer.so.$TRT_MAJOR" "$TRT_ROOT/lib/libnvinfer.so"
    ln -sf "$SITE/tensorrt_libs/libnvonnxparser.so.$TRT_MAJOR" "$TRT_ROOT/lib/libnvonnxparser.so"
    # CMake's FindCUDAToolkit only needs a toolkit-shaped directory and something that answers
    # `nvcc --version`; nothing here is compiled with nvcc (the SDK has no .cu files).
    ln -sfn "$SITE/nvidia/cuda_runtime/include" "$CUDA_ROOT/include"
    ln -sf "$SITE/nvidia/cuda_runtime/lib/libcudart.so.12" "$CUDA_ROOT/lib64/libcudart.so"
    ln -sf "$SITE/nvidia/cuda_runtime/lib/libcudart.so.12" "$CUDA_ROOT/lib64/libcudart.so.12"
    ln -sfn "$CUDA_ROOT/lib64" "$CUDA_ROOT/lib"
    printf '#!/bin/sh\necho "nvcc: NVIDIA (R) Cuda compiler driver"\necho "Cuda compilation tools, release 12.9, V12.9.86"\n' \
        > "$CUDA_ROOT/bin/nvcc"
    chmod +x "$CUDA_ROOT/bin/nvcc"
    CMAKE_ARGS+=(
        -DIVIT_USE_TENSORRT=ON -DTENSORRT_ROOT="$TRT_ROOT" -DCUDAToolkit_ROOT="$CUDA_ROOT"
        -DCUDAToolkit_NVCC_EXECUTABLE="$CUDA_ROOT/bin/nvcc"
        -DCMAKE_CXX_FLAGS="-I$SITE/nvidia/cuda_nvcc/include"   # CUDA's crt/ headers
    )
    LIBPATH="$SITE/tensorrt_libs:$SITE/nvidia/cuda_runtime/lib:$LIBPATH"
else
    CMAKE_ARGS+=(-DIVIT_USE_TENSORRT=OFF)
fi

step "Configure"
export LD_LIBRARY_PATH="$LIBPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake "${CMAKE_ARGS[@]}" | tee "$WORK/configure.log"
if [ "$TENSORRT" = 1 ] && ! grep -q "TensorRT:       ON" "$WORK/configure.log"; then
    echo "TensorRT was requested but CMake disabled it" >&2; exit 1
fi

step "Build"
cmake --build "$BUILD" -j "$JOBS"
if [ "$TENSORRT" = 1 ]; then
    # grep without -q: under pipefail, -q exits early and ldd's SIGPIPE fails the check at random.
    ldd "$BUILD/lib/libivit.so" | grep libnvinfer >/dev/null || { echo "libivit does not link TensorRT" >&2; exit 1; }
fi

step "C++ tests"
ctest --test-dir "$BUILD" --output-on-failure

step "Stage the Python package"
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -r "$ROOT/python/ivit" "$STAGE/ivit"
mv "$ROOT"/python/ivit/_ivit_core*.so "$STAGE/ivit/"

step "Python tests"
cd "$ROOT"
IVIT_C_LIBRARY="$(ls "$BUILD"/lib/libivit.so.*.*.* | head -1)" \
PYTHONPATH="$STAGE" "$PY" -m pytest tests/python -q -p no:cacheprovider -rs \
    2>&1 | tee "$WORK/pytest.log"

step "Done"
cat <<EOF
To run your own code against this build, in a new terminal:
  export LD_LIBRARY_PATH="$LIBPATH\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
  export PYTHONPATH="$STAGE"
  "$PY" -c 'import ivit; print(ivit.list_devices())'
C++ programs from this build are in $BUILD/bin (same LD_LIBRARY_PATH).
EOF
