#!/usr/bin/env bash
# Install the OpenVINO edition (optionally with the TensorRT backend) on Ubuntu/Debian; Python
# packages stay in a venv.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SDK_DIR/.venv"
VENV_EXPLICIT=0
MODE=python
INSTALL_SYSTEM=1
WITH_DEV=0
WITH_TENSORRT=0
CHECK_ONLY=0
JOBS=2

usage() {
    cat <<'HELP'
Usage: bash scripts/install_sdk.sh [options]

Installs the OpenVINO edition (add --tensorrt for the NVIDIA backend too). Requires
Ubuntu/Debian and Internet access.
System packages use apt-get (sudo if needed); Python packages use a venv.

  --venv PATH       Use/create this venv. Default: the virtual environment that is active in
                    this shell ($VIRTUAL_ENV), otherwise repository/.venv
  --mode MODE       python or cpp (default: python)
  --dev             Include Python development tools / C++ GoogleTest
  --tensorrt        Also build the TensorRT backend. Needs the NVIDIA driver and the CUDA /
                    TensorRT development packages installed first (SDK manual 3.4); fails if
                    they are missing instead of building without TensorRT
  --skip-system     Do not install system packages; require them beforehand
  --jobs NUMBER     Limit native build parallelism (default: 2)
  --check           Check prerequisites only; install nothing
  -h, --help        Show this help

Examples:
  bash scripts/install_sdk.sh
  bash scripts/install_sdk.sh --dev
  bash scripts/install_sdk.sh --tensorrt
  bash scripts/install_sdk.sh --mode cpp
  bash scripts/install_sdk.sh --skip-system --venv /tmp/ivit-venv

NVIDIA drivers, CUDA, TensorRT and Intel GPU/NPU drivers are not installed by this script.
For --tensorrt set TENSORRT_ROOT / CUDAToolkit_ROOT if they are not in the system paths.
HELP
}
log() { printf '[iVIT] %s\n' "$*"; }
die() { printf '[iVIT] ERROR: %s\n' "$*" >&2; exit 1; }
trap 'printf "[iVIT] Failed at line %s. Check the error above; rerun after resolving it.\n" "$LINENO" >&2' ERR

while (($#)); do
    case "$1" in
        --venv|--mode|--jobs)
            (($# >= 2)) || die "$1 requires a value"
            [[ -n "$2" && "$2" != --* ]] || die "$1 requires a value"
            case "$1" in
                --venv) VENV_DIR="$2"; VENV_EXPLICIT=1 ;;
                --mode) MODE="$2" ;;
                --jobs) JOBS="$2" ;;
            esac
            shift 2 ;;
        --skip-system) INSTALL_SYSTEM=0; shift ;;
        --dev) WITH_DEV=1; shift ;;
        --tensorrt) WITH_TENSORRT=1; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (use --help)" ;;
    esac
done
# An already-active virtual environment is where the user expects the install to go (as with
# pip). Ignoring it silently created a second .venv and left the active one empty.
if ((!VENV_EXPLICIT)) && [[ -n "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "$VIRTUAL_ENV/pyvenv.cfg" && -x "$VIRTUAL_ENV/bin/python3" ]]; then
        VENV_DIR="$VIRTUAL_ENV"
        log "A virtual environment is active; installing into it: $VENV_DIR (use --venv PATH to choose another)"
    else
        log "Ignoring VIRTUAL_ENV=$VIRTUAL_ENV (not a usable venv); using $VENV_DIR"
    fi
fi
[[ "$MODE" == python || "$MODE" == cpp ]] || die '--mode must be python or cpp'
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die '--jobs must be a positive integer'
[[ -f /etc/os-release ]] || die 'Cannot identify the operating system'
# shellcheck source=/dev/null
. /etc/os-release
[[ "$ID" == ubuntu || "$ID" == debian ]] || die "Unsupported OS: $ID; install dependencies manually"

check_native() {
    command -v python3 >/dev/null || die 'Missing python3; install python3 and python3-venv'
    python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
    command -v c++ >/dev/null || die 'Missing C++ compiler; install build-essential'
    command -v pkg-config >/dev/null || die 'Missing pkg-config'
    pkg-config --exists opencv4 || die 'Missing OpenCV C++ development package; install libopencv-dev (opencv-python is insufficient)'
    log "OpenCV C++: $(pkg-config --modversion opencv4)"
}

# --tensorrt: the SDK only compiles the backend when CUDA and the TensorRT headers/libraries are
# present, and pip would otherwise finish "successfully" without it. Check before the long build.
check_tensorrt() {
    local nvcc="" header="" dir
    if command -v nvcc >/dev/null; then nvcc="$(command -v nvcc)"
    elif [[ -x "${CUDAToolkit_ROOT:-/usr/local/cuda}/bin/nvcc" ]]; then nvcc="${CUDAToolkit_ROOT:-/usr/local/cuda}/bin/nvcc"; fi
    for dir in "${TENSORRT_ROOT:+$TENSORRT_ROOT/include}" /usr/include/x86_64-linux-gnu \
               /usr/include/aarch64-linux-gnu /usr/include /usr/local/tensorrt/include; do
        if [[ -n "$dir" && -f "$dir/NvInfer.h" ]]; then header="$dir"; break; fi
    done
    [[ -n "$nvcc" ]] || die '--tensorrt needs the CUDA toolkit (nvcc not found). Install cuda-nvcc and cuda-cudart-dev from the NVIDIA apt repository (SDK manual 3.4) or set CUDAToolkit_ROOT'
    [[ -n "$header" ]] || die '--tensorrt needs the TensorRT development files (NvInfer.h not found). Install libnvinfer-dev and libnvonnxparsers-dev (SDK manual 3.4) or set TENSORRT_ROOT'
    log "CUDA compiler: $nvcc"
    log "TensorRT headers: $header"
}

if ((CHECK_ONLY)); then
    check_native
    if ((WITH_TENSORRT)); then check_tensorrt; fi
    python3 -c 'import venv, ensurepip'
    log 'Native prerequisites found. CMake/OpenVINO and SDK compilation are checked during installation.'
    exit 0
fi

if ((INSTALL_SYSTEM)); then
    command -v apt-get >/dev/null || die 'apt-get not found'
    SUDO=()
    if ((EUID != 0)); then
        command -v sudo >/dev/null || die 'sudo not found; ask an administrator to install dependencies, then use --skip-system'
        SUDO=(sudo)
    fi
    PACKAGES=(build-essential python3 python3-venv libopencv-dev pkg-config)
    if [[ "$MODE" == python ]]; then
        PACKAGES+=(python3-dev)
    fi
    if [[ "$MODE" == cpp ]] && ((WITH_DEV)); then
        PACKAGES+=(libgtest-dev)
    fi
    log "Installing system dependencies: ${PACKAGES[*]}"
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y "${PACKAGES[@]}"
fi
check_native
if ((WITH_TENSORRT)); then check_tensorrt; fi

# Backend selection. --tensorrt also makes configuring fail when a requested backend is missing.
TRT_STATE=OFF
CMAKE_EXTRA=()
if ((WITH_TENSORRT)); then
    TRT_STATE=ON
    CMAKE_EXTRA+=(IVIT_REQUIRE_BACKENDS=ON)
    if [[ -n "${TENSORRT_ROOT:-}" ]]; then CMAKE_EXTRA+=("TENSORRT_ROOT=$TENSORRT_ROOT"); fi
    if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then CMAKE_EXTRA+=("CUDAToolkit_ROOT=$CUDAToolkit_ROOT"); fi
fi
PIP_CMAKE=(); CMAKE_D=()
for item in "${CMAKE_EXTRA[@]}"; do PIP_CMAKE+=("-Ccmake.define.$item"); CMAKE_D+=("-D$item"); done

# Refuse to replace an unrelated directory or use a symlinked environment.
[[ ! -L "$VENV_DIR" ]] || die 'Venv path must not be a symlink'
if [[ -e "$VENV_DIR" ]]; then
    [[ -f "$VENV_DIR/pyvenv.cfg" && -x "$VENV_DIR/bin/python3" ]] || die "Existing path is not a usable venv: $VENV_DIR"
else
    log "Creating venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
VENV_DIR="$(cd -- "$VENV_DIR" && pwd)"
VENV_PY="$VENV_DIR/bin/python3"
"$VENV_PY" -c 'import sys; assert sys.prefix != sys.base_prefix; assert sys.version_info >= (3, 10); print("venv Python:", sys.executable)'
# Ensure CMake's python3 discovery also uses this venv.
export PATH="$VENV_DIR/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"

log 'Installing build tools and OpenVINO inside the venv'
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install cmake ninja 'scikit-build-core>=0.8' 'pybind11>=2.11' 'openvino>=2026.0'
OV_CMAKE_DIR="$("$VENV_PY" -c 'from pathlib import Path; import openvino; print(Path(openvino.__file__).parent / "cmake")')"
[[ -f "$OV_CMAKE_DIR/OpenVINOConfig.cmake" ]] || die 'OpenVINO wheel does not contain OpenVINOConfig.cmake; install a native OpenVINO distribution'

if [[ "$MODE" == python ]]; then
    EXTRAS=openvino
    if ((WITH_DEV)); then EXTRAS=dev,openvino; fi
    log 'Building and installing Python SDK'
    "$VENV_PY" -m pip install -e "$SDK_DIR[$EXTRAS]" \
        -Ccmake.define.IVIT_USE_OPENVINO=ON \
        "-Ccmake.define.IVIT_USE_TENSORRT=$TRT_STATE" \
        "${PIP_CMAKE[@]}" \
        "-Ccmake.define.OpenVINO_DIR=$OV_CMAKE_DIR"
    "$VENV_PY" -m pip check
    "$VENV_PY" - <<'PYVERIFY'
import ivit
assert ivit.is_cpp_available(), "C++ bindings unavailable: " + str(getattr(ivit, "_CPP_ERROR", None))
devices = ivit.list_devices()
assert any(d.backend == "openvino" and d.is_available for d in devices), "No available OpenVINO device"
print("SDK version:", ivit.__version__)
for device in devices:
    print(device.id, device.name, device.backend, device.is_available)
PYVERIFY
    if ((WITH_TENSORRT)); then
        "$VENV_PY" - <<'PYTRT'
import ivit
if any(d.backend == "tensorrt" and d.is_available for d in ivit.list_devices()):
    print("TensorRT backend: built in and an NVIDIA GPU is available")
else:
    print("TensorRT backend: built in, but no usable NVIDIA GPU was found (check the driver with nvidia-smi)")
PYTRT
    fi
else
    BUILD_DIR="$SDK_DIR/build/install-openvino"
    TESTS=OFF
    if ((WITH_DEV)); then TESTS=ON; fi
    log "Building C++ SDK: $BUILD_DIR"
    cmake -S "$SDK_DIR" -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DIVIT_BUILD_PYTHON=OFF -DIVIT_BUILD_EXAMPLES=ON \
        "-DIVIT_BUILD_TESTS=$TESTS" \
        -DIVIT_USE_OPENVINO=ON "-DIVIT_USE_TENSORRT=$TRT_STATE" \
        "${CMAKE_D[@]}" \
        "-DOpenVINO_DIR=$OV_CMAKE_DIR"
    cmake --build "$BUILD_DIR" --parallel "$JOBS"
    "$BUILD_DIR/bin/simple_inference" devices
    if ((WITH_DEV)); then
        ctest --test-dir "$BUILD_DIR" --output-on-failure --no-tests=error
    fi
fi
log 'Installation checks completed; model accuracy/inference still requires a compatible model.'
if [[ "${VIRTUAL_ENV:-}" == "$VENV_DIR" ]]; then
    log 'This virtual environment is already active in this terminal.'
else
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log "Note: this terminal has a different virtual environment active ($VIRTUAL_ENV); it does not contain the SDK."
    fi
    printf 'Activate the venv in each new terminal: source %q/bin/activate\n' "$VENV_DIR"
fi
