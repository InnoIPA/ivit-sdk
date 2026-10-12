#!/usr/bin/env bash
# Install the OpenVINO edition on Ubuntu/Debian; Python packages stay in a venv.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SDK_DIR/.venv"
MODE=python
INSTALL_SYSTEM=1
WITH_DEV=0
CHECK_ONLY=0
JOBS=2

usage() {
    cat <<'HELP'
Usage: bash scripts/install_sdk.sh [options]

Installs the OpenVINO edition. Requires Ubuntu/Debian and Internet access.
System packages use apt-get (sudo if needed); Python packages use a venv.

  --venv PATH       Use/create this venv (default: repository/.venv)
  --mode MODE       python or cpp (default: python)
  --dev             Include Python development tools / C++ GoogleTest
  --skip-system     Do not install system packages; require them beforehand
  --jobs NUMBER     Limit native build parallelism (default: 2)
  --check           Check prerequisites only; install nothing
  -h, --help        Show this help

Examples:
  bash scripts/install_sdk.sh
  bash scripts/install_sdk.sh --dev
  bash scripts/install_sdk.sh --mode cpp
  bash scripts/install_sdk.sh --skip-system --venv /tmp/ivit-venv

NVIDIA drivers, CUDA, TensorRT and Intel GPU/NPU drivers are not installed.
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
                --venv) VENV_DIR="$2" ;;
                --mode) MODE="$2" ;;
                --jobs) JOBS="$2" ;;
            esac
            shift 2 ;;
        --skip-system) INSTALL_SYSTEM=0; shift ;;
        --dev) WITH_DEV=1; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (use --help)" ;;
    esac
done
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

if ((CHECK_ONLY)); then
    check_native
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
        -Ccmake.define.IVIT_USE_TENSORRT=OFF \
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
else
    BUILD_DIR="$SDK_DIR/build/install-openvino"
    TESTS=OFF
    if ((WITH_DEV)); then TESTS=ON; fi
    log "Building C++ SDK: $BUILD_DIR"
    cmake -S "$SDK_DIR" -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DIVIT_BUILD_PYTHON=OFF -DIVIT_BUILD_EXAMPLES=ON \
        "-DIVIT_BUILD_TESTS=$TESTS" \
        -DIVIT_USE_OPENVINO=ON -DIVIT_USE_TENSORRT=OFF \
        "-DOpenVINO_DIR=$OV_CMAKE_DIR"
    cmake --build "$BUILD_DIR" --parallel "$JOBS"
    "$BUILD_DIR/bin/simple_inference" devices
    if ((WITH_DEV)); then
        ctest --test-dir "$BUILD_DIR" --output-on-failure --no-tests=error
    fi
fi
log 'Installation checks completed; model accuracy/inference still requires a compatible model.'
printf 'Activate the venv in each new terminal: source %q/bin/activate\n' "$VENV_DIR"
