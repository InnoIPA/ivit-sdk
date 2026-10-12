# Repository Guidelines

## Project Structure & Module Organization

iVIT-SDK provides computer vision inference through C++17 and Python 3.9+ APIs. Public C++ headers live in `include/ivit/`; implementations in `src/` mirror the `core`, `runtime`, `vision`, and `utils` modules. Python code and bindings live under `python/`. Keep backend-specific behavior in runtime modules.

Tests are organized into `tests/python/`, `tests/cpp/`, and `tests/integration/`. Runnable examples live in `examples/python/` and `examples/cpp/`, with sample assets in `examples/data/`. Documentation is in `docs/`; build and distribution tooling is in `cmake/`, `scripts/`, and `packaging/`.

## Build, Test, and Development Commands

Run commands from the repository root:

- `python -m pip install -e ".[dev]"`: install the editable SDK and Python development tools; native compilation requires OpenCV development libraries and CMake 3.18+.
- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DIVIT_BUILD_TESTS=ON`: configure native builds and tests. Install GoogleTest to enable C++ test targets.
- `cmake --build build -j`: compile the SDK, bindings, examples, and enabled tests.
- `pytest tests/python`: run the Python suite.
- `ctest --test-dir build --output-on-failure`: run registered C++ tests.
- `bash scripts/ci/build_and_test.sh [--tensorrt]`: build from a clean venv and run the C++ and Python suites exactly as CI does.
- `python examples/python/01_quickstart.py`: exercise the Python quickstart; it fetches `yolox-s` through the Model Zoo (set `IVIT_ZOO_MIRROR` where the release URL is unreachable).

## Coding Style & Naming Conventions

Use four-space indentation and follow neighboring code. Python uses `snake_case` functions and modules, `PascalCase` classes, and Black with an 88-character line limit. Run `black python/` before submitting Python changes. Development dependencies also include Flake8 and mypy. C++ uses `.hpp` headers, `.cpp` implementations, the `ivit` namespace, and `PascalCase` types; format changed files with clang-format as described in the README.

## Testing Guidelines

Use pytest files named `test_*.py` with `test_*` functions. C++ tests use GoogleTest. Add regression coverage for changed behavior, especially inference results and public APIs. Run `pytest --cov=ivit` for coverage; no minimum percentage is configured. Document unavailable hardware, model assets, or optional backends when reporting skipped tests.

## Commit & Pull Request Guidelines

Git metadata is absent from this checkout, so historical commit conventions cannot be verified. Use concise imperative subjects, such as `Fix TensorRT device selection`. Pull requests should explain the problem, behavior changes, relevant issues, and validation commands/results. Include backend and hardware details for inference changes and screenshots for visualization changes.

## Configuration & Generated Assets

Keep downloaded models, `deps/`, build outputs, runtime caches, generated TensorRT engines, and `.env` secrets out of commits; these paths are ignored. Describe any additional backend setup in the relevant documentation.
