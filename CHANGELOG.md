# Changelog

Versions follow [semver](https://semver.org). The C ABI (`ivit_abi_version()`) only
gains functions within a major version. The C++ API/ABI may change in a minor
release, so `libivit.so` uses the SONAME `MAJOR.MINOR` (for example `libivit.so.1.1`)
and C++ applications must be rebuilt after a minor upgrade.

## 1.1.0

### Added
- Log sink: `ivit::set_log_callback`, Python `ivit.set_log_callback`, C
  `ivit_set_log_callback` / `ivit_set_log_level`, C# `Ivit.SetLogHandler` / `Ivit.SetLogLevel`.
- `ivit_version_string()` (C), `Ivit.Version` (C#), `IVIT_VERSION_NUMBER` (C++).
- Model Zoo integrity checking: `ivit/zoo/checksums.json`, `sha256=` argument,
  `IVIT_ZOO_REQUIRE_CHECKSUM`, and `scripts/update_zoo_checksums.py`.
- Zoo `onnx_url` may be a Google Drive share link (`drive.google.com/file/d/<id>/view`, `open?id=`,
  `uc?id=`); it is downloaded through Drive's direct-download endpoint. A web page returned instead
  of the file (not shared publicly, wrong link, download quota exceeded) is a clear error, and the
  SHA256 check still applies. Mirror files are named after the model (`<model>.onnx`), since share
  links carry no file name.
- The five YOLOX ONNX files are hosted on Google Drive (their former GitHub release URLs returned
  404). All five were downloaded from Drive and matched their registered SHA256.
- `IVIT_ZOO_MIRROR` (http(s) base URL or local directory) for Zoo downloads: tried first, with
  fallback to `onnx_url`; checksum mismatches are never retried elsewhere.
- `LoadConfig.num_instances` / `ivit_load_options_v2.num_instances` (TensorRT only).
- C ABI additions (ABI stays v1, nothing removed or changed): `ivit_model_info_json`,
  `ivit_load_options_v2` / `ivit_model_load_v2`, `ivit_model_predict_ex`,
  `ivit_model_predict_batch`, `ivit_infer_options_v1`, `ivit_image_v1` (gray or BGR),
  `ivit_last_error_category`. C#: `Model.InfoJson()`, `IvitException.Category`.
- `InferConfig.agnostic_nms` (C++, Python `Detector.predict(..., agnostic_nms=)`).
- Continuous integration (`.github/workflows/ci.yml`): fast checks, a full OpenVINO build with
  the C++ and Python suites, a TensorRT-enabled build (compiled and linked without a GPU), and an
  optional self-hosted GPU job. All native jobs run `scripts/ci/build_and_test.sh`, the same script
  for local use: `scripts/ci/build_and_test.sh [--tensorrt]`.
- `libivit.so` SONAME (`MAJOR.MINOR`) for non-wheel builds.

### Changed
- **TensorRT 11.4** (was 10.12) for CI and the x86_64 NVIDIA release (`11.4.0.106-1+cuda12.9`;
  the deb now depends on `libnvinfer11`, `libnvonnxparsers11`). TensorRT 11 has only strongly
  typed networks and removed the FP16 builder flag, so `precision="fp16"` with an FP32 model now
  rewrites the parsed network to FP16 (weights, casts to float, and FP32 casts at the inputs and
  outputs so the engine I/O stays float32). Measured on an RTX 2070 SUPER: yolox-s, resnet18 and
  deeplabv3-mobilenetv3 FP16 match FP32 (same detections and top-1, 99.97% equal mask pixels) and
  run 1.3–1.8x faster. A model that is already FP16 builds an FP16 engine even with
  `precision="fp32"`. Builds against TensorRT 10 (JetPack) keep the old FP16 builder flag.
  Engines cached by TensorRT 10 are rebuilt (the cache key includes the TensorRT version).
- **OpenVINO 2026.4.1** for the Intel release (was 2025.4.1, while CI already tested 2026.4.1),
  from Intel's unversioned apt repository (`openvino/2025` stops at 2025.4.1). The Intel deb now
  also depends on the CPU and AUTO plugins and the ONNX/IR frontends, and recommends the GPU/NPU
  plugins: OpenVINO's core package has no devices or model readers, so depending on it alone
  installed an SDK that could not run a model.
- TensorRT engines are cached in `~/.cache/ivit/engines` by default (were written next to the
  model). `LoadConfig.cache_dir` and `set_cache_dir()` still take precedence.
- Python 3.10 or newer is required (was declared as 3.9+): OpenVINO 2026 publishes no Python 3.9
  wheels, so a 3.9 install could not get a working backend. The full suite passes on 3.10 and 3.12.
- `ivitConfig.cmake` looks for the TensorRT major version the SDK was built against.
- SDK tarballs are `.tar.xz` (were `.tar.gz`): with TensorRT 11.4 the NVIDIA tarball is 2.35 GB
  as gzip, over GitHub's 2 GiB per-asset limit, and 1.41 GB as xz.
- All SDK output now goes to stderr by default (previously some messages used stdout).
  Applications that parsed SDK text from stdout must switch to the log callback.
- `LogLevel` moved to `ivit/core/log.hpp` (still included by `ivit/ivit.hpp`).
- Zoo downloads write to a temporary file, verify, then atomically replace; cached
  files with a known checksum are re-verified. Downloads use a network timeout.
- Corrected the SDK manual where it contradicted the code (Detector decoder
  selection, `classes`/`max_detections`, `use_cache`).

### Added
- `bash scripts/install_sdk.sh --tensorrt`: builds the Python SDK (or `--mode cpp`) with the
  TensorRT backend. It checks for `nvcc` and the TensorRT headers first and stops with a pointer
  to the manual (3.4) instead of building without TensorRT.
- CMake option `IVIT_REQUIRE_BACKENDS` (default OFF): configuring fails, with the fix, when a
  requested backend (OpenVINO, CUDA/TensorRT) is not found, instead of dropping it with a warning
  that pip hides.

### Removed
- OpenVINO `.pdmodel` (PaddlePaddle) loading: never tested or documented, and the Intel deb has
  no Paddle frontend. Loading one now fails with an unsupported-format error.
- Dead code: `TensorRTRuntime::set_device`/`dtype_to_nv`, `TensorRTLogger::set_verbose` (TensorRT
  verbose output was already never forwarded), Python `ivit.devices.DeviceType`,
  `ivit.train.exporter.export_model` (not exported from `ivit.train`, so its own example failed),
  `ivit.utils.profiler.ScopedTimer`, and unused imports.
- `scripts/download_deps.sh` and the `deps/install` lookup in CMake (`IVIT_DEPS_DIR`): it fetched
  OpenVINO 2024.6 and only told you to download TensorRT by hand, and no install path used it.
- The `tensorrt` and `cuda` pip extras: the SDK never loads the pip `tensorrt` package, and
  `tensorrt>=8.6` could pull a different major than the SDK was built with.
- Unused Black and mypy settings and dev dependencies (neither ever ran; CI runs Flake8).
- Dead or unsupported code: `python/ivit/core/model.py` (only used by three tests that were always
  skipped, VideoSource moved to C++), the `ivit serve` command (an unauthenticated stdlib HTTP
  server; the SDK does not provide a server), the Zoo's `_export_rtdetr_onnx` and
  `_export_mmpose_onnx` (they raised ImportError on every call; `download(..., from_source=True)`
  for those models now fails with a clear message), the unused `sphinx` dev dependencies and
  readthedocs URL (there is no docs build), and the two process logs
  `docs/development/sdk-adjustments.zh-TW.md` and `release-cleanup.zh-TW.md` (superseded by this
  changelog and the manual's limits section).
- The experimental HailoRT and QNN backends (off by default, never verified on hardware; QNN had
  never been compiled): `IVIT_USE_HAILORT` / `IVIT_USE_QNN`, their runtimes,
  `BackendType::HailoRT` / `QNN`, `Layout::HailoNMS` and the `hailo_nms` decoder, the `hailo*`,
  `qnn:*`, `htp`, `hexagon`, `dsp` and `iq*` device names (they now fall back to the OpenVINO CPU
  like any other unknown name), and `.hef` / `.serialized` model routing. `backend="hailort"` /
  `"qnn"` now fails with "Unknown backend". `BackendType::Auto` is now 2 (was 3).
- RTMPose (`rtmpose-s`, `rtmpose-m`) and RT-DETR (`rtdetr-l`, `rtdetr-x`) from the Model Zoo: the
  SDK has no decoder for either, and no RT-DETR ONNX existed (its download URL returned 404 and there
  was no export recipe). The Zoo now lists 16 models: 11 ready, 5 raw.
- Model Zoo `format=` option (`ivit.zoo.download/load`, `ivit zoo download -f`) and
  `ModelInfo.formats`: only ONNX was ever produced; `format="openvino"` / `"tensorrt"` silently
  returned a path that did not exist. The Zoo now always returns ONNX, and `ivit.load()` converts it
  for the target backend (cached) as before.
- C++ `ivit::Profiler` / `ProfileReport` (`ivit/utils/profiler.hpp`): untested and unused, with no
  Python binding. Python keeps `ivit.utils.Profiler`.
- `ivit.train.trainer` module (a shim for `from ivit.train.trainer import Trainer`);
  `from ivit.train import Trainer` is unchanged.
- `packaging/package_sdk.sh.in` and the `package_sdk.sh` it generated; `scripts/build_tarball.sh`
  builds the tarball.
- The `task` parameter of `get_best_device()` (C++ and Python): it was ignored. The signature is
  now `get_best_device(priority="performance")`.
- The `task` parameter of `ivit.devices.best()`, also ignored.
- `InferConfig::enable_profiling` (C++): nothing read it and Python never bound it. Per-layer
  profiling is `LoadConfig.enable_profiling` (OpenVINO).
- Public API that could never do anything (1.1 has not shipped to customers, so it is removed
  rather than deprecated):
  - `Model::configure_openvino` / `configure_tensorrt` / `configure_qnn` (C++ and Python) always
    threw; set runtime options in `LoadConfig` when loading. Their `OpenVINOConfig`,
    `TensorRTConfig`, `QNNConfig` types and `ivit/core/runtime_config.hpp` are removed with them.
  - `Model::shutdown_async()` did nothing (there is no executor to stop).
  - `TaskType::InstanceSegmentation`, `PoseEstimation`, `FaceDetection`, `FaceRecognition`, `OCR`,
    `AnomalyDetection`: no decoder produces them. `TaskType::Unknown` is now 3 (was 9).
  - `Results::poses`, `Pose`, `Keypoint`, `Visualizer::draw_poses`: no decoder fills them, so they
    were always empty.
  - The C ABI and C# bindings are unchanged.
- The pure-Python `ivit.core.result` / `ivit.core.types` containers and `ivit.utils.Visualizer`.
  Inference always returned the native `ivit.Results`, `ivit.Detection`, `ivit.BBox`; the Python
  look-alikes had no way to receive a result, and their tests asserted defaults the SDK does not
  have (`max_detections == 300`, `conf_threshold == 0.5`, `precision is None`) and "NMS" tests that
  never called NMS. Use `results.visualize(image)` for drawing. Their tests were replaced by tests
  of the native types and by real end-to-end NMS tests (per-class vs `agnostic_nms`, also through
  `predict_async`). `ivit.utils` keeps `Profiler`, `VideoStream`/`VideoWriter` and the metrics
  helpers.
- Docker: the `docker/` directory (release and verification images, compose file), `.dockerignore`,
  `scripts/build_docker.sh`, the Docker build/push job and `skip_docker` input in the release
  workflow, the `docker` mode of `scripts/verify_package.sh` and the `--skip-docker` option of
  `scripts/release.sh`. The SDK is installed directly (`scripts/install_sdk.sh`, source build, or
  tarball/deb); nothing in the supported install paths used containers. Images previously pushed to
  the container registry are not deleted by this change.
- Code that could never run: the `Model` async executor (`ensure_executor()` was never called,
  so its thread and queue never started), OpenVINO's
  `set_async()` / `use_async_` branch (nothing could enable it), `discover_qnn_devices()` and
  `qnn_is_available()` (declared, never defined), commented-out "XXX" platform templates in
  `device_manager.cpp` (see `docs/hardware-extension.md`), and the MambaVision export helper.
- Intel VPU (Myriad) support: device discovery, the `vpu`/`myriad` device names, `ivit.devices.vpu()`
  and `DeviceType.VPU` / `ivit.D.VPU`. The hardware is end-of-life and OpenVINO removed the MYRIAD
  plugin in 2023, so these could not find a device.
- Nine Python tests that never called the SDK (they exercised numpy/OpenCV on data they built).

### Performance
- Preprocessing: ~2x faster (resize straight into the padded canvas, vectorised split/convert,
  mean/std steps skipped when they are the identity). Output is bit-identical to 1.0.
- Detection decoding: ~8x faster (yolox-s: 2.5 ms -> 0.3 ms). YOLOv5/YOLOX-style anchors whose
  objectness is below the confidence threshold are skipped (valid for sigmoid outputs, which are
  <= 1); YOLOv8 class scores are scanned class-major. Output is identical to 1.0.
- TensorRT `num_instances` (1-16, default 1 = previous behaviour): one model owns that many
  execution contexts (each with its own CUDA stream and device buffers), so concurrent callers
  run on the GPU together instead of queueing behind one lock. Measured end to end on an RTX
  2070 SUPER, FP16, 4 caller threads: yolox-nano 797 -> 1344 FPS (2 instances, +69%), yolox-s
  348 -> 434 (+25%), yolox-m 225 -> 296 (4 instances, +32%). Gains shrink as the model gets
  GPU-bound or the CPU side (pre/post-processing) saturates; each instance costs extra GPU
  memory (activation buffers), so raise it only as far as throughput improves.
- Measured (not changed): OpenVINO gains nothing from extra threads or models on CPU/iGPU - the
  device is already saturated - so `num_instances` is rejected there rather than ignored.

### Fixed
- Release workflow: `softprops/action-gh-release@v1` no longer runs on GitHub Actions; updated to v2.
- Release workflow now (a) fails early unless the tag matches the version declared in the source
  (`scripts/ci/check_version.sh`; `-rc1`/`-beta` suffixes allowed), (b) runs the full CI first and
  starts packaging only if it passes, and (c) reports a failed CI gate as a failure instead of
  "completed successfully".
- OpenVINO `precision="fp32"` is now honored on CPU and Intel GPU. Previously no hint was
  set, so OpenVINO used its device default (FP16 on Intel GPUs, BF16 on AMX CPUs).
  YOLOX decoded boxes on a UHD 770 were off by up to ~30 px. NPU is unchanged.
- Zoo ONNX export no longer leaves weights in a `<name>.onnx.data` sidecar (newer torch
  versions do this); they are merged into the single `.onnx` file so the cached file
  loads on its own and the SHA256 covers the weights.
- Async submissions (`predict_async`) silently dropped `agnostic_nms`; the option is now copied
  with the rest of the inference options.

- `ivit.zoo.load()` loaded every model without a task, so results had no decoded detections or
  classes, and torchvision models got no ImageNet normalization (resnet18 on bus.jpg: top-1 at 14%
  instead of "minibus" at 90%). Each model now carries the load settings its export needs
  (`ModelInfo.load_options`, `ModelInfo.ready`), applied by `zoo.load()`; keyword arguments
  override them. Verified for all 11 ready models through `zoo.load()` against the original
  framework (YOLOX classes, torchvision top-5, DeepLabV3 masks 100% equal). The 5 models the SDK
  cannot decode (torchvision detectors) are listed as "raw": `download()` works,
  `zoo.load()` refuses with an explanation unless `task=` is given. Downloads and exports write a
  `<model>.labels.txt` so C/C++ users of the bare file get class names.
- `get_best_device()` (C++) ranked by device type only and picked an Intel iGPU over an NVIDIA GPU
  for "performance"; it now uses the same ranking as `ivit.devices.best()` (NVIDIA > Intel
  discrete > iGPU > NPU > CPU; efficiency: NPU first; memory: discrete GPU first).
- `ivit.core.get_backend_for_device("gpu:0")` returned "tensorrt"; `gpu` is an Intel GPU
  (OpenVINO). It now follows the native device-name rules.
- `split_dataset(val_ratio=...)` ignored `val_ratio`, and reseeded the global `random` module. The
  default 0.8/0.2 split is unchanged.
- Release packaging: `download_deps.sh` could never download OpenVINO (sourcing `/etc/os-release`
  overwrote its version, and the archive URL had the wrong directory, file name and arch suffix);
  `verify_package.sh` stopped after its first check under `set -e` and its deb checks grepped their
  own output, so they always passed; `include(CPack)` ran before the CPack settings, so .deb builds
  failed (no maintainer) and lost their dependencies and file name; the tarball job lacked
  pybind11; Intel jobs installed an Ubuntu 22.04 OpenVINO build on Ubuntu 24.04. Packaging now
  also fails when CMake drops the requested backend, and verify_package.sh checks that libivit.so
  links it. Verified locally: Intel tarball and .deb build and verify; an NVIDIA build without
  TensorRT is refused.
- NVIDIA release jobs: moved from Ubuntu 22.04 with CUDA 12.2 (`Jimver/cuda-toolkit`) to
  Ubuntu 24.04 with CUDA 12.9 and TensorRT 10.12.0.36 from NVIDIA's ubuntu2404 apt repository, with
  every TensorRT package version pinned (unpinned, apt installs TensorRT 11). The Jetson job now
  checks that its runner has JetPack instead of installing an x86 CUDA toolkit.
- Tarballs: `deps/lib` stored a full copy of a library for every symlink name (Intel 61 MB, now
  39 MB); now one file per library plus symlinks. `ivitConfig.cmake` now uses the bundled
  OpenVINO / TensorRT in `deps/lib`; before, `find_package(ivit)` from either tarball failed unless
  the backend (and, for NVIDIA, the CUDA toolkit) was installed separately. Verified locally: both
  tarballs build a client with only the tarball present and run inference (CPU; RTX 2070 SUPER).
- .deb dependencies now name the backend runtimes the library was linked against
  (`libopenvino-2025.4.1`; `libnvinfer10, libnvonnxparsers10, cuda-cudart-12-9`) instead of
  `openvino (>= 2024.0)` / `libnvinfer-dev`, which let apt pick an incompatible major version.
- `ivit` CLI: `devices` crashed (`DeviceInfo.available` is `is_available`); `benchmark` crashed
  reading the input shape; `predict` crashed (it passed `conf=`/`iou=` keywords the binding does
  not take, and called `stream(show=, save=)` and `Results.show()`, which do not exist) and
  `--save` wrote JSON under an image name; `export -o` ignored the file name; `zoo info <unknown>`
  and SDK errors exited 0 or with a traceback. `benchmark` / `predict` now accept Model Zoo names
  (with their verified settings) and `--task`, say when a model file has no task (so nothing is decoded), use the
  task's default confidence unless `--conf` is given, and `--save` writes the annotated image or
  video. Covered by `tests/python/test_cli.py`.
- `ivit convert` with a `.pt` / `.pth` input failed on PyTorch 2.6+ (`torch.load` now defaults to
  `weights_only=True`, so neither TorchScript nor pickled models loaded), always exported a
  640x640 input, and `-f openvino` / `-f tensorrt` passed the `.pt` straight to the native converter.
  TorchScript now converts directly; a pickled `nn.Module` needs `--trust-pickle` (unpickling runs
  code from the file); a state dict or unreadable file fails with a one-line reason. `--input-size
  N` or `H,W` sets the input (default 640), and `.pt` → OpenVINO / TensorRT exports ONNX into the
  output directory first and converts that.
- C++ examples: `detection_demo`, `classification_demo` and `video_demo` aborted without a display
  (`cv::imshow` with no X/Wayland session); they now open a window only when a display exists, and
  the image demos also save `detection_result.jpg` / `classification_result.jpg`. `video_demo`
  reports an error instead of dividing by zero when no frame was read. `simple_inference benchmark`
  built a `Detector`, so any non-YOLO model failed; it now loads the model without a task and sizes
  the dummy image from the model input. `simple_inference` exited 0 after a failed command; it now
  exits 1.
- `examples/python/01_quickstart.py` wrote its output into `examples/data/images/`; it now writes
  `quickstart_output.jpg` to the current directory like the other examples.
- `ivit.vision.Detector` defaulted to `conf_threshold=0.25` and `max_detections=300`, unlike
  `ivit.Detector`, C++ `Detector` and `InferConfig` (0.5 and 100); all now use 0.5 / 0.45 / 100.
- The Intel tarball had no OpenVINO device: Intel's apt OpenVINO keeps the CPU/GPU/NPU plugins in
  `openvino-<version>/` next to `libopenvino.so`, and only the libraries beside it were bundled
  (the pip layout used by CI keeps them together, so CI never saw it). The directory is now
  bundled; a customer program built against the tarball runs on CPU, GPU and AUTO.
- TensorRT: a load failed (after building the engine) when the engine cache could not be written,
  e.g. a model in a read-only directory with the default cache next to it. A cache that cannot
  be written is now a warning; the built engine is used and rebuilt on the next load.
- `simple_inference clear-cache` printed the directory it would clear but passed an empty path,
  so nothing was deleted from disk.
- `ivit info` reported whether pip `openvino`/`tensorrt` packages were importable, so an apt
  install with TensorRT said "Not installed". It now shows the backends the SDK loaded and their
  runtime versions.
- The YOLOX training error suggested `pip install 'ivit-sdk[training-yolox]'` (no such extra) and
  `pip install yolox` (does not build); it now says how to use the YOLOX source tree.
- Project URLs in `pyproject.toml` point to github.com/InnoIPA/ivit-sdk.
- Callback events `pre_process`, `infer_start`, `infer_end` and `post_process` could be registered
  but never fired. `predict`/`predict_batch` now fire them in that order.
- Found by following the docs as a first-time user:
  - The manual's classification examples (Python 6.1, C++ 7.2) printed all 1000 class scores and,
    for the Zoo's resnet18, gave a wrong top-1 without any error (the example used the default
    input contract; ImageNet models need mean/std). They now print the top 5 and show the
    mean/std settings; the manual also says where pip OpenVINO's libraries are for
    `LD_LIBRARY_PATH`.
  - `simple_inference detect` on a Zoo YOLOX silently reported 0 objects (default contract, wrong
    decoder). It now explains why and points to `detection_demo ... yolox`; `classify` points to
    `classification_demo ... imagenet`. The usage text no longer splits the `convert` examples.
  - `scripts/ci/build_and_test.sh` let CMake pick any Python on PATH, so with a second Python
    installed it built the extension for that one and the venv could not import it (31
    unrelated-looking test failures). It now builds for the venv's interpreter.
  - Following the TensorRT path: `build_and_test.sh --tensorrt` ended with just "Done", and running
    what it built failed with `libnvinfer.so.11: cannot open shared object file`. It now prints
    the `LD_LIBRARY_PATH`/`PYTHONPATH` to use and the manual (3.5) explains them. The TensorRT
    headers are included as SYSTEM headers, which removes 50 deprecation warnings per build that
    came from NVIDIA's own headers.
  - Installing NVIDIA's packages as `release.yml` did failed: `libnvinfer-dev` needs
    `libnvinfer-safe-headers-dev` of the same build, and with it unpinned apt chose the cuda13.4
    one ("unmet dependencies"). Both release jobs would have failed on GitHub at that step. It is
    now pinned, and the manual (3.4) documents the full apt command.
  - There was no documented way to get TensorRT in a Python from-source install (3.2 only showed
    `IVIT_USE_TENSORRT=OFF`, and pip silently built without TensorRT when it was missing). See
    `--tensorrt` and `IVIT_REQUIRE_BACKENDS` above.
  - `.[dev]` did not include `onnx`, so after the documented install about 20 test files were
    skipped without a hint. It is now in `.[dev]`.
- Found by auditing the finished branch (each reproduced before fixing):
  - **Crash at interpreter exit** after `submit_inference` (about 60% of runs with a real model
    in flight): the detached worker threads kept running while the inference runtime and Python
    were torn down. The SDK now waits for them at exit (Python `atexit`, C++ static destruction),
    and there is `ivit.wait_for_pending_work()` / `ivit::wait_for_pending_work()`. Python callbacks
    no longer try to take the GIL once the interpreter is finalizing.
  - **TensorRT accepted any input shape or dtype** that fit the device buffer and returned wrong
    numbers (stale or reinterpreted memory); OpenVINO already rejected them. Both now reject.
  - **OpenVINO outputs of an unrepresentable type** (int16, float64, bfloat16...) were sized as
    4 bytes per element: an over-read for 16-bit types and half the data dropped for float64. They
    now fail with a clear error; TensorRT does the same at load time.
  - **Segmentation with `resize_mode="letterbox_top_left"`** returned a misaligned mask (content
    squeezed into the top ~27%; 60% agreement with the correct mask); centered letterboxing of
    images with odd padding was off by a pixel row/column. The mask is now cropped to the area
    the image occupies.
  - **`predict_tta`**: boxes from `rotate*` / `scale_*` were never mapped back (some fell outside
    the image), segmentation silently returned an empty mask, unknown names and an empty list were
    accepted, and the merge ignored classes, truncated boxes to integers and ignored
    `max_detections`. All augmentations are now mapped back and verified against OpenCV; the merge
    uses the same NMS as `predict`; unsupported input raises.
  - **`Results.save`** crashed on paths shorter than 5 characters, ignored write failures, and let
    the file extension override `format` (contradicting the manual). `format` alone decides and
    failures raise.
  - **C#**: `Predict` defaulted to `confidence = 0.5`, so a classifier returned no classes unless
    0 was passed. It defaults to -1 (task default) like every other interface;
    `ivit_model_predict_bgr` now accepts exactly -1.
  - OpenVINO `get_devices` redirected process-wide stderr with `dup2`, which two concurrent loads
    could race and leave stderr pointing at /dev/null (the same hack was already removed from
    `DeviceManager`).
  - `install_sdk.sh` ignored an already-active virtual environment: run inside a fresh venv it
    created a separate `.venv` in the repository and left the active one without the SDK (the
    next `import ivit` failed with `ModuleNotFoundError`). It now installs into the active
    environment, says so, and warns when the terminal has a different one active; `--venv` still
    wins and without an active environment it creates `.venv` as before.
  - **Model Zoo cache**: two processes downloading the same model shared one `<name>.onnx.part`
    and corrupted each other (spurious SHA256 mismatch or FileNotFoundError); an interrupted or
    failed export, or weight download, left a half-written `.onnx`/`.pth` that the next call
    returned as cached. Downloads use per-call temporary names; exports are built in a private
    directory and published (`.onnx` last) only when complete.
  - **CLI**: `ivit zoo download <unknown>`, network failures and a missing optional package
    ended in a Python traceback (now `Error: ...`, exit code 1; Ctrl-C exits 130); `ivit predict`
    treated RTSP/RTMP URLs and `.webm`/`.m4v`/... files as images; its output for URL sources had
    no usable name. `ivit export -t intel_*` always printed the same OpenVINO IR regardless of
    target, because an IR does not depend on the device; the help now says so.
  - Smaller: `ivit.load("")` raised a bare `RuntimeError` (now `ModelLoadError`);
    `Results.topk(-1)` and `set_log_level("bogus")` are rejected with `ValueError`;
    `devices.gpu(7)` returned `gpu:0`; `devices.best("bogus")` silently used another ranking;
    `visualize()` threw an OpenCV exception for grayscale/BGRA canvases and
    `Visualizer::create_comparison` divided by zero for `cols=0`; `Detector::predict_video` treated
    "720p.mp4" as camera 720 and the Python `Detector.predict_video` ended silently for a source
    that cannot be opened; `DeviceManager::get_device_status` reported unknown ids (`gpu:1`,
    `npu` without an NPU) as available under the CPU's name and threw on `cuda:abc`;
    `predict_concurrent` skipped `BatchEnd` when an image failed; `set_cache_dir` was an
    unsynchronised global; OpenVINO's own exception type escaped `infer`.
  - **Packaging**: a failed Python wheel build was swallowed and the tarball shipped without it,
    and `verify_package.sh` only checked for the wheel when one existed. The wheel is now required.
  - Docs that disagreed with the code: `max_detections` default (100, not 300), top-level
    `ivit.Detector`/`Classifier`/`Segmentor` are slimmer native classes, not `ivit.vision.*`, the
    `ivit.load` docstring offered Model Zoo names, and the module docstring called a
    non-existent `results.show()`. Python exception hierarchy: catch `ivit.IVITError`,
    `ivit.ModelLoadError`, ... (native errors are instances of those); the `ivit.core.*` classes
    are Python-side subclasses with causes/suggestions and do not catch native errors.
  - **Release workflow**: a manual run with a subset of platforms failed in `prepare` (multi-line
    `jq` output in `GITHUB_OUTPUT`); a tag push always included the Jetson job, which blocks the
    whole release when no self-hosted arm64 runner exists (now opt-in through the repository
    variable `IVIT_JETSON_RUNNER`); release notes of a pre-release tag listed file names that do
    not exist. `.gitignore` did not cover `output_seg*` written by an example; `docs/README.md`
    said CI runs on every push (it runs on `main`, PRs, manual runs and release tags).
  - The Zoo export error said `pip install torch torchvision`, after which the export still failed
    on `onnxscript`. The message now names every dependency (`.[zoo]`), and a missing `onnx` /
    `onnxscript` during export is reported the same way instead of as a torch-internal traceback.
- The release notes named the Jetson deb `..._arm64-jetson.deb`; CMake builds `..._arm64-nvidia.deb`.
- `ivit.zoo.list_models(task=...)` returned `[]` for an unknown task name (for example
  `"classification"`, the `LoadConfig` spelling); it now raises `ValueError` listing `classify`,
  `detect` and `segment`.
- `ivit.utils.Profiler.benchmark()` raised `AttributeError` (it called the removed Python
  runtime's `model._runtime.infer`) and its precision was always "fp32" (it read attributes that do
  not exist). It now times full `model.predict()` calls on an image sized from the model input
  (or `input_shape=` / `image=`), reports the loaded precision, and the report says "not measured"
  for memory instead of 0 MB. `MetricsCollector` listeners may now unsubscribe while being notified.
  `ivit.utils` (profiler, video, metrics) is covered by `tests/python/test_utils.py`.
- Docs/text that no longer matched the code: the manual's export claims for `convert_model` /
  `clear_cache`, the removed Python runtime, `shutdown_async`, `predict_concurrent`, the install
  log sections; `ivit.vision`'s "pure Python fallback" (there is none); the YOLOX export error that
  recommended `pip install yolox` (its PyPI release does not build); `AGENTS.md`'s `docker/`.
- Examples: all used unverified `yolov8n`-style models (and pointed at AGPL-licensed Ultralytics);
  the Python ones also put the source tree first on `sys.path`, hiding an installed SDK. They now
  use Model Zoo names with their verified settings; the C++ demos gained `yolox` / `imagenet` options.

### Behaviour changes
- Building from source with pip OpenVINO now requires `openvino>=2026.0` (`install_sdk.sh`, the
  `[openvino]` extra, CI). Earlier pip releases use the old C++ ABI and fail to link with the system
  OpenCV (2025.4.1 verified failing, 2026.4.1 working). apt/tgz OpenVINO builds are unaffected.
- `InferConfig.conf_threshold` now defaults to -1, meaning "task default": detection still uses
  0.5 (unchanged), classification now uses 0 and returns every class. Before, the 0.5 default made
  a classifier with several similar scores return an empty list. Code that reads
  `InferConfig().conf_threshold` now sees -1; setting a value in [0,1] behaves exactly as before.
  Negative values other than -1 are rejected instead of being silently treated as unset.
  `ivit_infer_options_v1.confidence` follows the same rule; the older `ivit_model_predict_bgr`
  and `Detector.predict(conf_threshold=0.25)` are unchanged.
- A labels list (explicit, `labels_path` or a `.labels.txt` sidecar) whose length differs from the
  model's output class count now fails at load with `labels has N entries but the model outputs M
  classes`; before, it failed at the first prediction. Applies wherever the class count follows from
  the declared output shape (classification, segmentation and YOLO-style detection).

### Compatibility
- Wire-compatible with 1.0.0 for the C ABI. C++ applications must be rebuilt.
- Python 3.9–3.12 wheel layout unchanged.

## 1.0.0
- First consolidated source release: shared preprocessing/decoder contract, C ABI v1 and the C#
  wrapper, content-fingerprinted caches, Python async futures, TensorRT resource management.
