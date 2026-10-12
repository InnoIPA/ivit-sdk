"""Release metadata stays consistent and the manual stays truthful."""

import ast
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def versions():
    cmake = re.search(r"project\(ivit VERSION (\d+\.\d+\.\d+)", read("CMakeLists.txt")).group(1)
    pyproject = re.search(r'^version = "([^"]+)"', read("pyproject.toml"), re.M).group(1)
    init = re.search(r'__version__ = "([^"]+)"', read("python/ivit/__init__.py")).group(1)
    header = read("include/ivit/ivit.hpp")
    parts = [
        re.search(rf"#define IVIT_VERSION_{name} (\d+)", header).group(1)
        for name in ("MAJOR", "MINOR", "PATCH")
    ]
    string = re.search(r'#define IVIT_VERSION_STRING "([^"]+)"', header).group(1)
    return {
        "CMakeLists.txt": cmake,
        "pyproject.toml": pyproject,
        "python/ivit/__init__.py": init,
        "ivit.hpp MAJOR.MINOR.PATCH": ".".join(parts),
        "ivit.hpp STRING": string,
    }


def test_all_version_declarations_agree():
    found = versions()
    assert len(set(found.values())) == 1, found


def test_changelog_and_manual_mention_current_version():
    version = versions()["CMakeLists.txt"]
    assert f"## {version}" in read("CHANGELOG.md")
    assert f"1.{version.split('.')[1]}.0" in read("docs/sdk-manual.zh-TW.md").splitlines()[2]
    assert version in read("docs/sdk-manual.zh-TW.md").splitlines()[2]


def test_release_workflow_default_matches_version():
    version = versions()["CMakeLists.txt"]
    assert f"default: '{version}'" in read(".github/workflows/release.yml")


def test_shared_library_declares_soname_for_native_builds():
    cmake = read("CMakeLists.txt")
    assert "SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}" in cmake


def test_manual_python_examples_are_valid_syntax():
    manual = read("docs/sdk-manual.zh-TW.md")
    blocks = re.findall(r"```python\n(.*?)```", manual, re.S)
    assert blocks
    for index, block in enumerate(blocks):
        try:
            ast.parse(block)
        except SyntaxError as error:  # pragma: no cover - failure path
            raise AssertionError(f"manual python block #{index + 1}: {error}\n{block}")


def test_manual_relative_links_resolve():
    manual_path = ROOT / "docs/sdk-manual.zh-TW.md"
    manual = manual_path.read_text(encoding="utf-8")
    missing = []
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", manual):
        if "://" in target or target.startswith("mailto:"):
            continue
        if not (manual_path.parent / target).resolve().exists():
            missing.append(target)
    assert not missing, missing


def test_manual_does_not_repeat_corrected_claims():
    manual = read("docs/sdk-manual.zh-TW.md")
    assert "依檔名判斷模型類型" not in manual.replace("不會依檔名判斷模型類型", "")
    assert "目前未傳入底層單張 predict" not in manual
    assert "未完整傳遞到 runtime 的磁碟快取設定" not in manual


# ------------------------------------------------------------------ continuous integration


def test_ci_workflow_runs_the_shared_script_for_every_backend_job():
    workflow = read(".github/workflows/ci.yml")
    assert workflow.count("scripts/ci/build_and_test.sh") >= 3  # openvino, tensorrt, gpu
    assert "--tensorrt" in workflow
    assert (ROOT / "scripts/ci/build_and_test.sh").exists()
    # Invoked through bash so the check survives a checkout that dropped the executable bit.
    assert "run: bash scripts/ci/build_and_test.sh" in workflow


def test_ci_script_cleans_up_the_source_tree_and_tracks_the_pinned_tensorrt_release():
    script = read("scripts/ci/build_and_test.sh")
    assert "trap cleanup EXIT" in script  # the build drops _ivit_core*.so into python/ivit
    assert "set -euo pipefail" in script
    # The headers must come from the same TensorRT release as the pip libraries.
    assert 'TRT_VERSION="11.4"' in script and 'TRT_TAG="v11.4"' in script
    assert 'ONNX_PARSER_BRANCH="release/11.4-GA"' in script
    assert 'libnvinfer.so.$TRT_MAJOR' in script  # the soname follows the pinned major
    assert "tensorrt-cu12==$TRT_VERSION" in script


def test_gpu_job_fails_when_tensorrt_tests_were_skipped():
    workflow = read(".github/workflows/ci.yml")
    assert "TensorRT GPU required" in workflow  # the skip reason the GPU job treats as failure
    assert "TensorRT GPU required" in read("tests/python/test_tensorrt_cache_contract.py")
    assert "tee \"$WORK/pytest.log\"" in read("scripts/ci/build_and_test.sh")


def test_release_workflow_uses_no_end_of_life_actions():
    release = read(".github/workflows/release.yml")
    assert "action-gh-release@v1" not in release  # v1 no longer runs on GitHub Actions


# ------------------------------------------------------------------ release gating


def run_version_check(version):
    return subprocess.run(
        ["bash", str(ROOT / "scripts/ci/check_version.sh"), version],
        capture_output=True, text=True,
    )


def test_version_check_accepts_the_source_version_in_tag_and_prerelease_forms():
    current = versions()["CMakeLists.txt"]
    for form in (current, f"v{current}", f"v{current}-rc1", f"{current}-beta"):
        result = run_version_check(form)
        assert result.returncode == 0, (form, result.stdout, result.stderr)


def test_version_check_rejects_a_tag_that_differs_from_the_source():
    major, minor, patch = versions()["CMakeLists.txt"].split(".")
    for other in (f"{major}.{minor}.{int(patch) + 1}", f"{major}.{int(minor) + 1}.{patch}", "9.9.9"):
        result = run_version_check(f"v{other}")
        assert result.returncode == 1, other
        assert "MISMATCH" in result.stdout


def test_version_check_rejects_malformed_versions():
    for bad in ("banana", "1.1", "v1.1.0.0", "1.1.x", ""):
        assert run_version_check(bad).returncode == 2, bad


def test_release_waits_for_ci_and_checks_the_version_first():
    release = read(".github/workflows/release.yml")
    assert "uses: ./.github/workflows/ci.yml" in release
    for job in ("build-tarball", "build-deb"):
        # the job header must be followed directly by its needs line
        assert re.search(rf"\n  {job}:\n    needs: \[prepare, ci\]\n", release), job
    assert "bash scripts/ci/check_version.sh" in release


def test_docker_is_gone_and_nothing_still_refers_to_it():
    assert not (ROOT / "docker").exists()
    assert not (ROOT / ".dockerignore").exists()
    assert not (ROOT / "scripts/build_docker.sh").exists()
    for relative in (
        ".github/workflows/release.yml", ".github/workflows/ci.yml", "scripts/release.sh",
        "scripts/verify_package.sh", "scripts/package_source.py", "README.md", "docs/README.md",
        "docs/deployment/release-automation.md",
    ):
        assert "docker" not in read(relative).lower(), relative


def test_notify_reports_a_failed_or_skipped_ci_gate_as_failure():
    release = read(".github/workflows/release.yml")
    notify = release[release.index("\n  notify:"):]
    assert "needs: [ci, prepare" in notify  # CI failing skips the builds; it must still be seen
    assert "contains(needs.*.result, 'cancelled')" in notify


# ------------------------------------------------------------------ removed items stay removed


def test_removed_files_and_settings_do_not_come_back():
    for gone in (
        "python/ivit/core/model.py",  # only existed for tests that were permanently skipped
        # A parallel pure-Python Results/Detection/BBox and a Visualizer that could only draw those.
        # Nothing in the SDK could produce them (inference returns the native types), so they were
        # untouched by real use while their tests asserted defaults the SDK does not have.
        "python/ivit/core/result.py",
        "python/ivit/core/types.py",
        "python/ivit/utils/visualizer.py",
        "docs/development/sdk-adjustments.zh-TW.md",
        "docs/development/release-cleanup.zh-TW.md",
        "packaging/package_sdk.sh.in",  # superseded by scripts/build_tarball.sh; nothing ran it
        "python/ivit/train/trainer.py",  # compatibility shim for an import nobody used
        "include/ivit/utils/profiler.hpp",  # C++ Profiler: untested, unused, no binding
        "src/utils/profiler.cpp",
    ):
        assert not (ROOT / gone).exists(), gone
    pyproject = read("pyproject.toml")
    assert "sphinx" not in pyproject and "readthedocs" not in pyproject  # there is no docs build
    cli = read("python/ivit/cli.py")
    assert "cmd_serve" not in cli and "HTTPServer" not in cli  # unauthenticated, not a supported feature


def test_no_markdown_link_points_at_a_missing_file():
    missing = []
    for doc in [ROOT / "README.md", ROOT / "CHANGELOG.md", *(ROOT / "docs").rglob("*.md")]:
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", doc.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("mailto:"):
                continue
            if not (doc.parent / target).resolve().exists():
                missing.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not missing, missing


def test_dead_internals_stay_removed():
    """Code that could never run: keep it from creeping back."""
    sources = {
        str(p.relative_to(ROOT)): p.read_text(encoding="utf-8", errors="ignore")
        for p in [*(ROOT / "src").rglob("*.[ch]pp"), *(ROOT / "include").rglob("*.hpp"),
                  *(ROOT / "python/ivit").rglob("*.py")]
    }
    gone = {
        "ensure_executor": "executor that nothing ever started",
        "executor_running_": "executor that nothing ever started",
        "use_async_": "OpenVINO async branch with no way to enable it",
        "discover_qnn_devices": "declared, never defined",
        "qnn_is_available": "declared, never defined",
        "IVIT_HAS_XXX": "commented-out template; see docs/hardware-extension.md",
        "load_mambavision_for_export": "MambaVision support was removed",
        "MYRIAD": "Intel VPU is end-of-life and no longer supported by OpenVINO",
    }
    found = [f"{path}: {token} ({why})" for token, why in gone.items()
             for path, text in sources.items() if token in text]
    assert not found, found


def test_ignored_options_stay_removed():
    common = read("include/ivit/core/common.hpp")
    infer = common[common.index("struct InferConfig {"):]
    infer = infer[:infer.index("};")]
    assert "enable_profiling" not in infer  # nothing read it; LoadConfig.enable_profiling is the real one
    assert "task" not in read("python/ivit/devices.py").split("def best(", 1)[1].split(")", 1)[0]
    # The aarch64 deb is named <arch>-<backend> by CMake, i.e. arm64-nvidia.
    assert "arm64-jetson.deb" not in read(".github/workflows/release.yml")


def test_cpp_examples_run_headless_and_report_failures():
    # Without a display OpenCV's GUI aborts the process, so a window is only opened when one exists.
    for name in ("detection_demo", "classification_demo", "video_demo"):
        source = read(f"examples/cpp/{name}.cpp")
        assert "has_display()" in source, name
    simple = read("examples/cpp/simple_inference.cpp")
    assert "_exit(g_exit_code)" in simple and simple.count("g_exit_code = 1;") == 6
    assert "load_model(model_path, config)" in simple  # benchmark loads without a task


def test_removed_public_api_stays_removed():
    sources = {
        path: read(path)
        for path in ("include/ivit/core/model.hpp", "include/ivit/core/result.hpp",
                     "include/ivit/core/common.hpp", "include/ivit/utils/visualizer.hpp",
                     "python/ivit/_binding.cpp", "python/ivit/__init__.py")
    }
    gone = ("shutdown_async", "configure_openvino", "configure_tensorrt", "configure_qnn",
            "OpenVINOConfig", "TensorRTConfig", "QNNConfig", "struct Pose", "struct Keypoint",
            "poses", "draw_poses", "InstanceSegmentation", "PoseEstimation", "FaceDetection",
            "FaceRecognition", "AnomalyDetection", "OCR")
    found = [f"{path}: {token}" for token in gone for path, text in sources.items()
             if re.search(rf"\b{token}\b", text)]
    assert not found, found
    assert not (ROOT / "include/ivit/core/runtime_config.hpp").exists()



# ------------------------------------------------------------------ packaging defects stay fixed


def test_unused_dependency_downloader_stays_removed():
    # Every install path uses pip (CI, install_sdk.sh) or apt (release); deps/ was never read.
    assert not (ROOT / "scripts/download_deps.sh").exists()
    assert "IVIT_DEPS_DIR" not in read("CMakeLists.txt")


def test_verify_package_counts_checks_and_greps_the_package_not_its_own_output():
    script = read("scripts/verify_package.sh")
    assert "((PASS++))" not in script and "((FAIL++))" not in script  # exits under set -e at 0
    assert not re.search(r'check "[^"]*" [^\n]*\| *grep', script)    # piped grep saw check's output
    assert "grep -q" not in script  # with pipefail, -q kills dpkg-deb with SIGPIPE
    assert not re.search(r"\| *grep -q", read("scripts/ci/build_and_test.sh"))  # same for ldd
    assert "links_backend" in script


def test_packaging_refuses_to_build_without_the_requested_backend():
    for name in ("scripts/build_tarball.sh", "scripts/build_deb.sh"):
        script = read(name)
        assert 'tee "$BUILD_DIR/configure.log"' in script, name
        assert 'require_backend "$BUILD_DIR/configure.log"' in script, name


def test_cpack_is_included_after_its_settings():
    cmake = read("CMakeLists.txt")
    include_at = cmake.index("include(CPack)")
    for setting in ("CPACK_PACKAGE_CONTACT", "CPACK_DEBIAN_PACKAGE_DEPENDS", "CPACK_PACKAGE_FILE_NAME"):
        assert cmake.rindex(setting) < include_at, setting


def test_release_runners_match_their_apt_sources():
    release = read(".github/workflows/release.yml")
    assert "openvino ubuntu24 main" in release and "OV=2026.4.1" in release
    assert "matrix.platform == 'aarch64-jetson' && 'self-hosted-arm64' || 'ubuntu-24.04'" in release
    assert "ubuntu-22.04" not in release
    assert "ubuntu22 main" not in release
    assert "cuda/repos/ubuntu2404/x86_64/cuda-keyring" in release
    assert "Jimver/cuda-toolkit" not in release  # replaced by pinned apt packages
    assert '"pybind11>=2.11"' in release  # the tarball builds the Python bindings


def test_release_pins_tensorrt_so_apt_cannot_pick_a_newer_major():
    release = read(".github/workflows/release.yml")
    assert release.count("TRT_VERSION: '11.4.0.106-1+cuda12.9'") == 2  # tarball and deb jobs
    for package in ("libnvinfer11", "libnvinfer-dev", "libnvinfer-headers-dev",
                    "libnvonnxparsers11", "libnvonnxparsers-dev"):
        assert f'"{package}=${{TRT_VERSION}}"' in release, package
    assert 'CUDAToolkit_ROOT=${CUDA_HOME}' in release


def test_release_pins_openvino_2026_from_the_unversioned_intel_repo():
    release = read(".github/workflows/release.yml")
    # The year-named repo (openvino/2025) stops at 2025.4.1; 2026 releases are only here.
    assert release.count("https://apt.repos.intel.com/openvino ubuntu24 main") == 2
    assert "openvino/2025" not in release
    assert release.count("OV=2026.4.1") == 2
    assert release.count('"libopenvino-dev-$OV" "libopenvino-intel-cpu-plugin-$OV"') == 2


def test_deb_depends_on_the_backend_runtimes_it_was_linked_against():
    cmake = read("CMakeLists.txt")
    # OpenVINO's core package has no device plugins or model readers; depend on those too.
    for part in ("libopenvino-${_ivit_ov}", "libopenvino-intel-cpu-plugin-${_ivit_ov}",
                 "libopenvino-auto-plugin-${_ivit_ov}", "libopenvino-onnx-frontend-${_ivit_ov}",
                 "libopenvino-ir-frontend-${_ivit_ov}"):
        assert part in cmake, part
    assert "libopenvino-intel-gpu-plugin-${_ivit_ov}" in cmake.split("CPACK_DEBIAN_PACKAGE_RECOMMENDS", 1)[1]
    assert "libnvinfer${IVIT_TENSORRT_MAJOR}, libnvonnxparsers${IVIT_TENSORRT_MAJOR}" in cmake
    # The CMake config looks for the same TensorRT major the SDK was built against.
    assert "libnvinfer.so.@IVIT_TENSORRT_MAJOR@" in read("cmake/ivitConfig.cmake.in")
    assert "cuda-cudart-${CUDAToolkit_VERSION_MAJOR}-${CUDAToolkit_VERSION_MINOR}" in cmake


def test_intel_tarball_bundles_the_openvino_plugin_directory():
    # apt/tgz OpenVINO keeps CPU/GPU/NPU plugins in <libdir>/openvino-<version>/; a tarball
    # without that directory had no OpenVINO device at all.
    cmake = read("CMakeLists.txt")
    assert '"${OV_LIB_DIR}/openvino-${OpenVINO_VERSION}"' in cmake
    assert 'install(DIRECTORY "${IVIT_BUNDLE_OV_PLUGIN_DIR}" DESTINATION ${CMAKE_INSTALL_PREFIX}/deps/lib)' in cmake


def test_sdk_tarballs_use_xz_to_stay_under_the_github_asset_limit():
    # TensorRT 11.4: the NVIDIA tarball is 2.35 GB as .tar.gz (over 2 GiB), 1.41 GB as .tar.xz.
    assert 'tar -cJf "${SDK_NAME}.tar.xz"' in read("scripts/build_tarball.sh")
    release = read(".github/workflows/release.yml")
    assert ".tar.gz" not in release and release.count("ivit-sdk-*.tar.xz") == 2
    assert 'tar -xf "$tarball"' in read("scripts/verify_package.sh")


def test_tarball_bundle_keeps_symlinks_and_its_cmake_config_uses_them():
    cmake = read("CMakeLists.txt")
    assert "file(CREATE_LINK" in cmake  # one copy per library; other names are symlinks
    config = read("cmake/ivitConfig.cmake.in")
    assert '"${PACKAGE_PREFIX_DIR}/deps/lib"' in config
    bundle_check = config.index("_ivit_bundle_dir")
    assert bundle_check < config.index("find_dependency(OpenVINO")
    assert bundle_check < config.index("find_dependency(CUDAToolkit)")


def test_hailort_qnn_rtmpose_and_rtdetr_stay_removed():
    for path in ("src/runtime/hailort_runtime.cpp", "src/runtime/qnn_runtime.cpp",
                 "include/ivit/runtime/hailort_runtime.hpp", "include/ivit/runtime/qnn_runtime.hpp"):
        assert not (ROOT / path).exists(), path
    sources = {path: read(path) for path in (
        "CMakeLists.txt", "cmake/ivitConfig.cmake.in", "include/ivit/core/common.hpp",
        "src/core/device_manager.cpp", "src/runtime/runtime_factory.cpp",
        "src/vision/processing.cpp", "python/ivit/_binding.cpp", "python/ivit/zoo/registry.py",
    )}
    found = [f"{path}: {token}" for token in ("HAILO", "Hailo", "hailo", "QNN", "qnn", "rtmpose", "rtdetr")
             for path, text in sources.items() if token in text]
    assert not found, found
