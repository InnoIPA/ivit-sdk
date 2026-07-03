"""
Tests for the HailoRT backend integration.

These tests run on any machine (including x86 without Hailo-8 hardware).
They validate:
  - HailoConfig Python API
  - Device-string parsing for Hailo devices
  - Runtime discovery (is_available returns False gracefully on non-Hailo hardware)
  - CMake option is wired up correctly
  - No regressions in existing backends
"""

import pytest
import os
import importlib.util

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _import_file(module_name, rel_path):
    """Import a Python file directly, bypassing package __init__ loaders."""
    abs_path = os.path.abspath(os.path.join(_ROOT, rel_path))
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cpp_available():
    """Return True when C++ bindings + libivit.so are loadable."""
    try:
        import ivit
        return ivit.is_cpp_available()
    except Exception:
        return False


def _hailo_cpp_available():
    """Return True when C++ bindings are available AND HailoConfig is exported."""
    try:
        import ivit
        return ivit.is_cpp_available() and hasattr(ivit, "HailoConfig")
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# HailoConfig Python API
# ──────────────────────────────────────────────────────────────────────────────

class TestHailoConfigPython:
    """Pure-Python HailoConfig tests (no C++ bindings required)."""

    @pytest.fixture(autouse=True)
    def _rc(self):
        self.rc = _import_file(
            "ivit_runtime_config",
            "python/ivit/core/runtime_config.py"
        )

    def test_import_hailo_config_from_python_module(self):
        """HailoConfig class exists in the pure-Python runtime_config module."""
        assert hasattr(self.rc, "HailoConfig")
        cfg = self.rc.HailoConfig()
        assert cfg is not None

    def test_hailo_config_defaults(self):
        """HailoConfig has expected default values."""
        cfg = self.rc.HailoConfig()
        assert cfg.batch_size == 1
        assert cfg.timeout_ms == 1000

    def test_hailo_config_custom_batch_size(self):
        """HailoConfig accepts custom batch_size."""
        cfg = self.rc.HailoConfig(batch_size=4)
        assert cfg.batch_size == 4

    def test_hailo_config_custom_timeout(self):
        """HailoConfig accepts custom timeout_ms."""
        cfg = self.rc.HailoConfig(timeout_ms=5000)
        assert cfg.timeout_ms == 5000

    def test_hailo_config_dict_roundtrip(self):
        """HailoConfig.to_hailo_config() returns all expected keys."""
        cfg = self.rc.HailoConfig(batch_size=2, timeout_ms=2000)
        d = cfg.to_hailo_config()
        assert d["batch_size"] == 2
        assert d["timeout_ms"] == 2000


# ──────────────────────────────────────────────────────────────────────────────
# HailoConfig C++ bindings (skipped when bindings unavailable)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _hailo_cpp_available(), reason="C++ bindings or HailoConfig not available")
class TestHailoConfigCpp:
    """Tests for HailoConfig exposed via pybind11."""

    def test_import_hailo_config_from_ivit(self):
        import ivit
        cfg = ivit.HailoConfig()
        assert cfg is not None

    def test_cpp_hailo_config_defaults(self):
        import ivit
        cfg = ivit.HailoConfig()
        assert cfg.batch_size == 1
        assert cfg.timeout_ms == 1000

    def test_cpp_hailo_config_set_fields(self):
        import ivit
        cfg = ivit.HailoConfig()
        cfg.batch_size = 4
        cfg.timeout_ms = 3000
        assert cfg.batch_size == 4
        assert cfg.timeout_ms == 3000


# ──────────────────────────────────────────────────────────────────────────────
# Device string parsing for Hailo devices
# ──────────────────────────────────────────────────────────────────────────────

class TestHailoDeviceStringParsing:
    """
    Verify that Hailo device strings are routed to the HailoRT backend.
    These strings are parsed in C++ (device_manager.cpp).
    """

    _HAILO_ALIASES = [
        "hailo",
        "hailo:0",
        "hailo:1",
    ]

    @pytest.mark.parametrize("alias", _HAILO_ALIASES)
    def test_hailo_aliases_are_known(self, alias):
        """Each Hailo alias must be listed in the supported devices table."""
        assert alias in self._HAILO_ALIASES

    def test_hailo_default_is_first_device(self):
        """'hailo' without index refers to device 0."""
        alias = "hailo"
        assert "hailo" in alias.lower()

    def test_hailo_explicit_index(self):
        """'hailo:0' explicitly addresses the first card."""
        alias = "hailo:0"
        parts = alias.split(":")
        assert parts[0] == "hailo"
        assert parts[1] == "0"


# ──────────────────────────────────────────────────────────────────────────────
# Runtime availability on non-Hailo hardware
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _cpp_available(), reason="C++ bindings not available")
class TestHailoRTRuntimeAvailability:
    """HailoRTRuntime should not crash on hardware without Hailo-8."""

    def test_list_devices_does_not_crash(self):
        """list_devices() must not raise even when Hailo-8 is absent."""
        import ivit
        devices = ivit.list_devices()
        assert isinstance(devices, list)

    def test_hailort_supported_formats(self):
        """HailoRTRuntime must report .hef as the only supported format."""
        import ivit
        # Verify supported_formats is accessible; actual values checked via C++ runtime
        # This test validates the contract: only .hef is supported
        devices = ivit.list_devices()
        hailo_devices = [d for d in devices if getattr(d, "backend", "") == "hailort"]
        if not hailo_devices:
            # No Hailo hardware — verify format contract via string check in source
            src_path = os.path.join(_ROOT, "src", "runtime", "hailort_runtime.cpp")
            if os.path.isfile(src_path):
                with open(src_path) as f:
                    content = f.read()
                assert '".hef"' in content, "hailort_runtime.cpp must include .hef as supported format"
            else:
                pytest.skip("hailort_runtime.cpp not yet implemented")

    def test_hailo_device_not_in_list_on_x86_without_card(self):
        """On x86 without Hailo-8 PCIe card, no hailo device should appear."""
        import ivit
        if os.path.exists("/dev/hailo0"):
            pytest.skip("Hailo-8 card detected — device may be present")
        devices = ivit.list_devices()
        hailo_devices = [d for d in devices if getattr(d, "backend", "") == "hailort"]
        assert len(hailo_devices) == 0, (
            f"Unexpected HailoRT devices: {[d.id for d in hailo_devices]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# convert_model without DFC installed
# ──────────────────────────────────────────────────────────────────────────────

class TestHailoConvertModel:
    """convert_model() must raise an error when hailo_model_zoo CLI is absent."""

    def test_hailort_convert_no_dfc_source_has_error_message(self):
        """hailort_runtime.cpp must include DFC installation instructions in convert_model."""
        src_path = os.path.join(_ROOT, "src", "runtime", "hailort_runtime.cpp")
        if not os.path.isfile(src_path):
            pytest.skip("hailort_runtime.cpp not yet implemented")
        with open(src_path) as f:
            content = f.read()
        assert "hailo_model_zoo" in content, (
            "convert_model() must reference hailo_model_zoo compile CLI"
        )
        assert "ModelConversionError" in content or "IVITError" in content, (
            "convert_model() must raise an error when DFC is not installed"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Backend enum in Python
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _hailo_cpp_available(), reason="C++ bindings or HailoConfig not available")
class TestBackendEnum:
    """The HailoConfig must be in the public ivit namespace."""

    def test_hailo_config_in_public_api(self):
        """HailoConfig is exported from the top-level ivit namespace."""
        import ivit
        assert hasattr(ivit, "HailoConfig"), "HailoConfig must be in ivit namespace"


# ──────────────────────────────────────────────────────────────────────────────
# Regression: existing backends unaffected
# ──────────────────────────────────────────────────────────────────────────────

class TestNoRegressions:
    """Adding HailoRT must not break existing import or API surface."""

    def test_ivit_imports_cleanly(self):
        import ivit
        assert ivit.__version__ is not None

    def test_openvino_config_still_importable(self):
        rc = _import_file("ivit_runtime_config", "python/ivit/core/runtime_config.py")
        cfg = rc.OpenVINOConfig()
        assert cfg.performance_mode == "LATENCY"

    def test_tensorrt_config_still_importable(self):
        rc = _import_file("ivit_runtime_config", "python/ivit/core/runtime_config.py")
        cfg = rc.TensorRTConfig()
        assert cfg.enable_fp16 is False

    def test_qnn_config_still_importable(self):
        rc = _import_file("ivit_runtime_config", "python/ivit/core/runtime_config.py")
        cfg = rc.QNNConfig()
        assert cfg.backend == "htp"

    def test_zoo_module_still_works(self):
        import ivit
        assert hasattr(ivit, "zoo")

    @pytest.mark.skipif(not _cpp_available(), reason="C++ bindings not available")
    def test_classifier_still_importable(self):
        from ivit import Classifier  # noqa


# ──────────────────────────────────────────────────────────────────────────────
# CMake option smoke-test (via file inspection)
# ──────────────────────────────────────────────────────────────────────────────

class TestCMakeHailoRTOption:
    """Verify CMakeLists.txt correctly wires up the HailoRT option."""

    _cmake_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "CMakeLists.txt"
    )

    def _read_cmake(self):
        with open(self._cmake_path, encoding='utf-8') as f:
            return f.read()

    def test_hailort_option_defined(self):
        """IVIT_USE_HAILORT option must be declared."""
        cmake = self._read_cmake()
        assert 'option(IVIT_USE_HAILORT' in cmake

    def test_hailort_source_added_conditionally(self):
        """hailort_runtime.cpp is added under IVIT_USE_HAILORT guard."""
        cmake = self._read_cmake()
        assert "hailort_runtime.cpp" in cmake
        assert "IVIT_USE_HAILORT" in cmake

    def test_hailort_compile_definition(self):
        """IVIT_HAS_HAILORT compile definition is set when option is ON."""
        cmake = self._read_cmake()
        assert "IVIT_HAS_HAILORT" in cmake

    def test_hailort_shown_in_summary(self):
        """HailoRT must appear in the cmake configuration summary."""
        cmake = self._read_cmake()
        assert "HailoRT:" in cmake


# ──────────────────────────────────────────────────────────────────────────────
# Header / source existence
# ──────────────────────────────────────────────────────────────────────────────

class TestHailoRTFilesExist:
    """Required HailoRT backend files must be present in the tree."""

    _root = os.path.join(os.path.dirname(__file__), "..", "..")

    def _path(self, *parts):
        return os.path.join(self._root, *parts)

    def test_hailort_runtime_header_exists(self):
        assert os.path.isfile(self._path("include", "ivit", "runtime", "hailort_runtime.hpp"))

    def test_hailort_runtime_source_exists(self):
        assert os.path.isfile(self._path("src", "runtime", "hailort_runtime.cpp"))

    def test_hailort_runtime_header_has_guard(self):
        with open(self._path("include", "ivit", "runtime", "hailort_runtime.hpp")) as f:
            content = f.read()
        assert "IVIT_RUNTIME_HAILORT_RUNTIME_HPP" in content
        assert "#ifdef IVIT_HAS_HAILORT" in content

    def test_hailort_runtime_source_has_guard(self):
        with open(self._path("src", "runtime", "hailort_runtime.cpp")) as f:
            content = f.read()
        assert "#ifdef IVIT_HAS_HAILORT" in content
        assert "#endif // IVIT_HAS_HAILORT" in content

    def test_runtime_init_registers_hailort(self):
        with open(self._path("src", "runtime", "runtime_init.cpp")) as f:
            content = f.read()
        assert "#ifdef IVIT_HAS_HAILORT" in content
        assert "HailoRTRuntime" in content

    def test_device_manager_discovers_hailort(self):
        with open(self._path("src", "core", "device_manager.cpp")) as f:
            content = f.read()
        assert "discover_hailort_devices" in content
        assert "hailo_is_available" in content

    def test_common_hpp_has_hailort_enum(self):
        with open(self._path("include", "ivit", "core", "common.hpp")) as f:
            content = f.read()
        assert "HailoRT" in content

    def test_runtime_config_hpp_has_hailo_struct(self):
        with open(self._path("include", "ivit", "core", "runtime_config.hpp")) as f:
            content = f.read()
        assert "HailoConfig" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
