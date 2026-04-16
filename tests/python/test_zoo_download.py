"""Tests for Model Zoo ONNX direct download support."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ivit.zoo.registry import (
    ModelInfo,
    MODEL_REGISTRY,
    download,
    _download_onnx_direct,
    _MIN_ONNX_SIZE,
)


_ONNX_URL_BASE = "https://github.com/innodisk-mannywang/ivit-sdk/releases/download/models-v1.0"

# Models that should have onnx_url set
_MODELS_WITH_ONNX_URL = [
    "yolox-nano",
    "yolox-tiny",
    "yolox-s",
    "yolox-m",
    "yolox-l",
    "rtdetr-l",
    "rtdetr-x",
    "mambavision-t",
    "mambavision-s",
    "mambavision-b",
    "mambavision-l",
    "mambavision-l2",
    "rtmpose-s",
    "rtmpose-m",
]


class TestModelInfoOnnxUrl:
    """Tests for the onnx_url field on ModelInfo."""

    def test_onnx_url_default_is_none(self):
        """onnx_url should default to None."""
        info = ModelInfo(
            name="test",
            task="detect",
            description="test model",
            input_size=(640, 640),
            num_classes=80,
            formats=["onnx"],
        )
        assert info.onnx_url is None

    def test_onnx_url_can_be_set(self):
        """onnx_url can be explicitly set."""
        url = "https://example.com/model.onnx"
        info = ModelInfo(
            name="test",
            task="detect",
            description="test model",
            input_size=(640, 640),
            num_classes=80,
            formats=["onnx"],
            onnx_url=url,
        )
        assert info.onnx_url == url

    @pytest.mark.parametrize("model_name", _MODELS_WITH_ONNX_URL)
    def test_models_have_onnx_url(self, model_name):
        """All target models should have onnx_url set."""
        info = MODEL_REGISTRY[model_name]
        assert info.onnx_url is not None
        assert info.onnx_url.startswith(_ONNX_URL_BASE)
        assert info.onnx_url.endswith(".onnx")

    def test_torchvision_models_have_no_onnx_url(self):
        """Torchvision models should not have onnx_url."""
        torchvision_models = [
            name for name, info in MODEL_REGISTRY.items()
            if info.source == "torchvision"
        ]
        assert len(torchvision_models) > 0
        for name in torchvision_models:
            assert MODEL_REGISTRY[name].onnx_url is None, (
                f"Torchvision model '{name}' should not have onnx_url"
            )

    def test_all_nine_models_accounted_for(self):
        """All expected models with onnx_url should be accounted for (updated for MambaVision)."""
        models_with_url = [
            name for name, info in MODEL_REGISTRY.items()
            if info.onnx_url is not None
        ]
        assert len(models_with_url) == 14
    def test_all_models_with_onnx_url_accounted_for(self):
        """Registry models with onnx_url must match the expected list exactly."""
        models_with_url = sorted(
            name for name, info in MODEL_REGISTRY.items()
            if info.onnx_url is not None
        )
        assert models_with_url == sorted(_MODELS_WITH_ONNX_URL)


class TestDownloadRouting:
    """Tests for download routing logic (direct ONNX vs. source conversion)."""

    @patch("ivit.zoo.registry._download_onnx_direct")
    @patch("ivit.zoo.registry._get_cache_dir")
    def test_download_uses_direct_onnx_by_default(self, mock_cache_dir, mock_direct):
        """download() should use _download_onnx_direct when onnx_url exists."""
        tmp_dir = Path("/tmp/test_zoo_cache")
        mock_cache_dir.return_value = tmp_dir
        expected_path = tmp_dir / "yolox-s.onnx"
        mock_direct.return_value = expected_path

        result = download("yolox-s", format="onnx")

        mock_direct.assert_called_once()
        call_args = mock_direct.call_args
        assert call_args[0][0] == "yolox-s"  # name
        assert "yolox-s.onnx" in call_args[0][1]  # onnx_url
        assert result == expected_path

    @patch("ivit.zoo.registry._export_yolox_onnx")
    @patch("ivit.zoo.registry._get_cache_dir")
    def test_download_from_source_skips_direct(self, mock_cache_dir, mock_export):
        """download(from_source=True) should skip direct download and use exporter."""
        tmp_dir = Path("/tmp/test_zoo_cache")
        mock_cache_dir.return_value = tmp_dir
        expected_path = tmp_dir / "yolox-s.onnx"
        mock_export.return_value = expected_path

        result = download("yolox-s", format="onnx", from_source=True)

        mock_export.assert_called_once()
        assert result == expected_path

    @patch("ivit.zoo.registry._export_torchvision_onnx")
    @patch("ivit.zoo.registry._get_cache_dir")
    def test_download_torchvision_goes_to_exporter(self, mock_cache_dir, mock_export):
        """Torchvision models (no onnx_url) should go to torchvision exporter."""
        tmp_dir = Path("/tmp/test_zoo_cache")
        mock_cache_dir.return_value = tmp_dir
        expected_path = tmp_dir / "resnet18.onnx"
        mock_export.return_value = expected_path

        result = download("resnet18", format="onnx")

        mock_export.assert_called_once()
        assert result == expected_path

    @patch("ivit.zoo.registry._download_onnx_direct")
    @patch("ivit.zoo.registry._get_cache_dir")
    def test_download_non_onnx_format_skips_direct(self, mock_cache_dir, mock_direct):
        """Non-ONNX format should never use direct download even if onnx_url exists."""
        tmp_dir = Path("/tmp/test_zoo_cache")
        mock_cache_dir.return_value = tmp_dir

        # openvino format for yolox-s — should not call _download_onnx_direct
        # It won't have a conversion path either, but the routing should skip direct
        try:
            download("yolox-s", format="openvino")
        except Exception:
            pass  # May fail for missing converter, that's OK

        mock_direct.assert_not_called()

    @patch("ivit.zoo.registry._download_onnx_direct")
    @patch("ivit.zoo.registry._get_cache_dir")
    def test_download_rtdetr_uses_direct_onnx(self, mock_cache_dir, mock_direct):
        """RT-DETR models should use direct ONNX download by default."""
        tmp_dir = Path("/tmp/test_zoo_cache")
        mock_cache_dir.return_value = tmp_dir
        expected_path = tmp_dir / "rtdetr-l.onnx"
        mock_direct.return_value = expected_path

        result = download("rtdetr-l", format="onnx")

        mock_direct.assert_called_once()
        assert result == expected_path

    @patch("ivit.zoo.registry._download_onnx_direct")
    @patch("ivit.zoo.registry._get_cache_dir")
    def test_download_rtmpose_uses_direct_onnx(self, mock_cache_dir, mock_direct):
        """RTMPose models should use direct ONNX download by default."""
        tmp_dir = Path("/tmp/test_zoo_cache")
        mock_cache_dir.return_value = tmp_dir
        expected_path = tmp_dir / "rtmpose-s.onnx"
        mock_direct.return_value = expected_path

        result = download("rtmpose-s", format="onnx")

        mock_direct.assert_called_once()
        assert result == expected_path

    @patch("ivit.zoo.registry._get_cache_dir")
    def test_download_cached_model_returns_immediately(self, mock_cache_dir, tmp_path):
        """If model file already exists, download() should return it without downloading."""
        mock_cache_dir.return_value = tmp_path
        cached_file = tmp_path / "yolox-s.onnx"
        cached_file.write_bytes(b"fake onnx content")

        result = download("yolox-s", format="onnx")

        assert result == cached_file

    @patch("ivit.zoo.registry._download_file")
    @patch("ivit.zoo.registry._get_cache_dir")
    def test_download_onnx_direct_validates_file_size(self, mock_cache_dir, mock_dl, tmp_path):
        """_download_onnx_direct should reject files smaller than _MIN_ONNX_SIZE."""
        mock_cache_dir.return_value = tmp_path
        onnx_path = tmp_path / "test.onnx"

        def fake_download(url, dest, desc=None):
            dest.write_bytes(b"tiny")  # < _MIN_ONNX_SIZE
            return dest

        mock_dl.side_effect = fake_download

        with pytest.raises(RuntimeError, match="too small"):
            _download_onnx_direct("test", "https://example.com/test.onnx", onnx_path)

        # File should be cleaned up
        assert not onnx_path.exists()

    @patch("ivit.zoo.registry._download_file")
    @patch("ivit.zoo.registry._get_cache_dir")
    def test_download_onnx_direct_accepts_valid_file(self, mock_cache_dir, mock_dl, tmp_path):
        """_download_onnx_direct should accept files >= _MIN_ONNX_SIZE."""
        mock_cache_dir.return_value = tmp_path
        onnx_path = tmp_path / "test.onnx"

        def fake_download(url, dest, desc=None):
            dest.write_bytes(b"x" * _MIN_ONNX_SIZE)
            return dest

        mock_dl.side_effect = fake_download

        result = _download_onnx_direct("test", "https://example.com/test.onnx", onnx_path)

        assert result == onnx_path
        assert onnx_path.exists()
