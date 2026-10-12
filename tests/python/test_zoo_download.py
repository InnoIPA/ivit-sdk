"""Tests for Model Zoo ONNX direct download support."""

import pytest
from pathlib import Path
from unittest.mock import patch

from ivit.zoo.registry import (
    ModelInfo,
    MODEL_REGISTRY,
    download,
    get_model_info,
    load as registry_load,
    _download_onnx_direct,
    _MIN_ONNX_SIZE,
)



# Models that should have onnx_url set
_MODELS_WITH_ONNX_URL = [
    "yolox-nano",
    "yolox-tiny",
    "yolox-s",
    "yolox-m",
    "yolox-l",
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
            onnx_url=url,
        )
        assert info.onnx_url == url

    @pytest.mark.parametrize("model_name", _MODELS_WITH_ONNX_URL)
    def test_models_have_onnx_url(self, model_name):
        """Every pre-converted model has a usable https onnx_url."""
        from ivit.zoo.registry import _google_drive_file_id

        url = MODEL_REGISTRY[model_name].onnx_url
        assert url is not None and url.startswith("https://")
        assert _google_drive_file_id(url) or url.endswith(".onnx"), url
        if model_name.startswith("yolox-"):
            assert _google_drive_file_id(url), url  # hosted on Google Drive

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

    def test_all_pre_converted_models_accounted_for(self):
        """Exactly the 5 YOLOX models have onnx_url."""
        models_with_url = [
            name for name, info in MODEL_REGISTRY.items()
            if info.onnx_url is not None
        ]
        assert len(models_with_url) == 5


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

        result = download("yolox-s")

        mock_direct.assert_called_once()
        call_args = mock_direct.call_args
        assert call_args[0][0] == "yolox-s"  # name
        assert call_args[0][1] == MODEL_REGISTRY["yolox-s"].onnx_url
        assert result == expected_path

    @patch("ivit.zoo.registry._export_yolox_onnx")
    @patch("ivit.zoo.registry._get_cache_dir")
    def test_download_from_source_skips_direct(self, mock_cache_dir, mock_export):
        """download(from_source=True) should skip direct download and use exporter."""
        tmp_dir = Path("/tmp/test_zoo_cache")
        mock_cache_dir.return_value = tmp_dir
        expected_path = tmp_dir / "yolox-s.onnx"
        mock_export.return_value = expected_path

        result = download("yolox-s", from_source=True)

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

        result = download("resnet18")

        mock_export.assert_called_once()
        assert result == expected_path

    def test_download_has_no_format_option(self):
        """Zoo models are ONNX only; ivit.load() converts them per backend."""
        import inspect

        assert "format" not in inspect.signature(download).parameters
        assert "format" not in inspect.signature(registry_load).parameters
        assert not hasattr(get_model_info("yolox-s"), "formats")

    @patch("ivit.zoo.registry._get_cache_dir")
    def test_download_cached_model_returns_immediately(self, mock_cache_dir, tmp_path):
        """A cached file is returned without downloading when no checksum is registered.

        Without a registered checksum the cache cannot be verified and is trusted; models
        with a checksum are re-verified (see test_zoo_checksum.py).
        """
        mock_cache_dir.return_value = tmp_path
        cached_file = tmp_path / "yolox-s.onnx"
        cached_file.write_bytes(b"fake onnx content")

        with patch("ivit.zoo.registry._expected_sha256", return_value=None):
            result = download("yolox-s")

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


def test_list_models_filters_by_zoo_task_and_rejects_unknown_names():
    from ivit.zoo import list_models

    assert "yolox-s" in list_models(task="detect")
    assert "resnet18" in list_models(task="classify") and "yolox-s" not in list_models(task="classify")
    assert "deeplabv3-resnet50" in list_models(task="segment")
    # "classification" is the LoadConfig spelling, not a zoo task: say so instead of returning [].
    with pytest.raises(ValueError, match="classify"):
        list_models(task="classification")
