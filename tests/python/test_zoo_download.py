"""Tests for Model Zoo ONNX direct download support."""

import hashlib
import os
import threading

import pytest
from pathlib import Path
from unittest.mock import patch

from ivit.zoo import registry
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
    def test_download_from_source_skips_direct(self, mock_cache_dir, mock_export, tmp_path):
        """download(from_source=True) should skip direct download and use exporter."""
        mock_cache_dir.return_value = tmp_path
        expected_path = tmp_path / "yolox-s.onnx"
        # exporters now write into a private staging directory and are handed that path
        mock_export.side_effect = lambda name, onnx_path, info, force: onnx_path.write_bytes(b"onnx")

        result = download("yolox-s", from_source=True)

        mock_export.assert_called_once()
        assert result == expected_path
        assert expected_path.read_bytes() == b"onnx"

    @patch("ivit.zoo.registry._export_torchvision_onnx")
    @patch("ivit.zoo.registry._get_cache_dir")
    def test_download_torchvision_goes_to_exporter(self, mock_cache_dir, mock_export, tmp_path):
        """Torchvision models (no onnx_url) should go to torchvision exporter."""
        mock_cache_dir.return_value = tmp_path
        expected_path = tmp_path / "resnet18.onnx"
        mock_export.side_effect = lambda name, onnx_path, info: onnx_path.write_bytes(b"onnx")

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


class TestAtomicCache:
    """Exports and downloads must never leave a half-written file that looks cached."""

    @pytest.fixture
    def cache(self, tmp_path):
        with patch("ivit.zoo.registry._get_cache_dir", return_value=tmp_path):
            yield tmp_path

    @pytest.mark.parametrize("failure", [RuntimeError("export failed"), KeyboardInterrupt()])
    def test_failed_or_interrupted_export_leaves_nothing_in_the_cache(self, cache, failure):
        def exporter(name, onnx_path, info):
            onnx_path.write_bytes(b"half written")  # what an interrupted torch.onnx.export leaves
            raise failure

        with patch("ivit.zoo.registry._export_torchvision_onnx", side_effect=exporter):
            with pytest.raises(type(failure)):
                download("resnet18")
        assert list(cache.iterdir()) == []  # no .onnx, no staging directory

        def good(name, onnx_path, info):
            onnx_path.write_bytes(b"complete")

        with patch("ivit.zoo.registry._export_torchvision_onnx", side_effect=good) as exporter:
            assert download("resnet18").read_bytes() == b"complete"  # re-exported, not "cached"
            assert exporter.call_count == 1

    def test_export_publishes_sidecars_before_the_onnx(self, cache):
        order = []
        real_replace = os.replace

        def spy(src, dst):
            order.append(Path(dst).name)
            real_replace(src, dst)

        def exporter(name, onnx_path, info):
            onnx_path.write_bytes(b"onnx")
            onnx_path.with_suffix(".labels.txt").write_text("a\nb\n")

        with patch("ivit.zoo.registry._export_torchvision_onnx", side_effect=exporter), patch(
            "ivit.zoo.registry.os.replace", side_effect=spy
        ):
            download("resnet18")
        assert order == ["resnet18.labels.txt", "resnet18.onnx"]  # a visible .onnx is complete

    def test_an_exporter_that_produces_nothing_is_an_error(self, cache):
        with patch("ivit.zoo.registry._export_torchvision_onnx"):
            with pytest.raises(RuntimeError, match="did not produce"):
                download("resnet18")

    def test_interrupted_weight_download_leaves_no_truncated_file(self, tmp_path):
        class Response:
            headers = {"content-length": "10"}
            def raise_for_status(self): pass
            def close(self): pass
            def iter_content(self, chunk_size):
                yield b"12345"
                raise ConnectionError("connection dropped")

        with patch("requests.get", return_value=Response()):
            with pytest.raises(ConnectionError):
                registry._download_file("https://example.invalid/w.pth", tmp_path / "w.pth")
        assert list(tmp_path.iterdir()) == []

    def test_concurrent_downloads_of_the_same_model_do_not_corrupt_each_other(self, tmp_path):
        payload = bytes(range(256)) * 20
        digest = hashlib.sha256(payload).hexdigest()
        barrier = threading.Barrier(2)

        def slow_fetch(source, partial, desc):
            with open(partial, "wb") as f:
                f.write(payload[:2500])
                barrier.wait(timeout=10)  # both downloads are mid-file at the same time
                f.write(payload[2500:])

        results, errors = [], []

        def run():
            try:
                results.append(registry._download_onnx_direct(
                    "m", "https://example.invalid/m.onnx", tmp_path / "m.onnx", force=True, sha256=digest))
            except BaseException as error:  # noqa: BLE001 - report every failure
                errors.append(error)

        with patch("ivit.zoo.registry._fetch_into", side_effect=slow_fetch), patch(
            "ivit.zoo.registry._mirror_source", return_value=None
        ):
            threads = [threading.Thread(target=run) for _ in range(2)]
            [t.start() for t in threads]
            [t.join() for t in threads]
        assert not errors, errors
        assert hashlib.sha256((tmp_path / "m.onnx").read_bytes()).hexdigest() == digest
        assert [p.name for p in tmp_path.iterdir()] == ["m.onnx"]  # no .part files left
