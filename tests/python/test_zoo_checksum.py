"""Model Zoo integrity checking for pre-converted ONNX downloads."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from ivit.zoo import registry
from ivit.zoo.registry import _MIN_ONNX_SIZE, _download_onnx_direct, download

PAYLOAD = b"x" * (_MIN_ONNX_SIZE + 10)
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://example.invalid/model.onnx"


@pytest.fixture
def fetch(monkeypatch):
    """Replace the network with a fake that writes `content` and counts calls."""
    state = {"calls": 0, "content": PAYLOAD}

    def fake(url, dest, desc=None):
        state["calls"] += 1
        Path(dest).write_bytes(state["content"])
        return dest

    monkeypatch.setattr(registry, "_download_file", fake)
    monkeypatch.delenv("IVIT_ZOO_REQUIRE_CHECKSUM", raising=False)
    return state


def test_matching_checksum_installs_file_and_leaves_no_partial(tmp_path, fetch):
    target = tmp_path / "m.onnx"
    assert _download_onnx_direct("m", URL, target, sha256=DIGEST) == target
    assert target.read_bytes() == PAYLOAD
    assert not (tmp_path / "m.onnx.part").exists()


def test_mismatch_discards_download_and_never_creates_target(tmp_path, fetch):
    fetch["content"] = b"y" * (_MIN_ONNX_SIZE + 10)
    target = tmp_path / "m.onnx"
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        _download_onnx_direct("m", URL, target, sha256=DIGEST)
    assert not target.exists()
    assert not (tmp_path / "m.onnx.part").exists()


def test_failed_download_keeps_previous_good_file_intact(tmp_path, fetch):
    target = tmp_path / "m.onnx"
    target.write_bytes(PAYLOAD)
    fetch["content"] = b"tiny"
    with pytest.raises(RuntimeError, match="too small"):
        _download_onnx_direct("m", URL, target, force=True, sha256=DIGEST)
    assert target.read_bytes() == PAYLOAD
    assert not (tmp_path / "m.onnx.part").exists()


def test_verified_cache_is_not_downloaded_again(tmp_path, fetch):
    target = tmp_path / "m.onnx"
    target.write_bytes(PAYLOAD)
    assert _download_onnx_direct("m", URL, target, sha256=DIGEST) == target
    assert fetch["calls"] == 0


def test_tampered_cache_is_replaced(tmp_path, fetch):
    target = tmp_path / "m.onnx"
    target.write_bytes(b"z" * (_MIN_ONNX_SIZE + 10))
    _download_onnx_direct("m", URL, target, sha256=DIGEST)
    assert fetch["calls"] == 1
    assert target.read_bytes() == PAYLOAD


def test_unchecksummed_download_still_works_by_default(tmp_path, fetch):
    target = tmp_path / "unregistered.onnx"
    assert _download_onnx_direct("unregistered", URL, target) == target


def test_require_checksum_refuses_before_any_network_access(tmp_path, fetch, monkeypatch):
    monkeypatch.setenv("IVIT_ZOO_REQUIRE_CHECKSUM", "1")
    with pytest.raises(RuntimeError, match="IVIT_ZOO_REQUIRE_CHECKSUM"):
        _download_onnx_direct("unregistered", URL, tmp_path / "m.onnx")
    assert fetch["calls"] == 0


def test_require_checksum_allows_download_with_explicit_digest(tmp_path, fetch, monkeypatch):
    monkeypatch.setenv("IVIT_ZOO_REQUIRE_CHECKSUM", "1")
    _download_onnx_direct("unregistered", URL, tmp_path / "m.onnx", sha256=DIGEST.upper())
    assert (tmp_path / "m.onnx").exists()


def test_malformed_digest_is_rejected(tmp_path, fetch):
    with pytest.raises(ValueError, match="64 hex"):
        _download_onnx_direct("m", URL, tmp_path / "m.onnx", sha256="abc")
    assert fetch["calls"] == 0


def test_registry_and_checksums_file_are_consulted(tmp_path, fetch, monkeypatch):
    monkeypatch.setattr(registry, "_load_checksums", lambda: {"m": DIGEST})
    assert registry._expected_sha256("m") == DIGEST
    other = hashlib.sha256(b"other").hexdigest()
    assert registry._expected_sha256("m", other) == other


def test_download_rejects_sha256_for_non_direct_routes(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "_get_cache_dir", lambda: tmp_path)
    with pytest.raises(ValueError, match="sha256 only applies"):
        download("resnet18", sha256=DIGEST)


def test_download_passes_sha256_to_direct_route(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "_get_cache_dir", lambda: tmp_path)
    seen = {}

    def fake(name, url, path, force=False, sha256=None):
        seen["sha256"] = sha256
        return path

    monkeypatch.setattr(registry, "_download_onnx_direct", fake)
    download("yolox-s", sha256=DIGEST)
    assert seen["sha256"] == DIGEST


def test_shipped_checksums_file_is_well_formed_and_matches_registry():
    data = json.loads(registry._CHECKSUMS_PATH.read_text(encoding="utf-8"))
    assert data["schema"] == 1
    for name, digest in data["sha256"].items():
        assert name in registry.MODEL_REGISTRY, name
        assert registry.MODEL_REGISTRY[name].onnx_url, name
        assert registry._SHA256_RE.match(digest), name


def test_update_script_records_digests_from_a_directory(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/update_zoo_checksums.py"
    spec = importlib.util.spec_from_file_location("update_zoo_checksums", script)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    models = tmp_path / "models"
    models.mkdir()
    (models / "yolox-s.onnx").write_bytes(PAYLOAD)
    output = tmp_path / "checksums.json"
    assert tool.main(["--from-dir", str(models), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["sha256"] == {"yolox-s": DIGEST}


def test_inline_external_data_makes_export_self_contained(tmp_path):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.arange(8, dtype=np.float32).reshape(2, 4), "w")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "m.onnx"
    onnx.save_model(
        model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
        location="m.onnx.data", size_threshold=0,
    )
    assert (tmp_path / "m.onnx.data").exists()

    assert registry._inline_external_data(path) == path
    assert not (tmp_path / "m.onnx.data").exists()
    merged = onnx.load(str(path), load_external_data=False)
    assert all(t.data_location != onnx.TensorProto.EXTERNAL for t in merged.graph.initializer)
    assert np.array_equal(numpy_helper.to_array(merged.graph.initializer[0]), np.arange(8).reshape(2, 4))


def test_inline_external_data_is_a_noop_without_sidecar(tmp_path):
    path = tmp_path / "m.onnx"
    path.write_bytes(b"graph")
    assert registry._inline_external_data(path) == path
    assert path.read_bytes() == b"graph"


# ---------------------------------------------------------------- mirror support


def test_mirror_source_url_and_directory_forms(monkeypatch):
    url = "yolox-s.onnx"
    monkeypatch.delenv("IVIT_ZOO_MIRROR", raising=False)
    assert registry._mirror_source(url) is None
    monkeypatch.setenv("IVIT_ZOO_MIRROR", "https://mirror.corp/zoo/")
    assert registry._mirror_source(url) == "https://mirror.corp/zoo/yolox-s.onnx"
    monkeypatch.setenv("IVIT_ZOO_MIRROR", "file:///mnt/models")
    assert registry._mirror_source(url) == "/mnt/models/yolox-s.onnx"
    monkeypatch.setenv("IVIT_ZOO_MIRROR", "/mnt/models/")
    assert registry._mirror_source(url) == "/mnt/models/yolox-s.onnx"


def test_local_directory_mirror_serves_verified_file_without_network(tmp_path, fetch, monkeypatch):
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "model.onnx").write_bytes(PAYLOAD)
    monkeypatch.setenv("IVIT_ZOO_MIRROR", str(mirror))
    target = tmp_path / "cache" / "model.onnx"
    _download_onnx_direct("model", URL, target, sha256=DIGEST)
    assert target.read_bytes() == PAYLOAD
    assert fetch["calls"] == 0


def test_missing_mirror_file_falls_back_to_origin(tmp_path, fetch, monkeypatch):
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    monkeypatch.setenv("IVIT_ZOO_MIRROR", str(mirror))
    target = tmp_path / "model.onnx"
    _download_onnx_direct("model", URL, target, sha256=DIGEST)
    assert fetch["calls"] == 1
    assert target.read_bytes() == PAYLOAD


def test_unusable_mirror_file_falls_back_to_origin(tmp_path, fetch, monkeypatch):
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "model.onnx").write_bytes(b"<html>404</html>")
    monkeypatch.setenv("IVIT_ZOO_MIRROR", str(mirror))
    target = tmp_path / "model.onnx"
    _download_onnx_direct("model", URL, target, sha256=DIGEST)
    assert fetch["calls"] == 1
    assert target.read_bytes() == PAYLOAD


def test_tampered_mirror_is_an_error_not_a_silent_fallback(tmp_path, fetch, monkeypatch):
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "model.onnx").write_bytes(b"y" * (_MIN_ONNX_SIZE + 10))
    monkeypatch.setenv("IVIT_ZOO_MIRROR", str(mirror))
    target = tmp_path / "model.onnx"
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        _download_onnx_direct("model", URL, target, sha256=DIGEST)
    assert fetch["calls"] == 0
    assert not target.exists()
    assert not (tmp_path / "model.onnx.part").exists()


def test_http_mirror_is_tried_before_origin(tmp_path, monkeypatch):
    seen = []

    def fake(url, dest, desc=None):
        seen.append(url)
        if "mirror.corp" in url:
            raise OSError("connection refused")
        Path(dest).write_bytes(PAYLOAD)
        return dest

    monkeypatch.setattr(registry, "_download_file", fake)
    monkeypatch.setenv("IVIT_ZOO_MIRROR", "https://mirror.corp/zoo")
    _download_onnx_direct("model", URL, tmp_path / "model.onnx", sha256=DIGEST)
    assert seen == ["https://mirror.corp/zoo/model.onnx", URL]


# ---------------------------------------------------------------- Google Drive links

DRIVE_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"
DRIVE_DIRECT = (
    f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t"
)


@pytest.mark.parametrize("link", [
    f"https://drive.google.com/file/d/{DRIVE_ID}/view?usp=sharing",
    f"https://drive.google.com/file/d/{DRIVE_ID}/view?usp=drive_link",
    f"https://drive.google.com/open?id={DRIVE_ID}",
    f"https://drive.google.com/uc?export=download&id={DRIVE_ID}",
    DRIVE_DIRECT,
])
def test_google_drive_share_links_become_direct_downloads(link):
    assert registry._direct_download_url(link) == DRIVE_DIRECT


@pytest.mark.parametrize("url", [
    URL,
    "https://github.com/org/repo/releases/download/v1/yolox-s.onnx",
    "https://drive.google.com/drive/folders/",       # no file id
    f"http://drive.google.com/file/d/{DRIVE_ID}/view",  # not https
    f"https://drive.google.com.evil.example/file/d/{DRIVE_ID}/view",
])
def test_other_urls_are_left_alone(url):
    assert registry._direct_download_url(url) == url


class FakeResponse:
    def __init__(self, content_type, body):
        self.headers = {"content-type": content_type, "content-length": str(len(body))}
        self._body = body
        self.closed = False

    status_code = 200

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} Client Error")

    def iter_content(self, chunk_size):
        yield self._body

    def close(self):
        self.closed = True


def test_drive_web_page_instead_of_file_is_a_clear_error(tmp_path, monkeypatch):
    requests = pytest.importorskip("requests")
    pytest.importorskip("tqdm")
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        return FakeResponse("text/html; charset=utf-8", b"<html>quota exceeded</html>")

    monkeypatch.setattr(requests, "get", fake_get)
    dest = tmp_path / "m.onnx"
    with pytest.raises(RuntimeError, match="Anyone with the link"):
        registry._download_file(f"https://drive.google.com/file/d/{DRIVE_ID}/view", dest)
    assert seen == [DRIVE_DIRECT]
    assert not dest.exists()


def test_drive_not_found_explains_the_sharing_setting(tmp_path, monkeypatch):
    requests = pytest.importorskip("requests")
    pytest.importorskip("tqdm")
    response = FakeResponse("text/html", b"")
    response.status_code = 404
    monkeypatch.setattr(requests, "get", lambda url, **kw: response)
    with pytest.raises(RuntimeError, match="404.*Anyone with the link"):
        registry._download_file(f"https://drive.google.com/file/d/{DRIVE_ID}/view", tmp_path / "m")


def test_drive_file_download_writes_the_bytes(tmp_path, monkeypatch):
    requests = pytest.importorskip("requests")
    pytest.importorskip("tqdm")
    monkeypatch.setattr(
        requests, "get", lambda url, **kw: FakeResponse("application/octet-stream", PAYLOAD)
    )
    dest = tmp_path / "m.onnx"
    registry._download_file(f"https://drive.google.com/file/d/{DRIVE_ID}/view", dest)
    assert dest.read_bytes() == PAYLOAD


def test_mirror_names_files_after_the_model_for_drive_links(tmp_path, fetch, monkeypatch):
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "model.onnx").write_bytes(PAYLOAD)
    monkeypatch.setenv("IVIT_ZOO_MIRROR", str(mirror))
    target = tmp_path / "cache" / "model.onnx"
    drive_link = f"https://drive.google.com/file/d/{DRIVE_ID}/view?usp=sharing"
    _download_onnx_direct("model", drive_link, target, sha256=DIGEST)
    assert target.read_bytes() == PAYLOAD
    assert fetch["calls"] == 0


def test_shipped_yolox_checksums_reject_a_forged_cache(tmp_path, monkeypatch):
    """A fake file named like a registered model must not be returned as-is."""
    assert registry._expected_sha256("yolox-s") is not None
    monkeypatch.setattr(registry, "_get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(registry, "_download_file", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    (tmp_path / "yolox-s.onnx").write_bytes(b"fake onnx content")
    with pytest.raises(OSError):
        download("yolox-s")


def test_all_hosted_yolox_models_have_checksums():
    for name in ("yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l"):
        assert registry._expected_sha256(name), name


def test_the_dead_exporters_are_gone():
    assert not hasattr(registry, "_export_rtdetr_onnx")
    assert not hasattr(registry, "_export_mmpose_onnx")


# ------------------------------------------------------------------ zoo.load applies the contract


def capture_ivit_load(monkeypatch, tmp_path):
    import sys
    import types

    calls = []
    fake = types.ModuleType("ivit")
    fake.load = lambda path, device="auto", **options: calls.append((path, device, options)) or "model"
    monkeypatch.setitem(sys.modules, "ivit", fake)
    monkeypatch.setattr(registry, "download", lambda name: tmp_path / f"{name}.onnx")
    return calls


def test_zoo_load_applies_the_verified_contract(monkeypatch, tmp_path):
    calls = capture_ivit_load(monkeypatch, tmp_path)
    registry.load("yolox-s", device="cpu")
    path, device, options = calls[-1]
    assert path.endswith("yolox-s.onnx") and device == "cpu"
    assert options["task"] == "detection" and options["decoder"] == "yolox_decoded"
    assert options["input_color"] == "bgr" and options["input_scale"] == 1.0
    assert options["labels"] == registry.COCO_LABELS and len(options["labels"]) == 80

    registry.load("resnet18")
    options = calls[-1][2]
    assert options["task"] == "classification"
    assert options["mean"] == [0.485, 0.456, 0.406] and options["std"] == [0.229, 0.224, 0.225]


def test_zoo_load_lets_keyword_arguments_override_the_contract(monkeypatch, tmp_path):
    calls = capture_ivit_load(monkeypatch, tmp_path)
    registry.load("yolox-s", backend="tensorrt", labels=["only"], use_cache=False)
    options = calls[-1][2]
    assert options["labels"] == ["only"] and options["backend"] == "tensorrt"
    assert options["decoder"] == "yolox_decoded"  # the rest of the contract stays


def test_zoo_load_refuses_models_without_an_sdk_decoder(monkeypatch, tmp_path):
    calls = capture_ivit_load(monkeypatch, tmp_path)
    for name in ("fasterrcnn-resnet50", "retinanet-resnet50"):
        with pytest.raises(ValueError, match="no SDK decoder"):
            registry.load(name)
    assert calls == []  # nothing was loaded or downloaded
    registry.load("fasterrcnn-resnet50", task="")  # explicit opt-in: loads without a decoder
    assert calls[-1][2] == {"task": ""}


def test_ready_models_are_exactly_the_verified_families():
    ready = {n for n, i in registry.MODEL_REGISTRY.items() if i.ready}
    assert ready == {
        "yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l",
        "resnet18", "resnet50", "mobilenetv3", "efficientnet-b0",
        "deeplabv3-resnet50", "deeplabv3-mobilenetv3",
    }
    for name, info in registry.MODEL_REGISTRY.items():
        task = info.load_options.get("task")
        if info.ready:  # the contract matches the registry's task
            assert {"detect": "detection", "classify": "classification",
                    "segment": "segmentation"}[info.task] == task, name


def test_download_writes_a_labels_sidecar_for_models_that_have_labels(tmp_path, fetch, monkeypatch):
    monkeypatch.setattr(registry, "_get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(registry, "_expected_sha256", lambda name, explicit=None: None)
    path = download("yolox-s")
    sidecar = path.with_suffix(".labels.txt")
    assert sidecar.read_text(encoding="utf-8").splitlines() == registry.COCO_LABELS
    sidecar.write_text("custom\n", encoding="utf-8")  # a user's own sidecar is never overwritten
    download("yolox-s")
    assert sidecar.read_text(encoding="utf-8") == "custom\n"
