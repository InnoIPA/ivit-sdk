"""Verify source export excludes artifacts and remains reproducible."""

import importlib.util
import json
import tarfile
from pathlib import Path

import pytest


def exporter():
    script = Path(__file__).resolve().parents[2] / "scripts/package_source.py"
    spec = importlib.util.spec_from_file_location("source_package", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_fixture(tmp_path):
    root = tmp_path / "source"
    for name, data in {
        "pyproject.toml": 'version = "1.0.0"\n',
        "README.md": "Source release\n",
        "python/ivit/__init__.py": "__version__ = '1.0.0'\n",
        "examples/data/images/bus.jpg": "sample asset",
        "examples/data/images/output.json": "generated result",
        "python/ivit/__pycache__/cached.pyc": "compiled",
        "python/ivit/_ivit_core.so": "compiled",
        "bindings/csharp/IvitSdk/obj/project.assets.json": "generated",
        ".venv/secret.txt": "local environment",
        ".env": "local secret",
        "build/cache.engine": "engine",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    return root


def test_source_export_excludes_generated_files(tmp_path):
    root = source_fixture(tmp_path)
    archive, manifest = exporter().package(root, tmp_path / "release")
    expected = {
        "README.md",
        "pyproject.toml",
        "python/ivit/__init__.py",
        "examples/data/images/bus.jpg",
    }
    records = json.loads(manifest.read_text())
    assert {row["path"] for row in records["files"]} == expected
    with tarfile.open(archive) as tar:
        assert {name.split("/", 1)[1] for name in tar.getnames()} == expected


def test_source_export_reproducible_and_no_overwrite(tmp_path):
    root = source_fixture(tmp_path)
    module = exporter()
    first, _ = module.package(root, tmp_path / "first")
    second, _ = module.package(root, tmp_path / "second")
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        module.package(root, tmp_path / "first")


def test_source_export_rejects_source_symlinks(tmp_path):
    root = source_fixture(tmp_path)
    (root / "python/ivit/leaked.txt").symlink_to(root / ".env")
    with pytest.raises(ValueError, match="symlinks require review"):
        exporter().package(root, tmp_path / "release")


def test_source_export_rejects_output_inside_source(tmp_path):
    root = source_fixture(tmp_path)
    with pytest.raises(ValueError, match="outside the source tree"):
        exporter().package(root, root / "docs/release")
