#!/usr/bin/env python3
"""Export a source release without local environments or generated assets."""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT_FILES = {
    "AGENTS.md",
    "README.md",
    "CHANGELOG.md",
    "CMakeLists.txt",
    "pyproject.toml",
    ".gitignore",
    "LICENSE",
    "NOTICE",
}
ROOT_DIRS = {
    "include",
    "src",
    "python",
    "bindings",
    "cmake",
    "scripts",
    "packaging",
    "examples",
    "tests",
    "docs",
    ".github",
}
SKIP_DIRS = {
    "__pycache__",
    "bin",
    "obj",
    "node_modules",
    ".pytest_cache",
    ".venv",
    "venv",
    "model_cache",
}
SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".pyd",
    ".engine",
    ".trt",
    ".plan",
    ".blob",
    ".log",
    ".onnx",
    ".pt",
    ".pth",
    ".hef",
}


def include_source(path):
    parts = path.parts
    if len(parts) == 1:
        return path.name in ROOT_FILES
    if parts[0] not in ROOT_DIRS:
        return False
    if any(
        p in SKIP_DIRS or p.endswith(".egg-info") or p.startswith(".env") for p in parts
    ):
        return False
    if path.suffix in SKIP_SUFFIXES or path.name == ".gitkeep":
        return False
    if parts[:2] == ("examples", "data"):
        if path.stem == "output":
            return False
        if path.stem.endswith(("_output", "_result")):
            return False
    return True


def package(root, output):
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Release output must be outside the source tree")
    # Source-only allowlist; do not traverse local venvs, builds or dependencies.
    candidates = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in sorted(ROOT_DIRS):
        base = root / name
        if base.is_symlink():
            raise ValueError(f"Source directory must not be a symlink: {base}")
        if base.is_dir():
            candidates.extend(base.rglob("*"))
    sources = []
    for file in sorted(candidates):
        relative = file.relative_to(root)
        if not include_source(relative):
            continue
        if file.is_symlink():
            raise ValueError(f"Source symlinks require review: {relative}")
        if file.is_file():
            sources.append(
                (
                    relative.as_posix(),
                    file.read_bytes(),
                    0o755 if file.stat().st_mode & 0o111 else 0o644,
                )
            )
    # Parse the declared package version without tomllib (added in Python 3.11).
    import re

    version = re.search(
        r'^version = "([0-9A-Za-z.+-]+)"$',
        (root / "pyproject.toml").read_text(),
        re.MULTILINE,
    )
    if not version:
        raise ValueError("Cannot determine source version")
    name = f"ivit-sdk-{version.group(1)}-source"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{name}.tar.gz"
    manifest = output / f"{name}.manifest.json"
    if archive.exists() or manifest.exists():
        raise FileExistsError("Release output already exists; choose a new directory")
    # Stable ordering, timestamps and ownership produce reproducible archives.
    with archive.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as tar:
                for relative, data, mode in sources:
                    info = tarfile.TarInfo(f"{name}/{relative}")
                    info.size, info.mode, info.mtime = len(data), mode, 0
                    tar.addfile(info, io.BytesIO(data))
    records = [
        {
            "path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for relative, data, _ in sources
    ]
    manifest.write_text(
        json.dumps(
            {
                "archive": archive.name,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "files": records,
            },
            indent=2,
        )
        + "\n"
    )
    return archive, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive, manifest = package(
        Path(__file__).resolve().parents[1], args.output.resolve()
    )
    print(archive)
    print(manifest)


if __name__ == "__main__":
    main()
