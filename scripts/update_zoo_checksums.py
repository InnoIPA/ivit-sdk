#!/usr/bin/env python3
"""Maintain python/ivit/zoo/checksums.json (SHA256 of pre-converted Zoo ONNX files).

Record checksums from a trusted build, not from whatever the download URL serves
today: hashing a download only proves it matches itself.

  # record from locally built/trusted files named <model>.onnx
  update_zoo_checksums.py --from-dir /path/to/trusted/models

  # verify the recorded checksums against the download URLs (exit 1 on mismatch)
  update_zoo_checksums.py --check

  # record from the URLs (trust-on-first-use; review the result before committing)
  update_zoo_checksums.py --download --models yolox-s yolox-m
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "python/ivit/zoo/registry.py"
DEFAULT_OUTPUT = ROOT / "python/ivit/zoo/checksums.json"


def load_registry():
    spec = importlib.util.spec_from_file_location("ivit_zoo_registry", REGISTRY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_of_url(url):
    import requests

    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("sha256", {}))


def write_checksums(path, entries):
    payload = {"schema": 1, "sha256": dict(sorted(entries.items()))}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-dir", type=Path, help="hash <dir>/<model>.onnx files")
    mode.add_argument("--download", action="store_true", help="hash the files served at onnx_url")
    mode.add_argument("--check", action="store_true", help="verify recorded checksums against onnx_url")
    parser.add_argument("--models", nargs="*", help="limit to these model names")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    registry = load_registry()
    candidates = {
        name: info for name, info in registry.MODEL_REGISTRY.items() if info.onnx_url
    }
    if args.models:
        unknown = sorted(set(args.models) - set(candidates))
        if unknown:
            parser.error(f"not pre-converted ONNX models: {', '.join(unknown)}")
        candidates = {name: candidates[name] for name in args.models}

    entries = read_checksums(args.output)
    failures = 0
    for name, info in candidates.items():
        try:
            if args.from_dir:
                path = args.from_dir / f"{name}.onnx"
                if not path.is_file():
                    print(f"SKIP {name}: {path} not found")
                    continue
                digest = registry._sha256_file(path)
            else:
                digest = sha256_of_url(info.onnx_url)
        except Exception as error:  # network or I/O failure for one model
            print(f"FAIL {name}: {error}")
            failures += 1
            continue
        if args.check:
            recorded = entries.get(name)
            if recorded is None:
                print(f"MISSING {name}: no recorded checksum")
                failures += 1
            elif recorded != digest:
                print(f"MISMATCH {name}: recorded {recorded}, served {digest}")
                failures += 1
            else:
                print(f"OK {name}")
        else:
            entries[name] = digest
            print(f"RECORD {name} {digest}")

    if not args.check:
        write_checksums(args.output, entries)
        print(f"Wrote {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
