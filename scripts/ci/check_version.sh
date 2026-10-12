#!/usr/bin/env bash
# Fail unless a release tag / version matches every version declared in the source tree.
#
#   scripts/ci/check_version.sh v1.1.0        # tag form
#   scripts/ci/check_version.sh 1.1.0-rc1     # pre-releases are allowed: the suffix is ignored
#
# Release file names come from CMakeLists.txt while the GitHub Release is named after the tag,
# so a mismatch would publish "1.1.1" notes next to files named 1.1.0.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <version-or-tag>" >&2
    exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

requested="${1#v}"
if ! [[ "$requested" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]]; then
    echo "error: '$1' is not a version like 1.2.3 or v1.2.3-rc1" >&2
    exit 2
fi
base="${requested%%-*}"

declare -A found=(
    ["CMakeLists.txt"]="$(sed -n 's/^project(ivit VERSION \([0-9][0-9.]*\).*/\1/p' "$ROOT/CMakeLists.txt" | head -1)"
    ["pyproject.toml"]="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
    ["python/ivit/__init__.py"]="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$ROOT/python/ivit/__init__.py" | head -1)"
    ["include/ivit/ivit.hpp"]="$(sed -n 's/^#define IVIT_VERSION_STRING "\(.*\)"/\1/p' "$ROOT/include/ivit/ivit.hpp" | head -1)"
)

status=0
echo "release version: $requested (source version must be $base)"
for file in "CMakeLists.txt" "pyproject.toml" "python/ivit/__init__.py" "include/ivit/ivit.hpp"; do
    value="${found[$file]}"
    if [ "$value" = "$base" ]; then
        printf '  ok        %-26s %s\n' "$file" "$value"
    else
        printf '  MISMATCH  %-26s %s\n' "$file" "${value:-<not found>}"
        status=1
    fi
done
if [ "$status" -ne 0 ]; then
    echo "error: bump the version in the files above (or fix the tag) before releasing" >&2
fi
exit "$status"
