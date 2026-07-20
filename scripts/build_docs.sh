#!/usr/bin/env bash
#
# 建置 iVIT-SDK 文件站台（MkDocs Material + Python API + Doxygen C++ sidecar）。
#
# 用法：
#   bash scripts/build_docs.sh          # 靜態建置 -> ./site/
#   bash scripts/build_docs.sh serve    # 本機預覽 http://127.0.0.1:8000
#
# 前置需求（在 ivitsdk conda env 內）：
#   pip install -e '.[docs]'
#   sudo apt install -y doxygen     # 或 conda install -c conda-forge doxygen
#
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) 確認 ivit 套件可被 import（mkdocstrings 需要能 import 才能擷取 docstring）。
if ! python -c "import ivit" 2>/dev/null; then
  echo "✗ 無法 import ivit。請先在 ivitsdk env 執行： pip install -e '.[docs]'" >&2
  exit 1
fi
if ! python -c "import ivit; assert ivit.is_cpp_available()" 2>/dev/null; then
  echo "⚠ ivit C++ 綁定不可用（is_cpp_available()=False）；Python 文件仍可產生，" >&2
  echo "  但 C++ 綁定型別的說明會較精簡。" >&2
fi

# 2) 確認文件工具已安裝。
command -v mkdocs >/dev/null || { echo "✗ 找不到 mkdocs。請執行： pip install -e '.[docs]'" >&2; exit 1; }
command -v doxygen >/dev/null || {
  echo "✗ 找不到 doxygen。請安裝： sudo apt install doxygen（或 conda install -c conda-forge doxygen）" >&2
  exit 1
}

# 3) 先產生 C++ sidecar（HTML 必須先存在於 docs/ 內，才會被 mkdocs 複製進 site/）。
echo "==> [1/2] Doxygen：產生 C++ API -> docs/api/cpp/html/"
doxygen Doxyfile

# 4) 建置或預覽 MkDocs 站台。
if [[ "${1:-build}" == "serve" ]]; then
  echo "==> [2/2] mkdocs serve （Ctrl-C 結束）"
  exec mkdocs serve
else
  # 註：不用 --strict。部分既有手寫文件（hardware-extension.md、
  # hailort-detection-flow.md）含指向 ../src/*.cpp 的 GitHub 相對連結，
  # MkDocs 視為 broken link；那是既有內容，與 API reference 無關。
  echo "==> [2/2] mkdocs build -> ./site/"
  mkdocs build
  echo "✓ 完成。開啟 site/index.html 檢視。"
fi
