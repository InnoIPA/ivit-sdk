# HailoRT 開發指令

## Build

```bash
# 基本 build（無 HailoRT）
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DIVIT_BUILD_EXAMPLES=ON \
    -DIVIT_BUILD_TESTS=OFF \
    -DIVIT_BUILD_PYTHON=ON
make -j$(nproc)

# 含 HailoRT build
cmake build -DIVIT_USE_HAILORT=ON
make -j$(nproc) -C build
```

---

## 安裝 / 更新 Python 套件

```bash
# 編譯完後同步到 site-packages（因為舊 .so 被 site-packages 優先載入）
cp build/lib/libivit.so \
   ~/anaconda3/envs/ivitsdk/lib/python3.10/site-packages/ivit/libivit.so

cp python/ivit/_ivit_core.cpython-310-x86_64-linux-gnu.so \
   ~/anaconda3/envs/ivitsdk/lib/python3.10/site-packages/ivit/_ivit_core.cpython-310-x86_64-linux-gnu.so
```

---

## 測試

```bash
# 跑所有 HailoRT 測試（含 slow/硬體測試）
conda run -n ivitsdk pytest tests/python/test_hailort_backend.py -v -m "slow or not slow"

# 只跑特定測試
conda run -n ivitsdk pytest tests/python/test_hailort_backend.py::TestHailoRTRuntimeAvailability::test_hailo_load_and_infer -v -s
```

---

## HailoRT 硬體診斷

```bash
# 確認硬體狀態
ls /dev/hailo*
lspci | grep -i hailo
hailortcli fw-control identify

# 解析 HEF 資訊（架構、input/output shape）
hailortcli parse-hef <model.hef>
```

---

## 注意事項

每次重新 build 後要記得手動 `cp` 兩個 `.so`（`libivit.so` 和 `_ivit_core.so`）到 site-packages，否則 Python 會繼續載入舊版本。
