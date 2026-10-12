# SDK 文件索引

## 使用與整合

- [SDK 使用手冊](sdk-manual.zh-TW.md)：安裝與建置、架構圖、C++／Python／C API、模型契約及範例的唯一完整入口。
- [C# wrapper](../bindings/csharp/README.md)：.NET 專用載入與生命週期說明。
- [二進位套件安裝](../packaging/INSTALL.md)：已建置 tarball 的安裝方式。

## 開發與交付

- [Repository Guidelines](../AGENTS.md)：貢獻規範。
- [新增後端](hardware-extension.md)：C++ runtime 擴充位置。
- [持續整合](../.github/workflows/ci.yml)：每次 push／PR 執行；本機以 `scripts/ci/build_and_test.sh [--tensorrt]` 重現同一套建置與測試。
- [Release automation](deployment/release-automation.md)：平台 tarball 與 deb 流程；各平台仍須獨立驗收。

## 驗證紀錄

- [CHANGELOG](../CHANGELOG.md)：各版本的功能、行為變更與已知限制
- [OpenVINO GPU／NPU](development/openvino-device-verification.zh-TW.md)
- [TensorRT](development/tensorrt-device-verification.zh-TW.md)

驗證紀錄描述特定日期、版本與設備，使用方式以使用手冊及實際程式碼為準。
