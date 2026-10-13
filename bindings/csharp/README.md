# C# integration

Requires .NET 8 and a built 64-bit native iVIT SDK. The wrapper uses the C ABI;
Python is not required. Linux is validated; Windows native builds remain pending.

```bash
dotnet build examples/csharp/Quickstart/Quickstart.csproj
export LD_LIBRARY_PATH=/path/to/sdk/lib:/path/to/openvino/libs
dotnet run --project examples/csharp/Quickstart -- classifier.onnx cpu
```

Reference `bindings/csharp/IvitSdk/IvitSdk.csproj` in your application:

```csharp
using IvitSdk;
using var model = new Model("classifier.onnx", device: "cpu", backend: "openvino");
// BGR uint8 HWC; use the actual image width, height and row stride.
using var result = model.Predict(bgrBytes, width, height, stride, confidence: 0);
Console.WriteLine(result.ToJson());
var (maskWidth, maskHeight, classes) = result.ClassMask();
```

`confidence` defaults to -1, the task default: 0.5 for detection and no filtering for
classification (every class is returned). Pass 0–1 for a specific threshold.

Use `ModelOptions` for decoder, labels file, layout, color, scale and mean/std.
Defaults are RGB, NCHW, pixel/255 and no additional normalization. Use the
normalization and label order from the model's training/export pipeline.

`Model` and `Prediction` own SafeHandles and implement `IDisposable`. Results
remain valid after model disposal. Native status failures become `IvitException`;
its `Status` matches `include/ivit/c_api.h`. This interface is synchronous.

## Version and logging

```csharp
Console.WriteLine(Ivit.Version);
Ivit.SetLogLevel("warning");
Ivit.SetLogHandler((level, message) => logger.Log(level.ToString(), message));
```

`IvitException.Category` separates model-load, unsupported-format, device-not-found and
inference failures; `Model.InfoJson()` returns the input/output shapes, labels and image contract.

By default the SDK writes to stderr only. The handler may run on any SDK thread
and must be thread-safe; pass `null` to restore stderr.
