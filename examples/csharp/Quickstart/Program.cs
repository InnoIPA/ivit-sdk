using System.Text.Json;
using IvitSdk;

if (args.Length < 1) throw new ArgumentException("Usage: Quickstart classifier.onnx [device]");
Console.WriteLine(Model.DevicesJson());
using var model = new Model(args[0], device: args.Length > 1 ? args[1] : "cpu");
using var prediction = model.Predict(new byte[32 * 32 * 3], 32, 32, confidence: 0);
string json = prediction.ToJson();
using var parsed = JsonDocument.Parse(json);
if (parsed.RootElement.GetProperty("classifications").GetArrayLength() == 0)
    throw new Exception("Expected classifications");
Console.WriteLine(json);
if (prediction.ClassMask().Values.Length != 0) throw new Exception("Unexpected class mask");
try {
    model.Predict(new byte[1], 32, 32);
    throw new Exception("Short buffer was accepted");
} catch (IvitException error) when (error.Status == 1) { }
model.Dispose();
// Results retain their own ownership after the model is disposed.
JsonDocument.Parse(prediction.ToJson()).Dispose();
try {
    model.Predict(new byte[32 * 32 * 3], 32, 32);
    throw new Exception("Disposed model was accepted");
} catch (ObjectDisposedException) { }
Console.WriteLine("C# lifecycle/buffer checks passed");
