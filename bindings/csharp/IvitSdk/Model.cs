using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace IvitSdk;

public sealed class ModelOptions
{
    public string Decoder { get; init; } = "auto";
    public string InputColor { get; init; } = "rgb";
    public string InputLayout { get; init; } = "nchw";
    public string ResizeMode { get; init; } = "auto";
    public string LabelsPath { get; init; } = "";
    public string OutputName { get; init; } = "";
    public float InputScale { get; init; } = 1.0f / 255.0f;
    public float[] Mean { get; init; } = [0, 0, 0];
    public float[] Std { get; init; } = [1, 1, 1];
    public int BatchSize { get; init; } = 1;
}

[StructLayout(LayoutKind.Sequential)]
internal struct LoadOptions
{
    internal uint Size, Version;
    [MarshalAs(UnmanagedType.LPUTF8Str)] internal string Device;
    [MarshalAs(UnmanagedType.LPUTF8Str)] internal string Backend;
    [MarshalAs(UnmanagedType.LPUTF8Str)] internal string Task;
    [MarshalAs(UnmanagedType.LPUTF8Str)] internal string Precision;
    [MarshalAs(UnmanagedType.LPUTF8Str)] internal string Decoder;
    [MarshalAs(UnmanagedType.LPUTF8Str)] internal string Color;
    [MarshalAs(UnmanagedType.LPUTF8Str)] internal string Layout;
    [MarshalAs(UnmanagedType.LPUTF8Str)] internal string Resize;
    [MarshalAs(UnmanagedType.LPUTF8Str)] internal string LabelsPath;
    [MarshalAs(UnmanagedType.LPUTF8Str)] internal string OutputName;
    internal float Scale;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)] internal float[] Mean;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)] internal float[] Std;
    internal int Cache, Batch;
}

/// <summary>Why a native call failed, finer than <see cref="IvitException.Status"/>.</summary>
public enum IvitErrorCategory
{
    None = 0, Generic = 1, ModelLoad = 2, UnsupportedFormat = 3, DeviceNotFound = 4, Inference = 5,
}

public sealed class IvitException(int status, string message,
    IvitErrorCategory category = IvitErrorCategory.None) : Exception(message)
{
    public int Status { get; } = status;
    public IvitErrorCategory Category { get; } = category;
}

internal sealed class ModelHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    internal ModelHandle(ulong value) : base(true) => SetHandle(new IntPtr(checked((long)value)));
    protected override bool ReleaseHandle() => Native.ModelRelease((ulong)handle.ToInt64()) == 0;
}

internal sealed class ResultHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    internal ResultHandle(ulong value) : base(true) => SetHandle(new IntPtr(checked((long)value)));
    protected override bool ReleaseHandle() => Native.ResultRelease((ulong)handle.ToInt64()) == 0;
}

internal static class Native
{
    private const string Library = "ivit";
    [DllImport(Library, EntryPoint = "ivit_model_load_ex", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int ModelLoadEx([MarshalAs(UnmanagedType.LPUTF8Str)] string path,
        ref LoadOptions options, out ulong handle);
    [DllImport(Library, EntryPoint = "ivit_abi_version", CallingConvention = CallingConvention.Cdecl)]
    internal static extern uint AbiVersion();
    [DllImport(Library, EntryPoint = "ivit_last_error_category", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int LastErrorCategory();
    [DllImport(Library, EntryPoint = "ivit_model_info_json", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int ModelInfo(ModelHandle model, [Out] byte[]? buffer, ulong capacity,
        out ulong required);
    [DllImport(Library, EntryPoint = "ivit_version_string", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr VersionString();
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void LogNative(int level, IntPtr message, IntPtr userData);
    [DllImport(Library, EntryPoint = "ivit_set_log_callback", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int SetLogCallback(LogNative? callback, IntPtr userData);
    [DllImport(Library, EntryPoint = "ivit_set_log_level", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int SetLogLevel([MarshalAs(UnmanagedType.LPUTF8Str)] string level);
    [DllImport(Library, EntryPoint = "ivit_model_load", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int ModelLoad(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string device,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string backend,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string task,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string precision, int cache, out ulong handle);
    [DllImport(Library, EntryPoint = "ivit_model_release", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int ModelRelease(ulong handle);
    [DllImport(Library, EntryPoint = "ivit_result_release", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int ResultRelease(ulong handle);
    [DllImport(Library, EntryPoint = "ivit_model_predict_bgr", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Predict(ModelHandle model, byte[] pixels, ulong bytes,
        int width, int height, ulong stride, float confidence, float iou, int maxDetections,
        out ulong result);
    [DllImport(Library, EntryPoint = "ivit_result_json", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int ResultJson(ResultHandle result, [Out] byte[]? buffer,
        ulong capacity, out ulong required);
    [DllImport(Library, EntryPoint = "ivit_result_mask_i32", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Mask(ResultHandle result, [Out] int[]? buffer, ulong capacity,
        out int width, out int height, out ulong required);
    [DllImport(Library, EntryPoint = "ivit_last_error", CallingConvention = CallingConvention.Cdecl)]
    private static extern int LastError([Out] byte[]? buffer, ulong capacity, out ulong required);
    [DllImport(Library, EntryPoint = "ivit_devices_json", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Devices([Out] byte[]? buffer, ulong capacity, out ulong required);

    internal delegate int Reader(byte[]? buffer, ulong capacity, out ulong required);
    internal static string Read(Reader reader)
    {
        int status = reader(null, 0, out ulong required);
        if (status != 3) Check(status);
        byte[] bytes = new byte[checked((int)required)];
        Check(reader(bytes, required, out ulong copied));
        return Encoding.UTF8.GetString(bytes, 0, checked((int)copied - 1));
    }
    internal static void Check(int status)
    {
        if (status == 0) return;
        LastError(null, 0, out ulong required);
        byte[] bytes = new byte[checked((int)required)];
        LastError(bytes, required, out _);
        throw new IvitException(status, Encoding.UTF8.GetString(bytes, 0, bytes.Length - 1),
            (IvitErrorCategory)LastErrorCategory());
    }
}

/// <summary>Synchronous inference through the native SDK. Input is uint8 BGR HWC.</summary>
public sealed class Model : IDisposable
{
    private readonly ModelHandle handle;
    public Model(string path, string device = "cpu", string backend = "auto",
        string task = "classification", string precision = "fp32", bool useCache = false,
        ModelOptions? options = null)
    {
        if (IntPtr.Size != 8) throw new PlatformNotSupportedException("64-bit process required");
        if (Native.AbiVersion() != 1) throw new NotSupportedException("Unsupported iVIT C ABI");
        options ??= new ModelOptions();
        if (options.Mean.Length != 3 || options.Std.Length != 3)
            throw new ArgumentException("Mean/Std must contain three values");
        var native = new LoadOptions {
            Size = (uint)Marshal.SizeOf<LoadOptions>(), Version = 1,
            Device = device, Backend = backend, Task = task, Precision = precision,
            Decoder = options.Decoder, Color = options.InputColor, Layout = options.InputLayout,
            Resize = options.ResizeMode, LabelsPath = options.LabelsPath, OutputName = options.OutputName,
            Scale = options.InputScale, Mean = (float[])options.Mean.Clone(), Std = (float[])options.Std.Clone(),
            Cache = useCache ? 1 : 0, Batch = options.BatchSize,
        };
        Native.Check(Native.ModelLoadEx(path, ref native, out ulong id));
        handle = new ModelHandle(id);
    }
    public Prediction Predict(byte[] bgr, int width, int height, int stride = 0,
        float confidence = -1f, float iou = 0.45f, int maxDetections = 100)
    {
        ArgumentNullException.ThrowIfNull(bgr);
        if (stride == 0) stride = checked(width * 3);
        if (stride < 0) throw new ArgumentOutOfRangeException(nameof(stride));
        Native.Check(Native.Predict(handle, bgr, (ulong)bgr.LongLength, width, height,
            (ulong)stride, confidence, iou, maxDetections, out ulong id));
        return new Prediction(new ResultHandle(id));
    }
    public static string DevicesJson() => Native.Read(Native.Devices);
    /// <summary>Model metadata as JSON: inputs/outputs, labels, backend and image contract.</summary>
    public string InfoJson() => Native.Read((byte[]? bytes, ulong capacity, out ulong required) =>
        Native.ModelInfo(handle, bytes, capacity, out required));
    public void Dispose() => handle.Dispose();
}

public sealed class Prediction : IDisposable
{
    private readonly ResultHandle handle;
    internal Prediction(ResultHandle handle) => this.handle = handle;
    public string ToJson() => Native.Read((byte[]? bytes, ulong capacity, out ulong required) =>
        Native.ResultJson(handle, bytes, capacity, out required));
    public (int Width, int Height, int[] Values) ClassMask()
    {
        int status = Native.Mask(handle, null, 0, out int width, out int height, out ulong required);
        if (status != 3) Native.Check(status);
        int[] values = new int[checked((int)required)];
        Native.Check(Native.Mask(handle, values, required, out width, out height, out _));
        return (width, height, values);
    }
    public void Dispose() => handle.Dispose();
}

public enum LogLevel { Debug = 0, Info = 1, Warning = 2, Error = 3 }

/// <summary>SDK-wide version and logging controls.</summary>
public static class Ivit
{
    private static Native.LogNative? logDelegate; // keeps the native callback alive
    private static bool exitHooked;

    /// <summary>Native SDK version, for example "1.1.0".</summary>
    public static string Version => Marshal.PtrToStringUTF8(Native.VersionString()) ?? "";

    /// <summary>One of debug, info, warning, error or off.</summary>
    public static void SetLogLevel(string level) => Native.Check(Native.SetLogLevel(level));

    /// <summary>
    /// Routes SDK messages to <paramref name="handler"/>; null restores stderr.
    /// The handler runs on arbitrary SDK threads, possibly concurrently, so it must be
    /// thread-safe. Exceptions it throws are swallowed rather than crashing native code.
    /// </summary>
    public static void SetLogHandler(Action<LogLevel, string>? handler)
    {
        if (handler is null)
        {
            Native.Check(Native.SetLogCallback(null, IntPtr.Zero));
            logDelegate = null;
            return;
        }
        Native.LogNative native = (level, message, _) =>
        {
            try { handler((LogLevel)level, Marshal.PtrToStringUTF8(message) ?? ""); }
            catch { /* never let managed exceptions unwind into native frames */ }
        };
        Native.Check(Native.SetLogCallback(native, IntPtr.Zero));
        logDelegate = native;
        if (!exitHooked)
        {
            exitHooked = true;
            AppDomain.CurrentDomain.ProcessExit += (_, _) => Native.SetLogCallback(null, IntPtr.Zero);
        }
    }
}
