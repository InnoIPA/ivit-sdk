#ifndef IVIT_C_API_H
#define IVIT_C_API_H

#include <stdint.h>

#if defined(_WIN32)
#if defined(IVIT_BUILDING_LIBRARY)
#define IVIT_C_API __declspec(dllexport)
#else
#define IVIT_C_API __declspec(dllimport)
#endif
#define IVIT_C_CALL __cdecl
#else
#define IVIT_C_API __attribute__((visibility("default")))
#define IVIT_C_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ABI v1: UTF-8 strings; opaque process-local IDs, zero is invalid.
 * Release each successful model/result handle once. Results outlive models.
 * Calls may run concurrently; release prevents new calls while active calls
 * retain ownership. Never pass invalid pointers. No C++ exception escapes.
 */
typedef uint64_t ivit_model_handle;
typedef uint64_t ivit_result_handle;
typedef int32_t ivit_status;
#define IVIT_C_OK 0
#define IVIT_C_INVALID_ARGUMENT 1
#define IVIT_C_INVALID_HANDLE 2
#define IVIT_C_BUFFER_TOO_SMALL 3
#define IVIT_C_RUNTIME_ERROR 4

IVIT_C_API uint32_t IVIT_C_CALL ivit_abi_version(void);
/* SDK version as a static UTF-8 string such as "1.1.0"; never NULL, never freed. */
IVIT_C_API const char *IVIT_C_CALL ivit_version_string(void);

/* Logging. Messages go to stderr unless a callback is installed.
 * level: 0=debug 1=info 2=warning 3=error. The callback may run on any SDK
 * thread, possibly concurrently, with a UTF-8 message that is valid only
 * during the call. It must not throw (C++ callers) or call back into the SDK.
 * Install before loading models; call ivit_set_log_callback(NULL, NULL)
 * before the callback or user_data is destroyed. Additive: ABI stays v1.
 */
#define IVIT_C_LOG_DEBUG 0
#define IVIT_C_LOG_INFO 1
#define IVIT_C_LOG_WARNING 2
#define IVIT_C_LOG_ERROR 3
typedef void(IVIT_C_CALL *ivit_log_callback)(int32_t level, const char *message, void *user_data);
IVIT_C_API ivit_status IVIT_C_CALL ivit_set_log_callback(ivit_log_callback callback,
                                                         void *user_data);
/* level: "debug", "info", "warning", "error" or "off". */
IVIT_C_API ivit_status IVIT_C_CALL ivit_set_log_level(const char *level);
/* Query UTF-8 errors without clearing them. required includes the NUL byte.
 * Error state is per-thread and replaced by the next non-error-query call.
 */
IVIT_C_API ivit_status IVIT_C_CALL ivit_last_error(char *buffer, uint64_t capacity,
                                                   uint64_t *required);
IVIT_C_API ivit_status IVIT_C_CALL ivit_devices_json(char *buffer, uint64_t capacity,
                                                     uint64_t *required);
IVIT_C_API ivit_status IVIT_C_CALL ivit_model_load(const char *path, const char *device,
                                                   const char *backend, const char *task,
                                                   const char *precision, int32_t use_cache,
                                                   ivit_model_handle *model);
/* Initialize struct_size=sizeof(options) before init. Strings may be NULL to
 * use defaults. Storage remains caller-owned through model_load_ex.
 */
typedef struct ivit_load_options_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    const char *device;
    const char *backend;
    const char *task;
    const char *precision;
    const char *decoder;
    const char *input_color;
    const char *input_layout;
    const char *resize_mode;
    const char *labels_path;
    const char *output_name;
    float input_scale;
    float mean[3];
    float stddev[3];
    int32_t use_cache;
    int32_t batch_size;
} ivit_load_options_v1;
IVIT_C_API ivit_status IVIT_C_CALL ivit_load_options_init(ivit_load_options_v1 *options);
IVIT_C_API ivit_status IVIT_C_CALL ivit_model_load_ex(const char *path,
                                                      const ivit_load_options_v1 *options,
                                                      ivit_model_handle *model);
IVIT_C_API ivit_status IVIT_C_CALL ivit_model_release(ivit_model_handle model);
/* Input: uint8 BGR HWC; stride is bytes per row, including optional padding.
 * bytes must cover the final row. Caller owns input until this call returns.
 * result is zero on failure. Thresholds must be finite and within [0,1].
 */
IVIT_C_API ivit_status IVIT_C_CALL ivit_model_predict_bgr(ivit_model_handle model,
                                                          const uint8_t *pixels, uint64_t bytes,
                                                          int32_t width, int32_t height,
                                                          uint64_t stride, float confidence,
                                                          float iou, int32_t max_detections,
                                                          ivit_result_handle *result);
IVIT_C_API ivit_status IVIT_C_CALL ivit_result_release(ivit_result_handle result);
/* JSON metadata/classifications/detections; required includes NUL.
 * Call with NULL/0 to query size (BUFFER_TOO_SMALL); then supply buffer.
 */
IVIT_C_API ivit_status IVIT_C_CALL ivit_result_json(ivit_result_handle result, char *buffer,
                                                    uint64_t capacity, uint64_t *required);
/* Semantic class mask: row-major int32 values; capacity/required in elements.
 * Empty mask yields width=height=required=0. Query with NULL/0 first.
 */
IVIT_C_API ivit_status IVIT_C_CALL ivit_result_mask_i32(ivit_result_handle result, int32_t *buffer,
                                                        uint64_t capacity, int32_t *width,
                                                        int32_t *height, uint64_t *required);

/* ---- Additions in SDK 1.1 (additive: everything above is unchanged, ABI stays v1) ---- */

/* Why the last call on this thread failed, finer than ivit_status (which keeps returning
 * IVIT_C_RUNTIME_ERROR for these). 0 after a successful call. Does not clear the error.
 */
#define IVIT_C_ERR_NONE 0
#define IVIT_C_ERR_GENERIC 1
#define IVIT_C_ERR_MODEL_LOAD 2
#define IVIT_C_ERR_UNSUPPORTED_FORMAT 3
#define IVIT_C_ERR_DEVICE_NOT_FOUND 4
#define IVIT_C_ERR_INFERENCE 5
IVIT_C_API int32_t IVIT_C_CALL ivit_last_error_category(void);

/* Model metadata as JSON: name, task, device, backend, inputs/outputs (name, shape, dtype,
 * layout; -1 = dynamic), labels and the image contract. Query the size with NULL/0 first.
 */
IVIT_C_API ivit_status IVIT_C_CALL ivit_model_info_json(ivit_model_handle model, char *buffer,
                                                        uint64_t capacity, uint64_t *required);

/* Load options with the backend tuning fields that ivit_load_options_v1 lacks. Set struct_size
 * = sizeof(options) before ivit_load_options_v2_init. NULL strings use defaults. All pointers
 * (including labels) are caller-owned and only read during ivit_model_load_v2.
 */
typedef struct ivit_load_options_v2 {
    uint32_t struct_size;
    uint32_t abi_version; /* 2 */
    const char *device;
    const char *backend;
    const char *task;
    const char *precision;
    const char *decoder;
    const char *input_color;
    const char *input_layout;
    const char *resize_mode;
    const char *labels_path;
    const char *output_name;
    const char *cache_dir;
    const char *performance_mode; /* "LATENCY" or "THROUGHPUT" (OpenVINO) */
    const char *const *labels;    /* optional explicit class names */
    uint64_t workspace_bytes;     /* TensorRT workspace limit */
    uint32_t label_count;
    int32_t use_cache;
    int32_t batch_size;     /* TensorRT profile OPT */
    int32_t max_batch_size; /* TensorRT profile MAX */
    int32_t num_threads;    /* OpenVINO CPU threads; 0 = automatic */
    int32_t enable_profiling;
    float input_scale;
    float mean[3];
    float stddev[3];
    int32_t num_instances; /* TensorRT execution contexts for concurrent callers, 1-16 */
} ivit_load_options_v2;
IVIT_C_API ivit_status IVIT_C_CALL ivit_load_options_v2_init(ivit_load_options_v2 *options);
IVIT_C_API ivit_status IVIT_C_CALL ivit_model_load_v2(const char *path,
                                                      const ivit_load_options_v2 *options,
                                                      ivit_model_handle *model);

/* An 8-bit image: channels 1 (gray) or 3 (BGR), rows `stride` bytes apart, bytes must cover
 * the final row. The caller owns the pixels until the call returns.
 */
typedef struct ivit_image_v1 {
    const uint8_t *pixels;
    uint64_t bytes;
    uint64_t stride;
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t reserved; /* must be 0 */
} ivit_image_v1;

/* Per-call inference options. Set struct_size before ivit_infer_options_init.
 * Leave confidence at -1 unless you want a specific score threshold; a classification model then
 * returns all classes, a detection model keeps boxes scoring at least 0.5.
 */
typedef struct ivit_infer_options_v1 {
    uint32_t struct_size;
    uint32_t abi_version; /* 1 */
    float confidence;     /* [0,1], or -1 (the default) for the task default: 0.5 for detection,
                             0 for classification so that every class is returned */
    float iou;            /* default 0.45, within [0,1] */
    int32_t max_detections; /* default 100, > 0 */
    int32_t agnostic_nms;   /* 0 = per class (default), 1 = across classes */
    const int32_t *classes; /* optional class-id filter */
    uint32_t class_count;
    uint32_t reserved; /* must be 0 */
} ivit_infer_options_v1;
IVIT_C_API ivit_status IVIT_C_CALL ivit_infer_options_init(ivit_infer_options_v1 *options);

/* options may be NULL for the defaults above. */
IVIT_C_API ivit_status IVIT_C_CALL ivit_model_predict_ex(ivit_model_handle model,
                                                         const ivit_image_v1 *image,
                                                         const ivit_infer_options_v1 *options,
                                                         ivit_result_handle *result);

/* Batch of 1..1024 images. `results` points to `count` handles; all succeed or none are
 * created (every entry is 0 on failure). Release each handle separately.
 */
IVIT_C_API ivit_status IVIT_C_CALL ivit_model_predict_batch(ivit_model_handle model,
                                                            const ivit_image_v1 *images,
                                                            uint32_t count,
                                                            const ivit_infer_options_v1 *options,
                                                            ivit_result_handle *results);

#ifdef __cplusplus
}
#endif
#endif
