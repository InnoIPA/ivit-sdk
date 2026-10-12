#include <ivit/c_api.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call)                                                                                \
    do {                                                                                           \
        if ((call) != IVIT_C_OK)                                                                   \
            return __LINE__;                                                                       \
    } while (0)
int main(int argc, char **argv) {
    if (argc != 2 || ivit_abi_version() != 1)
        return 1;
    ivit_model_handle model = 0;
    ivit_result_handle result = 0;
    uint64_t required = 0;
    const uint64_t stride = 100; /* 96 image bytes + 4 bytes padding */
    unsigned char *image = (unsigned char *)calloc(32, stride);
    if (!image)
        return 2;
    CHECK(ivit_model_load(argv[1], "cpu", "openvino", "classification", "fp32", 0, &model));
    if (ivit_model_predict_bgr(model, image, 1, 32, 32, stride, 0, .45f, 100, &result) !=
            IVIT_C_INVALID_ARGUMENT ||
        result != 0)
        return 3;
    CHECK(ivit_model_predict_bgr(model, image, 32 * stride, 32, 32, stride, 0, .45f, 100, &result));
    free(image);
    CHECK(ivit_model_release(model));
    if (ivit_model_release(model) != IVIT_C_INVALID_HANDLE)
        return 4;
    if (ivit_result_json(result, NULL, 0, &required) != IVIT_C_BUFFER_TOO_SMALL)
        return 5;
    char *json = (char *)malloc(required);
    if (!json)
        return 6;
    CHECK(ivit_result_json(result, json, required, &required));
    puts(json);
    free(json);
    CHECK(ivit_result_release(result));
    if (ivit_result_release(result) != IVIT_C_INVALID_HANDLE)
        return 7;
    return 0;
}
