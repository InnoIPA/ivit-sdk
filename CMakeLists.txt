cmake_minimum_required(VERSION 3.18)
project(ivit VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# Options
# ============================================================================
option(IVIT_BUILD_PYTHON "Build Python bindings" ON)
option(IVIT_BUILD_TESTS "Build tests" ON)
option(IVIT_BUILD_EXAMPLES "Build examples" ON)
option(IVIT_USE_OPENVINO "Enable OpenVINO backend" ON)
option(IVIT_USE_TENSORRT "Enable TensorRT backend" ON)
option(IVIT_USE_QNN "Enable QNN backend (Qualcomm IQ Series)" OFF)
option(IVIT_USE_HAILORT "Enable HailoRT backend (Hailo-8 PCIe AI accelerator)" OFF)

# 依賴打包選項
option(IVIT_BUNDLE_DEPS "Bundle dependencies into SDK for redistribution" OFF)
set(IVIT_DEPS_DIR "${CMAKE_SOURCE_DIR}/deps" CACHE PATH "Dependencies directory")

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# RPATH 設定 - 讓 SDK 能找到打包的依賴庫
# ============================================================================
if(IVIT_BUNDLE_DEPS)
    # 設定 RPATH 為相對路徑，讓 SDK 可以自動找到打包的依賴庫
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;$ORIGIN/../deps/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    # Build 時也使用相對路徑
    set(CMAKE_BUILD_RPATH "$ORIGIN;${CMAKE_BINARY_DIR}/lib;${IVIT_DEPS_DIR}/install/runtime/lib/intel64;${IVIT_DEPS_DIR}/install/tensorrt/lib")

    message(STATUS "Bundle mode enabled - dependencies will be packaged with SDK")
endif()

# ============================================================================
# Output directories
# ============================================================================
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 用於收集需要打包的依賴庫
set(IVIT_BUNDLE_LIBRARIES "" CACHE INTERNAL "Libraries to bundle")

# ============================================================================
# Find dependencies
# ============================================================================

# OpenCV (required)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio dnn)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "OpenCV lib dir: ${OpenCV_INSTALL_PATH}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")

# ============================================================================
# OpenVINO
# ============================================================================
if(IVIT_USE_OPENVINO)
    # 優先使用打包的 OpenVINO
    if(IVIT_BUNDLE_DEPS AND EXISTS "${IVIT_DEPS_DIR}/install/runtime/cmake")
        set(OpenVINO_DIR "${IVIT_DEPS_DIR}/install/runtime/cmake")
        message(STATUS "Using bundled OpenVINO from: ${OpenVINO_DIR}")
    endif()

    find_package(OpenVINO QUIET COMPONENTS Runtime)

    # 若未找到，嘗試偵測 pip 安裝的 OpenVINO
    if(NOT OpenVINO_FOUND)
        execute_process(
            COMMAND python3 -c "import openvino; import os; print(os.path.dirname(openvino.__file__))"
            OUTPUT_VARIABLE _OV_PIP_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _OV_PIP_RESULT
        )
        if(_OV_PIP_RESULT EQUAL 0 AND EXISTS "${_OV_PIP_DIR}/cmake")
            set(OpenVINO_DIR "${_OV_PIP_DIR}/cmake")
            message(STATUS "Using pip-installed OpenVINO from: ${OpenVINO_DIR}")
            find_package(OpenVINO QUIET COMPONENTS Runtime)
        endif()
    endif()
    if(OpenVINO_FOUND)
        message(STATUS "OpenVINO found: ${OpenVINO_VERSION}")
        add_compile_definitions(IVIT_HAS_OPENVINO)

        # 修正 ABI 設定：pip OpenVINO 使用舊 ABI，但系統 OpenCV 使用新 ABI
        # 強制使用新 ABI (C++11) 以與 OpenCV 相容
        # 注意：這要求 OpenVINO 能容忍 ABI 不匹配（通常可以，因為介面不直接傳遞 std::string）
        remove_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
        message(STATUS "Forcing _GLIBCXX_USE_CXX11_ABI=1 for OpenCV compatibility")

        # 收集需要打包的庫
        if(IVIT_BUNDLE_DEPS)
            get_target_property(OV_LIB_LOCATION openvino::runtime LOCATION)
            get_filename_component(OV_LIB_DIR ${OV_LIB_LOCATION} DIRECTORY)
            file(GLOB OV_LIBS "${OV_LIB_DIR}/*.so*")
            list(APPEND IVIT_BUNDLE_LIBRARIES ${OV_LIBS})
        endif()
    else()
        message(WARNING "OpenVINO not found, disabling OpenVINO backend")
        set(IVIT_USE_OPENVINO OFF)
    endif()
endif()

# ============================================================================
# TensorRT
# ============================================================================
if(IVIT_USE_TENSORRT)
    # 優先使用打包的 TensorRT
    if(IVIT_BUNDLE_DEPS AND EXISTS "${IVIT_DEPS_DIR}/install/tensorrt")
        set(TENSORRT_ROOT "${IVIT_DEPS_DIR}/install/tensorrt")
        message(STATUS "Using bundled TensorRT from: ${TENSORRT_ROOT}")
    endif()

    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS ${TENSORRT_ROOT} ${CUDAToolkit_LIBRARY_DIR}/.. /usr/local/tensorrt
            PATH_SUFFIXES include)
        find_library(TENSORRT_LIBRARY nvinfer
            HINTS ${TENSORRT_ROOT} ${CUDAToolkit_LIBRARY_DIR} /usr/local/tensorrt
            PATH_SUFFIXES lib lib64)
        find_library(NVONNXPARSER_LIBRARY nvonnxparser
            HINTS ${TENSORRT_ROOT} ${CUDAToolkit_LIBRARY_DIR} /usr/local/tensorrt
            PATH_SUFFIXES lib lib64)

        if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
            message(STATUS "TensorRT found: ${TENSORRT_LIBRARY}")
            add_compile_definitions(IVIT_HAS_TENSORRT)

            # 收集需要打包的庫
            if(IVIT_BUNDLE_DEPS)
                get_filename_component(TRT_LIB_DIR ${TENSORRT_LIBRARY} DIRECTORY)
                file(GLOB TRT_LIBS "${TRT_LIB_DIR}/libnvinfer*.so*" "${TRT_LIB_DIR}/libnvonnxparser*.so*")
                list(APPEND IVIT_BUNDLE_LIBRARIES ${TRT_LIBS})

                # CUDA Runtime libraries
                file(GLOB CUDA_RUNTIME_LIBS
                    "${CUDAToolkit_LIBRARY_DIR}/libcudart.so*"
                    "${CUDAToolkit_LIBRARY_DIR}/libcublas*.so*"
                )
                list(APPEND IVIT_BUNDLE_LIBRARIES ${CUDA_RUNTIME_LIBS})
            endif()
        else()
            message(WARNING "TensorRT not found, disabling TensorRT backend")
            set(IVIT_USE_TENSORRT OFF)
        endif()
    else()
        message(WARNING "CUDA not found, disabling TensorRT backend")
        set(IVIT_USE_TENSORRT OFF)
    endif()
endif()

# ============================================================================
# HailoRT (Hailo-8 PCIe AI Accelerator)
# ============================================================================
if(IVIT_USE_HAILORT)
    if(NOT DEFINED HAILORT_SDK_ROOT AND DEFINED ENV{HAILORT_SDK_ROOT})
        set(HAILORT_SDK_ROOT "$ENV{HAILORT_SDK_ROOT}")
    endif()

    find_path(HAILORT_INCLUDE_DIR hailo/hailort.hpp
        HINTS ${HAILORT_SDK_ROOT} /usr/include /usr/local/include
    )
    find_library(HAILORT_LIBRARY hailort
        HINTS ${HAILORT_SDK_ROOT} /usr/lib /usr/local/lib
    )

    if(HAILORT_INCLUDE_DIR AND HAILORT_LIBRARY)
        message(STATUS "HailoRT found: ${HAILORT_LIBRARY}")
        add_compile_definitions(IVIT_HAS_HAILORT)
    else()
        message(WARNING "HailoRT not found — disabling HailoRT backend. "
                        "Set HAILORT_SDK_ROOT to your SDK installation path "
                        "or install via: apt install hailoRT-dev")
        set(IVIT_USE_HAILORT OFF)
    endif()
endif()

# ============================================================================
# Library sources
# ============================================================================
set(IVIT_SOURCES
    # Core
    src/core/model.cpp
    src/core/device_manager.cpp
    src/core/result.cpp
    src/core/callback.cpp
    src/core/runtime_config.cpp
    src/core/video_source.cpp

    # Runtime
    src/runtime/runtime_factory.cpp
    src/runtime/runtime_init.cpp

    # Vision
    src/vision/classifier.cpp
    src/vision/detector.cpp
    src/vision/segmentor.cpp

    # Utils
    src/utils/profiler.cpp
    src/utils/visualizer.cpp
)

# Add backend-specific sources
if(IVIT_USE_OPENVINO)
    list(APPEND IVIT_SOURCES src/runtime/openvino_runtime.cpp)
endif()

if(IVIT_USE_TENSORRT)
    list(APPEND IVIT_SOURCES src/runtime/tensorrt_runtime.cpp)
endif()

if(IVIT_USE_HAILORT)
    list(APPEND IVIT_SOURCES src/runtime/hailort_runtime.cpp)
endif()

# ============================================================================
# Create library
# ============================================================================
add_library(ivit SHARED ${IVIT_SOURCES})

target_include_directories(ivit
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${OpenCV_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(ivit
    PUBLIC
        ${OpenCV_LIBRARIES}
)

# Link backend libraries
if(IVIT_USE_OPENVINO)
    target_link_libraries(ivit PRIVATE openvino::runtime)
endif()

if(IVIT_USE_TENSORRT)
    target_include_directories(ivit PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(ivit PRIVATE
        ${TENSORRT_LIBRARY}
        ${NVONNXPARSER_LIBRARY}
        CUDA::cudart
    )
endif()

if(IVIT_USE_HAILORT)
    target_include_directories(ivit PRIVATE ${HAILORT_INCLUDE_DIR})
    target_link_libraries(ivit PRIVATE ${HAILORT_LIBRARY})
endif()

# ============================================================================
# Python bindings
# ============================================================================
if(IVIT_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        find_package(pybind11 CONFIG REQUIRED)
    endif()

    pybind11_add_module(_ivit_core python/ivit/_binding.cpp)
    target_link_libraries(_ivit_core PRIVATE ivit)

    if(SKBUILD)
        # scikit-build-core controls the output directory
        install(TARGETS _ivit_core LIBRARY DESTINATION .)
        install(TARGETS ivit LIBRARY DESTINATION .)
    else()
        set_target_properties(_ivit_core PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/ivit
        )
    endif()
endif()

# ============================================================================
# Tests
# ============================================================================
if(IVIT_BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)

    if(GTest_FOUND)
        add_executable(test_model tests/cpp/test_model.cpp)
        target_link_libraries(test_model PRIVATE ivit GTest::gtest_main)
        add_test(NAME test_model COMMAND test_model)

        add_executable(test_inference tests/cpp/test_inference.cpp)
        target_link_libraries(test_inference PRIVATE ivit GTest::gtest_main)
        add_test(NAME test_inference COMMAND test_inference)
    else()
        message(STATUS "Google Test not found, C++ tests disabled")
    endif()
endif()

# ============================================================================
# Examples
# ============================================================================
if(IVIT_BUILD_EXAMPLES)
    add_executable(simple_inference examples/cpp/simple_inference.cpp)
    target_link_libraries(simple_inference PRIVATE ivit ${OpenCV_LIBRARIES})

    add_executable(classification_demo examples/cpp/classification_demo.cpp)
    target_link_libraries(classification_demo PRIVATE ivit ${OpenCV_LIBRARIES})

    add_executable(detection_demo examples/cpp/detection_demo.cpp)
    target_link_libraries(detection_demo PRIVATE ivit ${OpenCV_LIBRARIES})

    add_executable(video_demo examples/cpp/video_demo.cpp)
    target_link_libraries(video_demo PRIVATE ivit ${OpenCV_LIBRARIES})

    add_executable(si_quickstart examples/cpp/si_quickstart.cpp)
    target_link_libraries(si_quickstart PRIVATE ivit ${OpenCV_LIBRARIES})

    add_executable(embedded_optimization examples/cpp/embedded_optimization.cpp)
    target_link_libraries(embedded_optimization PRIVATE ivit ${OpenCV_LIBRARIES})

    add_executable(backend_service examples/cpp/backend_service.cpp)
    target_link_libraries(backend_service PRIVATE ivit ${OpenCV_LIBRARIES})

    add_executable(data_analysis examples/cpp/data_analysis.cpp)
    target_link_libraries(data_analysis PRIVATE ivit ${OpenCV_LIBRARIES})
endif()

# ============================================================================
# Install
# ============================================================================
include(GNUInstallDirs)

install(TARGETS ivit
    EXPORT ivitTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ivit
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT ivitTargets
    FILE ivitTargets.cmake
    NAMESPACE ivit::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ivit
)

# ============================================================================
# Install bundled dependencies
# ============================================================================
if(IVIT_BUNDLE_DEPS AND IVIT_BUNDLE_LIBRARIES)
    # 建立 deps/lib 目錄並複製依賴庫
    install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX}/deps/lib)

    foreach(lib ${IVIT_BUNDLE_LIBRARIES})
        if(EXISTS ${lib} AND NOT IS_DIRECTORY ${lib})
            get_filename_component(lib_name ${lib} NAME)
            # 解析 symlink
            get_filename_component(lib_real ${lib} REALPATH)

            # 安裝實際檔案
            install(FILES ${lib_real}
                DESTINATION ${CMAKE_INSTALL_PREFIX}/deps/lib
                RENAME ${lib_name}
            )
        endif()
    endforeach()

    # 建立環境設定腳本
    file(WRITE "${CMAKE_BINARY_DIR}/setup_env.sh" [[
#!/bin/bash
# iVIT-SDK Environment Setup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(dirname "$SCRIPT_DIR")"

export LD_LIBRARY_PATH="${SDK_ROOT}/lib:${SDK_ROOT}/deps/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${SDK_ROOT}/python:${PYTHONPATH}"

echo "iVIT-SDK environment configured."
]])

    install(PROGRAMS "${CMAKE_BINARY_DIR}/setup_env.sh"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    )
endif()

# ============================================================================
# Package configuration
# ============================================================================
if(IVIT_BUNDLE_DEPS)
    # 建立打包腳本 (使用 configure_file 展開 CMake 變數)
    configure_file(
        "${CMAKE_SOURCE_DIR}/packaging/package_sdk.sh.in"
        "${CMAKE_BINARY_DIR}/package_sdk.sh"
        @ONLY
    )
endif()

# ============================================================================
# CPack DEB 設定
# ============================================================================
include(CPack)

set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_NAME "ivit-sdk")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_CONTACT "Innodisk AI Team <ai@innodisk.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "iVIT-SDK: Unified CV inference SDK for Intel and NVIDIA platforms")
set(CPACK_PACKAGE_VENDOR "Innodisk")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://www.innodisk.com")

# DEB 特定設定
set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")

# 依平台設定系統依賴
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libopencv-dev (>= 4.0)")
if(IVIT_USE_OPENVINO)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, openvino (>= 2024.0)")
endif()
if(IVIT_USE_TENSORRT)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, libnvinfer-dev")
endif()

# 架構與平台後綴
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
else()
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
endif()

# 用平台後綴區分不同後端的 .deb
if(IVIT_USE_TENSORRT AND NOT IVIT_USE_OPENVINO)
    set(CPACK_PACKAGE_FILE_NAME "ivit-sdk_${PROJECT_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-nvidia")
elseif(IVIT_USE_OPENVINO AND NOT IVIT_USE_TENSORRT)
    set(CPACK_PACKAGE_FILE_NAME "ivit-sdk_${PROJECT_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}-intel")
else()
    set(CPACK_PACKAGE_FILE_NAME "ivit-sdk_${PROJECT_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "iVIT-SDK Configuration Summary")
message(STATUS "==============================")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Backends:")
message(STATUS "  OpenVINO:       ${IVIT_USE_OPENVINO}")
message(STATUS "  TensorRT:       ${IVIT_USE_TENSORRT}")
message(STATUS "  QNN:            ${IVIT_USE_QNN}")
message(STATUS "  HailoRT:        ${IVIT_USE_HAILORT}")
message(STATUS "")
message(STATUS "Components:")
message(STATUS "  Python:         ${IVIT_BUILD_PYTHON}")
message(STATUS "  Tests:          ${IVIT_BUILD_TESTS}")
message(STATUS "  Examples:       ${IVIT_BUILD_EXAMPLES}")
message(STATUS "")
message(STATUS "Packaging:")
message(STATUS "  Bundle deps:    ${IVIT_BUNDLE_DEPS}")
if(IVIT_BUNDLE_DEPS)
    message(STATUS "  Deps dir:       ${IVIT_DEPS_DIR}")
    list(LENGTH IVIT_BUNDLE_LIBRARIES NUM_BUNDLE_LIBS)
    message(STATUS "  Libraries:      ${NUM_BUNDLE_LIBS} files to bundle")
endif()
message(STATUS "")
