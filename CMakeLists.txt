cmake_minimum_required(VERSION 3.18)
project(ivit VERSION 1.1.0 LANGUAGES CXX)

# ============================================================================
# Options
# ============================================================================
option(IVIT_BUILD_PYTHON "Build Python bindings" ON)
option(IVIT_BUILD_TESTS "Build tests" ON)
option(IVIT_BUILD_EXAMPLES "Build examples" ON)
option(IVIT_USE_OPENVINO "Enable OpenVINO backend" ON)
option(IVIT_USE_TENSORRT "Enable TensorRT backend" ON)

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

        # 收集需要打包的庫
        if(IVIT_BUNDLE_DEPS)
            get_target_property(OV_LIB_LOCATION openvino::runtime LOCATION)
            get_filename_component(OV_LIB_DIR ${OV_LIB_LOCATION} DIRECTORY)
            file(GLOB OV_LIBS "${OV_LIB_DIR}/*.so*")
            list(APPEND IVIT_BUNDLE_LIBRARIES ${OV_LIBS})
            # Intel's apt/tgz packages keep the device plugins in openvino-<version>/ next to
            # libopenvino.so, and OpenVINO looks for them there at run time (the pip package keeps
            # them beside it, which the glob above already covers). Without this directory the
            # bundled runtime has no CPU/GPU/NPU device at all.
            set(IVIT_BUNDLE_OV_PLUGIN_DIR "")
            if(IS_DIRECTORY "${OV_LIB_DIR}/openvino-${OpenVINO_VERSION}")
                set(IVIT_BUNDLE_OV_PLUGIN_DIR "${OV_LIB_DIR}/openvino-${OpenVINO_VERSION}")
            endif()
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

        if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND NVONNXPARSER_LIBRARY)
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
# Library sources
# ============================================================================

set(IVIT_SOURCES
    # Core
    src/core/model.cpp
    src/core/c_api.cpp
    src/core/log.cpp
    src/core/device_manager.cpp
    src/core/result.cpp
    src/core/callback.cpp
    src/core/video_source.cpp

    # Runtime
    src/runtime/runtime_factory.cpp
    src/runtime/runtime_init.cpp

    # Vision
    src/vision/classifier.cpp
    src/vision/processing.cpp
    src/vision/detector.cpp
    src/vision/segmentor.cpp

    # Utils
    src/utils/visualizer.cpp
)

# Add backend-specific sources
if(IVIT_USE_OPENVINO)
    list(APPEND IVIT_SOURCES src/runtime/openvino_runtime.cpp)
endif()

if(IVIT_USE_TENSORRT)
    list(APPEND IVIT_SOURCES src/runtime/tensorrt_runtime.cpp)
endif()


# ============================================================================
# Create library
# ============================================================================
add_library(ivit SHARED ${IVIT_SOURCES})
add_library(ivit::ivit ALIAS ivit)
target_compile_features(ivit PUBLIC cxx_std_17)
target_compile_definitions(ivit PRIVATE IVIT_BUILDING_LIBRARY)
# The C++ API/ABI may change in a minor release (new config fields), so the SONAME
# carries MAJOR.MINOR. Only the C ABI (ivit_abi_version) is stable across minors.
# Wheels install a single plain libivit.so beside _ivit_core, so skip it there.
if(NOT SKBUILD)
    set_target_properties(ivit PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
endif()

target_include_directories(ivit
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
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
        # Both native libraries are installed beside each other in ivit/.
        # OpenVINO's Python wheel keeps its runtime in the sibling package.
        set_property(TARGET _ivit_core APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
        set_property(TARGET ivit APPEND PROPERTY INSTALL_RPATH
            "$ORIGIN" "$ORIGIN/../openvino/libs")
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

        add_executable(test_processing tests/cpp/test_processing.cpp)
        target_include_directories(test_processing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_processing PRIVATE ivit GTest::gtest_main)
        add_test(NAME test_processing COMMAND test_processing)
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

# TensorRT major version (= the libnvinfer soname) the SDK is built against; ivitConfig.cmake and
# the deb dependencies must name that exact major.
set(IVIT_TENSORRT_MAJOR "")
if(IVIT_USE_TENSORRT AND EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _ivit_trt_line
         REGEX "define (NV_TENSORRT_MAJOR|TRT_MAJOR_ENTERPRISE) [0-9]+")
    string(REGEX MATCH "[0-9]+" IVIT_TENSORRT_MAJOR "${_ivit_trt_line}")
endif()

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/ivitConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ivitConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ivit"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ivitConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ivitConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ivitConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ivit"
)

# ============================================================================
# Install bundled dependencies
# ============================================================================
if(IVIT_BUNDLE_DEPS AND IVIT_BUNDLE_LIBRARIES)
    # 建立 deps/lib 目錄並複製依賴庫
    install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX}/deps/lib)

    # Copy each real library once and recreate its other names (libfoo.so, libfoo.so.N) as
    # symlinks. Copying every name as a full file tripled the size: TensorRT alone (libnvinfer
    # 723 MB, three names) grew a tarball by ~1.4 GB.
    set(_ivit_bundled_real_names "")
    foreach(lib ${IVIT_BUNDLE_LIBRARIES})
        if(EXISTS ${lib} AND NOT IS_DIRECTORY ${lib})
            get_filename_component(lib_name ${lib} NAME)
            get_filename_component(lib_real ${lib} REALPATH)
            get_filename_component(real_name ${lib_real} NAME)
            if(NOT real_name IN_LIST _ivit_bundled_real_names)
                list(APPEND _ivit_bundled_real_names ${real_name})
                install(FILES ${lib_real} DESTINATION ${CMAKE_INSTALL_PREFIX}/deps/lib)
            endif()
            if(NOT lib_name STREQUAL real_name)
                install(CODE "file(CREATE_LINK \"${real_name}\"
                    \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/deps/lib/${lib_name}\" SYMBOLIC)")
            endif()
        endif()
    endforeach()

    if(IVIT_BUNDLE_OV_PLUGIN_DIR)
        install(DIRECTORY "${IVIT_BUNDLE_OV_PLUGIN_DIR}" DESTINATION ${CMAKE_INSTALL_PREFIX}/deps/lib)
    endif()

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
# CPack DEB 設定
# ============================================================================
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
# libivit.so links versioned sonames (libopenvino.so.2641, libnvinfer.so.11, libcudart.so.12), so
# depend on the runtime packages of the versions it was built against. Open ranges such as
# "libnvinfer-dev" let apt install another TensorRT major (a new soname) and break the SDK at run time.
# OpenVINO's core package has no devices or model readers: the CPU/AUTO plugins and the ONNX/IR
# frontends are separate packages, so depend on them; GPU/NPU plugins are recommended (apt installs
# recommendations by default) because they are only useful with that hardware.
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libopencv-dev (>= 4.0)")
if(IVIT_USE_OPENVINO)
    if(OpenVINO_VERSION)
        set(_ivit_ov "${OpenVINO_VERSION}")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, libopenvino-${_ivit_ov}, \
libopenvino-intel-cpu-plugin-${_ivit_ov}, libopenvino-auto-plugin-${_ivit_ov}, \
libopenvino-onnx-frontend-${_ivit_ov}, libopenvino-ir-frontend-${_ivit_ov}")
        set(CPACK_DEBIAN_PACKAGE_RECOMMENDS
            "libopenvino-intel-gpu-plugin-${_ivit_ov}, libopenvino-intel-npu-plugin-${_ivit_ov}")
    else()
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, openvino (>= 2024.0)")
    endif()
endif()
if(IVIT_USE_TENSORRT)
    if(IVIT_TENSORRT_MAJOR)
        set(CPACK_DEBIAN_PACKAGE_DEPENDS
            "${CPACK_DEBIAN_PACKAGE_DEPENDS}, libnvinfer${IVIT_TENSORRT_MAJOR}, libnvonnxparsers${IVIT_TENSORRT_MAJOR}")
    else()
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, libnvinfer-dev")
    endif()
    if(CUDAToolkit_VERSION_MAJOR AND CUDAToolkit_VERSION_MINOR)
        set(CPACK_DEBIAN_PACKAGE_DEPENDS
            "${CPACK_DEBIAN_PACKAGE_DEPENDS}, cuda-cudart-${CUDAToolkit_VERSION_MAJOR}-${CUDAToolkit_VERSION_MINOR}")
    endif()
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

# include(CPack) writes CPackConfig.cmake from the CPACK_* variables set so far, so it must come
# after all of them; included earlier, the maintainer, dependencies and file name were all lost.
include(CPack)

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
