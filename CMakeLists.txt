cmake_minimum_required(VERSION 3.18)
project(pls_pvc_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(USE_OPENMP "Enable OpenMP parallelism for CPU loops" ON)
option(USE_CUDA "Enable CUDA backend for GPU acceleration" OFF)

if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures" FORCE)
    else()
        message(WARNING "CUDA compiler not found — disabling USE_CUDA")
        set(USE_CUDA OFF)
    endif()
endif()

find_package(ZLIB REQUIRED)

if(USE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
endif()

set(CORE_SOURCES
    src/NiftiIO.cpp
    src/PSF.cpp
    src/Operators.cpp
    src/Guidance.cpp
    src/SplitBregmanPVC.cpp
)

if(USE_CUDA)
    list(APPEND CORE_SOURCES
        src/CudaOperators.cu
        src/CudaSplitBregmanPVC.cu
    )
endif()

add_library(pls_pvc_core ${CORE_SOURCES})

target_include_directories(pls_pvc_core PUBLIC include)
target_link_libraries(pls_pvc_core PUBLIC ZLIB::ZLIB)

if(USE_CUDA)
    target_compile_definitions(pls_pvc_core PUBLIC PVC_USE_CUDA=1)
    set_target_properties(pls_pvc_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(pls_pvc_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(pls_pvc_core PUBLIC PVC_USE_OPENMP=1)
endif()

add_executable(pls_pvc src/main.cpp)
target_link_libraries(pls_pvc PRIVATE pls_pvc_core)

add_executable(pls_pvc_smoke_test tests/smoke_test.cpp)
target_link_libraries(pls_pvc_smoke_test PRIVATE pls_pvc_core)

add_executable(pls_pvc_smoke_test_4d tests/smoke_test_4d.cpp)
target_link_libraries(pls_pvc_smoke_test_4d PRIVATE pls_pvc_core)

enable_testing()
add_test(NAME pls_pvc_smoke_test COMMAND pls_pvc_smoke_test)
add_test(NAME pls_pvc_smoke_test_4d COMMAND pls_pvc_smoke_test_4d)
set_tests_properties(pls_pvc_smoke_test PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
set_tests_properties(pls_pvc_smoke_test_4d PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")

if(USE_CUDA)
    add_executable(pls_pvc_cuda_smoke_test tests/cuda_test.cu)
    target_link_libraries(pls_pvc_cuda_smoke_test PRIVATE pls_pvc_core)
    set_target_properties(pls_pvc_cuda_smoke_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    add_test(NAME pls_pvc_cuda_smoke_test COMMAND pls_pvc_cuda_smoke_test)
endif()
