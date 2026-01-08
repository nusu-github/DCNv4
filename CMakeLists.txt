cmake_minimum_required(VERSION 3.18)
project("${SKBUILD_PROJECT_NAME}"
  VERSION "${SKBUILD_PROJECT_VERSION}"
  LANGUAGES CXX CUDA
)

# Python (skbuild / scikit-build-core)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# PyTorch (libtorch from pip/conda is typically found via CMAKE_PREFIX_PATH)
find_package(Torch REQUIRED)

# CUDA toolkit (for CUDA::cudart etc.)
find_package(CUDAToolkit REQUIRED)

# Sources
file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/dcnv4/csrc/cuda/*.cu")
set(CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dcnv4/csrc/ops.cpp")

# Extension module
Python_add_library(dcnv4_C MODULE ${CPP_SOURCES} ${CUDA_SOURCES} WITH_SOABI)

# Use torch flags
target_compile_options(dcnv4_C PRIVATE ${TORCH_CXX_FLAGS})
target_compile_features(dcnv4_C PRIVATE cxx_std_17)
set_property(TARGET dcnv4_C PROPERTY CUDA_STANDARD 17)

# Includes (do not quote list vars)
target_include_directories(dcnv4_C PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/dcnv4/csrc"
  ${TORCH_INCLUDE_DIRS}
)

# Definitions
target_compile_definitions(dcnv4_C PRIVATE
  WITH_CUDA
  TORCH_EXTENSION_NAME=dcnv4_C
)
target_compile_definitions(dcnv4_C PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    __CUDA_NO_FP4_CONVERSION_OPERATORS__
    __CUDA_NO_FP4_CONVERSIONS__
    __CUDA_NO_FP6_CONVERSION_OPERATORS__
    __CUDA_NO_FP6_CONVERSIONS__
    __CUDA_NO_FP8_CONVERSION_OPERATORS__
    __CUDA_NO_FP8_CONVERSIONS__
    __CUDA_NO_HALF_CONVERSIONS__
    __CUDA_NO_HALF_OPERATORS__
    __CUDA_NO_HALF2_OPERATORS__
    __CUDA_NO_BFLOAT16_CONVERSIONS__
    __CUDA_NO_BFLOAT16_OPERATORS__
    __CUDA_NO_BFLOAT162_OPERATORS__
  >
)

# Compile options
target_compile_options(dcnv4_C PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    -lineinfo
    --expt-relaxed-constexpr
    --extended-lambda
    --use_fast_math
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g>
  >
)

# Link libraries
target_link_libraries(dcnv4_C PRIVATE
  ${TORCH_LIBRARIES}
  CUDA::cudart
)

set_target_properties(dcnv4_C PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH_USE_ORIGIN ON
  INSTALL_RPATH_USE_LINK_PATH ON
)

install(TARGETS dcnv4_C LIBRARY DESTINATION dcnv4)
