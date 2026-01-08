cmake_minimum_required(VERSION 3.18...3.28)
project(
  "${SKBUILD_PROJECT_NAME}"
  VERSION "${SKBUILD_PROJECT_VERSION}"
  LANGUAGES CXX CUDA)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Collect source files
file(GLOB CUDA_SOURCES "src/cuda/*.cu")
set(CPP_SOURCES "src/vision.cpp")

# Create the extension module
Python_add_library(dcnv4_C MODULE ${CPP_SOURCES} ${CUDA_SOURCES} WITH_SOABI)

target_include_directories(dcnv4_C PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${TORCH_INCLUDE_DIRS}"
)

target_link_libraries(dcnv4_C PRIVATE
  "${TORCH_LIBRARIES}"
  CUDA::cudart
)

target_compile_definitions(dcnv4_C PRIVATE
  WITH_CUDA
  TORCH_EXTENSION_NAME=_C
)

# CUDA compile options
set_target_properties(dcnv4_C PROPERTIES
  CUDA_STANDARD 17
  CXX_STANDARD 17
)

target_compile_options(dcnv4_C PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    -DCUDA_HAS_FP16=1
    -D__CUDA_NO_HALF_OPERATORS__
    -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_HALF2_OPERATORS__
  >
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

install(TARGETS dcnv4_C LIBRARY DESTINATION dcnv4)
