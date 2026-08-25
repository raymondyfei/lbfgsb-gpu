cmake_minimum_required (VERSION 3.15)

project (cuLBFGSB)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

# Initialize the build type (Release, Debug, etc)
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: Debug Release."
    FORCE)
endif (NOT CMAKE_BUILD_TYPE)

add_definitions (-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

# Enable CUDA language explicitly (required with modern CMake)
enable_language(CUDA)

# Use modern FindCUDAToolkit instead of deprecated FindCUDA
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)

# Set NVCC flags for all CUDA compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --default-stream per-thread --expt-relaxed-constexpr -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -Xcudafe --diag_suppress=20014")

# Set CUDA architectures (replaces the old CUDA_NVCC_FLAGS arch settings)
set(CMAKE_CUDA_ARCHITECTURES 72 75 86 90)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  add_definitions (-DDEBUG)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif (CMAKE_BUILD_TYPE MATCHES Debug)

# Add directory with macros
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Add cmake file with useful functions
include (CmakeCommon)

include_directories (${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory (culbfgsb)
add_subdirectory (examples)