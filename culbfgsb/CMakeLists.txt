append_files (Headers "h" .)
append_files (Sources "cu" .)
append_files (Sources "cpp" .)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fpermissive")
endif()

option(BUILD_CULBFGSB_SHARED "Build Dynamically-linked CULBFGSB" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(BUILD_CULBFGSB_SHARED)
add_definitions(-DCULBFGSB_SHARED)
add_library(cuLBFGSB SHARED
  ${Sources}
  ${Headers})
else(BUILD_CULBFGSB_SHARED)
  add_library(cuLBFGSB STATIC ${Sources} ${Headers})
endif()

# Link cuBLAS and CUDA runtime (replaces cuda_add_cublas_to_target)
# Use full path to cublas library to ensure include dirs are set correctly
find_package(CUDA REQUIRED COMPONENTS cublas)
target_link_libraries(cuLBFGSB PUBLIC ${CUDA_cublas_LIBRARY} ${CUDA_cudart_LIBRARY})
target_include_directories(cuLBFGSB PUBLIC ${CUDA_cublas_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/culbfgsb.h DESTINATION ${CMAKE_BINARY_DIR}/include/culbfgsb)
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_CURRENT_SOURCE_DIR}")