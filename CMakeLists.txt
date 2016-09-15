project (HetGraph CXX)
cmake_minimum_required (VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

message(STATUS "CUDA LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA LIBRARY:   ${CUDA_CUDA_LIBRARY}")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -Wall -Wextra -Wno-unused-parameter -ggdb")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg -g3 -O3")

set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -lineinfo")
set (CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -O3")
set (CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -pg -g -G -lineinfo")


set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=compute_20")
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_20")
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=compute_35")
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=compute_52")
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

include_directories(${MPI_CXX_INCLUDE_PATH})
include_directories(${CUDA_TOOLKIT_INCLUDE})

set(CUDA_HOST_COMPILER "/usr/bin/g++")

set(LIBS ${LIBS} ${CUDA_CUDA_LIBRARY})
set(LIBS ${LIBS} ${CUDA_CUDART_LIBRARY})

add_definitions(-DUSE_TBB)
set(LIBS ${LIBS} tbb)

cuda_add_executable(main main.cu)
target_link_libraries(main ${LIBS})

cuda_add_executable(convert convert.cu)
target_link_libraries(convert ${LIBS})
