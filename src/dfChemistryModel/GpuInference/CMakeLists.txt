cmake_minimum_required(VERSION 3.0)

project(GpuInference)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_library(GpuInference SHARED GpuInference.cpp)
target_compile_features(GpuInference PUBLIC cxx_range_for)
target_link_libraries(GpuInference ${TORCH_LIBRARIES})
set_property(TARGET GpuInference PROPERTY CXX_STANDARD 14)