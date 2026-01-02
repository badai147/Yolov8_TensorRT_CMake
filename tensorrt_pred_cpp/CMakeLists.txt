cmake_minimum_required(VERSION 3.12)

project(yolov8 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# CUDA 路径
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
set(CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)

# OpenCV 路径
set(OPENCV_ROOT "E:/Amazing/opencv412/opencv")
set(OPENCV_BUILD_DIR "${OPENCV_ROOT}/build")

# TensorRT 路径
set(TENSORRT_ROOT "E:/Amazing/TensorRT-10.13.0.35")

# CUDA 配置
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")

# CUDA 包含目录和库
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
file(GLOB CUDA_LIB_FILES "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/*.lib")
set(CUDA_LIBRARIES ${CUDA_LIB_FILES})

# OpenCV 配置
set(OpenCV_DIR "${OPENCV_BUILD_DIR}")
find_package(OpenCV REQUIRED)

# TensorRT 配置
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
file(GLOB TENSORRT_LIB_FILES "${TENSORRT_ROOT}/lib/*.lib")
set(TENSORRT_LIBRARIES ${TENSORRT_LIB_FILES})

# 项目包含目录
set(PROJECT_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

# 项目库文件
set(PROJECT_LIBRARIES
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARIES}
)

# 项目库目录
set(PROJECT_LIB_DIRS
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
    "${TENSORRT_ROOT}/lib"
)

# 创建可执行文件
add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/yolov8.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/filesystem.hpp
)

# 设置包含目录
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_INCLUDE_DIRS})

# 设置库目录
target_link_directories(${PROJECT_NAME} PRIVATE ${PROJECT_LIB_DIRS})

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_LIBRARIES})
