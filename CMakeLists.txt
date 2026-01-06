cmake_minimum_required(VERSION 3.18)

project(CUDAProject LANGUAGES CXX CUDA) #CUDA会自动使用nvcc编译器编译.cu文件

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找 CUDA
find_package(CUDAToolkit REQUIRED)

# 添加可执行文件
add_executable(hello hello.cu)

# 设置 CUDA 架构（RTX 4060 使用 Ada Lovelace 架构 sm_89）
set_target_properties(hello PROPERTIES
    CUDA_ARCHITECTURES "89"
)

# 链接 CUDA 库
target_link_libraries(hello CUDA::cudart)
